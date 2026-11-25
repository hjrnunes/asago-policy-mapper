#!/usr/bin/env python3
"""Evaluate the prefilter threshold offline, on saved fixtures.

Sweeps the drop threshold on a tuning split of the fixture policies and
reports drop rate, recall loss against the recorded grounded ids, bypass
rate and latency percentiles on a held-out split. Policies are assigned
to splits by the fixed lists below, so the threshold is chosen on
different policies than the ones it is reported on. All assessments run
through the replay assessor, so no provider is contacted; latency
percentiles come from the ``latency_ms`` the answers recorder stored with
each answer (the recorded assessment cost per row), and are null when the
answers file carries none.

Threshold choice: the sweep picks the threshold with the least recall
loss on the tuning split, breaking ties by the most drops, then the
smallest threshold. Pass ``--max-recall-loss`` to instead keep the most
aggressive threshold whose recall loss stays within a budget.

Usage:
    python scripts/eval_prefilter.py <fixtures.jsonl> <answers.jsonl> [--report FILE] [--max-recall-loss F]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from asago_policy_mapper.assessors.replay import ReplayAssessor
from asago_policy_mapper.extract.screening import (
    RELEVANCE_QUESTION_V1,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
)

TUNING_POLICIES = frozenset(
    {
        "amadeus",
        "aus-gov",
        "cisco-supplier",
        "dhs-gov",
        "ebay",
        "firstsource",
        "gray",
        "icrc",
        "leicestershire_police",
        "llvm",
        "lse-marking",
        "npcc",
        "prosus",
        "sap",
    }
)
"""Policies the threshold sweep tunes on (the risk-selected battery, part 1)."""

HELD_OUT_POLICIES = frozenset(
    {
        "ars",
        "camden-borough-work",
        "eu-com",
        "fs-isac",
        "guy-nhs",
        "lenovo",
        "lse-legreg",
        "new-york-state",
        "ovic",
        "penn",
        "rdash-nhs",
        "st-johns",
        "vps",
    }
)
"""Policies the chosen threshold is reported on (the risk-selected battery, part 2)."""

SWEEP_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
"""Thresholds to sweep on the tuning split."""


def split_rows(rows: list[dict], tuning: frozenset[str], held_out: frozenset[str]) -> tuple[list[dict], list[dict]]:
    """Partition fixture rows by their policy's split; unknown policies land in neither."""
    tune = [r for r in rows if r["policy"] in tuning]
    hold = [r for r in rows if r["policy"] in held_out]
    return tune, hold


def _effective_p(answers: tuple) -> float | None:
    """Reduce one candidate's stored answers to the probability the rule sees.

    Mirrors DropThreshold: a NoAnswer anywhere bypasses (None marks
    that), a NaN probability cannot be compared and bypasses the same way
    (the rule keeps it with no_answer:invalid), otherwise the maximum
    probability decides (any high one keeps).
    """
    probabilities = [o.probability for o in answers if isinstance(o, ProbabilityAnswer)]
    if any(isinstance(o, NoAnswer) for o in answers) or not probabilities:
        return None
    if any(math.isnan(p) for p in probabilities):
        return None
    return max(probabilities)


_AssessedRow = tuple[dict, dict[str, float | None], float | None]
"""One assessed fixture row: the row, per-candidate effective p, recorded latency."""


def _assess_rows(rows: list[dict], assessor: ReplayAssessor) -> list[_AssessedRow]:
    """Assess every row through the assessor's public API.

    Returns one entry per row: the row itself, each candidate's effective
    probability (None for bypass), and the row's recorded assessment cost
    (None when the answers file carried no latency for it, in which case
    the report omits that row from the latency percentiles).
    """
    assessed = []
    for row in rows:
        cases = [
            DecisionCase(risk_id=c["risk_id"], name=c["name"], description=c["description"]) for c in row["candidates"]
        ]
        batch = assessor.assess(RELEVANCE_QUESTION_V1, row["context"], cases)
        effective = {case.risk_id: _effective_p(batch.answers.get(case.risk_id, ())) for case in cases}
        assessed.append((row, effective, batch.latency_ms if batch.latency_recorded else None))
    return assessed


def threshold_metrics(
    rows: list[tuple[dict, dict[str, float | None], float]],
    threshold: float,
) -> dict:
    """Measure one threshold over assessed fixture rows: drops, recall loss, bypasses.

    Every candidate at or below the threshold drops; a None effective
    probability (unavailable or absent) keeps the candidate (bypass),
    mirroring the rule the pipeline applies. Recall loss counts grounded
    candidates the threshold dropped, over all grounded candidates.
    """
    candidates = 0
    dropped = 0
    grounded_total = 0
    grounded_dropped = 0
    bypassed = 0
    for row, effective, _latency in rows:
        grounded = set(row["grounded_ids"])
        for candidate in row["candidates"]:
            candidates += 1
            rid = candidate["risk_id"]
            if rid in grounded:
                grounded_total += 1
            p = effective.get(rid)
            if p is None:
                bypassed += 1
                continue
            if p <= threshold:
                dropped += 1
                if rid in grounded:
                    grounded_dropped += 1
    return {
        "threshold": threshold,
        "candidates": candidates,
        "dropped": dropped,
        "drop_rate": dropped / candidates if candidates else 0.0,
        "grounded_total": grounded_total,
        "grounded_dropped": grounded_dropped,
        "recall_loss": grounded_dropped / grounded_total if grounded_total else 0.0,
        "bypassed": bypassed,
        "bypass_rate": bypassed / candidates if candidates else 0.0,
    }


def percentile(values: list[float], fraction: float) -> float:
    """The nearest-rank percentile of a list; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def choose_threshold(sweep: list[dict], max_recall_loss: float | None = None) -> dict:
    """Pick the threshold with the least recall loss, breaking ties by most drops.

    With ``max_recall_loss``, pick instead the most aggressive threshold
    whose recall loss stays within the budget (most drops, then smallest
    threshold); when no threshold fits the budget, fall back to the
    least-loss rule.
    """
    if max_recall_loss is not None:
        within = [m for m in sweep if m["recall_loss"] <= max_recall_loss]
        if within:
            return min(within, key=lambda m: (-m["dropped"], m["threshold"]))
    return min(sweep, key=lambda m: (m["recall_loss"], -m["dropped"], m["threshold"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the prefilter threshold offline",
        epilog=(
            "Threshold choice: least recall loss on the tuning split, then most "
            "drops, then the smallest threshold. With --max-recall-loss, the most "
            "aggressive threshold whose recall loss stays within the budget."
        ),
    )
    parser.add_argument("fixtures", type=Path, help="fixtures JSONL from build_prefilter_fixtures.py")
    parser.add_argument("answers", type=Path, help="answers JSONL keyed by (context_sha256, risk_id)")
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here (default: stdout only)")
    parser.add_argument(
        "--max-recall-loss",
        type=float,
        default=None,
        help="recall-loss budget for the chosen threshold (default: none; pick least recall loss)",
    )
    args = parser.parse_args(argv)
    if args.max_recall_loss is not None and (math.isnan(args.max_recall_loss) or args.max_recall_loss < 0):
        parser.error(f"--max-recall-loss must be a non-negative number; got {args.max_recall_loss!r}")

    rows = [json.loads(line) for line in args.fixtures.read_text().splitlines() if line.strip()]
    try:
        assessor = ReplayAssessor(args.answers)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    tuning, held_out = split_rows(rows, TUNING_POLICIES, HELD_OUT_POLICIES)
    if not tuning or not held_out:
        print("Error: both splits must be non-empty; check the policy lists", file=sys.stderr)
        return 1

    tuning_assessed = _assess_rows(tuning, assessor)
    held_assessed = _assess_rows(held_out, assessor)
    sweep = [threshold_metrics(tuning_assessed, t) for t in SWEEP_THRESHOLDS]
    chosen = choose_threshold(sweep, args.max_recall_loss)
    held_metrics = threshold_metrics(held_assessed, chosen["threshold"])
    latencies = [latency for _row, _effective, latency in held_assessed if latency is not None]
    held_metrics["latency_p50_ms"] = percentile(latencies, 0.5) if latencies else None
    held_metrics["latency_p95_ms"] = percentile(latencies, 0.95) if latencies else None
    held_metrics["latency_recorded"] = bool(latencies)

    report = {
        "tuning_policies": sorted(TUNING_POLICIES),
        "held_out_policies": sorted(HELD_OUT_POLICIES),
        "max_recall_loss": args.max_recall_loss,
        "sweep": sweep,
        "chosen_threshold": chosen["threshold"],
        "held_out": held_metrics,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
