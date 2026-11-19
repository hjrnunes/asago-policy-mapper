#!/usr/bin/env python3
"""Evaluate the prefilter threshold offline, on saved fixtures.

Sweeps the drop threshold on a tuning split of the fixture policies and
reports drop rate, recall loss against the recorded grounded ids, bypass
rate and latency percentiles on a held-out split. Policies are assigned
to splits by the fixed lists below, so the threshold is chosen on
different policies than the ones it is reported on. All assessments run
through the replay assessor, so no provider is contacted.

Usage:
    python scripts/eval_prefilter.py <fixtures.jsonl> <outcomes.jsonl> [--report FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from asago_policy_mapper.assessors.replay import ReplayAssessor
from asago_policy_mapper.extract.decide import (
    RISK_RELEVANCE_V1,
    DecisionCase,
    Probability,
    Unavailable,
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


def _effective_p(outcomes: list) -> float | None:
    """Reduce one candidate's stored outcomes to the probability the rule sees.

    Mirrors ThresholdDrop: an Unavailable anywhere bypasses (None marks
    that), otherwise the maximum probability decides (any high one keeps).
    """
    probabilities = [o.p for o in outcomes if isinstance(o, Probability)]
    if any(isinstance(o, Unavailable) for o in outcomes) or not probabilities:
        return None
    return max(probabilities)


def _outcome_lookups(assessor: ReplayAssessor) -> tuple[dict[tuple[str, str], float], set[tuple[str, str]]]:
    """Flatten the replay store into effective-p lookups and bypass keys, keyed by sha."""
    outcomes: dict[tuple[str, str], float] = {}
    unavailable: set[tuple[str, str]] = set()
    for key, batch in assessor._outcomes.items():
        effective = _effective_p(batch)
        if effective is None:
            unavailable.add(key)
        else:
            outcomes[key] = effective
    return outcomes, unavailable


def threshold_metrics(
    rows: list[dict],
    outcomes: dict[tuple[str, str], float],
    unavailable: set[tuple[str, str]],
    threshold: float,
) -> dict:
    """Measure one threshold over fixture rows: drops, recall loss, bypasses.

    Every candidate at or below the threshold drops; unavailable or absent
    outcomes keep the candidate (bypass), mirroring the rule the pipeline
    will apply. Recall loss counts grounded candidates the threshold
    dropped, over all grounded candidates. Outcome keys are the rows'
    ``context_sha256`` values.
    """
    candidates = 0
    dropped = 0
    grounded_total = 0
    grounded_dropped = 0
    bypassed = 0
    for row in rows:
        grounded = set(row["grounded_ids"])
        for candidate in row["candidates"]:
            candidates += 1
            rid = candidate["risk_id"]
            if rid in grounded:
                grounded_total += 1
            lookup = (row["context_sha256"], rid)
            if lookup in unavailable:
                bypassed += 1
                continue
            p = outcomes.get(lookup)
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


def choose_threshold(sweep: list[dict]) -> dict:
    """Pick the threshold with the least recall loss, breaking ties by most drops."""
    return min(sweep, key=lambda m: (m["recall_loss"], -m["dropped"], m["threshold"]))


def _row_latencies(rows: list[dict], assessor: ReplayAssessor) -> list[float]:
    """One replay latency per row, measuring the row's assessment."""
    latencies = []
    for row in rows:
        cases = [
            DecisionCase(risk_id=c["risk_id"], name=c["name"], description=c["description"]) for c in row["candidates"]
        ]
        batch = assessor.assess(RISK_RELEVANCE_V1, row["context"], cases)
        latencies.append(batch.latency_ms)
    return latencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the prefilter threshold offline")
    parser.add_argument("fixtures", type=Path, help="fixtures JSONL from build_prefilter_fixtures.py")
    parser.add_argument("outcomes", type=Path, help="outcomes JSONL keyed by (context_sha256, risk_id)")
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here (default: stdout only)")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.fixtures.read_text().splitlines() if line.strip()]
    try:
        assessor = ReplayAssessor(args.outcomes)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    outcomes, unavailable = _outcome_lookups(assessor)

    tuning, held_out = split_rows(rows, TUNING_POLICIES, HELD_OUT_POLICIES)
    if not tuning or not held_out:
        print("Error: both splits must be non-empty; check the policy lists", file=sys.stderr)
        return 1

    sweep = [threshold_metrics(tuning, outcomes, unavailable, t) for t in SWEEP_THRESHOLDS]
    chosen = choose_threshold(sweep)
    held_metrics = threshold_metrics(held_out, outcomes, unavailable, chosen["threshold"])
    latencies = _row_latencies(held_out, assessor)
    held_metrics["latency_p50_ms"] = percentile(latencies, 0.5)
    held_metrics["latency_p95_ms"] = percentile(latencies, 0.95)

    report = {
        "tuning_policies": sorted(TUNING_POLICIES),
        "held_out_policies": sorted(HELD_OUT_POLICIES),
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
