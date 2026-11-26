#!/usr/bin/env python3
"""Record prefilter answers for saved fixtures, using a probability endpoint.

Runs an HTTP probability assessor over every fixture row from
``build_prefilter_fixtures.py`` and writes the answers file that
``ReplayAssessor`` and ``eval_prefilter.py`` read: one JSONL row per
(context_sha256, risk_id) carrying either ``p`` or an ``unavailable``
reason, plus the ``latency_ms`` of the assessment that produced it. The
latency is the provider's own round-trip time, so the offline evaluation
can report real latency percentiles instead of replay lookup time.

Answers are appended to the file one fixture row at a time and flushed
immediately, so a mid-run crash keeps every row already written instead
of losing the whole run. A re-run resumes: keys already present in the
answers file are skipped, so recorded answers are never re-requested
from the provider and no policy text is re-sent.

Usage:
    python scripts/record_prefilter_answers.py <fixtures.jsonl> <answers.jsonl> \
        --base-url URL --model MODEL [--api-key-env VAR] [--timeout-s S] [--max-batch N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
from asago_policy_mapper.extract.screening import (
    RELEVANCE_QUESTION_V1,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
)


def record_answers(rows: Sequence[dict], assessor) -> list[dict]:
    """Assess every fixture row's candidates and turn the batch into answer rows.

    Each answer row carries the fixture row's key pair, the answer
    (``p`` or an ``unavailable`` reason) and the assessment's
    ``latency_ms``. Candidates the assessor omitted entirely become
    ``unavailable: "missing"`` rows, so the answers file always covers
    every candidate the fixtures asked about.
    """
    answers: list[dict] = []
    for row in rows:
        cases = [
            DecisionCase(risk_id=c["risk_id"], name=c["name"], description=c["description"]) for c in row["candidates"]
        ]
        batch = assessor.assess(RELEVANCE_QUESTION_V1, row["context"], cases)
        for case in cases:
            case_answers = batch.answers.get(case.risk_id, ())
            answer = {"context_sha256": row["context_sha256"], "risk_id": case.risk_id, "latency_ms": batch.latency_ms}
            probability = next((o for o in case_answers if isinstance(o, ProbabilityAnswer)), None)
            unavailable = next((o for o in case_answers if isinstance(o, NoAnswer)), None)
            if unavailable is not None:
                answer["unavailable"] = unavailable.reason
                if unavailable.detail:
                    answer["detail"] = unavailable.detail
            elif probability is not None:
                answer["p"] = probability.probability
            else:
                answer["unavailable"] = "missing"
            answers.append(answer)
    return answers


def _row_answer_keys(row: dict) -> set[tuple[str, str]]:
    """The (context_sha256, risk_id) pairs one fixture row will record."""
    return {(row["context_sha256"], c["risk_id"]) for c in row["candidates"]}


def recorded_keys(path: Path) -> set[tuple[str, str]]:
    """Read the (context_sha256, risk_id) pairs an answers file already holds.

    Unreadable lines are ignored: the file is written by this script one
    line at a time, and the only corrupt line a crash can leave is the
    one being written when it happened.
    """
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            key, risk_id = raw.get("context_sha256"), raw.get("risk_id")
            if isinstance(key, str) and isinstance(risk_id, str):
                keys.add((key, risk_id))
    return keys


def record_to_file(rows: Sequence[dict], assessor, path: Path) -> tuple[int, int]:
    """Record answers incrementally, appending each fixture row's output as it is assessed.

    Returns (written, skipped): the answer rows written this run and the
    candidate keys already present in the file, which this run skipped so
    no already-recorded answer is re-requested from the provider. Each
    fixture row's rows are written and flushed before the next row is
    assessed, so a crash loses at most the row in flight.
    """
    done = recorded_keys(path)
    written = 0
    skipped = 0
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            row_keys = _row_answer_keys(row)
            pending = row_keys - done
            if not pending:
                skipped += len(row_keys)
                continue
            answers = record_answers([row], assessor)
            for answer in answers:
                handle.write(json.dumps(answer) + "\n")
            handle.flush()
            done |= row_keys
            written += len(answers)
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record prefilter answers for saved fixtures")
    parser.add_argument("fixtures", type=Path, help="fixtures JSONL from build_prefilter_fixtures.py")
    parser.add_argument("answers", type=Path, help="output answers JSONL for ReplayAssessor and eval_prefilter.py")
    parser.add_argument("--base-url", required=True, help="probability endpoint base URL")
    parser.add_argument("--model", required=True, help="model name the endpoint serves")
    parser.add_argument(
        "--api-key-env", default="PROB_API_KEY", help="env var holding the API key (default: %(default)s)"
    )
    parser.add_argument(
        "--timeout-s", type=float, default=10.0, help="per-request timeout in seconds (default: %(default)s)"
    )
    parser.add_argument("--max-batch", type=int, default=32, help="candidates per request (default: %(default)s)")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.fixtures.read_text().splitlines() if line.strip()]
    if not rows:
        print(f"Error: {args.fixtures} has no fixture rows", file=sys.stderr)
        return 1

    endpoint = ProbabilityEndpoint(
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_s=args.timeout_s,
        max_batch=args.max_batch,
    )
    assessor = HttpProbabilityAssessor(endpoint)
    written, skipped = record_to_file(rows, assessor, args.answers)
    print(f"Wrote {written} answer rows ({skipped} already recorded, skipped) to {args.answers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
