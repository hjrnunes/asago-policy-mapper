#!/usr/bin/env python3
"""Record prefilter answers for saved fixtures, using a probability endpoint.

Runs an HTTP probability assessor over every fixture row from
``build_prefilter_fixtures.py`` and writes the answers file that
``ReplayAssessor`` and ``eval_prefilter.py`` read: one JSONL row per
(context_sha256, risk_id) carrying either ``p`` or an ``unavailable``
reason, plus the ``latency_ms`` of the assessment that produced it. The
latency is the provider's own round-trip time, so the offline evaluation
can report real latency percentiles instead of replay lookup time.

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
    answers = record_answers(rows, assessor)
    with open(args.answers, "w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(answer) + "\n")
    scored = sum(1 for a in answers if "p" in a)
    print(f"Wrote {len(answers)} answers ({scored} scored) to {args.answers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
