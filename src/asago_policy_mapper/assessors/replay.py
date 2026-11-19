"""A replay assessor that serves stored outcomes from a JSONL file.

The offline evaluation (``scripts/eval_prefilter.py``) uses this assessor
to sweep thresholds and compare rules on saved outcomes without contacting
any provider. Rows are keyed by ``(context_sha256, risk_id)``, which the
fixture builder writes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from asago_policy_mapper.extract.decide import (
    AssessmentBatch,
    BinaryTask,
    DecisionCase,
    Outcome,
    Probability,
    Unavailable,
)


def context_sha256(context: str) -> str:
    """Hash a context string to the key the stored outcomes use."""
    return hashlib.sha256(context.encode()).hexdigest()


def _parse_row(raw: object) -> tuple[str, str, Outcome] | None:
    """Turn one stored row into its key and outcome, or None when unusable."""
    if not isinstance(raw, dict):
        return None
    key = raw.get("context_sha256")
    risk_id = raw.get("risk_id")
    if not isinstance(key, str) or not isinstance(risk_id, str):
        return None
    if "unavailable" in raw:
        reason = raw.get("unavailable")
        if reason in ("missing", "invalid", "timeout", "error"):
            return key, risk_id, Unavailable(reason)
        return None
    p = raw.get("p")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        return None
    return key, risk_id, Probability(float(p))


class ReplayAssessor:
    """A ``BinaryAssessor`` that answers from stored outcomes, offline.

    Emits ``Probability`` for rows with a valid ``p`` and ``Unavailable``
    for rows carrying an ``unavailable`` reason or for keys with no row at
    all. The file is read once, at construction. A missing or empty file
    raises ``ValueError``: an evaluation with nothing to replay would
    silently mark every candidate missing. Unusable lines are counted in
    ``malformed_lines`` and reported, not skipped quietly.
    """

    emits = frozenset({Probability, Unavailable})

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.assessor_id = f"replay:{self.path.name}"
        self.offsite = False
        self.malformed_lines = 0
        self._outcomes = self._load()

    def assess(self, task: BinaryTask, context: str, cases: Sequence[DecisionCase]) -> AssessmentBatch:
        t0 = time.perf_counter()
        key = context_sha256(context)
        outcomes: dict[str, tuple[Outcome, ...]] = {
            case.risk_id: tuple(self._outcomes.get((key, case.risk_id), ())) or (Unavailable("missing"),)
            for case in cases
        }
        latency_ms = (time.perf_counter() - t0) * 1000
        return AssessmentBatch(
            outcomes=outcomes,
            unmatched_ids=(),
            latency_ms=latency_ms,
            wire=None,
        )

    def _load(self) -> dict[tuple[str, str], list[Outcome]]:
        rows: dict[tuple[str, str], list[Outcome]] = {}
        if not self.path.exists():
            raise ValueError(f"answers file not found: {self.path}")
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:
                    self.malformed_lines += 1
                    continue
                parsed = _parse_row(raw)
                if parsed is None:
                    self.malformed_lines += 1
                    continue
                key, risk_id, outcome = parsed
                rows.setdefault((key, risk_id), []).append(outcome)
        if not rows:
            raise ValueError(f"answers file has no usable rows: {self.path}")
        if self.malformed_lines:
            print(
                f"Warning: {self.malformed_lines} malformed line(s) skipped in {self.path}",
                file=sys.stderr,
            )
        return rows
