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


_MAX_LATENCY_MS = 1e12
"""Upper bound for a recorded latency: far above any real round-trip, and
small enough that float() conversion cannot overflow."""


def _parse_row(raw: object) -> tuple[str, str, Outcome, float | None] | None:
    """Turn one stored row into its key, outcome and recorded latency, or None when unusable.

    Rows may carry ``latency_ms`` (the assessment cost the recorder
    measured); it is optional and absent from hand-written files. A
    latency that is not a number, is negative, NaN or absurdly large is
    treated as absent rather than poisoning the batch's reported latency.
    """
    if not isinstance(raw, dict):
        return None
    key = raw.get("context_sha256")
    risk_id = raw.get("risk_id")
    if not isinstance(key, str) or not isinstance(risk_id, str):
        return None
    latency = raw.get("latency_ms")
    recorded_ms = None
    if isinstance(latency, (int, float)) and not isinstance(latency, bool) and 0 <= latency <= _MAX_LATENCY_MS:
        recorded_ms = float(latency)
    if "unavailable" in raw:
        reason = raw.get("unavailable")
        if reason in ("missing", "invalid", "timeout", "error"):
            detail = raw.get("detail")
            outcome = Unavailable(reason, detail) if isinstance(detail, str) and detail else Unavailable(reason)
            return key, risk_id, outcome, recorded_ms
        return None
    p = raw.get("p")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        return None
    return key, risk_id, Probability(float(p)), recorded_ms


class ReplayAssessor:
    """A ``BinaryAssessor`` that answers from stored outcomes, offline.

    Emits ``Probability`` for rows with a valid ``p`` and ``Unavailable``
    for rows carrying an ``unavailable`` reason or for keys with no row at
    all. The file is read once, at construction. A missing path raises
    ``FileNotFoundError`` and a file with no usable rows raises
    ``ValueError``: an evaluation with nothing to replay would silently
    mark every candidate missing. Unusable lines are counted in
    ``malformed_lines`` and reported, not skipped quietly.

    Rows recorded by the answers recorder carry ``latency_ms``; a batch
    drawn from them reports the recorded assessment cost (the maximum the
    batch's rows carry), so an offline evaluation reports real latency.
    Batches with no recorded latency fall back to measuring the replay
    lookup, which carries no provider meaning; ``last_recorded`` says
    whether the previous batch's latency was recorded.
    """

    emits = frozenset({Probability, Unavailable})

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.assessor_id = f"replay:{self.path.name}"
        self.offsite = False
        self.malformed_lines = 0
        self.last_recorded = False
        self._latencies: dict[tuple[str, str], float] = {}
        self._outcomes = self._load()

    @property
    def recorded_latencies(self) -> bool:
        """Whether any loaded row carries recorded latency."""
        return bool(self._latencies)

    def assess(self, task: BinaryTask, context: str, cases: Sequence[DecisionCase]) -> AssessmentBatch:
        t0 = time.perf_counter()
        key = context_sha256(context)
        outcomes: dict[str, tuple[Outcome, ...]] = {
            case.risk_id: tuple(self._outcomes.get((key, case.risk_id), ())) or (Unavailable("missing"),)
            for case in cases
        }
        recorded = [self._latencies[(key, case.risk_id)] for case in cases if (key, case.risk_id) in self._latencies]
        self.last_recorded = bool(recorded)
        if recorded:
            latency_ms = max(recorded)
        else:
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
            raise FileNotFoundError(f"answers file not found: {self.path}")
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
                key, risk_id, outcome, recorded_ms = parsed
                rows.setdefault((key, risk_id), []).append(outcome)
                if recorded_ms is not None:
                    # Max on duplicates, matching the batch reduction in assess().
                    prior = self._latencies.get((key, risk_id))
                    self._latencies[(key, risk_id)] = max(prior, recorded_ms) if prior is not None else recorded_ms
        if not rows:
            raise ValueError(f"answers file has no usable rows: {self.path}")
        if self.malformed_lines:
            print(
                f"Warning: {self.malformed_lines} malformed line(s) skipped in {self.path}",
                file=sys.stderr,
            )
        return rows
