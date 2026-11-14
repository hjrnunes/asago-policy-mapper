"""Atomic numbering for pipeline-stage LLM call records.

Call sites derive each record's id from how many matching records the
collector already holds (``judge-001``, ``ground-002`` and so on). Doing the
count and the append in separate steps races when stage workers run in
parallel: two threads count the same number and both append it. Each site
keeps its own counting predicate (matching its id format), and
``append_numbered`` runs the count, the build and the append as one critical
section so the ids stay unique.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from asago_policy_mapper.extract.models import LLMCallRecord

_lock = threading.Lock()


def append_numbered(
    collector: list[LLMCallRecord],
    *,
    counts: Callable[[LLMCallRecord], bool],
    build: Callable[[int], LLMCallRecord],
) -> LLMCallRecord:
    """Append one record with a number one past the matching records.

    ``counts`` selects which records the numbering considers, so each call
    site keeps its legacy predicate and id format; ``build`` receives the
    next number and returns the record to append. Counting, building and
    appending happen under one lock, so concurrent workers get unique
    numbers.
    """
    with _lock:
        number = sum(1 for record in collector if counts(record)) + 1
        record = build(number)
        collector.append(record)
        return record
