import threading
import time

from asago_policy_mapper.extract.calllog import append_with_call_id
from asago_policy_mapper.extract.models import LLMCallRecord, _RiskEvidence


def _record(call_id: str, stage: str = "grounding") -> LLMCallRecord:
    return LLMCallRecord(
        call_id=call_id,
        stage=stage,  # type: ignore[arg-type]
        chunk_index=0,
        risk_ids=[],
        messages=[],
        response={},
        duration_ms=0.0,
        result_summary="",
    )


def test_append_with_call_id_returns_sequential_ids():
    collector: list[LLMCallRecord] = []
    for expected in ("x-001", "x-002", "x-003"):
        record = append_with_call_id(
            collector,
            counts=lambda r: r.stage == "grounding",
            build=lambda n: _record(f"x-{n:03d}"),
        )
        assert record.call_id == expected
    assert [r.call_id for r in collector] == ["x-001", "x-002", "x-003"]


def test_append_with_call_id_is_unique_under_threads():
    """Counting and appending happen as one critical section.

    32 threads hit the collector behind a barrier with a build that sleeps,
    so a check-then-append race would produce duplicate ids.
    """
    collector: list[LLMCallRecord] = []
    thread_count = 32
    barrier = threading.Barrier(thread_count)

    def build(n: int) -> LLMCallRecord:
        time.sleep(0.01)
        return _record(f"race-{n:03d}")

    def worker() -> None:
        barrier.wait()
        append_with_call_id(collector, counts=lambda r: r.stage == "grounding", build=build)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ids = sorted(r.call_id for r in collector)
    assert len(collector) == thread_count
    assert ids == [f"race-{n:03d}" for n in range(1, thread_count + 1)]


def test_record_ids_keep_legacy_counting():
    """An expansion record makes the next grounding id skip a number, as today.

    Exercises the production grounding site, not a hand-rolled predicate, so a
    change to the site's counting predicate fails here.
    """
    from asago_policy_mapper.extract.attribute import _record_grounding_call

    verdicts = [
        _RiskEvidence(risk_id="R-001", grounded=True, confidence="high", quotes=["text"]),
    ]
    collector: list[LLMCallRecord] = [_record("ground-expand-001", stage="grounding")]

    _record_grounding_call(
        collector,
        chunk_index=0,
        risk_ids=["R-001"],
        messages=[],
        verdicts=verdicts,
        duration_ms=0.0,
        grounded_count=1,
        batch_size=1,
    )

    assert [r.call_id for r in collector] == ["ground-expand-001", "ground-002"]


def test_append_with_call_id_counts_by_predicate():
    """Each site counts only its own stage; other stages leave numbering alone."""
    collector: list[LLMCallRecord] = [
        _record("judge-001", stage="judge"),
        _record("ground-001", stage="grounding"),
        _record("ground-variant-001", stage="variant_grounding"),
    ]

    record = append_with_call_id(
        collector,
        counts=lambda r: "expand" in r.call_id,
        build=lambda n: _record(f"ground-expand-{n:03d}", stage="grounding"),
    )

    assert record.call_id == "ground-expand-001"
    record = append_with_call_id(
        collector,
        counts=lambda r: r.stage == "judge",
        build=lambda n: _record(f"judge-{n:03d}", stage="judge"),
    )
    assert record.call_id == "judge-002"
    record = append_with_call_id(
        collector,
        counts=lambda r: r.stage == "variant_grounding",
        build=lambda n: _record(f"ground-variant-{n:03d}", stage="variant_grounding"),
    )
    assert record.call_id == "ground-variant-002"
    record = append_with_call_id(
        collector,
        counts=lambda r: r.stage == "causal_synthesis",
        build=lambda n: _record(f"causal-{n:03d}", stage="causal_synthesis"),
    )
    assert record.call_id == "causal-001"
