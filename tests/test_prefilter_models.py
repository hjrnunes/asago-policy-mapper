"""Model tests for the opt-in pre-grounding filter (FilteredCandidate, DecisionCallRecord, RetrievalStats).

The filter adds three optional fields to ``FilteredCandidate`` —
``dropped_by``, ``score`` and ``reason`` — which are set only when the
prefilter dropped the candidate; grounding rejections leave all three
``None``. ``ExtractionResult`` gains ``decision_calls``, a list of
``DecisionCallRecord`` objects holding the stage, chunk index, decision
log and exact request for each prefilter call, plus ``prefilter_ran``,
which carries the filter's presence (an empty ``decision_calls`` list
means the filter ran with nothing to assess); ``RetrievalStats`` gains
``prefilter_dropped``, the count of candidates the filter removed
(``grounding_filtered`` counts only grounding's own rejections). With
the filter off, none of the new fields may appear in serialized output,
so runs stay byte-identical to today's.
"""

import json

import pytest
from pydantic import ValidationError

from asago_policy_mapper.extract.models import (
    DecisionCallRecord,
    DecisionItem,
    DecisionLog,
    ExtractionResult,
    FilteredCandidate,
    RetrievalStats,
)
from asago_policy_mapper.extract.screening import AnswerBatch


def _result(**extra) -> ExtractionResult:
    return ExtractionResult(
        risks=[],
        source_documents=["policy.pdf"],
        retrieval_stats=RetrievalStats(
            total_chunks=0,
            total_candidates_retrieved=0,
            auto_accepted=0,
            llm_judged=0,
            grounding_filtered=0,
        ),
        **extra,
    )


def _decision_log() -> DecisionLog:
    return DecisionLog(
        question_id="risk-relevance",
        question_version="1",
        prompt_name="prefilter",
        prompt_sha256="0" * 64,
        assessor_id="http-probability:test-model",
        rule_id="drop-threshold/1",
        latency_ms=3.5,
        items=[
            DecisionItem(
                risk_id="R-001",
                answers=[{"kind": "probability", "probability": 0.05}],
                keep=False,
                reason="p<=0.10",
            )
        ],
        unmatched_ids=[],
    )


def _call_record(**overrides) -> DecisionCallRecord:
    fields = dict(
        stage="prefilter",
        chunk_index=2,
        request={"context": "policy text", "items": [{"id": "R-001", "text": "Model Bias: errors"}]},
        decision=_decision_log(),
    )
    fields.update(overrides)
    return DecisionCallRecord(**fields)


# --- FilteredCandidate: the three new optional fields ---


def test_filtered_candidate_defaults_match_grounding_rejection():
    """A grounding rejection constructs exactly as before the filter existed."""
    fc = FilteredCandidate(
        risk_id="R-001",
        risk_name="Model Bias",
        taxonomy="test",
        cross_encoder_score=0.9,
        rrf_score=0.03,
        bm25_rank=1,
        accepted_by="threshold",
        chunk_index=0,
    )
    assert fc.dropped_by is None
    assert fc.score is None
    assert fc.reason is None


def test_filtered_candidate_records_prefilter_drop():
    """A prefilter drop carries who dropped it, the score and the rule reason."""
    fc = FilteredCandidate(
        risk_id="R-001",
        risk_name="Model Bias",
        taxonomy="test",
        cross_encoder_score=0.9,
        rrf_score=0.03,
        bm25_rank=1,
        accepted_by="threshold",
        chunk_index=0,
        dropped_by="prefilter",
        score=0.05,
        reason="p<=0.10",
    )
    assert fc.dropped_by == "prefilter"
    assert fc.score == 0.05
    assert fc.reason == "p<=0.10"


def test_filtered_candidate_omits_unset_fields_when_dumped():
    """Grounding rejections serialize without any of the new fields.

    This is what keeps filter-off (and grounding-only) runs byte-identical:
    a candidate with no prefilter provenance must not grow keys.
    """
    fc = FilteredCandidate(
        risk_id="R-001",
        risk_name="Model Bias",
        taxonomy="test",
        cross_encoder_score=0.9,
        rrf_score=0.03,
        bm25_rank=1,
        accepted_by="threshold",
        chunk_index=0,
    )
    dumped = fc.model_dump()
    for key in ("dropped_by", "score", "reason"):
        assert key not in dumped
    assert set(dumped) == {
        "risk_id",
        "risk_name",
        "taxonomy",
        "cross_encoder_score",
        "rrf_score",
        "bm25_rank",
        "accepted_by",
        "chunk_index",
    }
    assert "dropped_by" not in json.loads(fc.model_dump_json())


def test_filtered_candidate_dropped_roundtrips():
    """A dropped candidate serializes its provenance and reads back unchanged."""
    fc = FilteredCandidate(
        risk_id="R-002",
        risk_name="Data Poisoning",
        taxonomy="test",
        cross_encoder_score=0.8,
        accepted_by="llm_judge",
        chunk_index=3,
        dropped_by="prefilter",
        score=0.01,
        reason="p<=0.10",
    )
    restored = FilteredCandidate(**fc.model_dump())
    assert restored == fc
    assert "dropped_by" in fc.model_dump()


# --- RetrievalStats.prefilter_dropped ---


def test_retrieval_stats_omits_prefilter_dropped_when_off():
    """Filter off: the serialized stats carry no prefilter_dropped key.

    This is what keeps filter-off runs byte-identical to pre-filter runs.
    """
    stats = RetrievalStats(
        total_chunks=0,
        total_candidates_retrieved=0,
        auto_accepted=0,
        llm_judged=0,
        grounding_filtered=0,
    )
    assert stats.prefilter_dropped is None
    dumped = stats.model_dump()
    assert "prefilter_dropped" not in dumped
    assert "prefilter_dropped" not in json.loads(stats.model_dump_json())
    # The full key set stays exactly the pre-filter set: any new
    # always-serialized key would change filter-off output.
    assert set(dumped) == {
        "total_chunks",
        "total_candidates_retrieved",
        "auto_accepted",
        "llm_judged",
        "auto_promoted",
        "grounding_filtered",
        "timing_ms",
    }


def test_retrieval_stats_records_prefilter_dropped_when_set():
    """Filter on: the count appears, even when the filter dropped nothing."""
    stats = RetrievalStats(
        total_chunks=0,
        total_candidates_retrieved=0,
        auto_accepted=0,
        llm_judged=0,
        grounding_filtered=0,
        prefilter_dropped=0,
    )
    assert stats.prefilter_dropped == 0
    assert stats.model_dump()["prefilter_dropped"] == 0

    stats = RetrievalStats(
        total_chunks=1,
        total_candidates_retrieved=4,
        auto_accepted=4,
        llm_judged=0,
        grounding_filtered=1,
        prefilter_dropped=2,
    )
    dumped = stats.model_dump()
    assert dumped["prefilter_dropped"] == 2
    # The two counters never overlap: grounding_filtered counts only what
    # grounding itself rejected.
    restored = RetrievalStats(**dumped)
    assert restored.prefilter_dropped == 2
    assert restored.grounding_filtered == 1


def test_extraction_result_stats_prefilter_dropped_roundtrip():
    """A filter-on result round-trips the drop count through serialization."""
    result = ExtractionResult(
        risks=[],
        source_documents=["policy.pdf"],
        retrieval_stats=RetrievalStats(
            total_chunks=0,
            total_candidates_retrieved=0,
            auto_accepted=0,
            llm_judged=0,
            grounding_filtered=0,
            prefilter_dropped=3,
        ),
    )
    restored = ExtractionResult(**result.model_dump())
    assert restored.retrieval_stats.prefilter_dropped == 3


# --- DecisionCallRecord ---


def test_decision_call_record_holds_stage_chunk_request_and_log():
    record = _call_record()
    assert record.stage == "prefilter"
    assert record.chunk_index == 2
    assert record.request["items"][0]["id"] == "R-001"
    assert record.decision.rule_id == "drop-threshold/1"


def test_decision_call_record_request_needs_no_credentials():
    """The request is plain data: any dict serializes without a key field."""
    record = _call_record()
    dumped = json.dumps(record.model_dump()["request"])
    assert "authorization" not in dumped.lower()
    assert "api_key" not in dumped.lower()


def test_decision_call_record_roundtrip():
    record = _call_record()
    restored = DecisionCallRecord(**record.model_dump())
    assert restored == record


def test_decision_call_record_requires_stage_and_request():
    """Stage, chunk_index and request are required; the log may be absent."""
    with pytest.raises(ValidationError):
        DecisionCallRecord(chunk_index=0, request={})
    with pytest.raises(ValidationError):
        DecisionCallRecord(stage="prefilter", chunk_index=0)
    bare = DecisionCallRecord(stage="prefilter", chunk_index=0, request={"items": []})
    assert bare.decision is None


def test_decision_call_record_serializes_decision_log():
    dumped = _call_record().model_dump()
    assert dumped["decision"]["rule_id"] == "drop-threshold/1"
    assert dumped["decision"]["items"][0]["risk_id"] == "R-001"


# --- ExtractionResult.decision_calls ---


def test_extraction_result_decision_calls_default_empty():
    result = _result()
    assert result.decision_calls == []


def test_extraction_result_omits_decision_calls_when_empty():
    """Filter off: the serialized result carries no decision_calls key at all."""
    dumped = _result().model_dump()
    assert "decision_calls" not in dumped
    assert "decision_calls" not in json.loads(_result().model_dump_json())


def test_extraction_result_decision_calls_roundtrip():
    result = _result(decision_calls=[_call_record()], prefilter_ran=True)
    restored = ExtractionResult(**result.model_dump())
    assert restored.decision_calls == result.decision_calls
    assert restored.prefilter_ran is True
    assert restored.decision_calls[0].decision.question_id == "risk-relevance"


def test_decision_call_record_accepts_all_stages():
    """The stage field is a plain string, not a closed literal list."""
    for stage in ("prefilter", "judge"):
        record = DecisionCallRecord(stage=stage, chunk_index=0, request={})
        assert record.stage == stage


# --- sanity that the screening types still compose with the new record ---


def test_decision_log_still_reduces_answers():
    """The prefilter's DecisionLog carries probability answers the screening seam builds."""
    log = _decision_log()
    assert isinstance(log.items[0].answers[0]["probability"], float)


def test_answer_batch_exchange_stays_optional():
    """Non-chat assessors (replay) still produce exchange=None batches."""
    batch = AnswerBatch(answers={}, unmatched_ids=(), latency_ms=0.0, exchange=None)
    assert batch.exchange is None
