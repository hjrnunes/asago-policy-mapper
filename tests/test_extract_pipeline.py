import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import stage_mock_client
from llm_errors import (
    authentication_failure,
    connection_failure,
    server_error,
    validation_exhaustion,
)

from asago_policy_mapper.extract.models import (
    EvidenceSpan,
    ExtractionResult,
    LLMCallRecord,
    RetrievalConfig,
    RetrievalScores,
    RiskMatch,
    ScoredCandidate,
    _CausalChain,
    _CausalChains,
    _RiskEvidence,
    _RiskEvidenceList,
)
from asago_policy_mapper.extract.parse import Chunk
from asago_policy_mapper.extract.pipeline import (
    PipelineInvariantError,
    _build_result_metadata,
    _build_retrieval_stats,
    _cap_accepted_candidates,
    _run_causal_synthesis,
    _run_causal_synthesis_stage,
    _run_grounding,
    _run_judge,
    build_risk_match,
    determine_accepted_by,
    run_extraction,
)
from asago_policy_mapper.extract.retrieve import ChunkResult, build_chunk_context_indices, build_chunk_contexts


def _make_risk(id, name, description, concern="", parent=""):
    return SimpleNamespace(
        id=id,
        name=name,
        description=description,
        concern=concern,
        risk_type="",
        isDefinedByTaxonomy="test-taxonomy",
        isPartOf=parent,
        exact_mappings=[],
        close_mappings=[],
        broad_mappings=[],
        narrow_mappings=[],
        related_mappings=[],
    )


MOCK_RISKS = [
    _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs favoring certain groups"),
    _make_risk("R-002", "Data Poisoning", "Malicious manipulation of training data"),
    _make_risk("R-003", "Privacy Violation", "Unauthorized use of personal data by AI"),
]


@pytest.mark.slow
def test_run_extraction_returns_extraction_result(mock_config, tmp_path):
    doc = tmp_path / "test.md"
    doc.write_text("AI systems must avoid bias and protect personal data. Training data integrity is critical.")

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.version == "0.4"
    assert result.source_documents == [str(doc)]
    assert result.retrieval_stats.total_chunks >= 1
    assert "service_tier" not in result.metadata


def test_run_extraction_empty_document(mock_config, tmp_path):
    doc = tmp_path / "empty.txt"
    doc.write_text("")

    mock_client = stage_mock_client()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.risks == []


def test_run_extraction_no_risks(mock_config, tmp_path):
    doc = tmp_path / "test.txt"
    doc.write_text("Some policy text.")

    mock_client = stage_mock_client()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=[],
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.risks == []
    assert result.retrieval_stats.total_chunks == 0


def test_run_extraction_no_chunks_returns_empty_result(mock_config, tmp_path, monkeypatch):
    """A parse that yields content but no chunks short-circuits to the empty result."""
    from asago_policy_mapper.extract import pipeline

    doc = tmp_path / "test.txt"
    doc.write_text("Some policy text.")
    monkeypatch.setattr(pipeline, "chunk_documents", lambda docs, max_tokens=512: [])

    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.risks == []
    assert result.retrieval_stats.total_chunks == 0


def test_metadata_records_structured_output_mode(mock_config):
    """A real structured-output client's resolved mode lands in the metadata."""
    from asago_policy_mapper.llm import StructuredOutputMode, _StructuredClient

    client = _StructuredClient.__new__(_StructuredClient)
    client.mode = StructuredOutputMode.JSON_SCHEMA_WITH_PROMPT

    metadata = _build_result_metadata(mock_config, RetrievalConfig(query_gen=False), {}, [], set(), client)

    assert metadata["structured_output_mode"] == "json_schema_with_prompt"


def test_metadata_omits_mode_without_real_client(mock_config):
    """A mocked or absent client leaves the structured_output_mode key out."""
    for client in (MagicMock(), None, SimpleNamespace(mode="json_schema")):
        metadata = _build_result_metadata(mock_config, RetrievalConfig(query_gen=False), {}, [], set(), client)
        assert "structured_output_mode" not in metadata


@pytest.mark.slow
def test_run_extraction_metadata_carries_client_mode(mock_config, tmp_path):
    """The pipeline passes its client to _build_result_metadata (wiring check)."""
    from asago_policy_mapper.llm import StructuredOutputMode, _StructuredClient

    doc = tmp_path / "test.md"
    doc.write_text("AI policy text about bias in model outputs and data governance.")

    client = _StructuredClient.__new__(_StructuredClient)
    client.mode = StructuredOutputMode.JSON_SCHEMA

    result = run_extraction(
        documents=[doc],
        client=client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert result.metadata["structured_output_mode"] == "json_schema"


def test_metadata_carries_query_gen_and_service_tier():
    """Query-gen runs record their queries, fallback chunks and service tier."""
    from asago_policy_mapper.extract.querygen import QueryResult

    config = SimpleNamespace(model="test-model", service_tier="flex")
    retrieval = RetrievalConfig(query_gen=True)
    query_results = [QueryResult(query="q-bias", chunk_indices=[0, 1], section="AI")]

    metadata = _build_result_metadata(config, retrieval, {"expanded_candidates": 2}, query_results, {3, 1})

    assert metadata["model"] == "test-model"
    assert metadata["service_tier"] == "flex"
    assert metadata["expansion_stats"] == {"expanded_candidates": 2}
    # The timestamp is UTC: it carries a +00:00 offset, not a naive one.
    assert metadata["timestamp"].endswith("+00:00")
    assert metadata["query_gen_queries"] == [{"query": "q-bias", "chunk_indices": [0, 1], "section": "AI"}]
    assert metadata["query_gen_fallback_chunks"] == [1, 3]


def test_metadata_omits_service_tier_when_unset():
    """Without a service tier the key stays out of the metadata."""
    config = SimpleNamespace(model="test-model", service_tier=None)

    metadata = _build_result_metadata(config, RetrievalConfig(query_gen=False), {}, [], set())

    assert "service_tier" not in metadata
    assert "query_gen_queries" not in metadata
    assert "query_gen_fallback_chunks" not in metadata


def test_run_extraction_passes_every_metadata_argument(mock_config, tmp_path, monkeypatch):
    """run_extraction forwards config, retrieval, stats, queries, fallbacks and client.

    Fast wiring check: the retrieval index is stubbed out so no embedding
    model loads; deleting or reordering a _build_result_metadata argument
    fails here instead of only in the slow suite.
    """
    from asago_policy_mapper.extract import pipeline
    from asago_policy_mapper.llm import StructuredOutputMode, _StructuredClient

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return []

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy text about data governance and risk management. "
        "The organization commits to bias detection and mitigation strategies "
        "across all machine learning development activities. "
        "Training data integrity is critical for model reliability and safety. "
        "Personal data must be protected during every processing stage. "
        "Third-party model suppliers undergo security review before deployment."
    )
    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())

    client = _StructuredClient.__new__(_StructuredClient)
    client.mode = StructuredOutputMode.JSON_SCHEMA

    result = run_extraction(
        documents=[doc],
        client=client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert result.metadata["model"] == mock_config.model
    assert result.metadata["structured_output_mode"] == "json_schema"
    assert result.metadata["expansion_stats"]["expanded_candidates"] == 0
    assert result.metadata["no_judge"] is False


def _chunk_result_judged_borderline(count=1, retrieved=None, auto_accepted=0):
    """Build one ChunkResult with `count` borderline candidates that were judged.

    The pipeline fills borderline_judged both when the LLM judge accepts
    candidates and when --no-judge auto-promotes them, so both counts come
    from the same field.
    """
    borderline = [
        ScoredCandidate(risk_id=f"R-B{i}", risk_name=f"R-B{i}", risk_description="d", cross_encoder_score=0.5)
        for i in range(count)
    ]
    return ChunkResult(
        chunk_index=0,
        source="doc.md",
        page=None,
        section=None,
        accepted=[],
        borderline=borderline,
        borderline_judged=list(borderline),
        stats={
            "candidates_retrieved": retrieved if retrieved is not None else count,
            "auto_accepted": auto_accepted,
            "borderline": count,
            "discarded": 0,
        },
    )


def _stats_plan(no_judge):
    """Resolve the plan whose judge mode matches the no_judge flag."""
    from asago_policy_mapper.extract.plan import resolve_plan

    retrieval = RetrievalConfig(query_gen=False, no_judge=no_judge)
    return resolve_plan(retrieval, prefilter=False, explicit=frozenset())


def test_retrieval_stats_no_judge_counts_auto_promoted():
    """Under --no-judge, llm_judged stays 0 and auto_promoted holds the count."""
    stats = _build_retrieval_stats(
        [_chunk_result_judged_borderline(2, retrieved=5, auto_accepted=1), _chunk_result_judged_borderline(1)],
        [object(), object()],
        {"judge_ms": 1.0},
        7,
        _stats_plan(no_judge=True),
    )
    assert stats.llm_judged == 0
    assert stats.auto_promoted == 3
    # The aggregates the run report relies on stay intact.
    assert stats.total_chunks == 2
    assert stats.total_candidates_retrieved == 6
    assert stats.auto_accepted == 1
    assert stats.grounding_filtered == 7
    assert stats.timing_ms == {"judge_ms": 1.0}


def test_retrieval_stats_defaults_to_normal_mode():
    """Omitting no_judge keeps the legacy llm_judged count (mutant guard for the default)."""
    stats = _build_retrieval_stats(
        [_chunk_result_judged_borderline(2)],
        [],
        {},
        0,
        _stats_plan(no_judge=False),
    )
    assert stats.llm_judged == 2
    assert stats.auto_promoted == 0


def test_retrieval_stats_missing_stat_keys_count_as_zero():
    """Chunk results whose stats lack the counters contribute zero, not an error."""
    cr = _chunk_result_judged_borderline(2)
    cr.stats = {}

    stats = _build_retrieval_stats([cr], [object()], {}, 0, _stats_plan(no_judge=False))

    assert stats.total_candidates_retrieved == 0
    assert stats.auto_accepted == 0


def test_retrieval_stats_normal_mode_keeps_llm_judged():
    """Normal mode keeps counting borderline_judged as llm_judged; auto_promoted stays 0."""
    stats = _build_retrieval_stats(
        [_chunk_result_judged_borderline(2), _chunk_result_judged_borderline(1)],
        [],
        {},
        0,
        _stats_plan(no_judge=False),
    )
    assert stats.llm_judged == 3
    assert stats.auto_promoted == 0


@pytest.mark.slow
def test_run_extraction_no_judge_reports_auto_promoted(mock_config, tmp_path):
    """An end-to-end --no-judge run reports zero llm_judged and its promotions in auto_promoted."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    # top_n_accept=0 with no score floor sends every retrieved candidate to
    # the borderline band, which --no-judge then promotes, so the test
    # exercises a nonzero count.
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(
            query_gen=False, no_judge=True, no_grounding=True, top_n_accept=0, min_score_floor=0.0
        ),
        ocr=False,
    )

    borderline_total = sum(cs.borderline for cs in result.chunks)
    assert borderline_total > 0
    assert result.retrieval_stats.llm_judged == 0
    assert result.retrieval_stats.auto_promoted == borderline_total


@pytest.mark.slow
def test_run_extraction_expands_variants_when_no_grounding(mock_config, tmp_path):
    """Ungrounded runs with variant catalogs re-expand chunk candidates to variants."""
    variant_risks = [
        _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs"),
        _make_risk(
            "unauthorized-processing---biometric-data",
            "Unauthorized Processing - Biometric Data",
            "Unauthorized processing of biometric data.",
        ),
        _make_risk(
            "unauthorized-processing---location-data",
            "Unauthorized Processing - Location Data",
            "Unauthorized processing of location data.",
        ),
    ]
    doc = tmp_path / "test.md"
    doc.write_text(
        "Policy covering unauthorized processing of biometric and location data. "
        "This document establishes controls for personal data handling across "
        "all processing activities undertaken by the organization."
    )

    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=variant_risks,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True, no_grounding=True),
        ocr=False,
    )

    # the collapsed parent candidates are expanded back to their variants
    variant_ids = {r.risk_id for r in result.risks}
    assert "unauthorized-processing---biometric-data" in variant_ids
    assert "unauthorized-processing---location-data" in variant_ids
    assert all(r.grounding_confidence == "ungrounded" for r in result.risks)


@pytest.mark.slow
def test_run_extraction_multiple_documents(mock_config, tmp_path):
    docs = []
    for i in range(3):
        doc = tmp_path / f"doc{i}.txt"
        doc.write_text(f"Policy document number {i} about AI governance.")
        docs.append(doc)

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=docs,
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert len(result.source_documents) == 3
    assert result.retrieval_stats.total_chunks >= 3


@pytest.mark.slow
def test_run_extraction_metadata(mock_config, tmp_path):
    doc = tmp_path / "test.txt"
    doc.write_text("AI risk document.")
    mock_config.service_tier = "flex"

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert result.metadata["model"] == "test-model"
    assert result.metadata["service_tier"] == "flex"
    assert result.metadata["top_n_accept"] == 10
    assert result.metadata["top_n_judge"] == 10


@pytest.mark.slow
def test_run_extraction_populates_chunks(mock_config, tmp_path):
    doc = tmp_path / "test.md"
    doc.write_text("AI systems must avoid bias and protect personal data. Training data integrity is critical.")

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert len(result.chunks) >= 1
    chunk = result.chunks[0]
    assert chunk.index == 0
    assert chunk.source == str(doc)
    assert len(chunk.text_preview) > 0
    assert len(chunk.text_preview) <= 200
    assert chunk.candidates_retrieved >= 0


@pytest.mark.slow
def test_run_extraction_populates_llm_calls(mock_config, tmp_path):
    """LLM calls list is present (may be empty if no borderline candidates)."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False),
        ocr=False,
    )

    assert isinstance(result.llm_calls, list)


@pytest.mark.slow
def test_run_extraction_no_judge_no_grounding(mock_config, tmp_path):
    """With no_judge+no_grounding, no LLM calls are made and all candidates become RiskMatch with empty evidence."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(no_judge=True, no_grounding=True, query_gen=False),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["no_judge"] is True
    assert result.metadata["no_grounding"] is True
    mock_client.chat.completions.create.assert_not_called()
    assert len(result.llm_calls) == 0
    for risk in result.risks:
        assert risk.evidence == []
        assert risk.grounding_confidence == "ungrounded"
    assert result.retrieval_stats.grounding_filtered == 0


@pytest.mark.slow
def test_run_extraction_no_grounding_with_judge(mock_config, tmp_path):
    """With no_grounding only, judge runs but grounding is skipped."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(no_grounding=True, query_gen=False),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["no_grounding"] is True
    assert result.metadata["no_judge"] is False
    for risk in result.risks:
        assert risk.evidence == []
        assert risk.grounding_confidence == "ungrounded"
    assert result.retrieval_stats.grounding_filtered == 0


@pytest.mark.slow
def test_run_extraction_no_judge_no_grounding_no_cross_encoder(mock_config, tmp_path):
    """With no_judge+no_grounding and no cross-encoder, pure BM25+semantic output."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(
            no_judge=True,
            no_grounding=True,
            use_cross_encoder=False,
            rrf_min_score=0.001,
            query_gen=False,
        ),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    mock_client.chat.completions.create.assert_not_called()
    for risk in result.risks:
        assert risk.accepted_by in ("rrf", "auto_promoted")


@pytest.mark.slow
def test_run_extraction_no_cross_encoder(mock_config, tmp_path):
    """With use_cross_encoder=False, no judge calls are made and metadata reflects the mode."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(use_cross_encoder=False, rrf_min_score=0.01, query_gen=False),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["use_cross_encoder"] is False
    assert result.metadata["rrf_min_score"] == 0.01
    assert result.metadata["cross_encoder_model"] is None
    judge_calls = [c for c in result.llm_calls if c.stage == "judge"]
    assert len(judge_calls) == 0


# --- determine_accepted_by unit tests ---


def _candidate(risk_id="R-001"):
    return ScoredCandidate(
        risk_id=risk_id,
        risk_name="Test",
        risk_description="desc",
        cross_encoder_score=0.8,
        rrf_score=0.5,
        rerank_source="cross_encoder",
        accepted_by="rank",
    )


def test_judge_promotion_stamps_llm_judge_label():
    """The LLM judge stamps its promotions at promotion time."""
    c = _candidate()
    stamped = c.model_copy(update={"accepted_by": "llm_judge"})
    assert determine_accepted_by(stamped) == "llm_judge"


def test_judge_promotion_stamps_auto_promoted_label():
    """Auto-promotion under --no-judge stamps its own label."""
    c = _candidate()
    stamped = c.model_copy(update={"accepted_by": "auto_promoted"})
    assert determine_accepted_by(stamped) == "auto_promoted"


def test_determine_accepted_by_unstamped_candidate_raises():
    """An unstamped candidate is an invariant failure, not a fallback.

    Retrieval, query generation and the judge stamp every accepted
    candidate with its selection label; a candidate without one never
    occurs in production.
    """
    c = _candidate()
    c = c.model_copy(update={"accepted_by": None})
    with pytest.raises(PipelineInvariantError):
        determine_accepted_by(c)


@pytest.mark.slow
def test_run_extraction_no_judge_with_grounding_accepted_by(mock_config, tmp_path):
    """Regression: no_judge=True with grounding should tag as 'auto_promoted', not 'llm_judge'."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(no_judge=True, no_grounding=False, query_gen=False),
        ocr=False,
    )

    for risk in result.risks:
        assert risk.accepted_by in ("auto_promoted", "rrf", "threshold"), (
            f"Expected auto_promoted/rrf/threshold but got '{risk.accepted_by}' for {risk.risk_id}"
        )


# --- build_risk_match unit tests ---


def test_build_risk_match_cross_encoder():
    c = _candidate()
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="threshold",
        grounding_confidence="high",
        evidence=[],
        use_cross_encoder=True,
    )
    assert m.confidence == c.cross_encoder_score
    assert m.scores.rrf_score == c.rrf_score
    assert m.accepted_by == "threshold"
    assert m.taxonomy == "test-tax"


def test_build_risk_match_rrf_mode():
    c = _candidate()
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="rrf",
        grounding_confidence="ungrounded",
        evidence=[],
        use_cross_encoder=False,
    )
    assert m.confidence == c.rrf_score


def test_build_risk_match_stamped_candidate_keeps_measured_confidence():
    """A reranker-stamped candidate keeps its measured score as confidence.

    The accepted_by label does not change the confidence derivation: only
    the candidate's stamps decide. An expansion stub that was never
    measured (rerank_source None) is the null case, covered in
    test_extract_score_semantics.
    """
    c = _candidate()
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="expansion",
        grounding_confidence="high",
        evidence=[],
    )
    assert m.confidence == c.cross_encoder_score
    assert m.confidence_source == "cross_encoder"


def test_build_risk_match_scores_override():
    c = _candidate()
    zero_scores = RetrievalScores(
        bm25_rank=0,
        embedding_distance=0.0,
        cross_encoder_score=0.0,
        rrf_score=0.0,
    )
    m = build_risk_match(
        c,
        taxonomy="test-tax",
        accepted_by="expansion",
        grounding_confidence="high",
        evidence=[],
        scores_override=zero_scores,
    )
    assert m.scores.bm25_rank == 0
    assert m.scores.cross_encoder_score == 0.0


# --- _run_expansion integration test ---

EXPANSION_RISKS = [
    _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs", parent="group-fairness"),
    _make_risk("R-002", "Data Bias", "Bias from training data distributions", parent="group-fairness"),
    _make_risk("R-003", "Privacy Violation", "Unauthorized use of personal data"),
]


@pytest.mark.slow
def test_run_extraction_expand_siblings(mock_config, tmp_path):
    """With expand_siblings, sibling risks of found risks are grounded and included."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.side_effect = lambda **kwargs: (
        _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=r["risk_id"],
                    grounded=True,
                    confidence="medium",
                    quotes=["bias detection and mitigation"],
                )
                for r in (
                    kwargs.get("messages", [{}])[-1].get("risks", [])
                    if isinstance(kwargs.get("messages", [{}])[-1], dict)
                    else []
                )
            ]
        )
        if kwargs.get("response_model")
        else SimpleNamespace(items=[])
    )

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=EXPANSION_RISKS,
        retrieval=RetrievalConfig(expand_siblings=True, no_judge=True, query_gen=False),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["expand_siblings"] is True
    expansion_risks = [r for r in result.risks if r.accepted_by == "expansion"]
    expansion_stats = result.metadata.get("expansion_stats", {})
    assert expansion_stats.get("expanded_candidates", 0) >= len(expansion_risks)


@pytest.mark.slow
def test_run_extraction_expand_no_siblings_when_no_grounding(mock_config, tmp_path):
    """Expansion requires grounding — skipped when no_grounding=True."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=EXPANSION_RISKS,
        retrieval=RetrievalConfig(expand_siblings=True, no_judge=True, no_grounding=True, query_gen=False),
        ocr=False,
    )

    assert result.metadata["expansion_stats"]["expanded_candidates"] == 0
    assert not any(r.accepted_by == "expansion" for r in result.risks)


# --- _run_judge integration test ---


@pytest.mark.slow
def test_run_extraction_judge_with_cross_encoder(mock_config, tmp_path):
    """With use_cross_encoder and judge enabled, borderline candidates go through judge."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = SimpleNamespace(items=[])

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(no_grounding=True, query_gen=False),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["no_judge"] is False
    assert result.metadata["use_cross_encoder"] is True
    for risk in result.risks:
        assert risk.accepted_by in ("threshold", "llm_judge")


# --- _run_causal_synthesis tests ---


def test_run_causal_synthesis_populates_fields():
    merged = [
        RiskMatch(
            risk_id="atlas-bias",
            risk_name="AI Bias",
            risk_description="Systematic bias.",
            confidence=0.85,
            confidence_source="cross_encoder",
            grounding_confidence="high",
            accepted_by="threshold",
            evidence=[
                EvidenceSpan(text="quote", document="doc.pdf", chunk_index=0),
                EvidenceSpan(text="quote2", document="doc.pdf", chunk_index=2),
            ],
            scores=RetrievalScores(bm25_rank=1, embedding_distance=0.2, cross_encoder_score=0.85, rrf_score=0.03),
        ),
    ]
    chunks = [
        SimpleNamespace(text="Chunk zero about credit scoring fairness."),
        SimpleNamespace(text="Chunk one irrelevant."),
        SimpleNamespace(text="Chunk two about discrimination in lending."),
    ]

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = _CausalChains(
        items=[
            _CausalChain(
                threat="Credit scoring discriminates",
                threat_source="Biased training data",
                vulnerability="No fairness audit",
                consequence="Applicants denied credit",
                impact="Financial exclusion",
            ),
        ]
    )

    from asago_policy_mapper.llm import LLMConfig

    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    collector: list[LLMCallRecord] = []

    result = _run_causal_synthesis(merged, chunks, mock_client, config, 4, collector)

    assert len(result) == 1
    assert result[0].threat == "Credit scoring discriminates"
    assert result[0].threat_source == "Biased training data"
    assert result[0].vulnerability == "No fairness audit"
    assert result[0].consequence == "Applicants denied credit"
    assert result[0].impact == "Financial exclusion"
    assert len(collector) == 1
    assert collector[0].stage == "causal_synthesis"


def test_run_causal_synthesis_skips_empty_results():
    merged = [
        RiskMatch(
            risk_id="atlas-bias",
            risk_name="AI Bias",
            risk_description="Bias.",
            confidence=0.85,
            confidence_source="cross_encoder",
            grounding_confidence="high",
            accepted_by="expansion",
            evidence=[
                EvidenceSpan(text="q", document="doc.pdf", chunk_index=0),
            ],
            scores=RetrievalScores(bm25_rank=1, embedding_distance=0.2, cross_encoder_score=0.85, rrf_score=0.03),
        ),
    ]
    chunks = [SimpleNamespace(text="Unrelated text about data retention.")]

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = _CausalChains(
        items=[_CausalChain(threat="", threat_source="", vulnerability="", consequence="", impact="")]
    )

    from asago_policy_mapper.llm import LLMConfig

    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    result = _run_causal_synthesis(merged, chunks, mock_client, config, 4, [])

    assert result[0].threat is None
    assert result[0].impact is None


def _causal_risk_and_chunks(risk_ids):
    merged = [
        RiskMatch(
            risk_id=risk_id,
            risk_name=f"Name {risk_id}",
            risk_description="Description.",
            confidence=0.85,
            confidence_source="cross_encoder",
            grounding_confidence="high",
            accepted_by="threshold",
            evidence=[EvidenceSpan(text="q", document="doc.pdf", chunk_index=0)],
            scores=RetrievalScores(bm25_rank=1, embedding_distance=0.2, cross_encoder_score=0.85, rrf_score=0.03),
        )
        for risk_id in risk_ids
    ]
    return merged, [SimpleNamespace(text="Policy text about the risk.")]


def _causal_client(failing_risk_id, error):
    def _create(*, model, response_model, messages):
        if any(failing_risk_id in m["content"] for m in messages):
            raise error
        return _CausalChains(
            items=[
                _CausalChain(
                    threat="Threat", threat_source="Source", vulnerability="Vuln", consequence="Cons", impact="Impact"
                )
            ]
        )

    client = stage_mock_client()
    client.chat.completions.create.side_effect = _create
    return client


def test_run_causal_synthesis_skips_risk_with_validation_exhaustion(caplog):
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b", "atlas-c"])
    client = _causal_client("atlas-b", validation_exhaustion())
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    with caplog.at_level(logging.WARNING, logger="asago_policy_mapper.extract.attribute"):
        result = _run_causal_synthesis(merged, chunks, client, config, 4, [])

    by_id = {m.risk_id: m for m in result}
    assert by_id["atlas-a"].threat == "Threat"
    assert by_id["atlas-c"].impact == "Impact"
    assert by_id["atlas-b"].threat is None
    assert by_id["atlas-b"].impact is None
    assert any("atlas-b" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.parametrize(
    "make_error",
    [connection_failure, server_error, authentication_failure],
    ids=["connection-error", "http-5xx", "http-401"],
)
def test_run_causal_synthesis_propagates_api_errors(make_error):
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b"])
    error = make_error()
    client = _causal_client("atlas-b", error)
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    with pytest.raises(type(error)) as excinfo:
        _run_causal_synthesis(merged, chunks, client, config, 4, [])

    assert excinfo.value is error


def test_run_causal_synthesis_marks_status_and_generated_provenance():
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a"])
    client = _causal_client("none-matches", None)
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    result = _run_causal_synthesis(merged, chunks, client, config, 4, [])

    risk = result[0]
    assert risk.causal_status == "generated"
    assert risk.threat == "Threat"
    assert risk.causal_provenance == {
        "threat": {"source": "generated"},
        "threat_source": {"source": "generated"},
        "vulnerability": {"source": "generated"},
        "consequence": {"source": "generated"},
        "impact": {"source": "generated"},
    }


def test_run_causal_synthesis_does_not_leak_marks_into_a_copied_risk():
    """A model_copy()'d risk shares its provenance dict; synthesis must not mutate the original's."""
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a"])
    client = _causal_client("none-matches", None)
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    original = merged[0].model_copy()

    result = _run_causal_synthesis(merged, chunks, client, config, 4, [])

    assert result[0].causal_provenance == {
        "threat": {"source": "generated"},
        "threat_source": {"source": "generated"},
        "vulnerability": {"source": "generated"},
        "consequence": {"source": "generated"},
        "impact": {"source": "generated"},
    }
    # The copy's source kept its own dict: no marks, no status.
    assert original.causal_provenance == {}
    assert original.causal_status is None


def test_run_causal_synthesis_marks_empty_answer_status():
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a"])
    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _CausalChains(
        items=[_CausalChain(threat="", threat_source="", vulnerability="", consequence="", impact="")]
    )
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    result = _run_causal_synthesis(merged, chunks, client, config, 4, [])

    risk = result[0]
    assert risk.causal_status == "empty_answer"
    assert risk.threat is None
    assert risk.causal_provenance == {}


def test_run_causal_synthesis_marks_no_evidence_status():
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a"])
    client = MagicMock()
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    # Evidence points at a chunk that does not exist in chunk_texts.
    merged[0].evidence[0].chunk_index = 9

    result = _run_causal_synthesis(merged, chunks, client, config, 4, [])

    risk = result[0]
    assert risk.causal_status == "no_evidence"
    assert risk.causal_provenance == {}
    client.chat.completions.create.assert_not_called()


def test_run_causal_synthesis_marks_validation_exhausted_status(caplog):
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a"])
    client = _causal_client("atlas-a", validation_exhaustion())
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    with caplog.at_level(logging.WARNING, logger="asago_policy_mapper.extract.attribute"):
        result = _run_causal_synthesis(merged, chunks, client, config, 4, [])

    risk = result[0]
    assert risk.causal_status == "validation_exhausted"
    assert risk.threat is None
    assert risk.causal_provenance == {}


def test_run_causal_synthesis_generated_fields_not_overwritten_by_later_merge():
    """A partially-empty chain marks only the fields that carry content."""
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a"])
    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _CausalChains(
        items=[_CausalChain(threat="Threat", threat_source="", vulnerability="Vuln", consequence="", impact="")]
    )
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")

    result = _run_causal_synthesis(merged, chunks, client, config, 4, [])

    risk = result[0]
    assert risk.causal_status == "generated"
    assert risk.threat == "Threat"
    assert risk.vulnerability == "Vuln"
    assert risk.threat_source is None
    assert risk.consequence is None
    assert risk.impact is None
    assert risk.causal_provenance == {"threat": {"source": "generated"}, "vulnerability": {"source": "generated"}}


def _causal_plan(retrieval):
    from asago_policy_mapper.extract.plan import resolve_plan

    return resolve_plan(retrieval, prefilter=False, explicit=frozenset())


def test_run_causal_synthesis_stage_marks_disabled_when_grounding_off():
    """Grounding off forces causal synthesis off, whatever the causal flag says.

    An ungrounded match carries no evidence, so there is nothing to
    synthesize from: the plan's truth table gates causal synthesis on
    grounding, and every match records ``disabled_no_grounding``.
    """
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b"])
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    # Causal synthesis is NOT turned off here; grounding off alone must gate it.
    retrieval = RetrievalConfig(query_gen=False, no_grounding=True)
    client = MagicMock()

    result = _run_causal_synthesis_stage(merged, chunks, client, config, _causal_plan(retrieval), [], 4, {})

    assert all(r.causal_status == "disabled_no_grounding" for r in result)
    client.chat.completions.create.assert_not_called()


def test_run_causal_synthesis_stage_marks_disabled_when_synthesis_off():
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b"])
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    retrieval = RetrievalConfig(query_gen=False, no_causal_synthesis=True)
    client = MagicMock()

    result = _run_causal_synthesis_stage(merged, chunks, client, config, _causal_plan(retrieval), [], 4, {})

    assert all(r.causal_status == "disabled_synthesis_off" for r in result)
    client.chat.completions.create.assert_not_called()


def test_run_causal_synthesis_stage_marks_disabled_when_client_none():
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b"])
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    retrieval = RetrievalConfig(query_gen=False)

    result = _run_causal_synthesis_stage(merged, chunks, None, config, _causal_plan(retrieval), [], 4, {})

    assert all(r.causal_status == "disabled_no_client" for r in result)


def test_run_causal_synthesis_stage_grounding_off_wins_over_synthesis_off():
    """With grounding and synthesis both off, the grounding reason is recorded."""
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b"])
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    retrieval = RetrievalConfig(query_gen=False, no_grounding=True, no_causal_synthesis=True)
    client = MagicMock()

    result = _run_causal_synthesis_stage(merged, chunks, client, config, _causal_plan(retrieval), [], 4, {})

    assert all(r.causal_status == "disabled_no_grounding" for r in result)
    client.chat.completions.create.assert_not_called()


def test_run_causal_synthesis_stage_synthesis_off_wins_over_client_none():
    """A plan-disabled run names the synthesis reason even without a client."""
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b"])
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    retrieval = RetrievalConfig(query_gen=False, no_causal_synthesis=True)

    result = _run_causal_synthesis_stage(merged, chunks, None, config, _causal_plan(retrieval), [], 4, {})

    assert all(r.causal_status == "disabled_synthesis_off" for r in result)


def test_run_causal_synthesis_stage_runs_synthesis_and_times_it():
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a", "atlas-b"])
    client = _causal_client("none-matches", None)
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    retrieval = RetrievalConfig(query_gen=False)
    timing: dict[str, float] = {}
    collector: list[LLMCallRecord] = []

    with patch("asago_policy_mapper.extract.pipeline._run_causal_synthesis", autospec=True) as run_synth:
        run_synth.return_value = merged
        result = _run_causal_synthesis_stage(
            merged, chunks, client, config, _causal_plan(retrieval), collector, 7, timing
        )

    run_synth.assert_called_once_with(merged, chunks, client, config, 7, collector)
    assert result is merged
    assert "causal_synthesis_ms" in timing


def test_run_causal_synthesis_stage_end_to_end_generated():
    from asago_policy_mapper.llm import LLMConfig

    merged, chunks = _causal_risk_and_chunks(["atlas-a"])
    client = _causal_client("none-matches", None)
    config = LLMConfig(base_url="http://localhost:8000/v1", model="test-model")
    retrieval = RetrievalConfig(query_gen=False)
    timing: dict[str, float] = {}
    collector: list[LLMCallRecord] = []

    result = _run_causal_synthesis_stage(merged, chunks, client, config, _causal_plan(retrieval), collector, 4, timing)

    assert result[0].causal_status == "generated"
    assert result[0].threat == "Threat"
    # The stage times itself under causal_synthesis_ms and records the call.
    assert "causal_synthesis_ms" in timing
    assert len(collector) == 1
    assert collector[0].stage == "causal_synthesis"


# --- query-gen integration test ---


@pytest.mark.slow
def test_run_extraction_query_gen_no_judge_no_grounding(mock_config, tmp_path):
    """With query_gen + no_judge + no_grounding, LLM generates queries and all candidates become ungrounded matches."""
    doc = tmp_path / "test.md"
    doc.write_text(
        "AI policy document about data governance and risk management. "
        "We must ensure bias detection and mitigation strategies are in place. "
        "Training data integrity is critical for model reliability."
    )

    from asago_policy_mapper.extract.querygen import GeneratedQueries

    mock_client = stage_mock_client()
    mock_client.chat.completions.create.return_value = GeneratedQueries(
        queries=["AI bias and fairness risks", "Training data integrity risks"]
    )

    result = run_extraction(
        documents=[doc],
        client=mock_client,
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=True, no_judge=True, no_grounding=True),
        ocr=False,
    )

    assert isinstance(result, ExtractionResult)
    assert result.metadata["query_gen"] is True
    assert "query_gen_queries" in result.metadata
    assert len(result.metadata["query_gen_queries"]) > 0
    assert "query_gen_fallback_chunks" in result.metadata
    assert len(result.llm_calls) > 0
    assert any(c.stage == "query_gen" for c in result.llm_calls)
    for risk in result.risks:
        assert risk.evidence == []
        assert risk.grounding_confidence == "ungrounded"


def test_cap_accepted_candidates_keeps_highest_rrf():
    low = ScoredCandidate(risk_id="low", risk_name="L", risk_description="", rrf_score=0.01)
    high = ScoredCandidate(risk_id="high", risk_name="H", risk_description="", rrf_score=0.9)
    mid = ScoredCandidate(risk_id="mid", risk_name="M", risk_description="", rrf_score=0.2)
    cr = ChunkResult(
        chunk_index=0,
        source="doc.md",
        page=None,
        section=None,
        accepted=[low, high, mid],
        borderline=[],
        borderline_judged=[],
        stats={"auto_accepted": 3, "candidates_retrieved": 3},
    )
    _cap_accepted_candidates([cr], limit=2)
    assert [c.risk_id for c in cr.accepted] == ["high", "mid"]
    assert cr.stats["auto_accepted"] == 2


# --- _run_judge unit tests ---


def _chunk_result_with_borderline(chunk_index=0, borderline_ids=("R-003",)):
    borderline = [_candidate(rid) for rid in borderline_ids]
    return ChunkResult(
        chunk_index=chunk_index,
        source="doc.md",
        page=None,
        section=None,
        accepted=[_candidate("R-001")],
        borderline=borderline,
        borderline_judged=[],
        stats={"auto_accepted": 1, "candidates_retrieved": 1 + len(borderline), "borderline": len(borderline)},
    )


def test_build_chunk_summaries_reports_rerank_unavailable():
    """The chunk summary carries rerank_unavailable from the chunk stats."""
    from asago_policy_mapper.extract.parse import Chunk
    from asago_policy_mapper.extract.pipeline import _build_chunk_summaries

    cr = ChunkResult(
        chunk_index=0,
        source="doc.md",
        page=None,
        section=None,
        accepted=[],
        borderline=[],
        borderline_judged=[],
        stats={"candidates_retrieved": 5, "auto_accepted": 0, "borderline": 0, "discarded": 0, "rerank_unavailable": 2},
    )
    chunks = [Chunk(text="Chunk one.", source="doc.md", index=0)]

    summaries = _build_chunk_summaries([cr], chunks)

    assert summaries[0].rerank_unavailable == 2


def test_build_chunk_summaries_rerank_unavailable_defaults_to_zero():
    """Chunks without the stat serialize rerank_unavailable as zero."""
    from asago_policy_mapper.extract.parse import Chunk
    from asago_policy_mapper.extract.pipeline import _build_chunk_summaries

    cr = ChunkResult(
        chunk_index=0,
        source="doc.md",
        page=None,
        section=None,
        accepted=[],
        borderline=[],
        borderline_judged=[],
        stats={"candidates_retrieved": 5, "auto_accepted": 0, "borderline": 0, "discarded": 0},
    )
    chunks = [Chunk(text="Chunk one.", source="doc.md", index=0)]

    summaries = _build_chunk_summaries([cr], chunks)
    summary = summaries[0]

    assert summary.rerank_unavailable == 0
    assert summary.model_dump()["rerank_unavailable"] == 0

    # The model default itself is zero, independent of the summary builder.
    from asago_policy_mapper.extract.models import ChunkSummary

    bare = ChunkSummary(
        index=0, source="doc.md", text_preview="p", candidates_retrieved=0, auto_accepted=0, borderline=0, discarded=0
    )
    assert bare.rerank_unavailable == 0
    assert bare.model_dump()["rerank_unavailable"] == 0


def _judge_retrieval(**overrides):
    from asago_policy_mapper.extract.models import RetrievalConfig

    return RetrievalConfig(query_gen=False, **overrides)


def _judge_plan(**overrides):
    from asago_policy_mapper.extract.plan import resolve_plan

    return resolve_plan(_judge_retrieval(**overrides), prefilter=False, explicit=frozenset())


def test_run_judge_no_judge_auto_promotes_borderline(mock_config):
    """With no_judge, every borderline candidate joins accepted and borderline_judged."""
    cr = _chunk_result_with_borderline()
    retrieval = _judge_retrieval(no_judge=True)

    _run_judge([cr], [], MagicMock(), mock_config, retrieval, [], _judge_plan(no_judge=True))

    assert [c.risk_id for c in cr.borderline_judged] == ["R-003"]
    assert [c.risk_id for c in cr.accepted] == ["R-001", "R-003"]


def test_run_judge_no_judge_skips_chunks_without_borderline(mock_config):
    """With no_judge, chunks without borderline candidates are left untouched."""
    cr = ChunkResult(
        chunk_index=1,
        source="doc.md",
        page=None,
        section=None,
        accepted=[_candidate("R-001")],
        borderline=[],
        borderline_judged=[],
        stats={"auto_accepted": 1, "candidates_retrieved": 1},
    )
    retrieval = _judge_retrieval(no_judge=True)

    _run_judge([cr], [], MagicMock(), mock_config, retrieval, [], _judge_plan(no_judge=True))

    assert cr.borderline_judged == []
    assert [c.risk_id for c in cr.accepted] == ["R-001"]


def test_run_judge_cross_encoder_judges_via_thread_pool(mock_config, monkeypatch):
    """With the cross-encoder path, each borderline chunk is judged concurrently;
    the chunk records the judged candidates and adds them to accepted."""
    from asago_policy_mapper.extract import pipeline

    cr0 = _chunk_result_with_borderline(0, ("R-003", "R-004"))
    cr0.borderline_judged = [_candidate("R-099")]  # stale judging is replaced, not extended
    cr1 = _chunk_result_with_borderline(1, ("R-005",))
    judged_by_chunk = {}

    def fake_judge_one(i, cr, client, model, records, judge_prompt, max_batch_size, context_text):
        judged_by_chunk[i] = (judge_prompt, max_batch_size, context_text)
        # judge keeps only the first borderline candidate of each chunk
        return i, [cr.borderline[0]]

    monkeypatch.setattr(pipeline, "_judge_one", fake_judge_one)
    retrieval = _judge_retrieval()
    contexts = ["context zero", "context one"]

    _run_judge([cr0, cr1], [], MagicMock(), mock_config, retrieval, [], _judge_plan(), chunk_contexts=contexts)

    assert judged_by_chunk[0] == ("judge_risk", 15, "context zero")
    assert judged_by_chunk[1] == ("judge_risk", 15, "context one")
    assert [c.risk_id for c in cr0.borderline_judged] == ["R-003"]
    assert [c.risk_id for c in cr0.accepted] == ["R-001", "R-003"]
    assert [c.risk_id for c in cr1.borderline_judged] == ["R-005"]
    assert [c.risk_id for c in cr1.accepted] == ["R-001", "R-005"]


def test_run_judge_cross_encoder_without_borderline_makes_no_calls(mock_config, monkeypatch):
    """With no borderline candidates the thread pool never submits work."""
    from asago_policy_mapper.extract import pipeline

    submitted = []

    class _NoPool:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args, **kwargs):
            submitted.append((fn.__name__, args))
            raise AssertionError("judge submitted work with no borderline candidates")

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", _NoPool)
    cr = ChunkResult(
        chunk_index=0,
        source="doc.md",
        page=None,
        section=None,
        accepted=[_candidate("R-001")],
        borderline=[],
        borderline_judged=[],
        stats={"auto_accepted": 1, "candidates_retrieved": 1},
    )
    retrieval = _judge_retrieval()

    _run_judge([cr], [], MagicMock(), mock_config, retrieval, [], _judge_plan())

    assert submitted == []
    assert cr.stats["auto_accepted"] == 1
    assert cr.borderline_judged == []


# --- grounding source-window threading tests ---


def _grounding_chunks():
    """Two documents: doc A chunks 0-2, doc B chunk 3."""
    return [
        Chunk(text="First chunk of doc A about data retention.", source="docA.md", index=0),
        Chunk(text="Second chunk: encryption at rest is mandatory.", source="docA.md", index=1),
        Chunk(text="Third chunk of doc A about audits.", source="docA.md", index=2),
        Chunk(text="Doc B: model theft is covered by insurance.", source="docB.md", index=3),
    ]


def _grounding_verdict_response(quotes_by_risk):
    return _RiskEvidenceList(
        items=[
            _RiskEvidence(risk_id=rid, grounded=True, confidence="high", quotes=quotes)
            for rid, quotes in quotes_by_risk.items()
        ]
    )


class _RecordingClient:
    """Mock client that records prompts and returns fixed verdicts."""

    def __init__(self, quotes_by_risk):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: _grounding_verdict_response(quotes_by_risk))
        )
        self.prompts = []

    def record(self, messages):
        self.prompts.append(messages)


def test_run_grounding_passes_same_document_windows(mock_config):
    """Chunk grounding verifies quotes against same-document neighbour windows."""
    chunks = _grounding_chunks()
    contexts = build_chunk_contexts(chunks)
    indices = build_chunk_context_indices(chunks)

    # Doc B's quote only exists in chunk 3 (another document); the doc A
    # quote lives in chunk 0, a neighbour of the grounded chunk 1.
    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _grounding_verdict_response(
        {"R-001": ["encryption at rest is mandatory.", "model theft is covered by insurance."]}
    )

    cr = ChunkResult(
        chunk_index=1,
        source="docA.md",
        page=None,
        section=None,
        accepted=[_candidate("R-001")],
        borderline=[],
        borderline_judged=[],
        stats={"auto_accepted": 1, "candidates_retrieved": 1},
    )
    retrieval = RetrievalConfig(query_gen=False, no_judge=True)

    matches, filtered, _ = _run_grounding(
        [cr],
        chunks,
        client,
        mock_config,
        retrieval,
        SimpleNamespace(get_taxonomy=lambda rid: "tax"),
        [],
        chunk_contexts=contexts,
        chunk_context_indices=indices,
    )

    assert len(matches) == 1
    evidence = matches[0].evidence
    by_text = {e.text: e for e in evidence}
    # Same-document neighbour quote: verified where it really is.
    assert by_text["encryption at rest is mandatory."].verification == "verified"
    assert by_text["encryption at rest is mandatory."].found_chunk_index == 1
    # Cross-document quote: the windows never include doc B, so not_found.
    assert by_text["model theft is covered by insurance."].verification == "not_found"
    assert by_text["model theft is covered by insurance."].found_chunk_index is None
    # chunk_index stays the chunk whose context was used.
    assert all(e.chunk_index == 1 for e in evidence)
    # The prompt was built from the context, unchanged.
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in sent if "encryption at rest" in m["content"])
    assert contexts[1] in user_content


def test_run_grounding_quote_from_neighbour_chunk_found(mock_config):
    """A quote living in a neighbour chunk records that neighbour as found."""
    chunks = _grounding_chunks()
    contexts = build_chunk_contexts(chunks)
    indices = build_chunk_context_indices(chunks)

    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _grounding_verdict_response(
        {"R-001": ["First chunk of doc A about data retention."]}
    )

    cr = ChunkResult(
        chunk_index=1,
        source="docA.md",
        page=None,
        section=None,
        accepted=[_candidate("R-001")],
        borderline=[],
        borderline_judged=[],
        stats={"auto_accepted": 1, "candidates_retrieved": 1},
    )
    retrieval = RetrievalConfig(query_gen=False, no_judge=True)

    matches, _, _ = _run_grounding(
        [cr],
        chunks,
        client,
        mock_config,
        retrieval,
        SimpleNamespace(get_taxonomy=lambda rid: "tax"),
        [],
        chunk_contexts=contexts,
        chunk_context_indices=indices,
    )

    e = matches[0].evidence[0]
    assert e.verification == "verified"
    # The quote lives in chunk 0, a neighbour admitted into chunk 1's context.
    assert e.found_chunk_index == 0
    assert e.chunk_index == 1


def test_run_grounding_without_indices_leaves_verification_none(mock_config):
    """No index list means the spans carry no verification label."""
    chunks = _grounding_chunks()
    contexts = build_chunk_contexts(chunks)

    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _grounding_verdict_response(
        {"R-001": ["encryption at rest is mandatory."]}
    )

    cr = ChunkResult(
        chunk_index=1,
        source="docA.md",
        page=None,
        section=None,
        accepted=[_candidate("R-001")],
        borderline=[],
        borderline_judged=[],
        stats={"auto_accepted": 1, "candidates_retrieved": 1},
    )
    retrieval = RetrievalConfig(query_gen=False, no_judge=True)

    matches, _, _ = _run_grounding(
        [cr],
        chunks,
        client,
        mock_config,
        retrieval,
        SimpleNamespace(get_taxonomy=lambda rid: "tax"),
        [],
        chunk_contexts=contexts,
        chunk_context_indices=None,
    )

    e = matches[0].evidence[0]
    assert e.verification is None
    assert e.found_chunk_index is None


def test_ground_one_bare_chunk_fallback_verifies_against_chunk(mock_config):
    """Without context text, _ground_one verifies against the bare chunk."""
    from asago_policy_mapper.extract.pipeline import _ground_one

    chunks = _grounding_chunks()
    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _grounding_verdict_response(
        {"R-001": ["Second chunk: encryption at rest is mandatory."]}
    )

    cr = ChunkResult(
        chunk_index=1,
        source="docA.md",
        page=None,
        section=None,
        accepted=[_candidate("R-001")],
        borderline=[],
        borderline_judged=[],
        stats={"auto_accepted": 1, "candidates_retrieved": 1},
    )

    _, merged = _ground_one(
        cr,
        chunks,
        client,
        "test-model",
        None,
        passes=1,
        max_batch_size=0,
        context_text="",
        context_source_windows=None,
    )

    e = merged["R-001"][0][0]
    assert e.verification == "verified"
    assert e.found_chunk_index == 1


def test_ground_variants_one_passes_windows(mock_config):
    """Variant grounding verifies quotes against the context's chunk windows."""
    from asago_policy_mapper.extract.pipeline import _ground_variants_one

    chunks = _grounding_chunks()
    contexts = build_chunk_contexts(chunks)
    indices = build_chunk_context_indices(chunks)

    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _grounding_verdict_response(
        {"child-risk": ["First chunk of doc A about data retention."]}
    )

    parent = RiskMatch(
        risk_id="parent-risk",
        risk_name="Parent",
        risk_description="Parent description",
        taxonomy="tax",
        confidence=0.9,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[
            EvidenceSpan(text="Second chunk: encryption at rest is mandatory.", document="docA.md", chunk_index=1)
        ],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.05),
    )

    _, grounded = _ground_variants_one(
        parent,
        chunks,
        {"parent-risk": [{"risk_id": "child-risk", "name": "Child", "description": "Child description"}]},
        client,
        "test-model",
        None,
        chunk_contexts=contexts,
        chunk_context_indices=indices,
    )

    e = grounded["child-risk"][0][0]
    assert e.verification == "verified"
    assert e.found_chunk_index == 0
    assert e.chunk_index == 1


# --- unattributed-span consumer guard tests ---


def _match_with_unattributed_evidence():
    """A merged risk whose evidence mixes attributed and unattributed spans."""
    return RiskMatch(
        risk_id="R-001",
        risk_name="Bias",
        risk_description="Bias risk.",
        taxonomy="tax",
        confidence=None,
        confidence_source="none",
        grounding_confidence="high",
        accepted_by="expansion",
        evidence=[
            EvidenceSpan(text="attributed quote.", document="docA.md", chunk_index=2, verification="verified"),
            EvidenceSpan(text="unattributed quote.", document="docA.md", chunk_index=None, verification="not_found"),
            EvidenceSpan(text="another unattributed.", document="docA.md", chunk_index=None, verification="not_found"),
        ],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.05),
    )


def test_consumers_skip_unattributed_spans():
    """chunk-risk map and found-risk chunks ignore spans without a chunk."""
    from asago_policy_mapper.extract.pipeline import _build_chunk_risk_map, _build_found_risk_chunks

    m = _match_with_unattributed_evidence()
    plan = _judge_plan()  # grounding on: attribute risks through evidence chunks

    chunk_risk_ids = _build_chunk_risk_map([], [m], plan)
    assert chunk_risk_ids == {2: ["R-001"]}

    found = _build_found_risk_chunks([m], [])
    assert found == {"R-001": {2}}


def test_ground_variants_one_skips_unattributed_evidence():
    """Variant grounding picks its context among spans that have a chunk."""
    from asago_policy_mapper.extract.pipeline import _ground_variants_one

    chunks = _grounding_chunks()
    contexts = build_chunk_contexts(chunks)
    indices = build_chunk_context_indices(chunks)

    client = MagicMock()
    client.for_stage = lambda stage: client
    client.chat.completions.create.return_value = _grounding_verdict_response(
        {"child-risk": ["encryption at rest is mandatory."]}
    )

    # The unattributed span sorts first on the tie-break key (higher
    # cross-encoder score wins), so the guard has to exclude it before
    # choosing the grounding context.
    parent = RiskMatch(
        risk_id="parent-risk",
        risk_name="Parent",
        risk_description="Parent description",
        taxonomy="tax",
        confidence=None,
        confidence_source="none",
        grounding_confidence="high",
        accepted_by="expansion",
        evidence=[
            EvidenceSpan(
                text="unattributed quote.",
                document="docA.md",
                chunk_index=None,
                verification="not_found",
                cross_encoder_score=0.99,
            ),
            EvidenceSpan(
                text="Second chunk: encryption at rest is mandatory.",
                document="docA.md",
                chunk_index=1,
                cross_encoder_score=0.9,
            ),
        ],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.05),
    )

    _, grounded = _ground_variants_one(
        parent,
        chunks,
        {"parent-risk": [{"risk_id": "child-risk", "name": "Child", "description": "Child description"}]},
        client,
        "test-model",
        None,
        chunk_contexts=contexts,
        chunk_context_indices=indices,
    )

    # Grounding ran against the chunk the attributed span points at.
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m_["content"] for m_ in sent if "encryption at rest" in m_["content"])
    assert contexts[1] in user_content
    e = grounded["child-risk"][0][0]
    assert e.chunk_index == 1


def test_ground_variants_one_all_unattributed_returns_empty():
    """A parent backed only by unattributed spans skips variant grounding."""
    from asago_policy_mapper.extract.pipeline import _ground_variants_one

    chunks = _grounding_chunks()
    client = MagicMock()

    parent = RiskMatch(
        risk_id="parent-risk",
        risk_name="Parent",
        risk_description="Parent description",
        taxonomy="tax",
        confidence=None,
        confidence_source="none",
        grounding_confidence="high",
        accepted_by="expansion",
        evidence=[EvidenceSpan(text="unattributed.", document="docA.md", chunk_index=None, verification="not_found")],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.05),
    )

    _, grounded = _ground_variants_one(
        parent,
        chunks,
        {"parent-risk": [{"risk_id": "child-risk", "name": "Child", "description": "Child description"}]},
        client,
        "test-model",
        None,
        chunk_contexts=None,
        chunk_context_indices=None,
    )

    assert grounded == {}
    client.chat.completions.create.assert_not_called()
