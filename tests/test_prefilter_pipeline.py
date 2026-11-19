"""Pipeline tests for the opt-in pre-grounding filter.

The filter runs once per chunk over ``cr.accepted`` after the judge, uses
exactly the grounding context (``chunk_contexts[i]``), and builds
filtered copies of each ``ChunkResult`` for grounding while the original
``accepted`` lists stay untouched (``_build_found_risk_chunks`` and the
expansion-context choice read them). A synthetic parent (``---`` id) is
sent with ``is_variant_parent=True``; a dropped parent never reaches
variant grounding.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.extract.models import DecisionCallRecord, RetrievalConfig, ScoredCandidate
from asago_policy_mapper.extract.pipeline import GroundingPrefilter, run_extraction
from asago_policy_mapper.extract.screening import (
    Answer,
    AnswerBatch,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
    Question,
)


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

VARIANT_RISKS = [
    _make_risk(
        "unauthorized-processing---biometric-data",
        "Unauthorized Processing - Biometric Data",
        "Processing biometric data without a lawful basis",
    ),
    _make_risk(
        "unauthorized-processing---location-data",
        "Unauthorized Processing - Location Data",
        "Processing location data without a lawful basis",
    ),
    _make_risk("R-003", "Privacy Violation", "Unauthorized use of personal data by AI"),
]


class ScriptedAssessor:
    """An assessor that answers from a (risk_id -> Answer) script.

    Records every question/context/cases triple it was asked, so tests assert
    on exactly what the stage sent. ProbabilityAnswer-only answer_types, like the real
    HTTP assessor.
    """

    answer_types = frozenset({ProbabilityAnswer})
    sends_offsite = False
    prompt_name = "scripted"

    def __init__(self, script: dict[str, Answer], assessor_id: str = "scripted:test"):
        self.script = script
        self.assessor_id = assessor_id
        self.calls: list[tuple[Question, str, list[DecisionCase]]] = []

    def assess(self, question, context, cases):
        self.calls.append((question, context, list(cases)))
        answers = {c.risk_id: (self.script.get(c.risk_id, NoAnswer("missing")),) for c in cases}
        return AnswerBatch(answers=answers, unmatched_ids=(), latency_ms=0.5, exchange=None)


class ThresholdRule:
    """DropThreshold semantics, re-implemented here for independence.

    Drop when every probability is <= 0.10; NoAnswer keeps with a
    ``no_answer:<reason>`` reason. (Using the real DropThreshold would couple
    these tests to the probability-assessor PR; the rule is pinned there.)
    """

    rule_id = "drop-threshold/1"

    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def decide(self, cases, batch):
        from asago_policy_mapper.extract.screening import Decision

        decisions = []
        for case in cases:
            answers = batch.answers.get(case.risk_id, ())
            unavailable = next((a for a in answers if isinstance(a, NoAnswer)), None)
            if unavailable is not None:
                decisions.append(Decision(case.risk_id, True, f"no_answer:{unavailable.reason}"))
            else:
                p = max(a.probability for a in answers if isinstance(a, ProbabilityAnswer))
                decisions.append(Decision(case.risk_id, p > self.threshold, f"p={p}"))
        return decisions


def _prefilter(script: dict[str, Answer], threshold: float = 0.10) -> GroundingPrefilter:
    return GroundingPrefilter(assessor=ScriptedAssessor(script), rule=ThresholdRule(threshold))


@pytest.fixture
def mock_config():
    from asago_policy_mapper.llm import LLMConfig

    return LLMConfig(base_url="http://localhost:8000/v1", model="test-model", max_context=0)


def _run(doc_text, risks, retrieval, client, mock_config, prefilter):
    doc = _write_doc(doc_text)
    return run_extraction(
        documents=[doc],
        client=client,
        config=mock_config,
        risks=risks,
        retrieval=retrieval,
        prefilter=prefilter,
        ocr=False,
    )


def _write_doc(text):
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    doc = tmp / "policy.md"
    doc.write_text(text)
    return doc


DOC = (
    "AI policy document about data governance and risk management. "
    "We must ensure bias detection and mitigation strategies are in place. "
    "Training data integrity is critical for model reliability. "
    "Personal data must be protected during every processing stage. "
    "Third-party model suppliers undergo security review before deployment."
)


def test_prefilter_refuses_no_grounding(mock_config, tmp_path):
    """prefilter + retrieval.no_grounding raises ValueError."""
    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    with pytest.raises(ValueError, match="^prefilter requires grounding; it filters what grounding sees"):
        _run(
            DOC,
            MOCK_RISKS,
            RetrievalConfig(no_grounding=True, query_gen=False),
            MagicMock(),
            mock_config,
            _prefilter({}),
        )


def test_prefilter_off_is_noop(mock_config, tmp_path, monkeypatch):
    """With prefilter=None the result equals the base result byte for byte.

    Volatile fields (timestamp, stage timings, call durations) differ
    between any two runs, so both runs are canonicalized the same way
    before comparing; everything else must match exactly.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        return {c.risk_id: ([], "medium") for c in candidates if c.risk_id != "R-003"}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    def _canonical(result):
        import json as _json

        data = result.model_dump()
        # The same volatile fields the snapshot harness strips: the run
        # timestamp, per-stage timings and per-call durations differ
        # between any two runs of the same code.
        data.pop("metadata", None)
        stats = data.get("retrieval_stats", {})
        if isinstance(stats, dict) and isinstance(stats.get("timing_ms"), dict):
            stats["timing_ms"] = sorted(stats["timing_ms"])
        for call in data.get("llm_calls", []):
            if isinstance(call, dict) and isinstance(call.get("duration_ms"), (int, float)):
                call["duration_ms"] = "T"
        volatile_free = _json.dumps(data, sort_keys=True)
        metadata = dict(result.metadata)
        metadata.pop("timestamp")
        return volatile_free, _json.dumps(metadata, sort_keys=True)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    args = dict(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        ocr=False,
    )
    base = run_extraction(**args)
    filtered_off = run_extraction(**args)  # prefilter=None is the default

    assert _canonical(base) == _canonical(filtered_off)
    assert "decision_calls" not in base.model_dump()
    assert "prefilter" not in base.metadata


def test_prefilter_records_exchange_requests_when_assessor_batches(mock_config, tmp_path, monkeypatch):
    """A batching assessor's call record holds the requests it actually sent.

    The HTTP probability assessor splits at ``max_batch``, so a chunk with
    more candidates than the batch limit produces several request bodies;
    the record must reflect those boundaries, not one synthetic whole-run
    request that was never sent.
    """
    from asago_policy_mapper.extract import pipeline

    class BatchingAssessor(ScriptedAssessor):
        """Sends two batches of one item each, reporting both in exchange."""

        def assess(self, question, context, cases):
            self.calls.append((question, context, list(cases)))
            answers = {c.risk_id: (ProbabilityAnswer(0.5),) for c in cases}
            return AnswerBatch(
                answers=answers,
                unmatched_ids=(),
                latency_ms=0.5,
                exchange={
                    "requests": [{"context": context, "items": [{"id": c.risk_id, "text": "x"}]} for c in cases],
                },
            )

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(
        pipeline,
        "ground_and_extract_evidence",
        lambda **kwargs: {},
    )

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=BatchingAssessor({}), rule=ThresholdRule()),
        ocr=False,
    )

    assert len(result.decision_calls) >= 1
    request = result.decision_calls[0].request
    # The exchange-reported requests land in the record as a list of bodies,
    # one per batch, with their boundaries preserved: every body carries
    # exactly its own case's id, in order, against the same context.
    assert isinstance(request, list)
    assert len(request) == len(MOCK_RISKS)
    assert [body["items"][0]["id"] for body in request] == [r.id for r in MOCK_RISKS]
    assert all(body["context"] == request[0]["context"] for body in request)
    # A list-valued request round-trips through the record's serialized form.
    restored = DecisionCallRecord(**result.decision_calls[0].model_dump())
    assert restored.request == request


def test_prefilter_drops_at_0_10_keeps_above(mock_config, tmp_path, monkeypatch):
    """p <= 0.10 drops before grounding; p > 0.10 is grounded."""
    from asago_policy_mapper.extract import pipeline

    grounded_ids: list[str] = []

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        grounded_ids.extend(c.risk_id for c in candidates)
        return {c.risk_id: ([], "medium") for c in candidates}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter(
            {"R-001": ProbabilityAnswer(0.05), "R-002": ProbabilityAnswer(0.11), "R-003": ProbabilityAnswer(0.10)}
        ),
        ocr=False,
    )

    # R-002 (0.11 > 0.10) reached grounding; R-001 and R-003 (<= 0.10) did not.
    assert "R-001" not in grounded_ids
    assert "R-003" not in grounded_ids
    assert "R-002" in grounded_ids
    # The filter stage is timed under its own key.
    assert "prefilter_ms" in result.retrieval_stats.timing_ms
    # The drops are recorded with provenance.
    dropped = {fc.risk_id: fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"}
    assert set(dropped) == {"R-001", "R-003"}
    assert dropped["R-001"].score == 0.05
    assert dropped["R-003"].score == 0.10


def test_prefilter_bypass_keeps_and_records(mock_config, tmp_path, monkeypatch):
    """Timeout/invalid/missing outcomes keep the candidate (fail-open)."""
    from asago_policy_mapper.extract import pipeline

    grounded_ids: list[str] = []

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        grounded_ids.extend(c.risk_id for c in candidates)
        return {}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter(
            {
                "R-001": NoAnswer("timeout"),
                "R-002": NoAnswer("invalid"),
                "R-003": NoAnswer("missing"),
            }
        ),
        ocr=False,
    )

    # Every candidate bypassed the filter and reached grounding.
    assert set(grounded_ids) == {"R-001", "R-002", "R-003"}
    assert not [fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"]
    # The decision log records the assessor and the bypass reasons.
    assert all(call.decision.assessor_id == "scripted:test" for call in result.decision_calls)
    items = [i for call in result.decision_calls for i in call.decision.items]
    reasons = {i.risk_id: i.reason for i in items}
    assert "timeout" in reasons["R-001"]
    assert "invalid" in reasons["R-002"]


def test_prefilter_drops_variant_group_together(mock_config, tmp_path, monkeypatch):
    """A dropped synthetic parent never reaches variant grounding."""
    from asago_policy_mapper.extract import pipeline

    grounded_ids: list[str] = []
    variant_calls: list[str] = []

    class _StubIndex:
        def __init__(self):
            self._variant_map = {
                "unauthorized-processing": [
                    {
                        "risk_id": "unauthorized-processing---biometric-data",
                        "name": "Unauthorized Processing - Biometric Data",
                        "description": "Processing biometric data without a lawful basis",
                        "taxonomy": "test-taxonomy",
                    },
                    {
                        "risk_id": "unauthorized-processing---location-data",
                        "name": "Unauthorized Processing - Location Data",
                        "description": "Processing location data without a lawful basis",
                        "taxonomy": "test-taxonomy",
                    },
                ]
            }

        @property
        def variant_map(self):
            return self._variant_map

        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            # The collapsed index returns the synthetic parent, not the variants.
            return [
                ScoredCandidate(
                    risk_id="unauthorized-processing",
                    risk_name="Unauthorized Processing (2 variants)",
                    risk_description=(
                        "Unauthorized Processing: risk of AI systems involving unauthorized "
                        "processing across categories including biometric-data, location-data."
                    ),
                    cross_encoder_score=0.9,
                ),
                ScoredCandidate(
                    risk_id="R-003",
                    risk_name="Privacy Violation",
                    risk_description="Unauthorized use of personal data by AI",
                    cross_encoder_score=0.85,
                ),
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

        def expand_variants(self, candidates):
            return candidates

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        grounded_ids.extend(c.risk_id for c in candidates)
        return {c.risk_id: ([], "medium") for c in candidates if not c.risk_id.startswith("unauthorized-processing")}

    def fake_ground_variants(
        *,
        chunk_text,
        parent_id,
        parent_name,
        parent_description,
        variants,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
    ):
        variant_calls.append(parent_id)
        return {}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)
    monkeypatch.setattr(pipeline, "ground_variants", fake_ground_variants)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    assessor = ScriptedAssessor({"unauthorized-processing": ProbabilityAnswer(0.02)})
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=VARIANT_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    # The parent was dropped: neither it nor any variant was grounded.
    assert "unauthorized-processing" not in grounded_ids
    assert variant_calls == []
    dropped = [fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"]
    assert {fc.risk_id for fc in dropped} == {"unauthorized-processing"}
    # The parent id carries no "---": the case was flagged synthetic by the
    # index's variant map, not by the id text.
    question, context, cases = assessor.calls[0]
    by_id = {c.risk_id: c for c in cases}
    assert by_id["unauthorized-processing"].is_variant_parent is True
    assert by_id["R-003"].is_variant_parent is False


def test_prefilter_preserves_expansion_context(mock_config, tmp_path, monkeypatch):
    """Dropped candidates still count for expansion-context choice.

    ``_build_found_risk_chunks`` and the expansion grouping read the
    original accepted lists; the filter must not shrink them. This test
    pins the filtered-copies rule directly: the original ChunkResult
    accepted lists are compared before and after the run, and the
    expansion stage is shown the originals, not the filtered copies.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        return {c.risk_id: ([], "medium") for c in candidates}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    expansion_risks = [
        _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs", parent="group-fairness"),
        _make_risk("R-004", "Data Bias", "Bias from training data distributions", parent="group-fairness"),
    ]

    expansion_stage_inputs = {}
    real_expansion_stage = pipeline._run_expansion_stage

    def spy_expansion_stage(merged, risks, chunk_results, *args, **kwargs):
        expansion_stage_inputs["accepted_ids"] = sorted(
            c.risk_id for cr in chunk_results if cr is not None for c in cr.accepted
        )
        return real_expansion_stage(merged, risks, chunk_results, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_run_expansion_stage", spy_expansion_stage)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=expansion_risks,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.01)}),
        ocr=False,
    )

    # The expansion stage saw the ORIGINAL accepted lists: R-001 (dropped by
    # the filter) is still there, so its group still provides expansion
    # context, exactly as a filter-off run would.
    assert "R-001" in expansion_stage_inputs["accepted_ids"]
    # And the dropped R-001 is absent from the grounded risks while its
    # sibling R-004 arrives through expansion (no blacklist: the expansion
    # itself may propose R-001 again).
    assert not any(r.risk_id == "R-001" and r.accepted_by != "expansion" for r in result.risks)


def test_dropped_candidates_distinguishable(mock_config, tmp_path, monkeypatch):
    """A prefilter drop and a grounding rejection differ by dropped_by/score/reason."""
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        # R-002 reaches grounding but is rejected by it.
        return {}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.03), "R-003": ProbabilityAnswer(0.5)}),
        ocr=False,
    )

    by_id = {fc.risk_id: fc for fc in result.grounding_filtered_candidates}
    assert by_id["R-001"].dropped_by == "prefilter"
    assert by_id["R-001"].score == 0.03
    assert by_id["R-001"].reason is not None
    # The drop keeps its retrieval provenance: the prefilter filtered it,
    # it did not accept it. Rank cutoff auto-accept labels as "rank".
    assert by_id["R-001"].accepted_by == "rank"
    # R-002 passed the filter, was sent to grounding, and was rejected there:
    # no prefilter provenance.
    assert by_id["R-002"].dropped_by is None
    assert by_id["R-002"].score is None
    assert by_id["R-002"].reason is None


def test_prefilter_skips_chunks_without_accepted_candidates(mock_config, tmp_path, monkeypatch):
    """Chunks with no accepted candidates pass through untouched, unassessed."""
    from asago_policy_mapper.extract import pipeline

    assessor = ScriptedAssessor({})

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return []

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(
        pipeline,
        "ground_and_extract_evidence",
        lambda **kwargs: {},
    )

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    # No chunk had accepted candidates, so the assessor was never called
    # and no decision call records exist.
    assert assessor.calls == []
    assert result.decision_calls == []


def test_metadata_records_offsite_and_fields(mock_config, tmp_path, monkeypatch):
    """metadata.prefilter records assessor/rule/threshold/host/sends_offsite/fields."""
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(
        pipeline,
        "ground_and_extract_evidence",
        lambda **kwargs: {},
    )

    assessor = ScriptedAssessor({"R-001": ProbabilityAnswer(0.5)}, assessor_id="scripted:test")
    assessor.sends_offsite = True
    # The real HTTP assessor carries its endpoint; the host is derived from
    # it for the metadata block.
    assessor.endpoint = SimpleNamespace(base_url="https://prob.example.com/v1")

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule(0.10)),
        ocr=False,
    )

    assert result.metadata["prefilter"] == {
        "assessor_id": "scripted:test",
        "rule_id": "drop-threshold/1",
        "threshold": 0.10,
        "endpoint_host": "prob.example.com",
        "sends_offsite": True,
        "fields_sent": ["chunk_context", "risk_name", "risk_description"],
    }


def test_prefilter_metadata_defaults_for_local_assessors(mock_config, tmp_path, monkeypatch):
    """A bare local assessor records empty ids, a None host and offsite False.

    ``getattr`` defaults exist for assessors without ``rule_id``-style
    attributes; this pins what the block shows for them: an empty
    assessor/rule id, ``None`` threshold and host, and a local run.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    # The threshold must survive: the default rule pins 0.10 in DropThreshold;
    # a rule with a different threshold must show it, not None.
    rule = SimpleNamespace(
        rule_id="custom/9",
        threshold=0.25,
        decide=lambda cases, batch: [
            __import__("asago_policy_mapper.extract.screening", fromlist=["Decision"]).Decision(c.risk_id, True, "keep")
            for c in cases
        ],
    )

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=ScriptedAssessor({}), rule=rule),
        ocr=False,
    )

    assert result.metadata["prefilter"] == {
        "assessor_id": "scripted:test",
        "rule_id": "custom/9",
        "threshold": 0.25,
        "endpoint_host": None,
        "sends_offsite": False,
        "fields_sent": ["chunk_context", "risk_name", "risk_description"],
    }


def test_prefilter_cases_carry_name_and_description(mock_config, tmp_path, monkeypatch):
    """The assessor sees each candidate's name and description verbatim.

    The fields_sent metadata promises ``risk_name`` and
    ``risk_description`` leave the machine; the cases must carry exactly
    what the index produced.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    assessor = ScriptedAssessor({})
    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    _, _, cases = assessor.calls[0]
    by_id = {c.risk_id: c for c in cases}
    assert by_id["R-001"].name == "Model Bias"
    assert by_id["R-001"].description == "Systematic errors in AI outputs favoring certain groups"


def test_prefilter_assessor_sees_the_probability_task(mock_config, tmp_path, monkeypatch):
    """The assessor is asked the grounding-likelihood question, not the judge's."""
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    assessor = ScriptedAssessor({})
    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    task, _, _ = assessor.calls[0]
    assert task is pipeline.GROUNDING_LIKELIHOOD_QUESTION_V1
    assert task.question_id == "grounding-likelihood"
    assert task.version == "1"


def test_prefilter_sends_the_chunk_context_the_grounder_sees(mock_config, tmp_path, monkeypatch):
    """The prefilter's context is exactly the chunk's grounding context.

    ``_run_prefilter_stage`` must hand ``chunk_contexts[i]`` to the
    assessor, unchanged; a None or empty context would silently change
    what the filter scores on. The grounder's context is captured
    independently, so the assessor and the grounder are shown to receive
    the same string rather than the same wrong string.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    grounding_contexts = []

    def fake_ground(chunk_text, candidates, client, model, document, chunk_index, **kwargs):
        grounding_contexts.append((chunk_index, chunk_text))
        return {}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    seen_contexts = []
    original_one_chunk = pipeline._prefilter_one_chunk

    def spy_one_chunk(cr, context, prefilter, index, retrieval):
        seen_contexts.append((cr.chunk_index, context))
        return original_one_chunk(cr, context, prefilter, index, retrieval)

    monkeypatch.setattr(pipeline, "_prefilter_one_chunk", spy_one_chunk)

    assessor = ScriptedAssessor({})
    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    # Every chunk with accepted candidates was asked with its own
    # grounding context, non-empty and one per chunk index.
    assert seen_contexts
    contexts = {i: c for i, c in seen_contexts}
    for i, c in seen_contexts:
        assert c and c == contexts[i]
    # The assessor saw exactly the strings the stage handed down, and
    # those strings are the same ones the grounder received for the same
    # chunk, captured independently of the stage.
    assessor_contexts = [ctx for _, ctx, _ in assessor.calls]
    assert [c for _, c in seen_contexts] == assessor_contexts
    assert grounding_contexts, "grounding must have run for the kept candidates"
    grounded_by_chunk = dict(grounding_contexts)
    for i, c in seen_contexts:
        assert c == grounded_by_chunk[i]
    first_context = assessor_contexts[0]
    assert DOC.split(".")[0].split()[0] in first_context


def test_prefilter_assesses_every_chunk_until_the_end(mock_config, tmp_path, monkeypatch):
    """A chunk with no accepted candidates does not stop later chunks.

    The stage skips empty chunks with ``continue``; mutating that to
    ``break`` would silently leave every later chunk unfiltered, so the
    empty chunk comes first and the chunk with candidates after it.
    """
    from asago_policy_mapper.extract import pipeline

    class _TwoChunkIndex:
        variant_map = {}
        risks = []

        def __init__(self):
            self.calls = 0

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            self.calls += 1
            # Chunk 0 is empty; chunk 1 gets candidates.
            if self.calls == 1:
                return []
            return [
                ScoredCandidate(risk_id="R-001", risk_name="Model Bias", risk_description="d", cross_encoder_score=0.9)
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    index = _TwoChunkIndex()
    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: index)
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    # A multi-chunk document: two sections produce at least two chunks.
    doc = tmp_path / "p.md"
    doc.write_text(
        "First section about data governance and risk management. " * 20
        + "\n\n## Second\n\n"
        + "Second section on bias detection and mitigation strategies. " * 20
    )
    assessor = ScriptedAssessor({"R-001": ProbabilityAnswer(0.01)})
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    # The chunk with candidates comes after the empty one: a break at the
    # empty chunk would leave decision_calls empty while accepted
    # candidates exist.
    assert result.decision_calls, "the stage must process every chunk with candidates"


def test_prefilter_drop_records_carry_full_provenance(mock_config, tmp_path, monkeypatch):
    """A drop record keeps taxonomy, rrf score, bm25 rank and accepted_by.

    The record documents why the candidate was retrievable at all before
    the prefilter dropped it; losing any field loses the audit trail.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(
                    risk_id="R-001",
                    risk_name="Model Bias",
                    risk_description="Systematic errors in AI outputs",
                    cross_encoder_score=0.9,
                    rrf_score=0.0123,
                    bm25_rank=7,
                )
            ]

        def get_taxonomy(self, rid):
            return f"taxonomy-{rid}"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.05)}),
        ocr=False,
    )

    dropped = [fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"]
    assert len(dropped) == 1
    fc = dropped[0]
    assert fc.taxonomy == "taxonomy-R-001"
    assert fc.rrf_score == 0.0123
    assert fc.bm25_rank == 7
    # Rank cutoff auto-accept labels as "rank"; the provenance survived the
    # prefilter's filtered copy.
    assert fc.accepted_by == "rank"
    assert fc.score == 0.05
    assert fc.reason == "p=0.05"


def test_prefilter_request_records_synthetic_body_keys(mock_config, tmp_path, monkeypatch):
    """An assessor without an exchange record carries the body it would have sent.

    The keys and per-item fields pin the fallback shape: ``context`` and
    ``items`` with ``id``/``text`` per case, exactly as the probability
    task's request format defines them, plus ``synthesized: true`` — the
    body was built for the record, not sent anywhere.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({}),
        ocr=False,
    )

    call = result.decision_calls[0]
    assert call.stage == "prefilter"
    assert call.chunk_index == 0
    assert isinstance(call.request, dict)
    assert set(call.request) == {"context", "items", "synthesized"}
    assert call.request["synthesized"] is True
    assert [i["id"] for i in call.request["items"]] == [r.id for r in MOCK_RISKS]
    assert call.request["items"][0]["text"] == f"{MOCK_RISKS[0].name}: {MOCK_RISKS[0].description}"
    # The decision log identifies the task, prompt, assessor and rule.
    log = call.decision
    assert log.question_id == "grounding-likelihood"
    assert log.question_version == "1"
    assert log.prompt_name == "prefilter"
    assert log.prompt_sha256 == ""
    assert log.assessor_id == "scripted:test"
    assert log.rule_id == "drop-threshold/1"
    assert log.latency_ms == 0.5
    assert log.unmatched_ids == []
    # Each item carries the raw answers the rule saw.
    r1 = next(i for i in log.items if i.risk_id == "R-001")
    assert r1.answers == [{"kind": "no_answer", "reason": "missing", "detail": ""}]
    assert r1.keep is True


def test_prefilter_copy_preserves_chunk_identity_fields(mock_config, tmp_path, monkeypatch):
    """The filtered copy keeps the chunk's source, page, section and stats.

    Grounding reads the copy; losing any identity field would misattribute
    the kept candidates' evidence.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(
                    risk_id=r.id,
                    risk_name=r.name,
                    risk_description=r.description,
                    cross_encoder_score=0.9,
                )
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(
        pipeline,
        "ground_and_extract_evidence",
        lambda **kwargs: {c.risk_id: ([], "medium") for c in kwargs["candidates"]},
    )

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.05)}),
        ocr=False,
    )

    # The grounded match for the kept candidate carries the chunk's
    # identity: R-002 survived the filter and grounded with evidence.
    kept = {m.risk_id: m for m in result.risks}
    assert "R-002" in kept
    assert "R-001" not in kept


def test_prefilter_copy_carries_chunk_metadata_to_grounding(mock_config, tmp_path, monkeypatch):
    """Grounding sees the copy's page, section and borderline list intact.

    ``_ground_one`` reads page/section from the ChunkResult it is given;
    the filtered copy must carry them, and the borderline list must stay
    the original object so judged provenance survives.
    """
    from asago_policy_mapper.extract import pipeline

    grounding_calls = []

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        grounding_calls.append({"page": page, "section": section, "document": document})
        return {c.risk_id: ([], "medium") for c in candidates}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    originals = {}
    real_one_chunk = pipeline._prefilter_one_chunk

    def spy_one_chunk(cr, context, prefilter, index, retrieval):
        copy, drops, call = real_one_chunk(cr, context, prefilter, index, retrieval)
        originals[cr.chunk_index] = (cr, copy)
        return copy, drops, call

    monkeypatch.setattr(pipeline, "_prefilter_one_chunk", spy_one_chunk)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.05)}),
        ocr=False,
    )

    for _, (original, copy) in originals.items():
        assert copy.source == original.source
        assert copy.page == original.page
        assert copy.section == original.section
        assert copy.borderline is original.borderline
        assert copy.borderline_judged is original.borderline_judged
    assert grounding_calls
    for call in grounding_calls:
        assert call["document"] == str(doc)


def test_prefilter_drop_keeps_auto_promoted_provenance(mock_config, tmp_path, monkeypatch):
    """A dropped borderline candidate keeps its auto_promoted provenance.

    With ``no_judge``, borderline candidates move into accepted and are
    recorded in ``borderline_judged``; their drop records must show
    ``auto_promoted`` (or ``llm_judge`` when judged), proving the drop
    record recomputes provenance from the original candidate, not a
    blanket guess.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    # top_n_accept=1: R-002 and R-003 land in borderline; no_judge moves
    # them into accepted with borderline_judged provenance.
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True, top_n_accept=1, top_n_judge=5),
        prefilter=_prefilter({"R-002": ProbabilityAnswer(0.05), "R-003": ProbabilityAnswer(0.05)}),
        ocr=False,
    )

    dropped = {fc.risk_id: fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"}
    assert set(dropped) == {"R-002", "R-003"}
    assert all(fc.accepted_by == "auto_promoted" for fc in dropped.values())


def test_prefilter_copy_stats_are_independent(mock_config, tmp_path, monkeypatch):
    """The copy's stats dict is a fresh dict, not the original's mapping.

    Grounding mutates nothing in the copy's stats today, but sharing the
    dict would let any later in-place change leak back into the original
    ChunkResult.
    """
    from asago_policy_mapper.extract import pipeline

    originals = {}
    real_one_chunk = pipeline._prefilter_one_chunk

    def spy_one_chunk(cr, context, prefilter, index, retrieval):
        copy, drops, call = real_one_chunk(cr, context, prefilter, index, retrieval)
        originals[cr.chunk_index] = (cr, copy)
        return copy, drops, call

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})
    monkeypatch.setattr(pipeline, "_prefilter_one_chunk", spy_one_chunk)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.05)}),
        ocr=False,
    )

    assert originals
    for _, (original, copy) in originals.items():
        assert copy.stats is not original.stats
        assert copy.stats == original.stats


def test_prefilter_accepted_by_flows_through_filter(mock_config, tmp_path, monkeypatch):
    """A kept candidate keeps the accepted_by its retrieval mode earns.

    With ``no_judge`` and cross-encoder ranking, auto-accepted candidates
    keep ``threshold`` provenance through the filter.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(
                    risk_id=r.id,
                    risk_name=r.name,
                    risk_description=r.description,
                    cross_encoder_score=0.75,
                )
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(
        pipeline,
        "ground_and_extract_evidence",
        lambda **kwargs: {c.risk_id: ([], "medium") for c in kwargs["candidates"]},
    )

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.05)}),
        ocr=False,
    )

    kept = {m.risk_id: m for m in result.risks}
    assert "R-002" in kept
    # Auto-accepted by rank cutoff: the provenance survived the filtered copy.
    assert kept["R-002"].accepted_by == "rank"


def test_prefilter_unmatched_ids_and_latency_recorded(mock_config, tmp_path, monkeypatch):
    """The decision log carries the assessor's unmatched ids and latency."""
    from asago_policy_mapper.extract import pipeline

    class UnmatchedAssessor(ScriptedAssessor):
        def assess(self, question, context, cases):
            self.calls.append((question, context, list(cases)))
            return AnswerBatch(
                answers={c.risk_id: (ProbabilityAnswer(0.5),) for c in cases},
                unmatched_ids=("R-009",),
                latency_ms=12.5,
                exchange=None,
            )

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=UnmatchedAssessor({}), rule=ThresholdRule()),
        ocr=False,
    )

    log = result.decision_calls[0].decision
    assert log.unmatched_ids == ["R-009"]
    assert log.latency_ms == 12.5


def test_prefilter_drop_with_no_outcomes_records_empty_payload(mock_config, tmp_path, monkeypatch):
    """A rule that drops a case the assessment never answered records no outcomes.

    The drop record and the decision item both fall back to an empty
    outcome list; ``answer_payloads(())`` is ``[]``, and the record's
    score is None because no probability exists.
    """
    from asago_policy_mapper.extract import pipeline
    from asago_policy_mapper.extract.screening import Decision

    class EmptyAssessor(ScriptedAssessor):
        def assess(self, question, context, cases):
            self.calls.append((question, context, list(cases)))
            return AnswerBatch(answers={}, unmatched_ids=(), latency_ms=0.5, exchange=None)

    class DropEverythingRule:
        rule_id = "drop-all/1"

        def decide(self, cases, batch):
            return [Decision(c.risk_id, False, "no-answer") for c in cases]

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=EmptyAssessor({}), rule=DropEverythingRule()),
        ocr=False,
    )

    call = result.decision_calls[0]
    item = call.decision.items[0]
    assert item.answers == []
    assert item.keep is False
    assert item.reason == "no-answer"
    dropped = {fc.risk_id: fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"}
    assert set(dropped) == {"R-001", "R-002", "R-003"}
    assert all(fc.score is None for fc in dropped.values())
    assert all(fc.reason == "no-answer" for fc in dropped.values())


def test_prefilter_dropped_score_takes_max_probability(mock_config, tmp_path, monkeypatch):
    """A drop with several probabilities records the maximum.

    DropThreshold drops only when every probability is at or below the
    threshold; the recorded score is the strongest probability seen.
    """
    from asago_policy_mapper.extract import pipeline

    class MultiAnswerAssessor(ScriptedAssessor):
        def assess(self, question, context, cases):
            self.calls.append((question, context, list(cases)))
            outcomes = {
                "R-001": (ProbabilityAnswer(0.02), ProbabilityAnswer(0.08)),
                "R-002": (ProbabilityAnswer(0.5),),
                "R-003": (ProbabilityAnswer(0.9),),
            }
            return AnswerBatch(answers=outcomes, unmatched_ids=(), latency_ms=0.5, exchange=None)

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=MultiAnswerAssessor({}), rule=ThresholdRule()),
        ocr=False,
    )

    dropped = {fc.risk_id: fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"}
    assert dropped["R-001"].score == 0.08
    assert "R-002" not in dropped


def test_prefilter_offsite_without_endpoint_is_none_host(mock_config, tmp_path, monkeypatch):
    """An assessor with no endpoint records a None host, not a crash."""
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    # Neither an endpoint nor a base_url: the host must fall back to None.
    assessor = ScriptedAssessor({})
    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    assert result.metadata["prefilter"]["endpoint_host"] is None


def test_prefilter_endpoint_host_ignores_non_string_base_url(mock_config, tmp_path, monkeypatch):
    """A non-string base_url yields a None host, not a traceback.

    The host derivation must not hand ``urlsplit`` a number or other
    non-string; the isinstance guard is the only protection.
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    assessor = ScriptedAssessor({})
    assessor.endpoint = SimpleNamespace(base_url=12345)
    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=assessor, rule=ThresholdRule()),
        ocr=False,
    )

    assert result.metadata["prefilter"]["endpoint_host"] is None


def test_prefilter_rule_without_rule_id_records_empty(mock_config, tmp_path, monkeypatch):
    """A rule with no rule_id records an empty string, not None or a filler.

    The log's ``rule_id`` and the metadata block both default the
    attribute to ``""``; a missing attribute must not crash or invent a
    value.
    """
    from asago_policy_mapper.extract import pipeline
    from asago_policy_mapper.extract.screening import Decision

    class NoIdRule:
        """Keeps everything; carries no rule_id attribute at all."""

        def decide(self, cases, batch):
            return [Decision(c.risk_id, True, "keep") for c in cases]

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=ScriptedAssessor({}), rule=NoIdRule()),
        ocr=False,
    )

    assert result.decision_calls[0].decision.rule_id == ""
    assert result.metadata["prefilter"]["rule_id"] == ""


def test_prefilter_assessor_without_optional_attributes_records_defaults(mock_config, tmp_path, monkeypatch):
    """An assessor with no assessor_id/offsite records empty id and False.

    The metadata block's getattr defaults exist for minimal assessors;
    with no candidates the stage never reads the id directly, so the
    defaults are the only values recorded.
    """
    from asago_policy_mapper.extract import pipeline

    class _EmptyIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return []

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _EmptyIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    # No assessor_id, no sends_offsite, no endpoint: the metadata defaults apply.
    bare = SimpleNamespace(assess=lambda question, context, cases: None, answer_types=frozenset())
    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=bare, rule=ThresholdRule()),
        ocr=False,
    )

    block = result.metadata["prefilter"]
    assert block["assessor_id"] == ""
    assert block["sends_offsite"] is False
    assert block["endpoint_host"] is None
    assert result.decision_calls == []


def test_prefilter_nonparent_drop_reentry_counted(mock_config, tmp_path, monkeypatch):
    """A dropped plain risk that expansion re-proposes counts as re-entered.

    The dropped-id catalog adds non-parent drops under their own ids;
    losing that (for example recording None) would silently zero the
    re-entry count.
    """
    from asago_policy_mapper.extract import pipeline

    sibling_risks = [
        _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs", parent="sibling-group"),
        _make_risk("R-002", "Data Poisoning", "Malicious manipulation of training data", parent="sibling-group"),
    ]

    class _SiblingIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in sibling_risks
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _SiblingIndex())
    monkeypatch.setattr(
        pipeline,
        "ground_and_extract_evidence",
        lambda **kwargs: {c.risk_id: ([], "medium") for c in kwargs["candidates"] if c.risk_id == "R-002"},
    )

    def ground_group(chunks, chunk_indices, risks, client, model, document, call_collector=None):
        return {r["risk_id"]: ([], "medium") for r in risks if r["risk_id"] == "R-001"}

    monkeypatch.setattr(pipeline, "ground_risk_group", ground_group)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=sibling_risks,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=_prefilter({"R-001": ProbabilityAnswer(0.01)}),
        ocr=False,
    )

    stats = result.metadata["expansion_stats"]
    # R-002 grounded; expansion proposed its sibling R-001 (dropped): one
    # re-entry, which then grounded.
    assert stats["prefilter_reentered"] == 1
    assert stats["prefilter_reentered_grounded"] == 1
    assert "prefilter_reentered_parents" not in stats
    # No blacklist: the re-entered risk is in the final output.
    assert "R-001" in {m.risk_id for m in result.risks}


def test_prefilter_one_chunk_copy_keeps_page_and_section():
    """The filtered copy carries the chunk's page and section verbatim.

    A direct unit call pins the copy's identity fields without depending
    on a document format that produces non-None metadata.
    """
    from asago_policy_mapper.extract.retrieve import ChunkResult

    cr = ChunkResult(
        chunk_index=4,
        source="doc.pdf",
        page=7,
        section="Data Governance",
        accepted=[
            ScoredCandidate(
                risk_id="R-001",
                risk_name="Model Bias",
                risk_description="d",
                cross_encoder_score=0.9,
                accepted_by="rank",
            ),
            ScoredCandidate(
                risk_id="R-002",
                risk_name="Data Poisoning",
                risk_description="d",
                cross_encoder_score=0.8,
                accepted_by="rank",
            ),
        ],
        borderline=[],
        borderline_judged=[],
        stats={"candidates_retrieved": 2, "auto_accepted": 2, "borderline": 0, "discarded": 0, "bm25_rescued": 0},
    )

    class _Index:
        variant_map = {}

        def get_taxonomy(self, rid):
            return "t"

    from types import SimpleNamespace as _NS

    import asago_policy_mapper.extract.pipeline as pl

    prefilter = GroundingPrefilter(assessor=ScriptedAssessor({"R-001": ProbabilityAnswer(0.01)}), rule=ThresholdRule())
    copy, drops, call = pl._prefilter_one_chunk(
        cr, "ctx", prefilter, _Index(), _NS(no_judge=True, use_cross_encoder=True)
    )

    assert copy.page == 7
    assert copy.section == "Data Governance"
    assert copy.source == "doc.pdf"
    assert copy.chunk_index == 4
    assert [c.risk_id for c in copy.accepted] == ["R-002"]
    assert [fc.risk_id for fc in drops] == ["R-001"]
    assert drops[0].chunk_index == 4
    assert call.chunk_index == 4
    assert call.request["context"] == "ctx"


def test_prefilter_stage_context_fallback_for_mismatched_lengths():
    """A chunk beyond the contexts list gets an empty context, not a crash.

    ``_run_prefilter_stage`` guards the index because nothing guarantees
    the caller passed one context per chunk; the unit test pins the
    fallback ("" and not the wrong chunk's context) directly.
    """
    from asago_policy_mapper.extract.retrieve import ChunkResult

    cr = ChunkResult(
        chunk_index=3,
        source="p.md",
        page=1,
        section=None,
        accepted=[
            ScoredCandidate(risk_id="R-001", risk_name="Model Bias", risk_description="d", cross_encoder_score=0.9)
        ],
        borderline=[],
        borderline_judged=[],
        stats={"candidates_retrieved": 1, "auto_accepted": 1, "borderline": 0, "discarded": 0, "bm25_rescued": 0},
    )

    seen = []

    import asago_policy_mapper.extract.pipeline as pl

    real_one_chunk = pl._prefilter_one_chunk

    def spy(cr, context, prefilter, index, retrieval):
        seen.append(context)
        return real_one_chunk(cr, context, prefilter, index, retrieval)

    class _Index:
        variant_map = {}

        def get_taxonomy(self, rid):
            return "t"

    from types import SimpleNamespace as _NS

    prefilter = _NS(
        assessor=ScriptedAssessor({}),
        rule=ThresholdRule(),
    )
    import unittest.mock as _mock

    with _mock.patch.object(pl, "_prefilter_one_chunk", spy):
        pl._run_prefilter_stage(
            [None, cr], ["only-one-context"], prefilter, _Index(), _NS(no_judge=True, use_cross_encoder=True)
        )

    # Position 1 is one past the single context (len == 1): the stage
    # falls back to "" rather than raising or borrowing another chunk's
    # context.
    assert seen == [""]


def test_prefilter_stage_empty_contexts_list():
    """An empty contexts list means every chunk falls back to an empty context."""
    from asago_policy_mapper.extract.retrieve import ChunkResult

    cr = ChunkResult(
        chunk_index=0,
        source="p.md",
        page=None,
        section=None,
        accepted=[
            ScoredCandidate(risk_id="R-001", risk_name="Model Bias", risk_description="d", cross_encoder_score=0.9)
        ],
        borderline=[],
        borderline_judged=[],
        stats={"candidates_retrieved": 1, "auto_accepted": 1, "borderline": 0, "discarded": 0, "bm25_rescued": 0},
    )

    seen = []
    import asago_policy_mapper.extract.pipeline as pl

    real_one_chunk = pl._prefilter_one_chunk

    def spy(cr, context, prefilter, index, retrieval):
        seen.append(context)
        return real_one_chunk(cr, context, prefilter, index, retrieval)

    class _Index:
        variant_map = {}

        def get_taxonomy(self, rid):
            return "t"

    from types import SimpleNamespace as _NS

    prefilter = _NS(assessor=ScriptedAssessor({}), rule=ThresholdRule())
    import unittest.mock as _mock

    with _mock.patch.object(pl, "_prefilter_one_chunk", spy):
        pl._run_prefilter_stage([cr], [], prefilter, _Index(), _NS(no_judge=True, use_cross_encoder=True))

    assert seen == [""]


def test_prefilter_stage_matching_context_is_passed_verbatim():
    """A chunk with a matching context gets exactly that context string."""
    from asago_policy_mapper.extract.retrieve import ChunkResult

    cr = ChunkResult(
        chunk_index=1,
        source="p.md",
        page=None,
        section=None,
        accepted=[
            ScoredCandidate(risk_id="R-001", risk_name="Model Bias", risk_description="d", cross_encoder_score=0.9)
        ],
        borderline=[],
        borderline_judged=[],
        stats={"candidates_retrieved": 1, "auto_accepted": 1, "borderline": 0, "discarded": 0, "bm25_rescued": 0},
    )

    seen = []
    import asago_policy_mapper.extract.pipeline as pl

    real_one_chunk = pl._prefilter_one_chunk

    def spy(cr, context, prefilter, index, retrieval):
        seen.append(context)
        return real_one_chunk(cr, context, prefilter, index, retrieval)

    class _Index:
        variant_map = {}

        def get_taxonomy(self, rid):
            return "t"

    from types import SimpleNamespace as _NS

    prefilter = _NS(assessor=ScriptedAssessor({}), rule=ThresholdRule())
    import unittest.mock as _mock

    with _mock.patch.object(pl, "_prefilter_one_chunk", spy):
        pl._run_prefilter_stage(
            [None, cr],
            ["chunk-zero-context", "chunk-one-context"],
            prefilter,
            _Index(),
            _NS(no_judge=True, use_cross_encoder=True),
        )

    # The None chunk is skipped (no context asked), and position 1 gets
    # its own context verbatim.
    assert seen == ["chunk-one-context"]


# --- fail-open on assessor exceptions ---


def test_prefilter_assessor_exception_keeps_everything(mock_config, tmp_path, monkeypatch, caplog):
    """An assess() that raises keeps all candidates and records the failure.

    The filter must never lose risks to a broken assessor: the exception
    becomes a ``NoAnswer("error", ...)`` answer per case, which the
    keep-on-no-answer rule keeps, and a warning carries the
    assessor id, chunk index and error out of the run. The recorded
    detail is redacted against the endpoint's key, the same discipline
    the assessor applies to its exchange responses.
    """
    import logging
    from types import SimpleNamespace as _NS

    from asago_policy_mapper.extract import pipeline

    monkeypatch.setenv("PROB_API_KEY", "sentinel-secret-key")

    class ExplodingAssessor(ScriptedAssessor):
        def __init__(self, script):
            super().__init__(script, assessor_id="http-probability/test")
            self.endpoint = _NS(api_key_env="PROB_API_KEY")

        def assess(self, question, context, cases):
            self.calls.append((question, context, list(cases)))
            raise RuntimeError("request failed with key sentinel-secret-key in it")

    grounded_ids: list[str] = []

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        grounded_ids.extend(c.risk_id for c in candidates)
        return {c.risk_id: ([], "medium") for c in candidates}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    with caplog.at_level(logging.WARNING, logger="asago_policy_mapper.extract.pipeline"):
        result = run_extraction(
            documents=[doc],
            client=MagicMock(),
            config=mock_config,
            risks=MOCK_RISKS,
            retrieval=RetrievalConfig(query_gen=False, no_judge=True),
            prefilter=GroundingPrefilter(assessor=ExplodingAssessor({}), rule=ThresholdRule()),
            ocr=False,
        )

    # Every candidate still reached grounding (fail-open, not fail-closed).
    for risk in MOCK_RISKS:
        assert risk.id in grounded_ids
    # Nothing was dropped.
    assert result.retrieval_stats.prefilter_dropped == 0
    assert [fc for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"] == []
    # The warning names the assessor, the chunk and the error.
    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and "request failed with key" in r.getMessage()
    ]
    assert warnings, f"no warning recorded in {caplog.records}"
    message = warnings[0].getMessage()
    assert "http-probability/test" in message
    assert "chunk 0" in message
    # The decision log records the failure the same way an unanswered
    # case is recorded: every item keeps with a no_answer reason, and the
    # detail carries no credential — the exception message is redacted
    # against the endpoint's key.
    decision = result.decision_calls[0].decision
    assert len(decision.items) == len(MOCK_RISKS)
    assert all(item.keep for item in decision.items)
    assert all(item.reason == "no_answer:error" for item in decision.items)
    serialized = json.dumps(result.model_dump())
    assert "sentinel-secret-key" not in serialized
    detail = decision.items[0].answers[0]["detail"]
    assert "sentinel-secret-key" not in detail
    assert "[redacted]" in detail


def test_prefilter_assessor_exception_survives_serialization(mock_config, tmp_path, monkeypatch):
    """The failure-built batch round-trips through the result model."""

    class ExplodingAssessor(ScriptedAssessor):
        def assess(self, question, context, cases):
            raise RuntimeError("boom")

    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", lambda **kwargs: {})

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=GroundingPrefilter(assessor=ExplodingAssessor({}), rule=ThresholdRule()),
        ocr=False,
    )

    data = result.model_dump()
    assert data["decision_calls"]
    restored = result.model_validate(data)
    assert restored.decision_calls[0].decision.items
    assert all(item.keep for item in restored.decision_calls[0].decision.items)


# --- the real assessor against the real rule, over a mock transport ---


def test_prefilter_real_assessor_end_to_end(mock_config, tmp_path, monkeypatch):
    """The real DropThreshold over the real HttpProbabilityAssessor filters.

    The assessor runs against an ``httpx.MockTransport`` so the exchange
    never leaves the machine, but every other piece is production code:
    the endpoint, the assessor's request building, batching and exchange
    recording, and the rule's drop decision. Pins three behaviours: the
    request appears in ``decision_calls`` without the API key (the key
    travels in the Authorization header — the handler asserts it really
    was sent — and never reaches the serialized result), the 0.10
    boundary drops at-and-below and keeps above, and the serialized
    result carries no credential.
    """
    import httpx

    from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
    from asago_policy_mapper.extract.screening import DropThreshold

    monkeypatch.setenv("PROB_API_KEY", "sentinel-secret-key")
    sent_bodies = []
    sent_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(json.loads(request.content.decode("utf-8")))
        sent_headers.append(dict(request.headers))
        # Two items at 0.10 (dropped: at the threshold) and one at 0.9
        # (kept: above it), in the endpoint's scores response shape.
        scores = [
            {"id": "R-001", "p": 0.10},
            {"id": "R-002", "p": 0.10},
            {"id": "R-003", "p": 0.90},
        ]
        return httpx.Response(200, json={"scores": scores})

    transport = httpx.MockTransport(handler)
    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:8080",
        model="prob-model",
        api_key_env="PROB_API_KEY",
        timeout_s=5.0,
        max_batch=16,
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=transport)
    rule = DropThreshold(assessor, 0.10)
    prefilter = GroundingPrefilter(assessor=assessor, rule=rule)

    grounded_ids: list[str] = []

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in MOCK_RISKS
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        call_collector=None,
        max_batch_size=0,
    ):
        grounded_ids.extend(c.risk_id for c in candidates)
        return {c.risk_id: ([], "medium") for c in candidates}

    from asago_policy_mapper.extract import pipeline

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    result = run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=MOCK_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=prefilter,
        ocr=False,
    )

    # Exactly one real exchange happened, through the transport.
    assert len(sent_bodies) == 1
    # The exchange was authenticated: the key really traveled in the
    # Authorization header, so its absence from the result is a real
    # property of the recording, not an unset variable.
    assert sent_headers[0].get("authorization") == "Bearer sentinel-secret-key"
    assert "sentinel-secret-key" not in json.dumps(sent_bodies)
    # The 0.10 boundary: at-threshold drops, above keeps.
    assert "R-003" in grounded_ids
    assert "R-001" not in grounded_ids
    assert "R-002" not in grounded_ids
    dropped = {fc.risk_id for fc in result.grounding_filtered_candidates if fc.dropped_by == "prefilter"}
    assert dropped == {"R-001", "R-002"}
    assert result.retrieval_stats.prefilter_dropped == 2
    # The exchange appears in decision_calls, and no serialized form
    # carries the credential.
    assert result.decision_calls
    serialized = json.dumps(result.model_dump())
    assert "sentinel-secret-key" not in serialized
    # The record holds the request the assessor actually sent.
    record = result.decision_calls[0]
    assert isinstance(record.request, list)
    assert record.request[0]["items"][0]["id"] == "R-001"
