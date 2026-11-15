"""Pipeline tests for the opt-in pre-grounding filter (asago-uzfb1.7 step 2).

The filter runs once per chunk over ``cr.accepted`` after the judge, uses
exactly the grounding context (``chunk_contexts[i]``), and builds
filtered copies of each ``ChunkResult`` for grounding while the original
``accepted`` lists stay untouched (``_build_found_risk_chunks`` and the
expansion-context choice read them). A synthetic parent (``---`` id) is
sent with ``is_variant_parent=True``; a dropped parent never reaches
variant grounding.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.extract.models import DecisionCallRecord, RetrievalConfig, ScoredCandidate
from asago_policy_mapper.extract.pipeline import Prefilter, run_extraction
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


def _prefilter(script: dict[str, Answer], threshold: float = 0.10) -> Prefilter:
    return Prefilter(assessor=ScriptedAssessor(script), rule=ThresholdRule(threshold))


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
    with pytest.raises(ValueError, match="prefilter requires grounding"):
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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


def test_prefilter_records_wire_requests_when_assessor_batches(mock_config, tmp_path, monkeypatch):
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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
        prefilter=Prefilter(assessor=BatchingAssessor({}), rule=ThresholdRule()),
        ocr=False,
    )

    assert len(result.decision_calls) >= 1
    request = result.decision_calls[0].request
    # The wire-reported requests land in the record as a list of bodies,
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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
        prefilter=Prefilter(assessor=assessor, rule=ThresholdRule()),
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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
    # it did not accept it.
    assert by_id["R-001"].accepted_by == "threshold"
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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
        prefilter=Prefilter(assessor=assessor, rule=ThresholdRule()),
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

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
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
        prefilter=Prefilter(assessor=assessor, rule=ThresholdRule(0.10)),
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
