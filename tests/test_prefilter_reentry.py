"""Re-entry tests for the opt-in pre-grounding filter.

Expansion behaviour does not change and there is no blacklist: a dropped
candidate can come back through sibling expansion. The pipeline counts
how often that happens — ``prefilter_reentered`` (dropped catalog ids
expansion proposes again), ``prefilter_reentered_grounded`` (re-entered
ids that then grounded), and a separate parent count, because a dropped
synthetic parent counts as its member variant ids.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.extract.models import RetrievalConfig, ScoredCandidate
from asago_policy_mapper.extract.pipeline import Prefilter, run_extraction
from asago_policy_mapper.extract.screening import (
    Answer,
    AnswerBatch,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
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


class ScriptedAssessor:
    """An assessor answering from a (risk_id -> Answer) script."""

    answer_types = frozenset({ProbabilityAnswer})
    sends_offsite = False
    prompt_name = "scripted"

    def __init__(self, script: dict[str, Answer], assessor_id: str = "scripted:test"):
        self.script = script
        self.assessor_id = assessor_id
        self.calls: list[tuple[object, str, list[DecisionCase]]] = []

    def assess(self, question, context, cases):
        self.calls.append((question, context, list(cases)))
        answers = {c.risk_id: (self.script.get(c.risk_id, NoAnswer("missing")),) for c in cases}
        return AnswerBatch(answers=answers, unmatched_ids=(), latency_ms=0.5, exchange=None)


class ThresholdRule:
    """DropThreshold semantics, re-implemented here for independence from the screening code."""

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


DOC = (
    "AI policy document about data governance and risk management. "
    "We must ensure bias detection and mitigation strategies are in place. "
    "Training data integrity is critical for model reliability. "
    "Personal data must be protected during every processing stage. "
    "Third-party model suppliers undergo security review before deployment."
)


@pytest.fixture
def mock_config():
    from asago_policy_mapper.llm import LLMConfig

    return LLMConfig(base_url="http://localhost:8000/v1", model="test-model", max_context=0)


# A catalog where the synthetic parent "unauthorized-processing" (variants
# x---biometric-data, x---location-data) and R-003 share the isPartOf group
# "privacy-group": grounding R-003 proposes the parent's variants as
# siblings, so a dropped parent re-enters through expansion.
VARIANT_RISKS = [
    _make_risk(
        "unauthorized-processing---biometric-data",
        "Unauthorized Processing - Biometric Data",
        "Processing biometric data without a lawful basis",
        parent="privacy-group",
    ),
    _make_risk(
        "unauthorized-processing---location-data",
        "Unauthorized Processing - Location Data",
        "Processing location data without a lawful basis",
        parent="privacy-group",
    ),
    _make_risk("R-003", "Privacy Violation", "Unauthorized use of personal data by AI", parent="privacy-group"),
]


class _VariantIndex:
    """A collapsed index over VARIANT_RISKS: parent plus R-003."""

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


def test_prefilter_reentry_counted(mock_config, tmp_path, monkeypatch):
    """A dropped candidate that expansion proposes again is counted, not blocked.

    The dropped synthetic parent counts as its member variant ids; the
    parent gets its own count. Re-entered ids that ground are counted
    separately. The grounded outcome matches a filter-off run: expansion
    behaviour is unchanged, with no blacklist.
    """
    from asago_policy_mapper.extract import pipeline

    def fake_ground(
        chunk_text,
        candidates,
        client,
        model,
        document,
        chunk_index,
        page=None,
        section=None,
        records=None,
        max_batch_size=0,
    ):
        return {c.risk_id: ([], "medium") for c in candidates if c.risk_id == "R-003"}

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
        records=None,
    ):
        return {}

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _VariantIndex())
    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)
    monkeypatch.setattr(pipeline, "ground_variants", fake_ground_variants)

    def ground_group(chunks, chunk_indices, risks, client, model, document, records=None):
        # Expansion grounds every risk it is asked about, including the
        # re-entered dropped variants.
        return {r["risk_id"]: ([], "medium") for r in risks if r["risk_id"].startswith("unauthorized-processing---")}

    monkeypatch.setattr(pipeline, "ground_risk_group", ground_group)

    doc = tmp_path / "p.md"
    doc.write_text(DOC)
    args = dict(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=VARIANT_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        ocr=False,
    )

    with_filter = run_extraction(
        **args,
        prefilter=Prefilter(
            assessor=ScriptedAssessor({"unauthorized-processing": ProbabilityAnswer(0.02)}), rule=ThresholdRule()
        ),
    )
    without_filter = run_extraction(**args)

    stats = with_filter.metadata["expansion_stats"]
    # The dropped parent was re-proposed as its two member variants.
    assert stats["prefilter_reentered"] == 2
    assert stats["prefilter_reentered_parents"] == 1
    # Both re-entered variants then grounded through expansion.
    assert stats["prefilter_reentered_grounded"] == 2
    # No blacklist: the re-entered variants are in the final risks, exactly
    # as in the filter-off run.
    reentered_ids = {
        "unauthorized-processing---biometric-data",
        "unauthorized-processing---location-data",
    }
    assert reentered_ids <= {r.risk_id for r in with_filter.risks}
    assert {r.risk_id for r in with_filter.risks} == {r.risk_id for r in without_filter.risks}


def test_prefilter_reentry_absent_without_drops(mock_config, tmp_path, monkeypatch):
    """With no prefilter drops, the re-entry keys do not appear in the stats.

    Filter-off runs keep the exact expansion_stats shape they had before
    the filter existed.
    """
    from asago_policy_mapper.extract import pipeline

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _VariantIndex())
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
        risks=VARIANT_RISKS,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=Prefilter(assessor=ScriptedAssessor({}), rule=ThresholdRule()),
        ocr=False,
    )

    # The assessor bypassed everything (missing -> keep), so nothing was
    # dropped and no re-entry keys appear.
    assert "prefilter_reentered" not in result.metadata["expansion_stats"]
    assert "prefilter_reentered_grounded" not in result.metadata["expansion_stats"]


def test_prefilter_reentry_zero_when_nothing_reproposed(mock_config, tmp_path, monkeypatch):
    """Drops that expansion never proposes again count zero, and the keys exist."""
    from asago_policy_mapper.extract import pipeline

    # A catalog with no expansion edges: R-001 and R-002 are unrelated.
    plain_risks = [
        _make_risk("R-001", "Model Bias", "Systematic errors in AI outputs"),
        _make_risk("R-002", "Data Poisoning", "Malicious manipulation of training data"),
    ]

    class _PlainIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
            return [
                ScoredCandidate(risk_id=r.id, risk_name=r.name, risk_description=r.description, cross_encoder_score=0.9)
                for r in plain_risks
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _PlainIndex())
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
        risks=plain_risks,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=Prefilter(assessor=ScriptedAssessor({"R-001": ProbabilityAnswer(0.01)}), rule=ThresholdRule()),
        ocr=False,
    )

    stats = result.metadata["expansion_stats"]
    assert stats["prefilter_reentered"] == 0
    assert stats["prefilter_reentered_grounded"] == 0
    assert "prefilter_reentered_parents" not in stats


def test_prefilter_reentry_parents_zero_when_expansion_empty(mock_config, tmp_path, monkeypatch):
    """A dropped parent with no expansion proposals records zero, not absence.

    The parent count must appear whenever a parent was dropped, even when
    expansion proposes nothing at all, so a run's stats always answer "how
    many dropped groups came back?".
    """
    from asago_policy_mapper.extract import pipeline

    # A variant catalog with no isPartOf edges: the parent collapses, but
    # nothing grounds alongside it to drive expansion.
    variant_risks = [
        _make_risk(
            "unauthorized-processing---biometric-data",
            "Unauthorized Processing - Biometric Data",
            "Processing biometric data without a lawful basis",
        ),
    ]

    class _LonelyVariantIndex:
        variant_map = {
            "unauthorized-processing": [
                {
                    "risk_id": "unauthorized-processing---biometric-data",
                    "name": "Unauthorized Processing - Biometric Data",
                    "description": "Processing biometric data without a lawful basis",
                    "taxonomy": "test-taxonomy",
                }
            ]
        }
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0):
            return [
                ScoredCandidate(
                    risk_id="unauthorized-processing",
                    risk_name="Unauthorized Processing (1 variant)",
                    risk_description=(
                        "Unauthorized Processing: risk of AI systems involving unauthorized "
                        "processing across categories including biometric-data."
                    ),
                    cross_encoder_score=0.9,
                )
            ]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _LonelyVariantIndex())
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
        risks=variant_risks,
        retrieval=RetrievalConfig(query_gen=False, no_judge=True),
        prefilter=Prefilter(
            assessor=ScriptedAssessor({"unauthorized-processing": ProbabilityAnswer(0.01)}), rule=ThresholdRule()
        ),
        ocr=False,
    )

    stats = result.metadata["expansion_stats"]
    assert stats["prefilter_reentered"] == 0
    assert stats["prefilter_reentered_grounded"] == 0
    assert stats["prefilter_reentered_parents"] == 0
