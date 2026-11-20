"""Tests for variant-selective grounding (ground_variants + pipeline integration)."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from asago_policy_mapper.extract.attribute import ground_variants
from asago_policy_mapper.extract.index import RiskIndex
from asago_policy_mapper.extract.models import (
    EvidenceSpan,
    LLMCallRecord,
    RetrievalConfig,
    RetrievalScores,
    RiskMatch,
    ScoredCandidate,
    _RiskEvidence,
    _RiskEvidenceList,
)
from asago_policy_mapper.extract.pipeline import (
    PipelineInvariantError,
    _assert_no_synthetic_parents,
    _ground_variants_one,
    _run_expansion,
    _run_variant_grounding,
    run_extraction,
)
from asago_policy_mapper.extract.retrieve import ChunkResult
from asago_policy_mapper.prompts import render_prompt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARENT_ID = "ai-risk-taxonomy-unauthorized-processing"
PARENT_NAME = "Unauthorized processing"
PARENT_DESC = "Processing of personal data without legal basis or consent."

VARIANTS = [
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---biometric-data-(facial-recognition)",
        "name": "Biometric data (facial recognition)",
        "description": "Unauthorized processing of biometric data for facial recognition.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---health-data",
        "name": "Health data",
        "description": "Unauthorized processing of health records.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---financial-data",
        "name": "Financial data",
        "description": "Unauthorized processing of financial records.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---location-data",
        "name": "Location data",
        "description": "Unauthorized processing of geolocation information.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---criminal-records",
        "name": "Criminal records",
        "description": "Unauthorized processing of criminal record data.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": "ai-risk-taxonomy-unauthorized-processing---children's-data",
        "name": "Children's data",
        "description": "Unauthorized processing of minors' personal data.",
        "taxonomy": "ai-risk-taxonomy",
    },
]

VARIANT_MAP = {PARENT_ID: VARIANTS}


def _make_parent_match(chunk_index=0):
    return RiskMatch(
        risk_id=PARENT_ID,
        risk_name=PARENT_NAME,
        risk_description=PARENT_DESC,
        taxonomy="ai-risk-taxonomy",
        confidence=0.85,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[
            EvidenceSpan(
                text="Biometric data processing without consent.",
                document="policy.pdf",
                chunk_index=chunk_index,
                cross_encoder_score=0.9,
            ),
        ],
        scores=RetrievalScores(
            bm25_rank=3,
            embedding_distance=0.15,
            cross_encoder_score=0.85,
            rrf_score=0.04,
        ),
    )


def _make_non_parent_match():
    return RiskMatch(
        risk_id="atlas-bias",
        risk_name="AI Bias",
        risk_description="Systematic bias in AI outputs.",
        taxonomy="ibm-risk-atlas",
        confidence=0.90,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[
            EvidenceSpan(
                text="AI systems must be fair.",
                document="policy.pdf",
                chunk_index=0,
            ),
        ],
        scores=RetrievalScores(
            bm25_rank=1,
            embedding_distance=0.1,
            cross_encoder_score=0.90,
            rrf_score=0.05,
        ),
    )


def _make_chunks(texts):
    return [SimpleNamespace(text=t, page=i, section=f"Section {i}") for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# ground_variants() unit tests
# ---------------------------------------------------------------------------


class TestGroundVariants:
    def test_selective_grounding(self):
        """LLM grounds 2 of 6 variants; only those 2 are returned."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[0]["risk_id"],
                    grounded=True,
                    confidence="high",
                    quotes=["facial recognition data was collected without consent"],
                ),
                _RiskEvidence(
                    risk_id=VARIANTS[1]["risk_id"],
                    grounded=False,
                    confidence="low",
                    quotes=[],
                    rejection_reason="No health data mentioned.",
                ),
                _RiskEvidence(
                    risk_id=VARIANTS[2]["risk_id"],
                    grounded=False,
                    confidence="low",
                    quotes=[],
                    rejection_reason="No financial data mentioned.",
                ),
                _RiskEvidence(
                    risk_id=VARIANTS[3]["risk_id"],
                    grounded=True,
                    confidence="medium",
                    quotes=["GPS tracking without user awareness"],
                ),
                _RiskEvidence(
                    risk_id=VARIANTS[4]["risk_id"],
                    grounded=False,
                    confidence="low",
                    quotes=[],
                ),
                _RiskEvidence(
                    risk_id=VARIANTS[5]["risk_id"],
                    grounded=False,
                    confidence="low",
                    quotes=[],
                ),
            ]
        )

        result = ground_variants(
            chunk_text="facial recognition data was collected without consent. GPS tracking without user awareness.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS,
            client=mock_client,
            model="test-model",
            document="policy.pdf",
            chunk_index=2,
            page=5,
            section="Privacy",
        )

        assert len(result) == 2
        assert VARIANTS[0]["risk_id"] in result
        assert VARIANTS[3]["risk_id"] in result

        evidence, confidence = result[VARIANTS[0]["risk_id"]]
        assert confidence == "high"
        assert len(evidence) == 1
        assert evidence[0].text == "facial recognition data was collected without consent"
        assert evidence[0].document == "policy.pdf"
        assert evidence[0].chunk_index == 2
        assert evidence[0].page == 5
        assert evidence[0].section == "Privacy"

        evidence_loc, conf_loc = result[VARIANTS[3]["risk_id"]]
        assert conf_loc == "medium"
        assert evidence_loc[0].text == "GPS tracking without user awareness"

    def test_none_grounded(self):
        """LLM grounds 0 variants; empty result."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=v["risk_id"],
                    grounded=False,
                    confidence="low",
                    quotes=[],
                )
                for v in VARIANTS
            ]
        )

        result = ground_variants(
            chunk_text="Generic personal data processing without specifics.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS,
            client=mock_client,
            model="test-model",
            document="policy.pdf",
            chunk_index=0,
        )

        assert result == {}

    def test_empty_variants(self):
        """No variants → no LLM call, empty result."""
        mock_client = MagicMock()

        result = ground_variants(
            chunk_text="Some text.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=[],
            client=mock_client,
            model="test-model",
            document="doc.pdf",
            chunk_index=0,
        )

        assert result == {}
        mock_client.chat.completions.create.assert_not_called()

    def test_ignores_unknown_variant_ids(self):
        """Verdicts for IDs not in the variant list are ignored."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id="some-unknown-id",
                    grounded=True,
                    confidence="high",
                    quotes=["Some evidence."],
                ),
            ]
        )

        result = ground_variants(
            chunk_text="Some text.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS[:1],
            client=mock_client,
            model="test-model",
            document="doc.pdf",
            chunk_index=0,
        )

        assert result == {}

    def test_skips_empty_quotes(self):
        """Grounded verdict with only empty quotes → not included."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[0]["risk_id"],
                    grounded=True,
                    confidence="high",
                    quotes=["", "  ", ""],
                ),
            ]
        )

        result = ground_variants(
            chunk_text="Some text.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS[:1],
            client=mock_client,
            model="test-model",
            document="doc.pdf",
            chunk_index=0,
        )

        assert VARIANTS[0]["risk_id"] not in result

    def test_captures_call_record(self, mock_client):
        """Call collector records a variant_grounding stage entry."""
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[0]["risk_id"],
                    grounded=True,
                    confidence="high",
                    quotes=["Biometric data collected."],
                ),
            ]
        )

        collector: list[LLMCallRecord] = []
        ground_variants(
            chunk_text="Biometric data collected.",
            parent_id=PARENT_ID,
            parent_name=PARENT_NAME,
            parent_description=PARENT_DESC,
            variants=VARIANTS[:2],
            client=mock_client,
            model="test-model",
            document="policy.pdf",
            chunk_index=3,
            call_collector=collector,
        )

        assert len(collector) == 1
        record = collector[0]
        assert record.stage == "variant_grounding"
        assert record.call_id == "ground-variant-001"
        assert record.chunk_index == 3
        assert record.risk_ids == [VARIANTS[0]["risk_id"], VARIANTS[1]["risk_id"]]
        assert "1/2 variants grounded" in record.result_summary
        assert PARENT_ID in record.result_summary
        assert record.duration_ms >= 0


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestGroundVariantsPrompt:
    def test_renders_with_all_fields(self):
        messages = render_prompt(
            "ground_variants",
            {
                "chunk_text": "Biometric data collected without consent.",
                "parent_id": PARENT_ID,
                "parent_name": PARENT_NAME,
                "parent_description": PARENT_DESC,
                "variants": VARIANTS[:2],
            },
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "variant" in messages[0]["content"].lower()
        assert PARENT_ID in messages[1]["content"]
        assert VARIANTS[0]["risk_id"] in messages[1]["content"]
        assert "Biometric data collected without consent." in messages[1]["content"]


# ---------------------------------------------------------------------------
# Pipeline integration: _ground_variants_one
# ---------------------------------------------------------------------------


class TestGroundVariantsOne:
    def test_returns_grounded_variants(self):
        """_ground_variants_one calls ground_variants and returns results."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[0]["risk_id"],
                    grounded=True,
                    confidence="high",
                    quotes=["biometric data"],
                ),
            ]
        )

        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["biometric data processing without consent."])

        pm, grounded = _ground_variants_one(
            parent,
            chunks,
            VARIANT_MAP,
            mock_client,
            "test-model",
            None,
        )

        assert pm is parent
        assert VARIANTS[0]["risk_id"] in grounded

    def test_no_variants_returns_empty(self):
        """Parent with no variants in the map → empty dict."""
        mock_client = MagicMock()
        parent = _make_parent_match()
        chunks = _make_chunks(["Some text."])

        pm, grounded = _ground_variants_one(
            parent,
            chunks,
            {},
            mock_client,
            "test-model",
            None,
        )

        assert grounded == {}
        mock_client.chat.completions.create.assert_not_called()

    def test_no_evidence_returns_empty(self):
        """Parent with no evidence spans → empty dict."""
        mock_client = MagicMock()
        parent = _make_parent_match()
        parent.evidence = []
        chunks = _make_chunks(["Some text."])

        pm, grounded = _ground_variants_one(
            parent,
            chunks,
            VARIANT_MAP,
            mock_client,
            "test-model",
            None,
        )

        assert grounded == {}
        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline integration: _run_variant_grounding
# ---------------------------------------------------------------------------


class TestRunVariantGrounding:
    def _make_config(self):
        return SimpleNamespace(model="test-model")

    def test_partitions_parent_and_non_parent(self):
        """Parent matches get variant grounding; non-parent pass through."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[0]["risk_id"],
                    grounded=True,
                    confidence="high",
                    quotes=["biometric data"],
                ),
            ]
        )

        parent = _make_parent_match(chunk_index=0)
        non_parent = _make_non_parent_match()
        chunks = _make_chunks(["biometric data was processed without consent."])

        result, removed = _run_variant_grounding(
            [parent, non_parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
        )

        result_ids = {m.risk_id for m in result}
        assert non_parent.risk_id in result_ids
        assert VARIANTS[0]["risk_id"] in result_ids
        assert PARENT_ID not in result_ids
        assert removed == []

    def test_variant_inherits_parent_scores(self):
        """Variant matches inherit the parent's retrieval scores."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[1]["risk_id"],
                    grounded=True,
                    confidence="medium",
                    quotes=["health records"],
                ),
            ]
        )

        parent = _make_parent_match(chunk_index=0).model_copy(update={"confidence_source": "cross_encoder"})
        chunks = _make_chunks(["health records were accessed."])

        result, removed = _run_variant_grounding(
            [parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
        )

        assert len(result) == 1
        variant = result[0]
        assert variant.risk_id == VARIANTS[1]["risk_id"]
        assert variant.risk_name == VARIANTS[1]["name"]
        assert variant.taxonomy == "ai-risk-taxonomy"
        assert variant.scores.bm25_rank == parent.scores.bm25_rank
        assert variant.scores.embedding_distance == parent.scores.embedding_distance
        assert variant.scores.cross_encoder_score == parent.scores.cross_encoder_score
        assert variant.scores.rrf_score == parent.scores.rrf_score
        assert variant.accepted_by == parent.accepted_by
        assert variant.grounding_confidence == "medium"
        assert variant.scores_inherited_from == parent.risk_id

    def test_variant_without_taxonomy_entry_gets_empty_taxonomy(self):
        """A variant entry lacking a taxonomy key yields an empty-string taxonomy."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[0]["risk_id"],
                    grounded=True,
                    confidence="high",
                    quotes=["biometric data"],
                ),
            ]
        )
        bare_variants = [{k: v for k, v in VARIANTS[0].items() if k != "taxonomy"}]
        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["biometric data processing."])

        result, removed = _run_variant_grounding(
            [parent],
            chunks,
            {PARENT_ID: bare_variants},
            mock_client,
            self._make_config(),
            None,
        )

        assert len(result) == 1
        assert result[0].risk_id == VARIANTS[0]["risk_id"]
        assert result[0].taxonomy == ""

    def test_no_parents_returns_unchanged(self):
        """When no parent matches exist, returns input unchanged."""
        non_parent = _make_non_parent_match()
        chunks = _make_chunks(["Some text."])

        result = _run_variant_grounding(
            [non_parent],
            chunks,
            VARIANT_MAP,
            MagicMock(),
            self._make_config(),
            None,
        )

        assert len(result[0]) == 1
        assert result[0][0].risk_id == non_parent.risk_id
        assert result[1] == []

    def test_zero_variants_grounded_drops_parent(self):
        """If LLM grounds zero variants, parent is dropped entirely."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=v["risk_id"],
                    grounded=False,
                    confidence="low",
                    quotes=[],
                )
                for v in VARIANTS
            ]
        )

        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["Generic data processing discussion."])

        result = _run_variant_grounding(
            [parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
        )

        assert len(result[0]) == 0
        assert len(result[1]) == 1

    def test_uses_chunk_contexts_when_provided(self):
        """When chunk_contexts is provided, uses padded text instead of raw chunk."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _RiskEvidenceList(
            items=[
                _RiskEvidence(
                    risk_id=VARIANTS[0]["risk_id"],
                    grounded=True,
                    confidence="high",
                    quotes=["biometric data from padded context"],
                ),
            ]
        )

        parent = _make_parent_match(chunk_index=0)
        chunks = _make_chunks(["short chunk"])
        padded = ["biometric data from padded context with extra surrounding text"]

        _run_variant_grounding(
            [parent],
            chunks,
            VARIANT_MAP,
            mock_client,
            self._make_config(),
            None,
            chunk_contexts=padded,
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        user_msg = messages[-1]["content"] if messages else ""
        assert "padded context" in user_msg


# ---------------------------------------------------------------------------
# Sibling expansion onto variant risks
# ---------------------------------------------------------------------------


def _catalog_risk(risk_id, name, description, taxonomy, parent):
    return SimpleNamespace(
        id=risk_id,
        name=name,
        description=description,
        concern="",
        isDefinedByTaxonomy=taxonomy,
        isPartOf=parent,
    )


def test_expansion_carries_variant_taxonomy_onto_expanded_variant_risks():
    """A sibling that is a ``---`` variant keeps its own taxonomy, never an empty string."""
    catalog = [
        _catalog_risk("R-001", "Model bias", "Systematic errors.", "other-taxonomy", "grp"),
        *[_catalog_risk(v["risk_id"], v["name"], v["description"], v["taxonomy"], "grp") for v in VARIANTS],
    ]
    found = RiskMatch(
        risk_id="R-001",
        risk_name="Model bias",
        risk_description="Systematic errors.",
        confidence=0.9,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[EvidenceSpan(text="quote", document="doc.md", chunk_index=0)],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.03),
    )

    def _ground_all(**kwargs):
        evidence = [EvidenceSpan(text="quote", document="doc.md", chunk_index=0)]
        return {r["risk_id"]: (evidence, "medium") for r in kwargs["risks"]}

    with (
        patch("asago_policy_mapper.extract.index._load_bi_encoder", return_value=(None, None, None)),
        patch("asago_policy_mapper.extract.index._load_cross_encoder", return_value=(None, None, False, False)),
        patch("asago_policy_mapper.extract.pipeline.ground_risk_group", side_effect=_ground_all),
    ):
        index = RiskIndex(catalog)
        merged, _, _ = _run_expansion(
            catalog,
            [found],
            [],
            _make_chunks(["chunk about biometric and health data"]),
            ["doc.md"],
            index,
            MagicMock(),
            SimpleNamespace(model="test-model"),
            1,
            None,
        )

    expanded = {m.risk_id: m for m in merged if m.accepted_by == "expansion"}
    assert set(expanded) == {v["risk_id"] for v in VARIANTS}
    assert {m.taxonomy for m in expanded.values()} == {"ai-risk-taxonomy"}


# ---------------------------------------------------------------------------
# Expansion helper unit tests
# ---------------------------------------------------------------------------


def _grounding_group(risk_ids, chunk_indices=(0,)):
    from asago_policy_mapper.extract.expand import GroundingGroup

    return GroundingGroup(
        parent="grp",
        risk_ids=list(risk_ids),
        risk_lookup={
            rid: {"risk_id": rid, "risk_name": f"name {rid}", "risk_description": f"desc {rid}"} for rid in risk_ids
        },
        chunk_indices=list(chunk_indices),
    )


def test_ground_expansion_group_passes_group_to_ground_risk_group():
    """One call to ground_risk_group with the group's chunks, risks and document."""
    from asago_policy_mapper.extract import pipeline

    calls = []

    def fake_ground_risk_group(**kwargs):
        calls.append(kwargs)
        return {}

    with patch.object(pipeline, "ground_risk_group", side_effect=fake_ground_risk_group):
        group = _grounding_group(["R-001", "R-002"], chunk_indices=(2, 5))
        result = pipeline._ground_expansion_group(
            group,
            _make_chunks(["a", "b", "c", "d", "e", "f"]),
            ["doc.md"],
            MagicMock(),
            SimpleNamespace(model="test-model"),
            None,
        )

    assert result == {}
    assert len(calls) == 1
    assert calls[0]["chunk_indices"] == [2, 5]
    assert [r["risk_id"] for r in calls[0]["risks"]] == ["R-001", "R-002"]
    assert calls[0]["document"] == "doc.md"
    assert calls[0]["model"] == "test-model"


def test_ground_expansion_group_multi_first_pass_wins():
    """Across passes, a risk grounded earlier keeps its evidence and confidence."""
    from asago_policy_mapper.extract import pipeline

    pass_results = [
        {"R-001": ([EvidenceSpan(text="first", document="doc.md", chunk_index=0)], "high")},
        {
            "R-001": ([EvidenceSpan(text="second", document="doc.md", chunk_index=1)], "low"),
            "R-002": ([EvidenceSpan(text="only-late", document="doc.md", chunk_index=1)], "medium"),
        },
    ]

    with patch.object(pipeline, "ground_risk_group", side_effect=lambda **kwargs: pass_results.pop(0)):
        group = _grounding_group(["R-001", "R-002"])
        merged = pipeline._ground_expansion_group_multi(
            group,
            _make_chunks(["a", "b"]),
            ["doc.md"],
            MagicMock(),
            SimpleNamespace(model="test-model"),
            None,
            passes=2,
        )

    assert set(merged) == {"R-001", "R-002"}
    evidence, confidence = merged["R-001"]
    assert evidence[0].text == "first"
    assert confidence == "high"
    evidence, _ = merged["R-002"]
    assert evidence[0].text == "only-late"


def test_ground_expansion_group_multi_empty_passes_change_nothing():
    """Passes that ground nothing leave earlier results untouched."""
    from asago_policy_mapper.extract import pipeline

    pass_results = [{"R-001": ([EvidenceSpan(text="first", document="doc.md", chunk_index=0)], "high")}, {}]

    with patch.object(pipeline, "ground_risk_group", side_effect=lambda **kwargs: pass_results.pop(0)):
        group = _grounding_group(["R-001"])
        merged = pipeline._ground_expansion_group_multi(
            group,
            _make_chunks(["a"]),
            ["doc.md"],
            MagicMock(),
            SimpleNamespace(model="test-model"),
            None,
            passes=2,
        )

    assert set(merged) == {"R-001"}
    evidence, confidence = merged["R-001"]
    assert evidence[0].text == "first"
    assert confidence == "high"


def test_build_expansion_matches_uses_risk_lookup_and_taxonomy():
    """Each grounded risk becomes an expansion match with null confidence."""
    from asago_policy_mapper.extract import pipeline

    grounded = {
        "R-001": ([EvidenceSpan(text="quote", document="doc.md", chunk_index=0)], "high"),
    }
    risk_lookup = {"R-001": {"name": "Model Bias", "description": "Systematic errors."}}
    index = SimpleNamespace(get_taxonomy=lambda rid: f"tax-{rid}")

    matches = pipeline._build_expansion_matches(grounded, index, risk_lookup)

    assert len(matches) == 1
    m = matches[0]
    assert m.risk_id == "R-001"
    assert m.risk_name == "Model Bias"
    assert m.risk_description == "Systematic errors."
    assert m.accepted_by == "expansion"
    assert m.confidence is None
    assert m.confidence_source == "none"
    assert m.grounding_confidence == "high"
    assert m.taxonomy == "tax-R-001"
    assert [e.text for e in m.evidence] == ["quote"]


def test_build_found_risk_chunks_unions_candidates_and_evidence():
    """Found chunks come from accepted candidates and merged evidence."""
    from asago_policy_mapper.extract.models import ScoredCandidate
    from asago_policy_mapper.extract.pipeline import _build_found_risk_chunks
    from asago_policy_mapper.extract.retrieve import ChunkResult

    cr0 = ChunkResult(
        chunk_index=0,
        source="doc.md",
        page=None,
        section=None,
        accepted=[ScoredCandidate(risk_id="R-001 extra", risk_name="a", risk_description="")],
        borderline=[],
        borderline_judged=[],
        stats={},
    )
    cr1 = ChunkResult(
        chunk_index=1,
        source="doc.md",
        page=None,
        section=None,
        accepted=[ScoredCandidate(risk_id="R-002", risk_name="b", risk_description="")],
        borderline=[],
        borderline_judged=[],
        stats={},
    )
    merged = [
        RiskMatch(
            risk_id="R-001",
            risk_name="a",
            risk_description="",
            confidence=0.9,
            confidence_source="cross_encoder",
            grounding_confidence="high",
            accepted_by="threshold",
            evidence=[EvidenceSpan(text="q", document="doc.md", chunk_index=2)],
            scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.03),
        ),
    ]

    found = _build_found_risk_chunks(merged, [cr0, cr1])

    assert found == {"R-001": {0, 2}, "R-002": {1}}


def test_ground_expansion_groups_counts_and_builds_matches():
    """Every group's grounded risks become matches and the grounded count adds up."""
    from asago_policy_mapper.extract import pipeline

    def fake_ground_risk_group(**kwargs):
        risk_ids = [r["risk_id"] for r in kwargs["risks"]]
        return {rid: ([EvidenceSpan(text=f"q-{rid}", document="doc.md", chunk_index=0)], "medium") for rid in risk_ids}

    groups = [_grounding_group(["R-005"]), _grounding_group(["R-006", "R-007"])]
    risk_lookup = {
        "R-005": {"name": "n5", "description": "d5"},
        "R-006": {"name": "n6", "description": "d6"},
        "R-007": {"name": "n7", "description": "d7"},
    }
    index = SimpleNamespace(get_taxonomy=lambda rid: f"tax-{rid}")

    with patch.object(pipeline, "ground_risk_group", side_effect=fake_ground_risk_group):
        matches, grounded_count = pipeline._ground_expansion_groups(
            groups,
            _make_chunks(["a"]),
            ["doc.md"],
            index,
            MagicMock(),
            SimpleNamespace(model="test-model"),
            2,
            None,
            1,
            risk_lookup,
        )

    assert grounded_count == 3
    assert sorted(m.risk_id for m in matches) == ["R-005", "R-006", "R-007"]
    assert all(m.accepted_by == "expansion" for m in matches)


def test_ground_expansion_groups_multi_pass_uses_union_worker():
    """With several passes the per-group first-wins union worker runs inside the pool."""
    from asago_policy_mapper.extract import pipeline

    multi_calls = []

    def fake_multi(group, chunks, documents, client, config, call_collector, passes):
        multi_calls.append((tuple(group.risk_ids), passes))
        return {"R-005": ([EvidenceSpan(text="q", document="doc.md", chunk_index=0)], "medium")}

    risk_lookup = {"R-005": {"name": "n5", "description": "d5"}}
    index = SimpleNamespace(get_taxonomy=lambda rid: f"tax-{rid}")

    with patch.object(pipeline, "_ground_expansion_group_multi", side_effect=fake_multi):
        matches, grounded_count = pipeline._ground_expansion_groups(
            [_grounding_group(["R-005"])],
            _make_chunks(["a"]),
            ["doc.md"],
            index,
            MagicMock(),
            SimpleNamespace(model="test-model"),
            1,
            None,
            3,
            risk_lookup,
        )

    assert multi_calls == [(("R-005",), 3)]
    assert grounded_count == 1
    assert [m.risk_id for m in matches] == ["R-005"]


def test_run_expansion_variant_map_routes_through_variant_grounding():
    """With a variant map, expansion matches are variant-grounded before merging."""
    catalog = [
        _catalog_risk("R-001", "Model bias", "Systematic errors.", "other-taxonomy", "grp"),
        *[_catalog_risk(v["risk_id"], v["name"], v["description"], v["taxonomy"], "grp") for v in VARIANTS],
    ]
    found = RiskMatch(
        risk_id="R-001",
        risk_name="Model bias",
        risk_description="Systematic errors.",
        confidence=0.9,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[EvidenceSpan(text="quote", document="doc.md", chunk_index=0)],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.03),
    )

    def _ground_all(**kwargs):
        evidence = [EvidenceSpan(text="quote", document="doc.md", chunk_index=0)]
        return {r["risk_id"]: (evidence, "medium") for r in kwargs["risks"]}

    with (
        patch("asago_policy_mapper.extract.index._load_bi_encoder", return_value=(None, None, None)),
        patch("asago_policy_mapper.extract.index._load_cross_encoder", return_value=(None, None, False, False)),
        patch("asago_policy_mapper.extract.pipeline.ground_risk_group", side_effect=_ground_all),
        patch(
            "asago_policy_mapper.extract.pipeline._run_variant_grounding",
            side_effect=lambda matches, *a, **k: (matches, []),
        ) as variant_grounding,
    ):
        index = RiskIndex(catalog)
        merged, stats, _ = _run_expansion(
            catalog,
            [found],
            [],
            _make_chunks(["chunk about biometric and health data"]),
            ["doc.md"],
            index,
            MagicMock(),
            SimpleNamespace(model="test-model"),
            1,
            None,
            variant_map=index.variant_map,
        )

    # the expanded variant parents went through variant grounding
    variant_grounding.assert_called_once()
    assert stats["expanded_candidates"] == len(VARIANTS)
    assert stats["expanded_grounded"] == len(VARIANTS)
    expanded = {m.risk_id for m in merged if m.accepted_by == "expansion"}
    assert expanded == {v["risk_id"] for v in VARIANTS}


def test_run_expansion_without_candidates_returns_merged_unchanged():
    """No expansion candidates means the merged list and zeroed stats come back as-is."""
    catalog = [
        _catalog_risk("R-001", "Model bias", "Systematic errors.", "other-taxonomy", "grp"),
    ]
    found = RiskMatch(
        risk_id="R-001",
        risk_name="Model bias",
        risk_description="Systematic errors.",
        confidence=0.9,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[EvidenceSpan(text="quote", document="doc.md", chunk_index=0)],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.03),
    )

    with (
        patch("asago_policy_mapper.extract.index._load_bi_encoder", return_value=(None, None, None)),
        patch("asago_policy_mapper.extract.index._load_cross_encoder", return_value=(None, None, False, False)),
        patch("asago_policy_mapper.extract.pipeline.ground_risk_group") as ground_group,
    ):
        index = RiskIndex(catalog)
        merged, stats, _ = _run_expansion(
            catalog,
            [found],
            [],
            _make_chunks(["chunk about model bias"]),
            ["doc.md"],
            index,
            MagicMock(),
            SimpleNamespace(model="test-model"),
            1,
            None,
        )

    # R-001 has no siblings in the catalog: nothing to ground, merged unchanged
    ground_group.assert_not_called()
    assert merged == [found]
    assert stats == {"expanded_candidates": 0, "expanded_grounded": 0, "expansion_groups": 0}


# ---------------------------------------------------------------------------
# Removed variant groups: recording (stage grounding)
# ---------------------------------------------------------------------------


def _client_grounding_nothing():
    """A client whose variant-grounding verdicts ground no variant."""
    client = MagicMock()
    client.chat.completions.create.return_value = _RiskEvidenceList(
        items=[_RiskEvidence(risk_id=v["risk_id"], grounded=False, confidence="low", quotes=[]) for v in VARIANTS]
    )
    return client


def test_removed_group_recorded_with_reason():
    """A parent whose variants all fail grounding is recorded, not silently dropped."""
    parent = _make_parent_match(chunk_index=0)
    chunks = _make_chunks(["Generic data processing discussion."])

    matches, removed = _run_variant_grounding(
        [parent],
        chunks,
        VARIANT_MAP,
        _client_grounding_nothing(),
        SimpleNamespace(model="test-model"),
        None,
    )

    assert matches == []
    assert len(removed) == 1
    record = removed[0]
    assert record.parent_id == PARENT_ID
    assert record.stage == "grounding"
    assert record.reason == "no_variant_grounded"
    assert record.attempted_variant_ids == [v["risk_id"] for v in VARIANTS]


def test_removed_group_recorded_once_per_parent():
    """Per-chunk parent matches for one group collapse into a single record."""
    matches, removed = _run_variant_grounding(
        [_make_parent_match(chunk_index=0), _make_parent_match(chunk_index=1)],
        _make_chunks(["chunk zero", "chunk one"]),
        VARIANT_MAP,
        _client_grounding_nothing(),
        SimpleNamespace(model="test-model"),
        None,
    )

    assert matches == []
    assert len(removed) == 1
    assert removed[0].parent_id == PARENT_ID


def test_parent_without_evidence_recorded():
    """A parent match with no evidence spans is recorded with the no-parent-evidence reason."""
    parent = _make_parent_match(chunk_index=0)
    parent.evidence = []
    chunks = _make_chunks(["Some text."])

    matches, removed = _run_variant_grounding(
        [parent],
        chunks,
        VARIANT_MAP,
        MagicMock(),
        SimpleNamespace(model="test-model"),
        None,
    )

    assert matches == []
    assert len(removed) == 1
    record = removed[0]
    assert record.parent_id == PARENT_ID
    assert record.reason == "no_parent_evidence"
    assert record.stage == "grounding"
    # no variant was ever sent to ground_variants
    assert record.attempted_variant_ids == []


def _client_grounding_chunk(chunk_text_marker, grounded_delay_s=0.0):
    """A client whose verdicts ground one variant, but only for one chunk's prompt.

    The grounded chunk's answer can be delayed so the rejected chunk always
    completes first: the completion order the record logic must survive.
    """

    def _create(**kwargs):
        messages = kwargs["messages"]
        user_text = messages[-1]["content"]
        if chunk_text_marker not in user_text:
            return _RiskEvidenceList(
                items=[
                    _RiskEvidence(risk_id=v["risk_id"], grounded=False, confidence="low", quotes=[]) for v in VARIANTS
                ]
            )
        if grounded_delay_s:
            time.sleep(grounded_delay_s)
        return _RiskEvidenceList(
            items=[_RiskEvidence(risk_id=VARIANTS[0]["risk_id"], grounded=True, confidence="high", quotes=["quote"])]
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


def test_removed_group_not_recorded_when_any_chunk_grounded_a_variant():
    """A parent grounded in one chunk and rejected in another is not recorded as removed.

    The grounding chunk's answer arrives after the rejection, so the
    empty-grounded result is folded into the outcome first.
    """
    chunks = _make_chunks(["chunk zero text", "chunk one text"])

    matches, removed = _run_variant_grounding(
        [_make_parent_match(chunk_index=0), _make_parent_match(chunk_index=1)],
        chunks,
        VARIANT_MAP,
        _client_grounding_chunk("chunk one text", grounded_delay_s=0.2),
        SimpleNamespace(model="test-model"),
        None,
    )

    # chunk 1 grounded a variant; the parent survives through it
    assert [m.risk_id for m in matches] == [VARIANTS[0]["risk_id"]]
    assert removed == []


def test_removed_reason_independent_of_completion_order():
    """The reason and removal stay the same whichever chunk grounds or finishes first."""
    results = []
    # each pass swaps which chunk grounds the variant and reverses the
    # submission order, so the surviving result comes from a different
    # chunk and a different completion position each time
    for marker, reverse in (("chunk one text", False), ("chunk zero text", True)):
        parents = [_make_parent_match(chunk_index=0), _make_parent_match(chunk_index=1)]
        if reverse:
            parents = list(reversed(parents))
        matches, removed = _run_variant_grounding(
            parents,
            _make_chunks(["chunk zero text", "chunk one text"]),
            VARIANT_MAP,
            _client_grounding_chunk(marker),
            SimpleNamespace(model="test-model"),
            None,
        )
        results.append((sorted(m.risk_id for m in matches), [(r.reason, r.attempted_variant_ids) for r in removed]))

    # whichever chunk grounded the variant, the parent survives and no
    # removal is recorded; whichever chunk was rejected, nothing leaked
    assert results[0] == results[1]
    assert results[0][0] == [VARIANTS[0]["risk_id"]]
    assert results[0][1] == []


def test_removed_reason_no_parent_evidence_across_mixed_chunks():
    """Evidence without an attributed chunk counts as no parent evidence, not a variant attempt."""
    parent_attributed = _make_parent_match(chunk_index=0)
    parent_unattributed = _make_parent_match(chunk_index=1)
    # the group stage could not place these quotes; no chunk can ground against them
    parent_unattributed.evidence = [
        EvidenceSpan(text="quote", document="doc.md", chunk_index=None, cross_encoder_score=0.5)
    ]

    matches, removed = _run_variant_grounding(
        [parent_attributed, parent_unattributed],
        _make_chunks(["chunk zero text", "chunk one text"]),
        VARIANT_MAP,
        _client_grounding_nothing(),
        SimpleNamespace(model="test-model"),
        None,
    )

    # chunk 0 had attributed evidence, so the variants were attempted
    assert matches == []
    assert [(r.reason, r.attempted_variant_ids) for r in removed] == [
        ("no_variant_grounded", [v["risk_id"] for v in VARIANTS])
    ]


def test_parent_with_only_unattributed_evidence_records_no_attempt():
    """A parent whose evidence names no chunk records no attempted variants."""
    parent = _make_parent_match(chunk_index=0)
    parent.evidence = [EvidenceSpan(text="quote", document="doc.md", chunk_index=None, cross_encoder_score=0.5)]
    client = MagicMock()

    matches, removed = _run_variant_grounding(
        [parent],
        _make_chunks(["chunk text"]),
        VARIANT_MAP,
        client,
        SimpleNamespace(model="test-model"),
        None,
    )

    assert matches == []
    assert len(removed) == 1
    record = removed[0]
    assert record.reason == "no_parent_evidence"
    assert record.attempted_variant_ids == []
    # no variant was ever sent to the model
    client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Removed variant groups: recording (stage expansion)
# ---------------------------------------------------------------------------


def test_expansion_variant_removal_recorded():
    """Expansion matches that variant grounding removes are recorded with stage expansion."""
    client = MagicMock()
    client.chat.completions.create.return_value = _RiskEvidenceList(
        items=[_RiskEvidence(risk_id=v["risk_id"], grounded=False, confidence="low", quotes=[]) for v in VARIANTS]
    )
    chunks = _make_chunks(["chunk about data processing"])

    def _ground_all(**kwargs):
        evidence = [EvidenceSpan(text="quote", document="doc.md", chunk_index=0)]
        return {r["risk_id"]: (evidence, "medium") for r in kwargs["risks"]}

    found = RiskMatch(
        risk_id="R-001",
        risk_name="Model bias",
        risk_description="Systematic errors.",
        confidence=0.9,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[EvidenceSpan(text="quote", document="doc.md", chunk_index=0)],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.03),
    )
    # the catalog carries the collapsed parent, so expansion can propose it as
    # a sibling candidate and route it through variant grounding
    catalog = [
        _catalog_risk("R-001", "Model bias", "Systematic errors.", "other-taxonomy", "grp"),
        _catalog_risk(PARENT_ID, PARENT_NAME, PARENT_DESC, "ai-risk-taxonomy", "grp"),
    ]

    with (
        patch(
            "asago_policy_mapper.extract.expand.build_expansion_graph",
            return_value={"R-001": {PARENT_ID}},
        ),
        patch(
            "asago_policy_mapper.extract.pipeline.ground_risk_group",
            side_effect=_ground_all,
        ),
    ):
        merged, stats, removed = _run_expansion(
            catalog,
            [found],
            [],
            chunks,
            ["doc.md"],
            SimpleNamespace(get_taxonomy=lambda rid: f"tax-{rid}"),
            client,
            SimpleNamespace(model="test-model"),
            1,
            None,
            variant_map=VARIANT_MAP,
            chunk_contexts=["expanded context text"],
        )

    # the expanded parent went through the real variant grounding, which
    # grounded nothing, so the parent is gone and the removal is recorded
    assert not any(m.risk_id == PARENT_ID for m in merged)
    assert len(removed) == 1
    record = removed[0]
    assert record.parent_id == PARENT_ID
    assert record.stage == "expansion"
    assert record.reason == "no_variant_grounded"
    assert record.attempted_variant_ids == [v["risk_id"] for v in VARIANTS]


def test_expansion_forwards_run_context_to_variant_grounding():
    """The expansion call passes the run's chunks, client, workers and contexts on."""
    client = MagicMock()
    chunks = _make_chunks(["chunk about data processing"])
    found = RiskMatch(
        risk_id="R-001",
        risk_name="Model bias",
        risk_description="Systematic errors.",
        confidence=0.9,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=[EvidenceSpan(text="quote", document="doc.md", chunk_index=0)],
        scores=RetrievalScores(bm25_rank=1, embedding_distance=0.1, cross_encoder_score=0.9, rrf_score=0.03),
    )
    catalog = [
        _catalog_risk("R-001", "Model bias", "Systematic errors.", "other-taxonomy", "grp"),
        _catalog_risk("unrelated-parent", "Unrelated", "A collapsible parent.", "other-taxonomy", "grp"),
    ]
    variant_map = {"unrelated-parent": [{"risk_id": "unrelated-parent---a", "name": "", "description": ""}]}

    def _ground_all(**kwargs):
        evidence = [EvidenceSpan(text="quote", document="doc.md", chunk_index=0)]
        return {r["risk_id"]: (evidence, "medium") for r in kwargs["risks"]}

    with (
        patch(
            "asago_policy_mapper.extract.expand.build_expansion_graph",
            return_value={"R-001": {"unrelated-parent"}},
        ),
        patch("asago_policy_mapper.extract.pipeline.ground_risk_group", side_effect=_ground_all),
        patch(
            "asago_policy_mapper.extract.pipeline._run_variant_grounding",
            return_value=([], []),
        ) as variant_grounding,
    ):
        _run_expansion(
            catalog,
            [found],
            [],
            chunks,
            ["doc.md"],
            SimpleNamespace(get_taxonomy=lambda rid: f"tax-{rid}"),
            client,
            SimpleNamespace(model="test-model"),
            3,
            None,
            variant_map=variant_map,
            chunk_contexts=["expanded context text"],
        )

    call = variant_grounding.call_args
    assert call.args[1] is chunks
    assert call.args[2] is variant_map
    assert call.args[3] is client
    assert call.args[4].model == "test-model"
    assert call.kwargs["max_workers"] == 3
    assert call.kwargs["chunk_contexts"] == ["expanded context text"]
    assert call.kwargs["stage"] == "expansion"


# ---------------------------------------------------------------------------
# Removed variant groups: the output invariant
# ---------------------------------------------------------------------------


def test_invariant_raises_on_leaked_parent():
    """A variant-map key reaching the output raises PipelineInvariantError."""
    merged = [_make_parent_match(chunk_index=0)]

    with pytest.raises(PipelineInvariantError, match=PARENT_ID):
        _assert_no_synthetic_parents(merged, VARIANT_MAP)


def test_invariant_skipped_without_variants():
    """With an empty variant map the invariant does not fire, even for parent-like ids."""
    merged = [_make_parent_match(chunk_index=0)]

    # No exception: the empty map means the run never collapsed variants.
    _assert_no_synthetic_parents(merged, {})


def test_invariant_skipped_without_grounding():
    """Ungrounded runs expand parents to variants, so the invariant is off."""
    merged = [_make_parent_match(chunk_index=0)]

    # No exception: no_grounding runs emit expanded variants, not parents.
    _assert_no_synthetic_parents(merged, VARIANT_MAP, grounding=False)


@pytest.mark.slow
def test_run_extraction_invariant_fires_on_leaked_parent(mock_config, tmp_path):
    """A parent that survives variant grounding fails the run at the invariant."""
    catalog = [
        _catalog_risk("R-001", "Model bias", "Systematic errors.", "other-taxonomy", ""),
        *[_catalog_risk(v["risk_id"], v["name"], v["description"], v["taxonomy"], "grp") for v in VARIANTS],
    ]
    doc = tmp_path / "policy.md"
    doc.write_text(
        "Policy covering unauthorized processing of biometric and location data "
        "across all processing activities the organization undertakes."
    )
    parent_candidate = ScoredCandidate(
        risk_id=PARENT_ID,
        risk_name=PARENT_NAME,
        risk_description=PARENT_DESC,
        bm25_rank=1,
        rrf_score=0.9,
        cross_encoder_score=0.9,
    )

    def _retrieve_parent(chunks, i, index, **kwargs):
        return ChunkResult(
            chunk_index=i,
            source=chunks[i].source,
            page=chunks[i].page,
            section=chunks[i].section,
            accepted=[parent_candidate],
            borderline=[],
            borderline_judged=[],
            stats={
                "candidates_retrieved": 1,
                "auto_accepted": 1,
                "borderline": 0,
                "discarded": 0,
                "bm25_rescued": 0,
            },
        )

    def _ground_nothing(**kwargs):
        return {}

    with (
        patch("asago_policy_mapper.extract.index._load_bi_encoder", return_value=(None, None, None)),
        patch("asago_policy_mapper.extract.index._load_cross_encoder", return_value=(None, None, False, False)),
        # retrieval surfaces the collapsed parent; grounding accepts it; a
        # broken variant stage passes it through untouched — exactly the leak
        # the invariant exists to catch
        patch("asago_policy_mapper.extract.pipeline.retrieve_chunk", side_effect=_retrieve_parent),
        patch(
            "asago_policy_mapper.extract.pipeline.ground_and_extract_evidence",
            side_effect=lambda **kwargs: {
                c.risk_id: ([EvidenceSpan(text="quote", document="doc.md", chunk_index=0)], "high")
                for c in kwargs["candidates"]
            },
        ),
        patch(
            "asago_policy_mapper.extract.pipeline._run_variant_grounding",
            side_effect=lambda matches, *a, **k: (matches, []),
        ),
        patch("asago_policy_mapper.extract.pipeline.ground_risk_group", side_effect=_ground_nothing),
    ):
        with pytest.raises(PipelineInvariantError, match="unauthorized-processing"):
            run_extraction(
                documents=[doc],
                client=MagicMock(),
                config=mock_config,
                risks=catalog,
                retrieval=RetrievalConfig(query_gen=False, no_judge=True),
                ocr=False,
            )
