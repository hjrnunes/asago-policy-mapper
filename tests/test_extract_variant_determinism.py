"""Deterministic evidence selection in variant grounding (issue #150 part 1).

_ground_variants_one grounds variants against the parent match's "best"
evidence span. That choice must not depend on the order the parent's
evidence arrived in, which upstream depends on the order parallel
grounding calls finished. The key matches merge_matches' evidence key:
(-cross_encoder_score, chunk_index, text) plus the same total-order
serialization tail.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.extract.models import EvidenceSpan, RetrievalScores, RiskMatch
from asago_policy_mapper.extract.pipeline import _ground_variants_one

PARENT_ID = "ai-risk-taxonomy-unauthorized-processing"
VARIANT_ID = f"{PARENT_ID}---health-data"
VARIANT_MAP = {
    PARENT_ID: [
        {
            "risk_id": VARIANT_ID,
            "name": "Health data",
            "description": "Unauthorized processing of health records.",
            "taxonomy": "ai-risk-taxonomy",
        }
    ]
}


def _evidence(text, chunk_index, score=0.5, document="policy.pdf"):
    return EvidenceSpan(
        text=text,
        document=document,
        chunk_index=chunk_index,
        cross_encoder_score=score,
    )


def _parent_match(spans):
    return RiskMatch(
        risk_id=PARENT_ID,
        risk_name="Unauthorized processing",
        risk_description="Processing of personal data without legal basis or consent.",
        taxonomy="ai-risk-taxonomy",
        confidence=0.85,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=list(spans),
        scores=RetrievalScores(
            bm25_rank=3,
            embedding_distance=0.15,
            cross_encoder_score=0.85,
            rrf_score=0.04,
        ),
    )


def _chunks(texts):
    return [SimpleNamespace(text=t, page=i, section=f"Section {i}") for i, t in enumerate(texts)]


@pytest.fixture
def ground_calls(monkeypatch):
    """Record the chunk_index and document each ground_variants call sees."""
    calls = []
    from asago_policy_mapper.extract import pipeline

    def fake_ground_variants(*, chunk_text, document, **kwargs):
        calls.append((chunk_text, kwargs["chunk_index"], document))
        return {VARIANT_ID: ([_evidence("health records", kwargs["chunk_index"])], "high")}

    monkeypatch.setattr(pipeline, "ground_variants", fake_ground_variants)
    return calls


def test_variant_best_evidence_deterministic(ground_calls):
    """best_ev is the minimum by (-score, chunk_index, text) for any order.

    All spans tie on cross-encoder score, so the lowest chunk index decides
    which chunk grounds the variants — regardless of the evidence order the
    parent match carries, which depends on completion order upstream.
    """
    chunks = _chunks(["chunk zero", "chunk one", "chunk two"])
    span_orders = [
        [_evidence("in chunk two", 2), _evidence("in chunk zero", 0), _evidence("in chunk one", 1)],
        [_evidence("in chunk one", 1), _evidence("in chunk two", 2), _evidence("in chunk zero", 0)],
        [_evidence("in chunk zero", 0), _evidence("in chunk one", 1), _evidence("in chunk two", 2)],
    ]

    for spans in span_orders:
        _ground_variants_one(_parent_match(spans), chunks, VARIANT_MAP, MagicMock(), "test-model", None)

    assert [ci for _, ci, _ in ground_calls] == [0, 0, 0]


def test_variant_best_evidence_prefers_higher_score_over_lower_chunk(ground_calls):
    """Score outranks chunk index: a higher-scored span in a later chunk wins."""
    chunks = _chunks(["chunk zero", "chunk one"])
    spans = [_evidence("low score text", 0, score=0.2), _evidence("high score text", 1, score=0.9)]

    for order in (spans, list(reversed(spans))):
        _ground_variants_one(_parent_match(order), chunks, VARIANT_MAP, MagicMock(), "test-model", None)

    assert [ci for _, ci, _ in ground_calls] == [1, 1]


def test_variant_best_evidence_breaks_ties_on_text(ground_calls):
    """Equal score and chunk fall back to text: the smaller text wins.

    The two spans tie on score and chunk index; the alpha-text span must be
    selected in both input orders, which the recorded document shows — the
    spans carry different documents so the winner is observable.
    """
    chunks = _chunks(["chunk zero"])
    spans = [
        _evidence("zeta text", 0, score=0.5, document="z.pdf"),
        _evidence("alpha text", 0, score=0.5, document="a.pdf"),
    ]

    for order in (spans, list(reversed(spans))):
        _ground_variants_one(_parent_match(order), chunks, VARIANT_MAP, MagicMock(), "test-model", None)

    assert [(ci, doc) for _, ci, doc in ground_calls] == [(0, "a.pdf"), (0, "a.pdf")]


def test_variant_best_evidence_breaks_ties_on_document(ground_calls):
    """Ties past score/chunk/text fall back to the serialization tail.

    Two spans from different documents tie on score, chunk index and
    text, so only the total-order tail — the same serialization the merge
    ordering keys use — can separate them. The lexicographically smaller
    document must win in both input orders; without the tail, min() kept
    whichever span arrived first.
    """
    chunks = _chunks(["chunk zero"])
    spans = [
        _evidence("shared text", 0, score=0.5, document="b.pdf"),
        _evidence("shared text", 0, score=0.5, document="a.pdf"),
    ]

    for order in (spans, list(reversed(spans))):
        _ground_variants_one(_parent_match(order), chunks, VARIANT_MAP, MagicMock(), "test-model", None)

    assert [(ci, doc) for _, ci, doc in ground_calls] == [(0, "a.pdf"), (0, "a.pdf")]


def test_variant_best_evidence_passes_selected_span_document():
    """The selected span's document reaches ground_variants unchanged."""
    chunks = _chunks(["chunk zero", "chunk one"])
    seen = {}

    from asago_policy_mapper.extract import pipeline

    def fake_ground_variants(*, chunk_text, document, **kwargs):
        seen["document"] = document
        seen["chunk_index"] = kwargs["chunk_index"]
        return {}

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "ground_variants", fake_ground_variants)
        spans = [
            _evidence("in chunk one", 1, document="late.pdf"),
            _evidence("in chunk zero", 0, document="early.pdf"),
        ]
        _ground_variants_one(_parent_match(spans), chunks, VARIANT_MAP, MagicMock(), "test-model", None)

    assert seen["document"] == "early.pdf"
    assert seen["chunk_index"] == 0
