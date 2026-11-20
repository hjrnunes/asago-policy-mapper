import math
from unittest.mock import MagicMock

from asago_policy_mapper.extract.models import LLMCallRecord, ScoredCandidate, _JudgeVerdict, _JudgeVerdicts
from asago_policy_mapper.extract.parse import Chunk
from asago_policy_mapper.extract.retrieve import (
    build_chunk_context_indices,
    build_chunk_contexts,
    build_padded_text,
    classify_by_rank,
    classify_by_threshold,
    classify_candidates,
    judge_borderline,
    retrieve_chunk,
)


def test_build_padded_text_middle_chunk():
    chunks = [
        Chunk(text="First chunk. It has two sentences.", source="a.pdf", index=0),
        Chunk(text="Middle chunk with content.", source="a.pdf", index=1),
        Chunk(text="Last chunk. Final sentence here.", source="a.pdf", index=2),
    ]
    padded = build_padded_text(chunks, chunk_index=1, context_sentences=2)
    assert "First chunk." in padded or "It has two sentences." in padded
    assert "Middle chunk with content." in padded
    assert "Last chunk." in padded or "Final sentence here." in padded


def test_build_padded_text_first_chunk():
    chunks = [
        Chunk(text="First chunk only.", source="a.pdf", index=0),
        Chunk(text="Second chunk text.", source="a.pdf", index=1),
    ]
    padded = build_padded_text(chunks, chunk_index=0, context_sentences=2)
    assert "First chunk only." in padded
    assert "Second chunk text." in padded


def test_build_padded_text_last_chunk():
    chunks = [
        Chunk(text="First chunk. With detail.", source="a.pdf", index=0),
        Chunk(text="Last chunk alone.", source="a.pdf", index=1),
    ]
    padded = build_padded_text(chunks, chunk_index=1, context_sentences=2)
    assert "Last chunk alone." in padded
    assert "First chunk." in padded or "With detail." in padded


def test_build_padded_text_single_chunk():
    chunks = [Chunk(text="Only chunk.", source="a.pdf", index=0)]
    padded = build_padded_text(chunks, chunk_index=0, context_sentences=2)
    assert padded == "Only chunk."


def test_build_padded_text_no_cross_document():
    chunks = [
        Chunk(text="Doc A last chunk.", source="a.pdf", index=0),
        Chunk(text="Doc B first chunk.", source="b.pdf", index=0),
    ]
    padded = build_padded_text(chunks, chunk_index=0, context_sentences=2)
    assert "Doc B first chunk." not in padded


def test_classify_candidates_rank_based():
    """Rank-based mode: top N accepted, next M borderline, rest discarded."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.70),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-004", risk_name="D", risk_description="d", cross_encoder_score=0.30),
        ScoredCandidate(risk_id="R-005", risk_name="E", risk_description="e", cross_encoder_score=0.10),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        top_n_accept=2,
        top_n_judge=2,
    )
    assert [c.risk_id for c in accepted] == ["R-001", "R-002"]
    assert [c.risk_id for c in borderline] == ["R-003", "R-004"]
    assert [c.risk_id for c in discarded] == ["R-005"]


def test_classify_candidates_rank_based_floor():
    """Min score floor rejects low-scoring candidates regardless of rank."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.10),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        top_n_accept=2,
        top_n_judge=2,
        min_score_floor=0.3,
    )
    assert [c.risk_id for c in accepted] == ["R-001", "R-002"]
    assert len(borderline) == 0
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_candidates_legacy_threshold():
    """Legacy threshold mode when threshold_high is set."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.15),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        threshold_high=0.7,
        threshold_low=0.3,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_candidates_bm25_rescue():
    """Candidates with strong BM25 rank are rescued to borderline even with low cross-encoder score."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.001, bm25_rank=5),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.001, bm25_rank=15),
        ScoredCandidate(risk_id="R-004", risk_name="D", risk_description="d", cross_encoder_score=0.001, bm25_rank=0),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        threshold_high=0.7,
        threshold_low=0.15,
        bm25_rescue_rank=10,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert set(c.risk_id for c in discarded) == {"R-003", "R-004"}


def test_classify_candidates_bm25_rescue_disabled():
    """With bm25_rescue_rank=0, no rescue happens."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.001, bm25_rank=5),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        threshold_high=0.7,
        threshold_low=0.15,
        bm25_rescue_rank=0,
    )
    assert len(accepted) == 0
    assert len(borderline) == 0
    assert len(discarded) == 1


def test_classify_candidates_bm25_rescue_rank_based():
    """BM25 rescue works in rank-based mode too."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.85),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.001, bm25_rank=5),
    ]
    accepted, borderline, discarded = classify_candidates(
        candidates,
        top_n_accept=1,
        top_n_judge=0,
        bm25_rescue_rank=10,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert len(discarded) == 0


def test_judge_borderline_accepts_relevant():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        items=[MagicMock(risk_id="R-001", relevant=True, justification="Text discusses bias.")]
    )
    candidates = [
        ScoredCandidate(
            risk_id="R-001",
            risk_name="Model Bias",
            risk_description="Systematic errors favoring certain groups.",
            cross_encoder_score=0.5,
        ),
    ]
    accepted = judge_borderline(
        candidates,
        chunk_text="The AI system may produce biased outcomes.",
        client=mock_client,
        model="test-model",
    )
    assert len(accepted) == 1
    assert accepted[0].risk_id == "R-001"


def test_judge_borderline_rejects_irrelevant():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        items=[MagicMock(risk_id="R-005", relevant=False, justification="No mention of jobs.")]
    )
    candidates = [
        ScoredCandidate(
            risk_id="R-005",
            risk_name="Workforce Displacement",
            risk_description="Automation leading to unemployment.",
            cross_encoder_score=0.4,
        ),
    ]
    accepted = judge_borderline(
        candidates,
        chunk_text="The AI system must be transparent.",
        client=mock_client,
        model="test-model",
    )
    assert len(accepted) == 0


def test_judge_borderline_empty_candidates():
    mock_client = MagicMock()
    accepted = judge_borderline([], chunk_text="Text.", client=mock_client, model="m")
    assert accepted == []
    mock_client.chat.completions.create.assert_not_called()


def test_judge_borderline_captures_call():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _JudgeVerdicts(
        items=[_JudgeVerdict(risk_id="R-001", relevant=True, justification="Relevant")]
    )
    candidates = [
        ScoredCandidate(
            risk_id="R-001",
            risk_name="Bias",
            risk_description="Model bias risk",
            cross_encoder_score=0.5,
        ),
        ScoredCandidate(
            risk_id="R-002",
            risk_name="Privacy",
            risk_description="Privacy risk",
            cross_encoder_score=0.4,
        ),
    ]

    collector: list[LLMCallRecord] = []
    result = judge_borderline(
        candidates,
        "Some chunk text about AI bias.",
        mock_client,
        "test-model",
        call_collector=collector,
        chunk_index=3,
    )

    assert len(result) == 1
    assert result[0].risk_id == "R-001"
    assert len(collector) == 1
    assert collector[0].stage == "judge"
    assert collector[0].chunk_index == 3
    assert "R-001" in collector[0].risk_ids
    assert "R-002" in collector[0].risk_ids
    assert collector[0].result_summary == "1/2 accepted"
    assert collector[0].duration_ms >= 0


def test_judge_borderline_no_collector():
    """Existing behavior: no collector arg, no error."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _JudgeVerdicts(items=[])
    candidates = [
        ScoredCandidate(
            risk_id="R-001",
            risk_name="Bias",
            risk_description="Model bias risk",
            cross_encoder_score=0.5,
        ),
    ]

    result = judge_borderline(candidates, "Text", mock_client, "test-model")
    assert result == []


def test_retrieve_chunk_no_cross_encoder():
    """With use_cross_encoder=False, all candidates go to accepted, no borderline."""
    from asago_policy_mapper.extract.retrieve import retrieve_chunk

    mock_index = MagicMock()
    mock_index.hybrid_search.return_value = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", rrf_score=0.02),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", rrf_score=0.015),
    ]
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]
    cr = retrieve_chunk(chunks, 0, mock_index, use_cross_encoder=False, rrf_min_score=0.01)
    assert len(cr.accepted) == 2
    assert len(cr.borderline) == 0
    assert len(cr.borderline_judged) == 0
    assert cr.stats["auto_accepted"] == 2
    assert cr.stats["borderline"] == 0
    assert cr.stats["rerank_unscored"] == 0
    mock_index.hybrid_search.assert_called_once()
    call_kwargs = mock_index.hybrid_search.call_args
    assert call_kwargs[1]["rrf_min_score"] == 0.01


def test_retrieve_chunk_records_rerank_unscored():
    """retrieve_chunk reports the chunk's own unscored count as rerank_unscored.

    The index counter is cumulative across searches, so the chunk stat is the
    delta across this chunk's hybrid_search call.
    """

    class _CountingIndex:
        rerank_unscored = 0

        def hybrid_search(self, *args, **kwargs):
            # Two candidates went unscored during this chunk's search.
            self.rerank_unscored += 2
            return [ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.9)]

    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(chunks, 0, _CountingIndex())

    assert cr.stats["rerank_unscored"] == 2


def test_retrieve_chunk_rerank_unscored_is_per_search_delta():
    """A chunk searched after earlier searches reports only its own unscored count."""

    class _CountingIndex:
        def __init__(self):
            self.rerank_unscored = 5  # unscored in earlier searches

        def hybrid_search(self, *args, **kwargs):
            return [ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.9)]

    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(chunks, 0, _CountingIndex())

    assert cr.stats["rerank_unscored"] == 0


def test_retrieve_chunk_rerank_unscored_defaults_to_zero():
    """Indexes without a rerank_unscored attribute report zero."""
    from types import SimpleNamespace as _NS

    index = _NS(hybrid_search=lambda *a, **k: [ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a")])
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(chunks, 0, index)

    assert cr.stats["rerank_unscored"] == 0


def test_count_unscored_reads_int_counter():
    """The helper returns the index counter verbatim when it is an int."""
    from asago_policy_mapper.extract.retrieve import _count_unscored

    class _Idx:
        rerank_unscored = 4

    assert _count_unscored(_Idx()) == 4


def test_count_unscored_missing_attribute_is_zero():
    """No counter attribute reads as zero, whatever the getattr default mutates to."""
    from asago_policy_mapper.extract.retrieve import _count_unscored

    class _Idx:
        pass

    assert _count_unscored(_Idx()) == 0


def test_count_unscored_non_int_is_zero():
    """Mock-style counters (auto-attributes) read as zero, not as a count."""
    from asago_policy_mapper.extract.retrieve import _count_unscored

    class _Idx:
        rerank_unscored = "not-an-int"

    assert _count_unscored(_Idx()) == 0


# --- classify_by_rank / classify_by_threshold direct tests ---


def test_classify_by_rank_routes_unscored_to_borderline():
    """NaN-scored (unscored) candidates join the borderline list, not the rank selection."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="R-NAN", risk_name="N", risk_description="n", cross_encoder_score=float("nan")),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.70),
    ]

    accepted, borderline, discarded = classify_by_rank(candidates, top_n_accept=1, top_n_judge=1)

    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002", "R-NAN"]
    assert discarded == []


def test_classify_by_threshold_routes_unscored_to_borderline():
    """Threshold mode also routes unscored candidates to the borderline list."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="R-NAN", risk_name="N", risk_description="n", cross_encoder_score=float("nan")),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.05),
    ]

    accepted, borderline, discarded = classify_by_threshold(candidates, threshold_high=0.8, threshold_low=0.2)

    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-NAN"]
    assert [c.risk_id for c in discarded] == ["R-002"]


def test_classify_by_rank_unscored_does_not_disturb_rank_selection():
    """Unscored candidates neither take an accept slot nor get discarded by the floor."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="R-NAN", risk_name="N", risk_description="n", cross_encoder_score=float("nan")),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.70),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.50),
    ]

    accepted, borderline, discarded = classify_by_rank(candidates, top_n_accept=2, top_n_judge=1, min_score_floor=0.6)

    assert [c.risk_id for c in accepted] == ["R-001", "R-002"]
    assert [c.risk_id for c in borderline] == ["R-NAN"]
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_by_rank_all_unscored_all_borderline():
    accepted, borderline, discarded = classify_by_rank(
        [ScoredCandidate(risk_id="R-NAN", risk_name="N", risk_description="n", cross_encoder_score=float("nan"))],
        top_n_accept=5,
        top_n_judge=5,
    )

    assert accepted == []
    assert [c.risk_id for c in borderline] == ["R-NAN"]
    assert discarded == []


def test_classify_by_rank_caps_unscored_at_top_n_judge():
    """No more than top_n_judge unscored candidates ride along to the judge."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="N-1", risk_name="N1", risk_description="n1", cross_encoder_score=float("nan")),
        ScoredCandidate(risk_id="N-2", risk_name="N2", risk_description="n2", cross_encoder_score=float("nan")),
        ScoredCandidate(risk_id="N-3", risk_name="N3", risk_description="n3", cross_encoder_score=float("nan")),
    ]

    accepted, borderline, discarded = classify_by_rank(candidates, top_n_accept=1, top_n_judge=2)

    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["N-1", "N-2"]
    assert [c.risk_id for c in discarded] == ["N-3"]


def test_classify_by_threshold_caps_unscored_at_top_n_judge():
    """Threshold mode caps unscored riders at top_n_judge too."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="N-1", risk_name="N1", risk_description="n1", cross_encoder_score=float("nan")),
        ScoredCandidate(risk_id="N-2", risk_name="N2", risk_description="n2", cross_encoder_score=float("nan")),
    ]

    accepted, borderline, discarded = classify_by_threshold(candidates, threshold_high=0.8, threshold_low=0.2)

    # Default top_n_judge is 5, so both ride along.
    assert [c.risk_id for c in borderline] == ["N-1", "N-2"]
    assert discarded == []

    accepted, borderline, discarded = classify_by_threshold(
        candidates, threshold_high=0.8, threshold_low=0.2, top_n_judge=1
    )

    assert [c.risk_id for c in borderline] == ["N-1"]
    assert [c.risk_id for c in discarded] == ["N-2"]


def test_classify_by_threshold_default_budget_is_five():
    """Six unscored riders: exactly five ride along under the default budget."""
    candidates = [
        ScoredCandidate(risk_id=f"N-{i}", risk_name=f"N{i}", risk_description="n", cross_encoder_score=float("nan"))
        for i in range(1, 7)
    ]

    accepted, borderline, discarded = classify_by_threshold(candidates, threshold_high=0.8, threshold_low=0.2)

    assert len(borderline) == 5
    assert len(discarded) == 1


def test_classify_candidates_forwards_top_n_judge_to_threshold_mode():
    """The threshold branch through classify_candidates honours the judge budget."""
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="N-1", risk_name="N1", risk_description="n1", cross_encoder_score=float("nan")),
        ScoredCandidate(risk_id="N-2", risk_name="N2", risk_description="n2", cross_encoder_score=float("nan")),
    ]

    accepted, borderline, discarded = classify_candidates(
        candidates, threshold_high=0.8, threshold_low=0.2, top_n_judge=1
    )

    assert [c.risk_id for c in borderline] == ["N-1"]
    assert [c.risk_id for c in discarded] == ["N-2"]


# --- retrieve_chunk: unscored candidates reach the judge as borderline ---


class _UnscoringIndex:
    """hybrid_search returns a ranked list with unscored (NaN) candidates last."""

    def __init__(self):
        self.rerank_unscored = 0

    def hybrid_search(self, *args, **kwargs):
        self.rerank_unscored += 1
        return [
            ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.95),
            ScoredCandidate(
                risk_id="R-NAN",
                risk_name="N",
                risk_description="n",
                cross_encoder_score=float("nan"),
                bm25_rank=1,
            ),
            ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.40),
        ]


def test_retrieve_chunk_routes_unscored_to_borderline():
    """Unscored candidates become borderline with a serializable score, and count."""
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(chunks, 0, _UnscoringIndex(), top_n_accept=1, top_n_judge=1)

    assert [c.risk_id for c in cr.accepted] == ["R-001"]
    assert [c.risk_id for c in cr.borderline] == ["R-002", "R-NAN"]
    assert cr.stats["rerank_unscored"] == 1
    assert cr.stats["bm25_rescued"] == 0


def test_retrieve_chunk_unscored_not_counted_as_bm25_rescued():
    """An unscored candidate with an eligible BM25 rank is not a BM25 rescue."""
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(
        chunks,
        0,
        _UnscoringIndex(),
        top_n_accept=1,
        top_n_judge=1,
        min_score_floor=0.6,
        bm25_rescue_rank=2,
    )

    # R-002 (0.40, bm25_rank unset) is discarded by the floor; the unscored
    # candidate is borderline by the unscored rule, not by rescue.
    assert [c.risk_id for c in cr.borderline] == ["R-NAN"]
    assert cr.stats["bm25_rescued"] == 0


def test_unscored_never_serialized_as_nan():
    """Candidates carried in the chunk result never expose NaN; unscored reads 0.0."""
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(chunks, 0, _UnscoringIndex(), top_n_accept=1, top_n_judge=5)

    carried = cr.accepted + cr.borderline
    assert all(math.isnan(c.cross_encoder_score) is False for c in carried)
    unscored = [c for c in carried if c.risk_id == "R-NAN"]
    assert len(unscored) == 1
    assert unscored[0].cross_encoder_score == 0.0
    # The serialized form must round-trip as a plain float, never NaN/null.
    dumped = unscored[0].model_dump()
    assert dumped["cross_encoder_score"] == 0.0
    assert not math.isnan(dumped["cross_encoder_score"])


def test_classify_by_rank_accepts_top_n():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.70),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.50),
    ]
    accepted, borderline, discarded = classify_by_rank(
        candidates,
        top_n_accept=1,
        top_n_judge=1,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_by_rank_floor_discards():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.80),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.30),
    ]
    accepted, borderline, discarded = classify_by_rank(
        candidates,
        top_n_accept=5,
        top_n_judge=5,
        min_score_floor=0.50,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert len(borderline) == 0
    assert [c.risk_id for c in discarded] == ["R-002"]


def test_classify_by_threshold_accepts_above_high():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.90),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.50),
        ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", cross_encoder_score=0.05),
    ]
    accepted, borderline, discarded = classify_by_threshold(
        candidates,
        threshold_high=0.80,
        threshold_low=0.20,
    )
    assert [c.risk_id for c in accepted] == ["R-001"]
    assert [c.risk_id for c in borderline] == ["R-002"]
    assert [c.risk_id for c in discarded] == ["R-003"]


def test_classify_by_threshold_bm25_rescue():
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.05, bm25_rank=2),
    ]
    accepted, borderline, discarded = classify_by_threshold(
        candidates,
        threshold_high=0.80,
        threshold_low=0.20,
        bm25_rescue_rank=5,
    )
    assert len(accepted) == 0
    assert [c.risk_id for c in borderline] == ["R-001"]
    assert len(discarded) == 0


# --- build_chunk_contexts tests ---


def test_build_chunk_contexts_expands_both_neighbors():
    chunks = [
        Chunk(text="previous chunk text", source="a.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next chunk text", source="a.pdf", index=2),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert "previous chunk text" in contexts[1]
    assert "core chunk text" in contexts[1]
    assert "next chunk text" in contexts[1]


def test_build_chunk_contexts_no_budget():
    chunks = [
        Chunk(text="previous text", source="a.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next text", source="a.pdf", index=2),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=3)
    assert contexts[1] == "core chunk text"


def test_build_chunk_contexts_different_source_skipped():
    chunks = [
        Chunk(text="prev from other doc", source="b.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next from other doc", source="c.pdf", index=2),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert contexts[1] == "core chunk text"


def test_build_chunk_contexts_first_chunk():
    chunks = [
        Chunk(text="core chunk", source="a.pdf", index=0),
        Chunk(text="next chunk", source="a.pdf", index=1),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert "core chunk" in contexts[0]
    assert "next chunk" in contexts[0]


def test_build_chunk_contexts_last_chunk():
    chunks = [
        Chunk(text="prev chunk", source="a.pdf", index=0),
        Chunk(text="core chunk", source="a.pdf", index=1),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert "prev chunk" in contexts[1]
    assert "core chunk" in contexts[1]


def test_build_chunk_contexts_single_chunk():
    chunks = [Chunk(text="only chunk here", source="a.pdf", index=0)]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert contexts[0] == "only chunk here"


def test_build_chunk_contexts_section_headings():
    """Section headings are inserted at section transitions."""
    chunks = [
        Chunk(text="intro text", source="a.pdf", index=0, section="Introduction"),
        Chunk(text="method text", source="a.pdf", index=1, section="Methods"),
        Chunk(text="more methods", source="a.pdf", index=2, section="Methods"),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    # Context for chunk 1 should have heading when transitioning from Introduction
    assert "intro text" in contexts[1]
    assert "## Methods" in contexts[1]
    # Context for chunk 2 should not repeat section heading (same section as chunk 1)
    assert "method text" in contexts[2]
    assert contexts[2].count("## Methods") <= 1


def test_build_chunk_contexts_symmetric_expansion():
    """Expansion alternates between before and after."""
    chunks = [
        Chunk(text="chunk zero with some words", source="a.pdf", index=0),
        Chunk(text="chunk one with some words", source="a.pdf", index=1),
        Chunk(text="core chunk", source="a.pdf", index=2),
        Chunk(text="chunk three with some words", source="a.pdf", index=3),
        Chunk(text="chunk four with some words", source="a.pdf", index=4),
    ]
    # Budget for 3 neighbors (core=2 words, each neighbor=5 words, budget=17 words)
    contexts = build_chunk_contexts(chunks, max_context_tokens=17)
    ctx = contexts[2]
    assert "core chunk" in ctx
    assert "chunk one with some words" in ctx
    assert "chunk three with some words" in ctx


# --- build_chunk_context_indices tests ---


def test_context_indices_match_context_text_order():
    """The indices for chunk i list the chunks whose texts build_chunk_contexts joins."""
    chunks = [
        Chunk(text="chunk zero with some words", source="a.pdf", index=0),
        Chunk(text="chunk one with some words", source="a.pdf", index=1),
        Chunk(text="core chunk", source="a.pdf", index=2),
        Chunk(text="chunk three with some words", source="a.pdf", index=3),
        Chunk(text="chunk four with some words", source="a.pdf", index=4),
    ]
    contexts = build_chunk_contexts(chunks, max_context_tokens=17)
    indices = build_chunk_context_indices(chunks, max_context_tokens=17)
    assert indices[2] == [0, 1, 2, 3]
    # Every listed chunk's text appears in the context; unlisted chunks do not.
    assert chunks[0].text in contexts[2]
    assert chunks[3].text in contexts[2]
    assert chunks[4].text not in contexts[2]


def test_context_indices_with_section_headings():
    """Indices stay raw chunk positions while the context gains synthetic headings."""
    chunks = [
        Chunk(text="intro text", source="a.pdf", index=0, section="Introduction"),
        Chunk(text="method text", source="a.pdf", index=1, section="Methods"),
        Chunk(text="more methods", source="a.pdf", index=2, section="Methods"),
    ]
    indices = build_chunk_context_indices(chunks, max_context_tokens=100)
    contexts = build_chunk_contexts(chunks, max_context_tokens=100)
    assert indices[1] == [0, 1, 2]
    # The context inserts a heading at the section transition; the index list
    # carries only real chunk positions, never the heading.
    assert contexts[1] == "intro text\n\n## Methods\n\nmethod text\n\nmore methods"


def test_context_indices_exclude_other_documents():
    """Neighbours from a different source never appear in the indices."""
    chunks = [
        Chunk(text="prev from other doc", source="b.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next from other doc", source="c.pdf", index=2),
    ]
    indices = build_chunk_context_indices(chunks, max_context_tokens=100)
    assert indices[1] == [1]


def test_context_indices_core_only_when_no_budget():
    """A spent budget yields just the core chunk's index."""
    chunks = [
        Chunk(text="previous text", source="a.pdf", index=0),
        Chunk(text="core chunk text", source="a.pdf", index=1),
        Chunk(text="next text", source="a.pdf", index=2),
    ]
    indices = build_chunk_context_indices(chunks, max_context_tokens=3)
    assert indices[1] == [1]


def test_context_indices_first_and_last_chunk():
    """Edge chunks expand toward both sides that exist within the budget."""
    chunks = [
        Chunk(text="first chunk", source="a.pdf", index=0),
        Chunk(text="middle chunk", source="a.pdf", index=1),
        Chunk(text="last chunk", source="a.pdf", index=2),
    ]
    indices = build_chunk_context_indices(chunks, max_context_tokens=100)
    assert indices[0] == [0, 1, 2]
    assert indices[2] == [0, 1, 2]


def test_context_indices_stop_at_document_edge():
    """A small budget expands an edge chunk only toward the existing side."""
    chunks = [
        Chunk(text="first chunk", source="a.pdf", index=0),
        Chunk(text="middle chunk", source="a.pdf", index=1),
        Chunk(text="last chunk", source="a.pdf", index=2),
    ]
    # Budget for exactly one neighbour (core=2 words, neighbours=2 words each).
    indices = build_chunk_context_indices(chunks, max_context_tokens=5)
    assert indices[0] == [0, 1]
    assert indices[2] == [1, 2]


def test_context_indices_default_budget():
    """The default budget is 1024 tokens, matching build_chunk_contexts."""
    chunks = [Chunk(text="w" if i > 0 else "core", source="a.pdf", index=i) for i in range(1200)]
    # Core chunk 0 spends 1 token; the default 1024 budget admits 1023 one-word
    # neighbours, for 1024 indices. A 1025 budget would admit one more.
    assert build_chunk_context_indices(chunks)[0] == list(range(1024))


def test_context_indices_parallel_to_chunks():
    """One index list per chunk, core index always present at its position."""
    chunks = [Chunk(text=f"chunk {i} text", source="a.pdf", index=i) for i in range(4)]
    indices = build_chunk_context_indices(chunks, max_context_tokens=100)
    assert len(indices) == len(chunks)
    for i, idx_list in enumerate(indices):
        assert i in idx_list
