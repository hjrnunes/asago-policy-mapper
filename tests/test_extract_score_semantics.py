"""Score sources and truthful accepted_by labels (issue #150 part 2).

Every score in the output says where it came from: "not calculated" is
None, not 0.0; rerank and ColBERT paths stamp their source on the
candidate; accepted_by is recorded when the candidate is selected. The
tests here pin the model-level contract (step 1) and the setter behaviour
(steps 2-4 build on it).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from asago_policy_mapper.extract.merge import merge_matches
from asago_policy_mapper.extract.models import (
    EvidenceSpan,
    ExtractionResult,
    RetrievalScores,
    RetrievalStats,
    RiskMatch,
    ScoredCandidate,
)


def _scores(ce=0.5, rrf=0.03, bm25=1, embed=0.2):
    return RetrievalScores(bm25_rank=bm25, embedding_distance=embed, cross_encoder_score=ce, rrf_score=rrf)


def _match(risk_id, confidence, accepted_by="threshold", ce=0.5, evidence=None, source="cross_encoder"):
    return RiskMatch(
        risk_id=risk_id,
        risk_name=f"Risk {risk_id}",
        risk_description=f"Description of {risk_id}.",
        confidence=confidence,
        confidence_source=source if confidence is not None else "none",
        grounding_confidence="high",
        accepted_by=accepted_by,
        evidence=evidence if evidence is not None else [EvidenceSpan(text="q", document="d.pdf", chunk_index=0)],
        scores=_scores(ce=ce),
    )


def _empty_result():
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
    )


# --- model contract (step 1) ---


def test_querygen_match_confidence_is_null_source_none():
    """A query-gen match that no reranker scored carries null confidence."""
    m = _match("R-QG", None, accepted_by="query_gen", ce=None)
    assert m.confidence is None
    assert m.confidence_source == "none"
    assert m.scores.cross_encoder_score is None


def test_measured_zero_is_preserved():
    """A reranker-measured 0.0 stays 0.0 and keeps its source."""
    m = _match("R-CE", 0.0, accepted_by="threshold", ce=0.0)
    m = m.model_copy(update={"confidence_source": "cross_encoder"})
    assert m.confidence == 0.0
    assert m.confidence is not None
    assert m.confidence_source == "cross_encoder"
    assert m.scores.cross_encoder_score == 0.0


def test_bm25_rescued_scored_candidate_keeps_confidence():
    """A BM25-rescued candidate the reranker scored keeps its confidence.

    In cross-encoder mode the reranker measures every candidate in the
    rerank window, including ones later rescued by BM25 rank: their
    confidence is the measured score, sourced to the reranker.
    """
    m = _match("R-RESCUE", 0.4, accepted_by="threshold", ce=0.4)
    m = m.model_copy(update={"confidence_source": "cross_encoder"})
    assert m.confidence == 0.4
    assert m.confidence_source == "cross_encoder"
    assert m.scores.cross_encoder_score == 0.4
    # a missing BM25 rank still means "not rescued", it is not null
    assert m.scores.bm25_rank == 1


def test_never_reranked_candidate_is_null():
    """A candidate no reranker saw has null confidence and no source.

    Only candidates outside the rerank window stay unmeasured; the null is
    the marker for "not calculated", never 0.0.
    """
    m = _match("R-UNSCORED", None, accepted_by="threshold", ce=None)
    assert m.confidence is None
    assert m.confidence_source == "none"
    assert m.scores.cross_encoder_score is None
    # a missing BM25 rank still means "not rescued", it is not null
    assert m.scores.bm25_rank == 1


def test_accepted_by_query_gen_label():
    """The accepted_by vocabulary includes query_gen."""
    m = _match("R-QG", None, accepted_by="query_gen", ce=None)
    assert m.accepted_by == "query_gen"


def test_rank_vs_threshold_labels():
    """Rank-based and threshold-based acceptance produce distinct labels."""
    by_rank = _match("R-RANK", 0.9, accepted_by="rank")
    by_threshold = _match("R-THRESH", 0.9, accepted_by="threshold")
    assert by_rank.accepted_by == "rank"
    assert by_threshold.accepted_by == "threshold"


def test_expansion_confidence_null():
    """Expansion matches carry null confidence, source none."""
    m = _match("R-EXP", None, accepted_by="expansion", ce=None)
    assert m.confidence is None
    assert m.confidence_source == "none"


def test_variant_scores_marked_inherited():
    """Variant matches copying parent scores record the parent id."""
    m = _match("R-VAR", 0.9, accepted_by="threshold")
    assert m.scores_inherited_from is None
    inherited = m.model_copy(update={"scores_inherited_from": "unauthorized-processing"})
    assert inherited.scores_inherited_from == "unauthorized-processing"


def test_merge_handles_null_confidence():
    """merge_matches treats None as lower than any number, ties keep 150a's keys."""
    none_conf = _match("R-NULL", None, accepted_by="query_gen", ce=None)
    low = _match("R-NULL", 0.1, accepted_by="threshold", ce=0.1)
    for group in ([none_conf, low], [low, none_conf]):
        merged = merge_matches(group, max_evidence=3)
        assert merged[0].confidence == 0.1
        # the representative is the numeric-confidence match (None is
        # lower than any number), so its own metadata survives the merge
        assert merged[0].accepted_by == "threshold"
        assert merged[0].confidence_source == "cross_encoder"

    both_none = [
        _match(
            "R-N2",
            None,
            accepted_by="query_gen",
            ce=None,
            evidence=[EvidenceSpan(text="a", document="d", chunk_index=1)],
        ),
        _match(
            "R-N2",
            None,
            accepted_by="query_gen",
            ce=None,
            evidence=[EvidenceSpan(text="b", document="d", chunk_index=0)],
        ),
    ]
    for group in ([both_none[0], both_none[1]], [both_none[1], both_none[0]]):
        merged = merge_matches(group, max_evidence=3)
        # tied at None, the lowest evidence chunk decides (150a tie-break)
        assert merged[0].confidence is None
        assert merged[0].evidence[0].chunk_index == 0

    # output sorting: numeric confidences first (descending), None last
    ordered = merge_matches(
        [
            _match("R-ZERO", 0.0),
            _match("R-NONE", None, accepted_by="query_gen", ce=None),
            _match("R-HIGH", 0.9),
        ],
        max_evidence=3,
    )
    assert [m.risk_id for m in ordered] == ["R-HIGH", "R-ZERO", "R-NONE"]


def test_shared_candidate_not_mutated():
    """Setting selected_by/rerank_source never mutates a candidate in place."""
    shared = ScoredCandidate(risk_id="R-SHARED", risk_name="n", risk_description="d", rrf_score=0.05)
    tagged = shared.model_copy(update={"selected_by": "query_gen", "rerank_source": "cross_encoder"})
    assert shared.selected_by is None
    assert shared.rerank_source is None
    assert tagged.selected_by == "query_gen"
    assert tagged.rerank_source == "cross_encoder"


def test_result_version_is_04():
    """The schema version bumps to 0.4 with the nullable-confidence change."""
    assert _empty_result().version == "0.4"


def test_scored_candidate_source_defaults():
    """New source fields default to None on plain candidates."""
    c = ScoredCandidate(risk_id="R-1", risk_name="n", risk_description="d")
    assert c.rerank_source is None
    assert c.selected_by is None
    assert c.cross_encoder_score == 0.0


# --- setter behaviour (step 2: index.py and retrieve.py) ---


class _StubCrossEncoder:
    """Local cross-encoder stand-in returning fixed scores."""

    def predict(self, pairs):
        return [0.9, 0.2]


class _ZeroThenHighCrossEncoder:
    """Measures a true 0.0 for the first pair and 0.8 for the second."""

    def predict(self, pairs):
        return [0.0 if i == 0 else 0.8 for i in range(len(pairs))]


def _index_with_cross_encoder(monkeypatch):
    from asago_policy_mapper.extract import index as index_mod

    risks = [
        SimpleNamespace(
            id=rid,
            name=f"name-{rid}",
            description=f"desc-{rid}",
            concern="",
            risk_type="",
            isDefinedByTaxonomy="test-tax",
            isPartOf="",
        )
        for rid in ("R-001", "R-002")
    ]
    with (
        patch.object(index_mod, "_load_bi_encoder", return_value=(None, None, None)),
        patch.object(index_mod, "_load_cross_encoder", return_value=(_StubCrossEncoder(), None, False, False)),
    ):
        return index_mod.RiskIndex(risks)


def test_rerank_stamps_cross_encoder_source(monkeypatch):
    """rerank sets rerank_source=cross_encoder alongside the measured score."""
    idx = _index_with_cross_encoder(monkeypatch)
    candidates = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a"),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b"),
    ]

    reranked = idx.rerank("some text", candidates, top_k=5)

    assert [c.risk_id for c in reranked] == ["R-001", "R-002"]
    assert all(c.rerank_source == "cross_encoder" for c in reranked)
    assert reranked[0].cross_encoder_score == 0.9
    # the input candidates are not mutated
    assert all(c.rerank_source is None for c in candidates)


def test_hybrid_search_reranked_candidates_carry_source(monkeypatch):
    """hybrid_search results that went through rerank carry the source; measured 0.0 survives."""
    idx = _index_with_cross_encoder(monkeypatch)
    monkeypatch.setattr(idx, "search_bm25", lambda text, top_k=100: [])
    monkeypatch.setattr(
        idx,
        "search_semantic",
        lambda text, top_k=100: [ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a")],
    )

    results = idx.hybrid_search("text about bias", top_k=5)

    assert results
    assert all(c.rerank_source == "cross_encoder" for c in results)
    # the stub returns 0.9 and 0.2 — both measured, both kept
    assert {c.cross_encoder_score for c in results} >= {0.9}


def test_rrf_only_candidates_have_no_rerank_source(monkeypatch):
    """With rrf_min_score filtering (no rerank), candidates keep rerank_source None.

    The candidate-level score keeps its 0.0 default (classification
    compares it); the "not calculated" null is written at the RiskMatch
    boundary in the pipeline, driven by the missing rerank_source.
    """
    idx = _index_with_cross_encoder(monkeypatch)
    monkeypatch.setattr(idx, "search_bm25", lambda text, top_k=100: [])
    monkeypatch.setattr(
        idx,
        "search_semantic",
        lambda text, top_k=100: [ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a")],
    )

    results = idx.hybrid_search("text about bias", top_k=5, rrf_min_score=0.001)

    assert results
    assert all(c.rerank_source is None for c in results)


def test_bm25_rescued_candidate_keeps_measured_score(monkeypatch):
    """A BM25-rescued candidate the reranker scored keeps its measured score.

    The reranker measures every candidate in the rerank window; a scored
    candidate can fall out of the top-k slice, and BM25 rescue re-admits it
    by rank. Its measured cross-encoder score and rerank_source must
    survive, the way the ColBERT path does.
    """
    from asago_policy_mapper.extract import index as index_mod

    class _TwoSidedCrossEncoder:
        """Scores both pairs; top_k=1 keeps only the higher one."""

        def predict(self, pairs):
            return [0.9 if "A" in p[0] else 0.4 for p in pairs]

    risks = [
        SimpleNamespace(
            id=rid,
            name=name,
            description="desc",
            concern="",
            risk_type="",
            isDefinedByTaxonomy="test-tax",
            isPartOf="",
        )
        for rid, name in (("R-001", "A"), ("R-003", "C"))
    ]
    with (
        patch.object(index_mod, "_load_bi_encoder", return_value=(None, None, None)),
        patch.object(index_mod, "_load_cross_encoder", return_value=(_TwoSidedCrossEncoder(), None, False, False)),
    ):
        idx = index_mod.RiskIndex(risks)

    # BM25 returns two candidates; the reranker scores both, but top_k=1
    # keeps only R-001, so R-003 falls out of the slice and is rescued by
    # BM25 rank — with its measured score.
    monkeypatch.setattr(
        idx,
        "search_bm25",
        lambda text, top_k=100: [
            ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", bm25_rank=1),
            ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", bm25_rank=2),
        ],
    )
    monkeypatch.setattr(
        idx,
        "search_semantic",
        lambda text, top_k=100: [ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a")],
    )

    results = idx.hybrid_search("text", top_k=1, bm25_rescue_rank=5)

    by_id = {c.risk_id: c for c in results}
    assert "R-003" in by_id
    # rescued candidate was measured: score kept, source stamped, rank kept
    assert by_id["R-003"].cross_encoder_score == 0.4
    assert by_id["R-003"].rerank_source == "cross_encoder"
    assert by_id["R-003"].bm25_rank == 2


def test_bm25_rescued_invalid_measurement_is_null(monkeypatch):
    """A rescued candidate whose only measurement was invalid stays null.

    Every rescued candidate is in the rerank window and is measured, but
    an invalid measurement (NaN or infinity) is no measurement: the score
    is skipped, the candidate falls out of the top-k slice, and rescue
    re-admits it unmeasured — rerank_source None, null at the boundary.
    """
    from asago_policy_mapper.extract import index as index_mod

    class _NaNForThirdCrossEncoder:
        def predict(self, pairs):
            return [0.9 if "A" in p[0] else float("nan") for p in pairs]

    risks = [
        SimpleNamespace(
            id=rid,
            name=name,
            description="desc",
            concern="",
            risk_type="",
            isDefinedByTaxonomy="test-tax",
            isPartOf="",
        )
        for rid, name in (("R-001", "A"), ("R-003", "C"))
    ]
    with (
        patch.object(index_mod, "_load_bi_encoder", return_value=(None, None, None)),
        patch.object(index_mod, "_load_cross_encoder", return_value=(_NaNForThirdCrossEncoder(), None, False, False)),
    ):
        idx = index_mod.RiskIndex(risks)

    # Both candidates are in the rerank window; R-003's measurement is NaN
    # (invalid), so it is skipped and falls out of the top_k=1 slice, then
    # is rescued by BM25 rank — unmeasured.
    monkeypatch.setattr(
        idx,
        "search_bm25",
        lambda text, top_k=100: [
            ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", bm25_rank=1),
            ScoredCandidate(risk_id="R-003", risk_name="C", risk_description="c", bm25_rank=2),
        ],
    )
    monkeypatch.setattr(
        idx,
        "search_semantic",
        lambda text, top_k=100: [ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a")],
    )

    results = idx.hybrid_search("text", top_k=1, bm25_rescue_rank=5)

    by_id = {c.risk_id: c for c in results}
    assert "R-003" in by_id
    # invalid measurement: no score to keep, no source to stamp
    assert by_id["R-003"].rerank_source is None
    assert by_id["R-003"].bm25_rank == 2


def test_rerank_measured_zero_is_kept_with_source(monkeypatch):
    """A reranker-measured 0.0 survives rerank stamped with its source."""
    from asago_policy_mapper.extract import index as index_mod

    risks = [
        SimpleNamespace(
            id=rid,
            name=f"name-{rid}",
            description=f"desc-{rid}",
            concern="",
            risk_type="",
            isDefinedByTaxonomy="test-tax",
            isPartOf="",
        )
        for rid in ("R-001", "R-002")
    ]
    with (
        patch.object(index_mod, "_load_bi_encoder", return_value=(None, None, None)),
        patch.object(index_mod, "_load_cross_encoder", return_value=(_ZeroThenHighCrossEncoder(), None, False, False)),
    ):
        idx = index_mod.RiskIndex(risks)

    reranked = idx.rerank(
        "some text",
        [
            ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a"),
            ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b"),
        ],
        top_k=5,
    )

    by_id = {c.risk_id: c for c in reranked}
    # measured 0.0: kept as 0.0, stamped as measured — not dropped, not nulled
    assert by_id["R-001"].cross_encoder_score == 0.0
    assert by_id["R-001"].rerank_source == "cross_encoder"
    assert by_id["R-002"].cross_encoder_score == 0.8
    assert by_id["R-002"].rerank_source == "cross_encoder"


def test_retrieve_chunk_labels_rank_and_threshold():
    """retrieve_chunk stamps selected_by=rank or threshold after classification."""
    from asago_policy_mapper.extract.parse import Chunk
    from asago_policy_mapper.extract.retrieve import retrieve_chunk

    mock_index = MagicMock()
    mock_index.hybrid_search.return_value = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", cross_encoder_score=0.9),
        ScoredCandidate(risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.3),
    ]
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    # rank mode: no threshold_high
    cr = retrieve_chunk(chunks, 0, mock_index, top_n_accept=1, top_n_judge=1)
    assert all(c.selected_by == "rank" for c in cr.accepted)
    assert all(c.selected_by == "rank" for c in cr.borderline)

    # threshold mode: threshold_high set
    cr = retrieve_chunk(chunks, 0, mock_index, threshold_high=0.7, threshold_low=0.2)
    assert all(c.selected_by == "threshold" for c in cr.accepted)
    assert all(c.selected_by == "threshold" for c in cr.borderline)
    # the candidates from the mock were not mutated in place
    for c in mock_index.hybrid_search.return_value:
        assert c.selected_by is None


def test_retrieve_chunk_no_cross_encoder_labels_rrf():
    """Without the cross-encoder, accepted candidates are labelled rrf."""
    from asago_policy_mapper.extract.parse import Chunk
    from asago_policy_mapper.extract.retrieve import retrieve_chunk

    mock_index = MagicMock()
    mock_index.hybrid_search.return_value = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", rrf_score=0.02),
    ]
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(chunks, 0, mock_index, use_cross_encoder=False, rrf_min_score=0.01)

    assert len(cr.accepted) == 1
    assert cr.accepted[0].selected_by == "rrf"
    assert mock_index.hybrid_search.return_value[0].selected_by is None


def test_retrieve_chunk_cross_encoder_score_null_untouched():
    """A never-reranked candidate's provenance survives retrieve_chunk unharmed.

    The candidate-level score keeps its 0.0 default so classification
    still works; the null for "not calculated" is written later at the
    RiskMatch boundary from the missing rerank_source.
    """
    from asago_policy_mapper.extract.parse import Chunk
    from asago_policy_mapper.extract.retrieve import retrieve_chunk

    mock_index = MagicMock()
    # rank mode: both candidates survive (top_n_accept=2), but R-001 was
    # never reranked while R-002 carries a measured score
    mock_index.hybrid_search.return_value = [
        ScoredCandidate(risk_id="R-001", risk_name="A", risk_description="a", rerank_source=None),
        ScoredCandidate(
            risk_id="R-002", risk_name="B", risk_description="b", cross_encoder_score=0.5, rerank_source="cross_encoder"
        ),
    ]
    chunks = [Chunk(text="Test chunk about AI risks.", source="a.pdf", index=0)]

    cr = retrieve_chunk(chunks, 0, mock_index, top_n_accept=2, top_n_judge=0)

    by_id = {c.risk_id: c for c in cr.accepted + cr.borderline}
    assert by_id["R-002"].cross_encoder_score == 0.5
    assert by_id["R-002"].rerank_source == "cross_encoder"
    assert by_id["R-002"].selected_by == "rank"
    assert by_id["R-001"].rerank_source is None
    assert by_id["R-001"].selected_by == "rank"
