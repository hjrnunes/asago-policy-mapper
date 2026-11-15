"""Score sources and truthful accepted_by labels (issue #150 part 2).

Every score in the output says where it came from: "not calculated" is
None, not 0.0; rerank and ColBERT paths stamp their source on the
candidate; accepted_by is recorded when the candidate is selected. The
tests here pin the model-level contract (step 1) and the setter behaviour
(steps 2-4 build on it).
"""

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


def _match(risk_id, confidence, accepted_by="threshold", ce=0.5, evidence=None):
    return RiskMatch(
        risk_id=risk_id,
        risk_name=f"Risk {risk_id}",
        risk_description=f"Description of {risk_id}.",
        confidence=confidence,
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


def test_bm25_rescued_unreranked_is_null():
    """A BM25-rescued candidate the reranker never saw has null confidence."""
    m = _match("R-RESCUE", None, accepted_by="threshold", ce=None)
    assert m.confidence is None
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
        assert merged[0].confidence_source == "none"

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
