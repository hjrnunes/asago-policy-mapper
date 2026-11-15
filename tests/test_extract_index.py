import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asago_policy_mapper.extract.index import (
    RiskIndex,
    _is_remote,
    _make_score_normalizer,
    _parse_remote_url,
    _RemoteBiEncoder,
    _RemoteCrossEncoder,
    _rerank_pairs,
    _rrf_fuse,
    _scored_candidates,
)
from asago_policy_mapper.extract.models import ScoredCandidate


def _make_risk(id, name, description, concern=""):
    return SimpleNamespace(
        id=id,
        name=name,
        description=description,
        concern=concern,
        risk_type="",
        isDefinedByTaxonomy="test-taxonomy",
        isPartOf="",
    )


RISKS = [
    _make_risk("R-001", "Model Bias", "Systematic errors in AI model outputs that favor certain groups over others"),
    _make_risk("R-002", "Data Poisoning", "Malicious manipulation of training data to compromise model integrity"),
    _make_risk("R-003", "Privacy Violation", "Unauthorized collection or use of personal data by AI systems"),
    _make_risk("R-004", "Lack of Transparency", "Inability to explain or understand AI decision-making processes"),
    _make_risk("R-005", "Workforce Displacement", "Automation of jobs leading to unemployment and economic disruption"),
]


@pytest.fixture
def index():
    return RiskIndex(RISKS)


@pytest.mark.slow
def test_search_bm25_exact_term(index):
    results = index.search_bm25("model bias", top_k=3)
    assert len(results) > 0
    assert results[0].risk_id == "R-001"
    assert results[0].bm25_rank == 1


@pytest.mark.slow
def test_search_bm25_no_match(index):
    results = index.search_bm25("quantum computing hardware", top_k=3)
    assert len(results) == 0 or results[0].bm25_rank > 0


@pytest.mark.slow
def test_search_bm25_respects_top_k(index):
    results = index.search_bm25("AI system", top_k=2)
    assert len(results) <= 2


@pytest.mark.slow
def test_search_semantic(index):
    results = index.search_semantic("biased AI predictions discriminating against minorities", top_k=3)
    assert len(results) > 0
    risk_ids = [r.risk_id for r in results]
    assert "R-001" in risk_ids


@pytest.mark.slow
def test_search_semantic_returns_distance(index):
    results = index.search_semantic("data privacy", top_k=1)
    assert len(results) == 1
    assert 0.0 <= results[0].embedding_distance <= 2.0


@pytest.mark.slow
def test_rerank(index):
    candidates = index.search_semantic("personal data collection without consent", top_k=5)
    reranked = index.rerank("personal data collection without consent", candidates, top_k=3)
    assert len(reranked) <= 3
    assert all(0.0 <= c.cross_encoder_score <= 1.0 for c in reranked)
    scores = [c.cross_encoder_score for c in reranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.slow
def test_hybrid_search(index):
    results = index.hybrid_search("training data manipulation attack", top_k=3)
    assert len(results) > 0
    assert all(c.rrf_score > 0 for c in results)
    assert all(0.0 <= c.cross_encoder_score <= 1.0 for c in results)
    risk_ids = [r.risk_id for r in results]
    assert "R-002" in risk_ids


@pytest.mark.slow
def test_hybrid_search_populates_all_scores(index):
    results = index.hybrid_search("bias in AI", top_k=3)
    for c in results:
        assert c.rrf_score > 0
        assert c.cross_encoder_score >= 0


def test_empty_risks():
    index = RiskIndex([])
    assert index.search_bm25("anything", top_k=5) == []
    assert index.search_semantic("anything", top_k=5) == []
    assert index.hybrid_search("anything", top_k=5) == []


def _index_with_predict(monkeypatch, score_sequence):
    """Build an empty index whose remote reranker returns a fixed score vector."""
    index = RiskIndex([])
    # The generative remote reranker uses the default clip normalizer, which
    # preserves NaN; rerank's handling is what this pin exercises.
    remote = SimpleNamespace(predict=lambda pairs: np.array(score_sequence[: len(pairs)], dtype=np.float64))
    monkeypatch.setattr(index, "_remote_cross_encoder", remote)
    return index


def test_rerank_nan_score_does_not_consume_top_k_slot(monkeypatch):
    """A NaN (no-answer) candidate rides along last; it must not displace a valid candidate from top_k."""
    index = _index_with_predict(monkeypatch, [float("nan"), 0.9, 0.8])
    candidates = [
        ScoredCandidate(risk_id="R-A", risk_name="A", risk_description="a", cross_encoder_score=0.5),
        ScoredCandidate(risk_id="R-B", risk_name="B", risk_description="b", cross_encoder_score=0.5),
        ScoredCandidate(risk_id="R-C", risk_name="C", risk_description="c", cross_encoder_score=0.5),
    ]

    reranked = index.rerank("query", candidates, top_k=2)

    # Both valid candidates keep their slots; the unscored one carries its
    # NaN marker at the tail of the list.
    assert [c.risk_id for c in reranked] == ["R-B", "R-C", "R-A"]
    assert [c.cross_encoder_score for c in reranked[:2]] == [0.9, 0.8]
    assert math.isnan(reranked[2].cross_encoder_score)


def test_rerank_equal_scores_keep_input_order(monkeypatch):
    """Equal measured scores keep the input order, which is RRF rank descending.

    The sort behind the top-k selection is stable, and hybrid_search
    constructs the rerank window in RRF-rank order, so ties resolve to
    RRF rank: a fused candidate that outranks another on RRF stays ahead
    when the reranker scores them equally. The pin holds the whole chain
    — construction order plus sort stability — which no other test
    exercises (all others use distinct scores).
    """
    index = _index_with_predict(monkeypatch, [0.8, 0.8, 0.8])
    # RRF-descending order, as hybrid_search would construct the window:
    # R-A outranks R-B on RRF, R-B outranks R-C.
    candidates = [
        ScoredCandidate(risk_id="R-A", risk_name="A", risk_description="a", rrf_score=0.032),
        ScoredCandidate(risk_id="R-B", risk_name="B", risk_description="b", rrf_score=0.030),
        ScoredCandidate(risk_id="R-C", risk_name="C", risk_description="c", rrf_score=0.028),
    ]

    reranked = index.rerank("query", candidates, top_k=3)

    # Equal scores, unchanged order: the tie fell back to RRF rank.
    assert [c.risk_id for c in reranked] == ["R-A", "R-B", "R-C"]
    assert [c.cross_encoder_score for c in reranked] == [0.8, 0.8, 0.8]


def test_rerank_counts_unscored(monkeypatch, caplog):
    """Unscored candidates ride at the tail carrying the NaN marker, and warn once each.

    Unscored candidates stay in the returned list, after every scored one,
    so the selection path can route them to the judge instead of dropping
    them.
    """
    index = _index_with_predict(monkeypatch, [float("nan"), 0.9, float("nan"), 0.8])
    candidates = [
        ScoredCandidate(risk_id="R-A", risk_name="A", risk_description="a"),
        ScoredCandidate(risk_id="R-B", risk_name="B", risk_description="b"),
        ScoredCandidate(risk_id="R-C", risk_name="C", risk_description="c"),
        ScoredCandidate(risk_id="R-D", risk_name="D", risk_description="d"),
    ]

    with caplog.at_level("WARNING", logger="asago_policy_mapper.extract.index"):
        reranked = index.rerank("query", candidates, top_k=4)

    assert [c.risk_id for c in reranked] == ["R-B", "R-D", "R-A", "R-C"]
    assert index.rerank_unavailable == 2
    skip_warnings = [r.getMessage() for r in caplog.records if "unscored" in r.getMessage()]
    assert skip_warnings == ["Candidate R-A unscored by reranker", "Candidate R-C unscored by reranker"]


def test_rerank_majority_unscored_raises(monkeypatch, caplog):
    """A majority-unscored rerank call raises: the reranker is broken, not chatty."""
    index = _index_with_predict(monkeypatch, [float("nan"), 0.9, float("nan"), float("nan")])
    candidates = [
        ScoredCandidate(risk_id="R-A", risk_name="A", risk_description="a"),
        ScoredCandidate(risk_id="R-B", risk_name="B", risk_description="b"),
        ScoredCandidate(risk_id="R-C", risk_name="C", risk_description="c"),
        ScoredCandidate(risk_id="R-D", risk_name="D", risk_description="d"),
    ]

    with caplog.at_level("WARNING", logger="asago_policy_mapper.extract.index"):
        with pytest.raises(
            RuntimeError,
            match=r"^Reranker left 3 of 4 candidates unscored; check the reranker endpoint and model$",
        ):
            index.rerank("query", candidates, top_k=4)

    # The count and warnings land before the raise, so the failure is
    # observable in the run's diagnostics.
    assert index.rerank_unavailable == 3
    skip_warnings = [r for r in caplog.records if "unscored by reranker" in r.message]
    assert len(skip_warnings) == 3


def test_rerank_half_unscored_does_not_raise(monkeypatch):
    """Exactly half unscored is not a majority: candidates still ride along."""
    index = _index_with_predict(monkeypatch, [float("nan"), 0.9, float("nan"), 0.8])
    candidates = [
        ScoredCandidate(risk_id="R-A", risk_name="A", risk_description="a"),
        ScoredCandidate(risk_id="R-B", risk_name="B", risk_description="b"),
        ScoredCandidate(risk_id="R-C", risk_name="C", risk_description="c"),
        ScoredCandidate(risk_id="R-D", risk_name="D", risk_description="d"),
    ]

    reranked = index.rerank("query", candidates, top_k=4)

    assert [c.risk_id for c in reranked] == ["R-B", "R-D", "R-A", "R-C"]
    assert index.rerank_unavailable == 2


def test_rerank_unavailable_counter_starts_at_zero():
    index = RiskIndex([])
    assert index.rerank_unavailable == 0


def test_rerank_unavailable_accumulates_across_calls(monkeypatch):
    """The counter accumulates across rerank calls, not resets per call."""
    index = _index_with_predict(monkeypatch, [float("nan"), 0.9])
    candidates = [
        ScoredCandidate(risk_id="R-A", risk_name="A", risk_description="a"),
        ScoredCandidate(risk_id="R-B", risk_name="B", risk_description="b"),
    ]

    index.rerank("query one", candidates, top_k=2)
    index.rerank("query two", candidates, top_k=2)

    assert index.rerank_unavailable == 2


@pytest.mark.slow
def test_risk_count(index):
    assert index.risk_count == 5


@pytest.mark.slow
def test_cross_encoder_property(index):
    assert index.cross_encoder is not None


@pytest.mark.slow
def test_no_cross_encoder():
    idx = RiskIndex(RISKS, cross_encoder_model=None)
    assert idx.cross_encoder is None
    assert idx.rerank("test", [], top_k=3) == []


@pytest.mark.slow
def test_hybrid_search_rrf_only(index):
    results = index.hybrid_search("training data manipulation attack", top_k=3, rrf_min_score=0.005)
    assert len(results) > 0
    assert all(c.rrf_score >= 0.005 for c in results)
    assert all(c.cross_encoder_score == 0.0 for c in results)


@pytest.mark.slow
def test_hybrid_search_rrf_only_filters_low_scores(index):
    all_results = index.hybrid_search("bias", top_k=50, rrf_min_score=0.001)
    high_results = index.hybrid_search("bias", top_k=50, rrf_min_score=0.02)
    assert len(high_results) <= len(all_results)
    assert all(c.rrf_score >= 0.02 for c in high_results)


def test_is_remote():
    assert _is_remote("https://host.example.com/v1/embeddings")
    assert _is_remote("http://localhost:8000/v1/embeddings")
    assert not _is_remote("all-mpnet-base-v2")
    assert not _is_remote("cross-encoder/ms-marco-MiniLM-L-12-v2")


def test_parse_remote_url():
    base, model = _parse_remote_url("https://bge-m3-model-serving.apps.example.com/v1/embeddings")
    assert base == "https://bge-m3-model-serving.apps.example.com/v1"
    assert model == "bge-m3"

    base2, model2 = _parse_remote_url(
        "https://gte-reranker-modernbert-base-model-serving.apps.rosa.example.com/v1/score"
    )
    assert base2 == "https://gte-reranker-modernbert-base-model-serving.apps.rosa.example.com/v1"
    assert model2 == "gte-reranker-modernbert-base"

    base3, model3 = _parse_remote_url("https://host.com")
    assert base3 == "https://host.com/v1"
    assert model3 == "host"

    base4, _ = _parse_remote_url("https://host.com/v1")
    assert base4 == "https://host.com/v1"


def test_get_taxonomy_resolves_collapsed_variant_ids():
    """Variants are collapsed out of the index, yet each keeps the taxonomy it was defined by."""
    risks = [
        _make_risk("R-001", "Model Bias", "Systematic errors in AI model outputs"),
        SimpleNamespace(
            id="tax-unauthorized-use---pii",
            name="Unauthorized use - PII",
            description="Unauthorized use of PII.",
            concern="",
            isDefinedByTaxonomy="variant-taxonomy",
            isPartOf="",
        ),
        SimpleNamespace(
            id="tax-unauthorized-use---health",
            name="Unauthorized use - Health",
            description="Unauthorized use of health records.",
            concern="",
            isDefinedByTaxonomy="variant-taxonomy",
            isPartOf="",
        ),
    ]
    with (
        patch("asago_policy_mapper.extract.index._load_bi_encoder", return_value=(None, None, None)),
        patch("asago_policy_mapper.extract.index._load_cross_encoder", return_value=(None, None, False, False)),
    ):
        index = RiskIndex(risks)

    assert index.get_taxonomy("R-001") == "test-taxonomy"
    assert index.get_taxonomy("tax-unauthorized-use") == "variant-taxonomy"
    assert index.get_taxonomy("tax-unauthorized-use---pii") == "variant-taxonomy"
    assert index.get_taxonomy("tax-unauthorized-use---health") == "variant-taxonomy"
    assert index.get_taxonomy("unknown-risk") == ""


def test_colbert_remote_rejected():
    with pytest.raises(ValueError, match="ColBERT models cannot be served remotely"):
        RiskIndex(RISKS, colbert_model="https://lateon.example.com/v1/embeddings")


# --- _rrf_fuse tests ---


def _sc(risk_id, ce_score=0.0, embed_dist=0.0, bm25_rank=0):
    return ScoredCandidate(
        risk_id=risk_id,
        risk_name=risk_id,
        risk_description="",
        cross_encoder_score=ce_score,
        embedding_distance=embed_dist,
        bm25_rank=bm25_rank,
    )


def test_rrf_fuse_basic():
    list_a = [_sc("R-001"), _sc("R-002")]
    list_b = [_sc("R-002", embed_dist=0.3), _sc("R-003", embed_dist=0.5)]
    rrf_scores, candidate_data, bm25_ranks = _rrf_fuse(list_a, list_b, rrf_k=60)
    assert "R-001" in rrf_scores
    assert "R-002" in rrf_scores
    assert "R-003" in rrf_scores
    assert rrf_scores["R-002"] > rrf_scores["R-001"]
    assert bm25_ranks["R-001"] == 1
    assert bm25_ranks["R-002"] == 2
    assert "R-003" not in bm25_ranks
    assert candidate_data["R-001"].risk_id == "R-001"


def test_rrf_fuse_first_occurrence_wins():
    list_a = [_sc("R-001", ce_score=0.9)]
    list_b = [_sc("R-001", ce_score=0.1)]
    _, candidate_data, _ = _rrf_fuse(list_a, list_b)
    assert candidate_data["R-001"].cross_encoder_score == 0.9


# --- _rerank_pairs and _scored_candidates tests ---


def test_rerank_pairs_score_models_reverse_the_pair():
    candidates = [_sc("R-001"), _sc("R-002")]
    pairs = _rerank_pairs(is_nli=False, text="chunk text", candidates=candidates)
    assert pairs == [("R-001: ", "chunk text"), ("R-002: ", "chunk text")]


def test_rerank_pairs_nli_models_statement_second():
    candidates = [_sc("R-001")]
    pairs = _rerank_pairs(is_nli=True, text="chunk text", candidates=candidates)
    assert pairs == [("chunk text", "This text discusses R-001")]


def test_scored_candidates_sorts_and_attaches_scores():
    candidates = [_sc("R-001"), _sc("R-002"), _sc("R-003")]
    scored = _scored_candidates(candidates, np.array([0.1, 0.9, 0.5]), top_k=2)
    assert [c.risk_id for c in scored] == ["R-002", "R-003"]
    assert [c.cross_encoder_score for c in scored] == [0.9, 0.5]


def test_scored_candidates_skips_invalid_scores(caplog):
    candidates = [_sc("R-001"), _sc("R-002")]
    with caplog.at_level("WARNING", logger="asago_policy_mapper.extract.index"):
        scored = _scored_candidates(candidates, np.array([np.nan, 0.4]), top_k=2)
    assert [c.risk_id for c in scored] == ["R-002"]
    assert scored[0].cross_encoder_score == 0.4
    # The exact rendered message: the skip is diagnosable from the log.
    assert [record.getMessage() for record in caplog.records] == ["Skipping candidate R-001 with invalid score"]


# --- _make_score_normalizer tests ---


def test_score_normalizer_clip():
    norm = _make_score_normalizer(is_nli=False, apply_sigmoid=False)
    raw = np.array([0.5, 1.5, -0.3])
    result = norm(raw)
    np.testing.assert_array_almost_equal(result, [0.5, 1.0, 0.0])


def test_score_normalizer_sigmoid():
    norm = _make_score_normalizer(is_nli=False, apply_sigmoid=True)
    raw = np.array([0.0])
    result = norm(raw)
    assert abs(result[0] - 0.5) < 1e-6


def test_score_normalizer_nli_2d():
    norm = _make_score_normalizer(is_nli=True, apply_sigmoid=False)
    raw = np.array([[1.0, 2.0, 3.0]])
    result = norm(raw)
    assert result.shape == (1,)
    assert result[0] > 0.5


# --- _RemoteBiEncoder tests ---


@patch("openai.OpenAI")
def test_remote_bi_encoder_model_name_override(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])])
    encoder = _RemoteBiEncoder("https://my-endpoint-model-serving.apps.example.com/v1", model_name="actual-qwen3-model")
    encoder.encode(["test"], normalize=False)
    call_kwargs = mock_client.embeddings.create.call_args.kwargs
    assert call_kwargs["model"] == "actual-qwen3-model"


@patch("openai.OpenAI")
def test_remote_bi_encoder_default_api_key(mock_openai_cls):
    _RemoteBiEncoder("https://bge-m3-model-serving.apps.example.com/v1/embeddings")
    mock_openai_cls.assert_called_once()
    assert mock_openai_cls.call_args.kwargs["api_key"] == "none"


@patch("openai.OpenAI")
def test_remote_bi_encoder_custom_api_key(mock_openai_cls):
    _RemoteBiEncoder("https://bge-m3-model-serving.apps.example.com/v1/embeddings", api_key="my-secret-token")
    mock_openai_cls.assert_called_once()
    assert mock_openai_cls.call_args.kwargs["api_key"] == "my-secret-token"


@patch("openai.OpenAI")
def test_remote_bi_encoder_encode(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3]),
            SimpleNamespace(index=1, embedding=[0.4, 0.5, 0.6]),
        ]
    )

    encoder = _RemoteBiEncoder("https://bge-m3-model-serving.apps.example.com/v1/embeddings")
    result = encoder.encode(["hello", "world"], normalize=True)

    assert result.shape == (2, 3)
    # Verify rows are L2-normalized (unit vectors)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
    mock_client.embeddings.create.assert_called_once()


@patch("openai.OpenAI")
def test_remote_bi_encoder_encode_batches(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    # First batch: 2 texts
    batch1_response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
        ]
    )
    # Second batch: 1 text
    batch2_response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[1.0, 1.0]),
        ]
    )
    mock_client.embeddings.create.side_effect = [batch1_response, batch2_response]

    encoder = _RemoteBiEncoder(
        "https://bge-m3-model-serving.apps.example.com/v1/embeddings",
        batch_size=2,
    )
    result = encoder.encode(["a", "b", "c"], normalize=True)

    assert result.shape == (3, 2)
    assert mock_client.embeddings.create.call_count == 2
    # Verify first call got batch of 2, second call got batch of 1
    calls = mock_client.embeddings.create.call_args_list
    assert calls[0].kwargs["input"] == ["a", "b"]
    assert calls[1].kwargs["input"] == ["c"]


# --- _RemoteCrossEncoder tests ---


@patch("httpx.Client")
def test_remote_cross_encoder_predict(mock_httpx_client_cls):
    mock_client = MagicMock()
    mock_httpx_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 1, "score": 0.42},
            {"index": 0, "score": 0.85},
        ]
    }
    mock_client.post.return_value = mock_response

    encoder = _RemoteCrossEncoder("https://gte-reranker-model-serving.apps.example.com/v1/score")
    pairs = [("risk desc A", "chunk text"), ("risk desc B", "chunk text")]
    result = encoder.predict(pairs)

    assert result.shape == (2,)
    # Results should be sorted by index: index 0 -> 0.85, index 1 -> 0.42
    np.testing.assert_allclose(result, [0.85, 0.42])
    mock_client.post.assert_called_once()
    mock_response.raise_for_status.assert_called_once()


@patch("httpx.Client")
def test_remote_cross_encoder_predict_empty(mock_httpx_client_cls):
    mock_client = MagicMock()
    mock_httpx_client_cls.return_value = mock_client

    encoder = _RemoteCrossEncoder("https://gte-reranker-model-serving.apps.example.com/v1/score")
    result = encoder.predict([])

    assert result.shape == (0,)
    mock_client.post.assert_not_called()
