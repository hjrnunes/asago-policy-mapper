"""Truth-table tests for the run-plan resolver.

The expected table in ``_MODE_ROWS`` is pinned from the pipeline's current
mode resolution (``run_extraction`` and its stage gates in
``extract/pipeline.py``, ``retrieve_chunk`` in ``extract/retrieve.py``):

- the judge is unreachable without the cross-encoder, because
  ``retrieve_chunk`` returns early with no borderline candidates and
  ``_run_judge`` checks ``use_cross_encoder``;
- query generation's fallback chunks go through per-chunk cross-encoder
  retrieval, so the judge stays reachable when query generation is on;
- expansion, variant grounding, causal synthesis and the prefilter are
  effective only when grounding is on (ungrounded matches carry no
  evidence, so ``synthesize_causal_chain`` skips them too).
"""

import itertools

import pytest

from asago_policy_mapper.extract.models import RetrievalConfig
from asago_policy_mapper.extract.plan import resolve_plan

# llm_stages building blocks, in pipeline order.
_QG = ("query_gen",)
_JUDGE = ("judge",)
_GROUND = ("grounding", "variant_grounding")
_EXPANSION = ("expansion",)
_CAUSAL = ("causal_synthesis",)

# Key: (query_gen, use_cross_encoder, no_judge, no_grounding, expand_siblings,
#       no_causal_synthesis) -> (judge, llm_stages).
_MODE_ROWS = {
    (False, False, False, False, False, False): ("none", _GROUND + _CAUSAL),
    (False, False, False, False, False, True): ("none", _GROUND),
    (False, False, False, False, True, False): ("none", _GROUND + _EXPANSION + _CAUSAL),
    (False, False, False, False, True, True): ("none", _GROUND + _EXPANSION),
    (False, False, False, True, False, False): ("none", ()),
    (False, False, False, True, False, True): ("none", ()),
    (False, False, False, True, True, False): ("none", ()),
    (False, False, False, True, True, True): ("none", ()),
    (False, False, True, False, False, False): ("none", _GROUND + _CAUSAL),
    (False, False, True, False, False, True): ("none", _GROUND),
    (False, False, True, False, True, False): ("none", _GROUND + _EXPANSION + _CAUSAL),
    (False, False, True, False, True, True): ("none", _GROUND + _EXPANSION),
    (False, False, True, True, False, False): ("none", ()),
    (False, False, True, True, False, True): ("none", ()),
    (False, False, True, True, True, False): ("none", ()),
    (False, False, True, True, True, True): ("none", ()),
    (False, True, False, False, False, False): ("llm", _JUDGE + _GROUND + _CAUSAL),
    (False, True, False, False, False, True): ("llm", _JUDGE + _GROUND),
    (False, True, False, False, True, False): ("llm", _JUDGE + _GROUND + _EXPANSION + _CAUSAL),
    (False, True, False, False, True, True): ("llm", _JUDGE + _GROUND + _EXPANSION),
    (False, True, False, True, False, False): ("llm", _JUDGE),
    (False, True, False, True, False, True): ("llm", _JUDGE),
    (False, True, False, True, True, False): ("llm", _JUDGE),
    (False, True, False, True, True, True): ("llm", _JUDGE),
    (False, True, True, False, False, False): ("auto_promote", _GROUND + _CAUSAL),
    (False, True, True, False, False, True): ("auto_promote", _GROUND),
    (False, True, True, False, True, False): ("auto_promote", _GROUND + _EXPANSION + _CAUSAL),
    (False, True, True, False, True, True): ("auto_promote", _GROUND + _EXPANSION),
    (False, True, True, True, False, False): ("auto_promote", ()),
    (False, True, True, True, False, True): ("auto_promote", ()),
    (False, True, True, True, True, False): ("auto_promote", ()),
    (False, True, True, True, True, True): ("auto_promote", ()),
    (True, False, False, False, False, False): ("none", _QG + _GROUND + _CAUSAL),
    (True, False, False, False, False, True): ("none", _QG + _GROUND),
    (True, False, False, False, True, False): ("none", _QG + _GROUND + _EXPANSION + _CAUSAL),
    (True, False, False, False, True, True): ("none", _QG + _GROUND + _EXPANSION),
    (True, False, False, True, False, False): ("none", _QG),
    (True, False, False, True, False, True): ("none", _QG),
    (True, False, False, True, True, False): ("none", _QG),
    (True, False, False, True, True, True): ("none", _QG),
    (True, False, True, False, False, False): ("none", _QG + _GROUND + _CAUSAL),
    (True, False, True, False, False, True): ("none", _QG + _GROUND),
    (True, False, True, False, True, False): ("none", _QG + _GROUND + _EXPANSION + _CAUSAL),
    (True, False, True, False, True, True): ("none", _QG + _GROUND + _EXPANSION),
    (True, False, True, True, False, False): ("none", _QG),
    (True, False, True, True, False, True): ("none", _QG),
    (True, False, True, True, True, False): ("none", _QG),
    (True, False, True, True, True, True): ("none", _QG),
    (True, True, False, False, False, False): ("llm", _QG + _JUDGE + _GROUND + _CAUSAL),
    (True, True, False, False, False, True): ("llm", _QG + _JUDGE + _GROUND),
    (True, True, False, False, True, False): ("llm", _QG + _JUDGE + _GROUND + _EXPANSION + _CAUSAL),
    (True, True, False, False, True, True): ("llm", _QG + _JUDGE + _GROUND + _EXPANSION),
    (True, True, False, True, False, False): ("llm", _QG + _JUDGE),
    (True, True, False, True, False, True): ("llm", _QG + _JUDGE),
    (True, True, False, True, True, False): ("llm", _QG + _JUDGE),
    (True, True, False, True, True, True): ("llm", _QG + _JUDGE),
    (True, True, True, False, False, False): ("auto_promote", _QG + _GROUND + _CAUSAL),
    (True, True, True, False, False, True): ("auto_promote", _QG + _GROUND),
    (True, True, True, False, True, False): ("auto_promote", _QG + _GROUND + _EXPANSION + _CAUSAL),
    (True, True, True, False, True, True): ("auto_promote", _QG + _GROUND + _EXPANSION),
    (True, True, True, True, False, False): ("auto_promote", _QG),
    (True, True, True, True, False, True): ("auto_promote", _QG),
    (True, True, True, True, True, False): ("auto_promote", _QG),
    (True, True, True, True, True, True): ("auto_promote", _QG),
}


def _retrieval(query_gen, use_cross_encoder, no_judge, no_grounding, expand_siblings, no_causal_synthesis):
    return RetrievalConfig(
        query_gen=query_gen,
        use_cross_encoder=use_cross_encoder,
        no_judge=no_judge,
        no_grounding=no_grounding,
        expand_siblings=expand_siblings,
        no_causal_synthesis=no_causal_synthesis,
    )


@pytest.mark.parametrize("prefilter", [False, True])
@pytest.mark.parametrize(
    "query_gen,use_cross_encoder,no_judge,no_grounding,expand_siblings,no_causal_synthesis",
    list(itertools.product((False, True), repeat=6)),
)
def test_plan_matrix(
    query_gen,
    use_cross_encoder,
    no_judge,
    no_grounding,
    expand_siblings,
    no_causal_synthesis,
    prefilter,
):
    flags = (query_gen, use_cross_encoder, no_judge, no_grounding, expand_siblings, no_causal_synthesis)
    expected_judge, expected_stages = _MODE_ROWS[flags]
    retrieval = _retrieval(*flags)
    plan = resolve_plan(retrieval, prefilter=prefilter, explicit=frozenset())
    grounding = not no_grounding
    assert plan.acquisition == ("query_gen" if query_gen else "per_chunk")
    assert plan.rerank == ("cross_encoder" if use_cross_encoder else "none")
    assert plan.judge == expected_judge
    assert plan.grounding is grounding
    assert plan.variant_grounding is grounding
    assert plan.expansion is (grounding and expand_siblings)
    assert plan.causal_synthesis is (grounding and not no_causal_synthesis)
    assert plan.prefilter is (prefilter and grounding)
    assert plan.judge_context_tokens == retrieval.judge_context_tokens
    assert plan.llm_stages == expected_stages
    assert plan.needs_llm is bool(expected_stages)
    assert plan.warnings == ()


def test_no_ce_no_grounding_no_querygen_needs_no_llm():
    # Without the cross-encoder no borderline candidates exist, so the judge
    # cannot run even though --no-judge was never passed: the run needs no
    # LLM at all.
    retrieval = RetrievalConfig(query_gen=False, use_cross_encoder=False, no_grounding=True)
    plan = resolve_plan(retrieval, prefilter=False, explicit=frozenset())
    assert plan.judge == "none"
    assert plan.llm_stages == ()
    assert plan.needs_llm is False


def test_query_gen_fallback_keeps_judge_reachable():
    # Query generation's fallback runs per-chunk cross-encoder retrieval, so
    # borderline candidates still exist and the judge still runs.
    retrieval = RetrievalConfig(query_gen=True, no_grounding=True)
    plan = resolve_plan(retrieval, prefilter=False, explicit=frozenset())
    assert plan.judge == "llm"
    assert plan.llm_stages == ("query_gen", "judge")
    assert plan.needs_llm is True


def test_expand_siblings_ignored_without_grounding():
    retrieval = RetrievalConfig(no_grounding=True, expand_siblings=True)
    plan = resolve_plan(retrieval, prefilter=False, explicit=frozenset())
    assert plan.expansion is False

    grounded = resolve_plan(RetrievalConfig(expand_siblings=True), prefilter=False, explicit=frozenset())
    assert grounded.expansion is True


def test_prefilter_requires_grounding():
    retrieval = RetrievalConfig(no_grounding=True)
    plan = resolve_plan(retrieval, prefilter=True, explicit=frozenset())
    assert plan.prefilter is False

    grounded = resolve_plan(RetrievalConfig(), prefilter=True, explicit=frozenset())
    assert grounded.prefilter is True


def test_rerank_label_reflects_colbert():
    # ColBERT replaces the bi-encoder and the cross-encoder, while the judge
    # still tracks use_cross_encoder: classify_candidates runs on the MaxSim
    # scores stored in cross_encoder_score.
    retrieval = RetrievalConfig(colbert_model="colbert-ir", query_gen=False)
    plan = resolve_plan(retrieval, prefilter=False, explicit=frozenset())
    assert plan.acquisition == "per_chunk"
    assert plan.rerank == "colbert"
    assert plan.judge == "llm"

    no_ce = RetrievalConfig(colbert_model="colbert-ir", query_gen=False, use_cross_encoder=False)
    plan = resolve_plan(no_ce, prefilter=False, explicit=frozenset())
    assert plan.rerank == "colbert"
    assert plan.judge == "none"


def test_judge_context_tokens_recorded():
    retrieval = RetrievalConfig(query_gen=False, judge_context_tokens=512)
    plan = resolve_plan(retrieval, prefilter=False, explicit=frozenset({"--judge-context-tokens"}))
    assert plan.judge_context_tokens == 512
    assert plan.warnings == ()
