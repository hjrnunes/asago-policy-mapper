"""Resolve the effective run plan from the requested retrieval options.

The plan is a pure, flag-level description of what one extraction run will
do; nothing in this module runs a stage or touches a client. ``resolve_plan``
mirrors the pipeline's current mode resolution (``run_extraction`` and its
stage gates in ``extract/pipeline.py``, ``retrieve_chunk`` in
``extract/retrieve.py``):

- the judge is unreachable without the cross-encoder, because
  ``retrieve_chunk`` returns early with no borderline candidates and
  ``_run_judge`` checks ``use_cross_encoder``;
- query generation's fallback chunks go through per-chunk cross-encoder
  retrieval, so the judge stays reachable when query generation is on;
- expansion, variant grounding, causal synthesis and the prefilter are
  effective only when grounding is on, because ungrounded matches carry no
  evidence (so ``synthesize_causal_chain`` skips them too).

Two runtime conditions sit outside the plan: a missing client skips every
LLM stage, and variant grounding also needs variant risks in the catalog.

``explicit`` holds the CLI option names the user actually passed, in dashed
form (for example ``"--no-judge"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from asago_policy_mapper.extract.models import RetrievalConfig


@dataclass(frozen=True)
class RunPlan:
    """What one extraction run effectively does, resolved from the flags.

    ``llm_stages`` lists the pipeline stages that may call the LLM, in run
    order; ``needs_llm`` is true when any of them is present.
    """

    acquisition: Literal["query_gen", "per_chunk"]
    # The reranker on the per-chunk path; query-generation searches stay
    # unreranked by default.
    rerank: Literal["cross_encoder", "colbert", "none"]
    judge: Literal["llm", "auto_promote", "none"]
    grounding: bool
    variant_grounding: bool
    expansion: bool
    causal_synthesis: bool
    prefilter: bool
    judge_context_tokens: int
    llm_stages: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def needs_llm(self) -> bool:
        """Tell whether any resolved stage calls the LLM."""
        return bool(self.llm_stages)


def resolve_plan(retrieval: RetrievalConfig, *, prefilter: bool, explicit: frozenset[str]) -> RunPlan:
    """Resolve the effective run plan for one combination of options."""
    judge = _resolve_judge(retrieval)
    grounding = not retrieval.no_grounding
    return RunPlan(
        acquisition="query_gen" if retrieval.query_gen else "per_chunk",
        rerank=_rerank_label(retrieval),
        judge=judge,
        grounding=grounding,
        variant_grounding=grounding,
        expansion=grounding and retrieval.expand_siblings,
        causal_synthesis=grounding and not retrieval.no_causal_synthesis,
        prefilter=prefilter and grounding,
        judge_context_tokens=retrieval.judge_context_tokens,
        llm_stages=_resolve_llm_stages(retrieval, judge),
        warnings=(),
    )


def _resolve_judge(retrieval: RetrievalConfig) -> Literal["llm", "auto_promote", "none"]:
    """Resolve the judge mode from the cross-encoder and judge flags.

    Without the cross-encoder, ``retrieve_chunk`` returns early with no
    borderline candidates, so no judge ever runs.
    """
    if not retrieval.use_cross_encoder:
        return "none"
    return "auto_promote" if retrieval.no_judge else "llm"


def _rerank_label(retrieval: RetrievalConfig) -> Literal["cross_encoder", "colbert", "none"]:
    """Label the per-chunk reranker; ColBERT replaces the cross-encoder."""
    if retrieval.colbert_model:
        return "colbert"
    return "cross_encoder" if retrieval.use_cross_encoder else "none"


def _resolve_llm_stages(retrieval: RetrievalConfig, judge: Literal["llm", "auto_promote", "none"]) -> tuple[str, ...]:
    """List the stages that may call the LLM, in run order.

    Grounding gates its own stage and the stages that only ever see grounded
    matches: variant grounding, sibling expansion and causal synthesis.
    """
    stages: list[str] = []
    if retrieval.query_gen:
        stages.append("query_gen")
    if judge == "llm":
        stages.append("judge")
    if not retrieval.no_grounding:
        stages.extend(("grounding", "variant_grounding"))
        if retrieval.expand_siblings:
            stages.append("expansion")
        if not retrieval.no_causal_synthesis:
            stages.append("causal_synthesis")
    return tuple(stages)
