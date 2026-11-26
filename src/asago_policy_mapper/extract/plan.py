"""Resolve the effective run plan from the requested retrieval options.

The plan is a pure, flag-level description of what one extraction run will
do; nothing in this module runs a stage or touches a client. ``resolve_plan``
mirrors the pipeline's current mode resolution (``run_extraction`` and its
stage gates in ``extract/pipeline.py``, ``retrieve_chunk`` in
``extract/retrieve.py``):

- the judge is unreachable without the cross-encoder, because
  ``retrieve_chunk`` returns early with no borderline candidates and the
  judge stage branches on the plan's judge mode;
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

RRF_MIN_SCORE_ERROR = "--rrf-min-score must not be negative: it is a floor, never a ceiling"


def validate_rrf_min_score(rrf_min_score: float) -> None:
    """Refuse a negative RRF floor: a floor must never behave as a ceiling.

    One validator for every entry point — the extract command, the
    pipeline entry and the battery runner — so the message and the rule
    stay in step.
    """
    if rrf_min_score < 0:
        raise ValueError(RRF_MIN_SCORE_ERROR)


@dataclass(frozen=True)
class RunPlan:
    """What one extraction run effectively does, resolved from the flags.

    ``llm_stages`` lists the pipeline stages that may call the LLM, in run
    order; ``needs_llm`` is true when any of them is present.
    """

    candidate_source: Literal["query_gen", "per_chunk"]
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
        candidate_source="query_gen" if retrieval.query_gen else "per_chunk",
        rerank=_rerank_label(retrieval),
        judge=judge,
        grounding=grounding,
        variant_grounding=grounding,
        expansion=grounding and retrieval.expand_siblings,
        causal_synthesis=grounding and not retrieval.no_causal_synthesis,
        prefilter=prefilter and grounding,
        judge_context_tokens=retrieval.judge_context_tokens,
        llm_stages=_resolve_llm_stages(retrieval, judge),
        warnings=_resolve_warnings(retrieval, judge, explicit),
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


def _resolve_warnings(
    retrieval: RetrievalConfig, judge: Literal["llm", "auto_promote", "none"], explicit: frozenset[str]
) -> tuple[str, ...]:
    """Warn about explicitly passed options that are redundant or ignored.

    Only options the user actually passed (``explicit``, in dashed CLI form)
    can warn, and overlaps never change the resolved modes.
    """
    warnings: list[str] = []
    if "--no-judge" in explicit and judge == "none":
        warnings.append("--no-judge is redundant with --no-cross-encoder: no borderline candidates exist to judge")
    if "--expand-siblings" in explicit and retrieval.no_grounding:
        warnings.append("--expand-siblings is ignored with --no-grounding: expansion needs grounded matches")
    warnings.extend(_rrf_min_score_warnings(retrieval, explicit))
    if "--judge-context-tokens" in explicit and judge != "llm":
        warnings.append("--judge-context-tokens is unused: the judge does not run as an LLM stage")
    return tuple(warnings)


def _rrf_min_score_warnings(retrieval: RetrievalConfig, explicit: frozenset[str]) -> list[str]:
    """Warn when an explicit --rrf-min-score does not behave as a floor.

    On per-chunk retrieval the floor applies only when the cross-encoder is
    off; every query-generation search applies it while the value is
    positive. ColBERT retrieval ignores the value entirely. Negative values
    never reach the plan: the CLI rejects them before the plan resolves.
    """
    if "--rrf-min-score" not in explicit:
        return []
    if retrieval.colbert_model:
        return ["--rrf-min-score is ignored with --colbert-model: ColBERT retrieval has no RRF floor"]
    if not retrieval.use_cross_encoder:
        return _no_cross_encoder_floor_warnings(retrieval)
    if not retrieval.query_gen:
        return ["--rrf-min-score is ignored in per-chunk cross-encoder mode: the floor needs --no-cross-encoder"]
    if retrieval.rrf_min_score == 0:
        return [
            "--rrf-min-score is zero: it sets no floor, leaves the per-chunk path "
            "unchanged, and query-generation searches fall back to the default floor"
        ]
    return []


def _no_cross_encoder_floor_warnings(retrieval: RetrievalConfig) -> list[str]:
    """Warn when an explicit --rrf-min-score drops the per-chunk floor.

    Without the cross-encoder the floor branch is the only per-chunk
    filter, so a zero value keeps every fused candidate (the reranker
    never runs without a cross-encoder). Negative values are the CLI's
    to reject; reached programmatically, they get no warning.
    """
    if retrieval.rrf_min_score != 0:
        return []
    if retrieval.query_gen:
        return [
            "--rrf-min-score is zero: the per-chunk floor is dropped (every fused "
            "candidate passes without a cross-encoder), and query-generation "
            "searches fall back to the default floor"
        ]
    return ["--rrf-min-score is zero: the per-chunk floor is dropped and every fused candidate passes"]
