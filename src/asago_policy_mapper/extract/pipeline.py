from __future__ import annotations

import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from asago_policy_mapper.extract.attribute import (
    _counts_causal,
    _counts_expansion,
    _counts_grounding,
    _counts_variant,
    ground_and_extract_evidence,
    ground_risk_group,
    ground_variants,
    synthesize_causal_chain_with_status,
)
from asago_policy_mapper.extract.calllog import merge_call_records
from asago_policy_mapper.extract.index import RiskIndex
from asago_policy_mapper.extract.merge import _tiebreak_key, merge_matches
from asago_policy_mapper.extract.models import (
    CONFIDENCE_FOLLOWS_MEASUREMENT,
    RERANKER_CONFIDENCE_SOURCES,
    CausalProvenanceValue,
    CausalSynthesisResult,
    ChunkSummary,
    DecisionCallRecord,
    DecisionItem,
    DecisionLog,
    EvidenceSpan,
    ExtractionResult,
    FilteredCandidate,
    LLMCallRecord,
    RemovedVariantGroup,
    RetrievalConfig,
    RetrievalScores,
    RetrievalStats,
    RiskMatch,
    ScoredCandidate,
)
from asago_policy_mapper.extract.parse import chunk_documents, parse_document
from asago_policy_mapper.extract.plan import RunPlan, resolve_plan, validate_rrf_min_score
from asago_policy_mapper.extract.querygen import generate_queries, group_chunks
from asago_policy_mapper.extract.retrieve import (
    ChunkResult,
    _counts_judge,
    build_chunk_context_indices,
    build_chunk_contexts,
    judge_borderline,
    retrieve_chunk,
)
from asago_policy_mapper.extract.screening import (
    GROUNDING_LIKELIHOOD_QUESTION_V1,
    AnswerBatch,
    Decision,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
    answer_payloads,
)
from asago_policy_mapper.llm import LLMConfig, StructuredOutputMode

logger = logging.getLogger(__name__)

# The causal-chain fields RiskMatch carries, in chain order.
_CAUSAL_FIELDS = ("threat", "threat_source", "vulnerability", "consequence", "impact")


class PipelineInvariantError(RuntimeError):
    """A pipeline stage received state that earlier stages guarantee impossible."""


_QUERY_GEN_MAX_CANDIDATES = 50

_PREFILTER_FIELDS_SENT = ["chunk_context", "risk_name", "risk_description"]


@dataclass(frozen=True)
class GroundingPrefilter:
    """The opt-in pre-grounding filter: an assessor plus a keep/drop rule.

    ``assessor`` is any ``Assessor`` that emits ``ProbabilityAnswer``
    answers (today the HTTP probability assessor or the replay assessor);
    ``rule`` reduces those answers to keep/drop decisions (today
    ``DropThreshold``). Provider settings never travel in here: the
    assessor carries its own endpoint configuration, and the run's
    ``RetrievalConfig`` stays free of them. The fields stay ``Any`` so
    concrete assessors (whose ``answer_types`` frozensets are invariant under
    mypy) satisfy the dataclass without protocol casts.
    """

    assessor: Any
    rule: Any


def _cap_accepted_candidates(chunk_results: list[ChunkResult], limit: int) -> None:
    """Keep the highest-RRF candidates when query-gen unions exceed the per-chunk cap."""
    for cr in chunk_results:
        if cr is None or len(cr.accepted) <= limit:
            continue
        cr.accepted = sorted(cr.accepted, key=lambda c: c.rrf_score, reverse=True)[:limit]
        cr.stats["auto_accepted"] = len(cr.accepted)
        cr.stats["candidates_retrieved"] = max(cr.stats.get("candidates_retrieved", 0), len(cr.accepted))


@contextmanager
def timed(timing, key):
    t0 = time.time()
    try:
        yield
    finally:
        timing[key] = (time.time() - t0) * 1000


def _assert_no_synthetic_parents(merged, variant_map, *, check: bool) -> None:
    """Raise when a synthetic parent risk id would reach the output.

    The check uses the variant map's keys — the collapsed parent ids — rather
    than guessing from the ``---`` delimiter, so it only fires on ids the run
    actually collapsed. ``check`` is the variant-grounding stage's own gate
    (variants collapsed, the plan grounds, a client is present): only a run
    whose stage could replace parents with grounded variants is held to the
    no-parents-in-output promise. Ungrounded runs emit the expanded variants
    directly, and a run without a client leaves parent matches in place
    legitimately, so neither is checked.
    """
    if not variant_map or not check:
        return
    parent_ids = set(variant_map)
    leaked = [m.risk_id for m in merged if m.risk_id in parent_ids]
    if leaked:
        raise PipelineInvariantError(f"Synthetic parent risk ids reached the output: {sorted(set(leaked))}")


_AGENTIC_PHRASES = {
    "agentic",
    "ai agent",
    "ai agents",
    "autonomous agent",
    "autonomous agents",
    "agent framework",
    "tool-calling",
    "tool calling",
    "function calling",
    "agentic ai",
    "agentic system",
    "agentic systems",
    "multi-agent",
    "multiagent",
}


def _document_discusses_agents(texts: list[str]) -> bool:
    combined = " ".join(texts).lower()
    return any(phrase in combined for phrase in _AGENTIC_PHRASES)


def _stage_context(chunk_contexts: list[str], chunk_index: int, chunks) -> str:
    """The context a stage scores or grounds one chunk against.

    The selection rule grounding and the prefilter share: pick the
    context by the chunk's index, not the chunk's position in a stage's
    list, and fall back to the chunk's own text when the context is
    missing or empty. Selecting by position holds only while every
    stage sees a dense list of every chunk; selecting by index is the
    invariant the pipeline's data model already promises
    (``chunk_index`` keys the chunks list).
    """
    context = chunk_contexts[chunk_index] if chunk_contexts and 0 <= chunk_index < len(chunk_contexts) else ""
    return _stage_context_value(context, chunks[chunk_index] if 0 <= chunk_index < len(chunks) else None)


def _stage_context_value(context: str, chunk) -> str:
    """The grounding text for one chunk: the stage context, else the chunk's text."""
    return context if context else (chunk.text if chunk is not None else "")


def _judge_one(
    i,
    cr,
    client,
    model,
    records,
    judge_prompt="judge_risk",
    max_batch_size=0,
    context_text="",
):
    judged = judge_borderline(
        cr.borderline,
        context_text,
        client,
        model,
        records=records,
        chunk_index=i,
        prompt_name=judge_prompt,
        max_batch_size=max_batch_size,
    )
    return i, judged


def _context_source_windows(
    context_text: str,
    context_source_windows: list[tuple[int, str]] | None,
    chunk_index: int,
    chunk,
) -> list[tuple[int, str]] | None:
    """Pick the source windows a grounding call verifies its quotes against.

    Explicit windows win. Without them, a context built from neighbours
    carries no window list (None: output predates verification), while a
    bare-chunk context — no neighbours joined — verifies against that chunk
    alone.
    """
    if context_source_windows is not None:
        return context_source_windows
    if not context_text:
        return [(chunk_index, chunk.text)]
    return None


def _ground_one(
    cr,
    chunks,
    client,
    model,
    records,
    passes=1,
    max_batch_size=0,
    context_text="",
    context_source_windows=None,
):
    chunk = chunks[cr.chunk_index]
    text = _stage_context_value(context_text, chunk)
    source_windows = _context_source_windows(context_text, context_source_windows, cr.chunk_index, chunk)
    merged: dict[str, tuple[list, str]] = {}
    for pass_num in range(1, passes + 1):
        grounded = ground_and_extract_evidence(
            chunk_text=text,
            candidates=cr.accepted,
            client=client,
            model=model,
            document=chunk.source,
            chunk_index=cr.chunk_index,
            page=chunk.page,
            section=chunk.section,
            records=records,
            max_batch_size=max_batch_size,
            source_windows=source_windows,
        )
        logger.info(
            "Grounding chunk %d/%d pass %d/%d: %d risks grounded",
            cr.chunk_index + 1,
            len(chunks),
            pass_num,
            passes,
            len(grounded),
        )
        for rid, val in grounded.items():
            if rid not in merged:
                merged[rid] = val
    return cr, merged


def determine_accepted_by(candidate):
    """Label how a candidate was accepted.

    The label is stamped where the acceptance happens: retrieval and query
    generation stamp selection labels, and the judge stamps
    ``llm_judge``/``auto_promoted`` when it promotes a borderline candidate.
    A candidate with no stamp cannot reach here in production, so the
    missing stamp is an invariant failure, not a fallback case.
    """
    if candidate.accepted_by is not None:
        return candidate.accepted_by
    raise PipelineInvariantError(
        f"candidate {candidate.risk_id} reached acceptance without a selection label; "
        "retrieval, query generation and the judge must stamp every accepted candidate"
    )


def _confidence_from_candidate(candidate, *, use_cross_encoder):
    """Derive (confidence, source) from the candidate's recorded stamps.

    A reranker-measured score keeps its value and names its reranker. A
    candidate no reranker scored (rerank_source None in cross-encoder
    mode) is "not calculated": null confidence, never 0.0. Without the
    cross-encoder the RRF ranking score is the confidence, labelled rrf.
    """
    if not use_cross_encoder:
        return candidate.rrf_score, "rrf"
    if candidate.rerank_source is None:
        return None, "none"
    return candidate.cross_encoder_score, candidate.rerank_source


def build_risk_match(
    candidate,
    *,
    taxonomy,
    accepted_by,
    grounding_confidence,
    evidence,
    use_cross_encoder=True,
    scores_override=None,
):
    confidence, confidence_source = _confidence_from_candidate(candidate, use_cross_encoder=use_cross_encoder)
    scores = scores_override or RetrievalScores(
        bm25_rank=candidate.bm25_rank,
        embedding_distance=candidate.embedding_distance,
        cross_encoder_score=candidate.cross_encoder_score if candidate.rerank_source is not None else None,
        cross_encoder_source=candidate.rerank_source,
        rrf_score=candidate.rrf_score,
    )
    return RiskMatch(
        risk_id=candidate.risk_id,
        risk_name=candidate.risk_name,
        risk_description=candidate.risk_description,
        taxonomy=taxonomy,
        confidence=confidence,
        confidence_source=confidence_source,
        grounding_confidence=grounding_confidence,
        accepted_by=accepted_by,
        evidence=list(evidence),
        scores=scores,
    )


def _collect_ungrounded(chunk_results, index, retrieval):
    matches = []
    for cr in chunk_results:
        for candidate in cr.accepted:
            accepted_by = determine_accepted_by(candidate)
            matches.append(
                build_risk_match(
                    candidate,
                    taxonomy=index.get_taxonomy(candidate.risk_id),
                    accepted_by=accepted_by,
                    grounding_confidence="ungrounded",
                    evidence=[],
                    use_cross_encoder=retrieval.use_cross_encoder,
                )
            )
    return matches


def _run_grounding(
    chunk_results,
    chunks,
    client,
    config,
    retrieval,
    index,
    call_collector,
    grounding_passes=1,
    grounding_batch_size=0,
    chunk_contexts=None,
    chunk_context_indices=None,
):
    all_matches = []
    all_filtered = []
    total_candidates = 0
    total_grounded = 0
    max_workers = config.max_concurrent

    def _source_windows(chunk_index: int):
        """Build the (index, text) windows a chunk's context was built from."""
        if chunk_context_indices is None:
            return None
        return [(i, chunks[i].text) for i in chunk_context_indices[chunk_index] if i < len(chunks)]

    ground_tasks = [cr for cr in chunk_results if cr.accepted]
    for cr in ground_tasks:
        total_candidates += len(cr.accepted)

    logger.info(
        "Grounding %d chunks with %d candidates (%d passes, batch_size=%d)",
        len(ground_tasks),
        total_candidates,
        grounding_passes,
        grounding_batch_size,
    )

    if ground_tasks:
        # One records list per task: the worker numbers within it alone, and
        # the fan-in merge below renumbers across tasks in task order. The
        # submit and merge loops walk the same list-position pairing, so the
        # order cannot drift between them.
        task_records: list[list[LLMCallRecord]] = [[] for _ in ground_tasks]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _ground_one,
                    cr,
                    chunks,
                    client,
                    config.model,
                    task_records[pos],
                    grounding_passes,
                    grounding_batch_size,
                    _stage_context(chunk_contexts, cr.chunk_index, chunks),
                    _source_windows(cr.chunk_index),
                ): cr
                for pos, cr in enumerate(ground_tasks)
            }
            for future in as_completed(futures):
                cr, grounded = future.result()
                total_grounded += len(grounded)
                grounded_ids = set(grounded.keys())
                for candidate in cr.accepted:
                    accepted_by = determine_accepted_by(candidate)
                    rid = candidate.risk_id
                    if rid in grounded_ids:
                        evidence, confidence = grounded[rid]
                        all_matches.append(
                            build_risk_match(
                                candidate,
                                taxonomy=index.get_taxonomy(rid),
                                accepted_by=accepted_by,
                                grounding_confidence=confidence,
                                evidence=evidence,
                                use_cross_encoder=retrieval.use_cross_encoder,
                            )
                        )
                    else:
                        all_filtered.append(
                            FilteredCandidate(
                                risk_id=rid,
                                risk_name=candidate.risk_name,
                                taxonomy=index.get_taxonomy(rid),
                                cross_encoder_score=candidate.cross_encoder_score
                                if candidate.rerank_source is not None
                                else None,
                                rrf_score=candidate.rrf_score,
                                bm25_rank=candidate.bm25_rank,
                                accepted_by=accepted_by,
                                chunk_index=cr.chunk_index,
                            )
                        )
        if call_collector is not None:
            for records in task_records:
                merge_call_records(call_collector, records, counts=_counts_grounding)

    grounding_filtered = total_candidates - total_grounded
    return all_matches, all_filtered, grounding_filtered


def _variant_evidence_key(e: EvidenceSpan) -> tuple:
    """Ordering key that picks the evidence span variants ground against.

    Cross-encoder score decides first (descending); ties break on the
    lowest chunk index, then text. The serialization tail — the same one
    the merge ordering keys use — makes the order total, so ties past
    those components never fall back to input order even if a future
    constructor mixes spans from different documents or sections into
    one parent's evidence.
    """
    return (-e.cross_encoder_score, e.chunk_index, e.text, _tiebreak_key(e))


def _variant_context_span(evidence: list[EvidenceSpan]) -> EvidenceSpan | None:
    """Pick the best evidence span that names a chunk to ground against.

    Variant grounding needs a chunk; spans the group stage could not
    attribute (chunk_index None) cannot provide one, so they are excluded
    before the tie-break key (cross-encoder score, then chunk index, then
    text) runs.
    """
    attributed = [e for e in evidence if e.chunk_index is not None]
    if not attributed:
        return None
    return min(attributed, key=_variant_evidence_key)


def _ground_variants_one(
    parent_match,
    chunks,
    variant_map,
    client,
    model,
    records,
    chunk_contexts=None,
    chunk_context_indices=None,
) -> tuple[RiskMatch, dict, bool]:
    """Run variant-selective grounding for a single parent-level match.

    Returns the parent match, the grounded variants keyed by variant id,
    and whether variants were sent to ground_variants at all: the caller
    folds that flag into the group's outcome instead of re-deriving the
    proceed decision from the worker's input.
    """
    parent_id = parent_match.risk_id
    variants = variant_map.get(parent_id, [])
    best_ev = _variant_context_span(parent_match.evidence) if variants else None
    if best_ev is None:
        # no variants, or no evidence span the group stage could attribute
        # to a chunk — either way nothing was sent to ground_variants
        return parent_match, {}, False

    # the span comes from _variant_context_span, whose filter keeps only
    # spans with a chunk_index; cast once for mypy instead of re-asserting
    ci = cast(int, best_ev.chunk_index)
    text = chunk_contexts[ci] if chunk_contexts and ci < len(chunk_contexts) else chunks[ci].text
    chunk = chunks[ci]

    source_windows = None
    if chunk_context_indices is not None and ci < len(chunk_context_indices):
        source_windows = [(i, chunks[i].text) for i in chunk_context_indices[ci] if i < len(chunks)]

    grounded = ground_variants(
        chunk_text=text,
        parent_id=parent_id,
        parent_name=parent_match.risk_name,
        parent_description=parent_match.risk_description,
        variants=variants,
        client=client,
        model=model,
        document=best_ev.document,
        chunk_index=ci,
        page=chunk.page if hasattr(chunk, "page") else None,
        section=chunk.section if hasattr(chunk, "section") else None,
        records=records,
        source_windows=source_windows,
    )
    return parent_match, grounded, True


def _variant_stub(variant_id, variant_info, parent_match):
    """Build the candidate a grounded variant stands in for at match time.

    The parent's scores carry over: a reranker measured them for the same
    matching work, so they still hold. A measured reranker score rides with
    its source; an unmeasured parent leaves rerank_source unset, so its null
    confidence stays null instead of posing as a measured 0.0.
    """
    stub = ScoredCandidate(
        risk_id=variant_id,
        risk_name=variant_info["name"],
        risk_description=variant_info["description"],
        bm25_rank=parent_match.scores.bm25_rank,
        embedding_distance=parent_match.scores.embedding_distance,
        rrf_score=parent_match.scores.rrf_score,
    )
    parent_source = parent_match.confidence_source
    parent_score = parent_match.scores.cross_encoder_score
    if parent_score is None:
        return stub
    return stub.model_copy(
        update={
            "cross_encoder_score": parent_score,
            "rerank_source": parent_source if parent_source in RERANKER_CONFIDENCE_SOURCES else None,
        }
    )


def _run_variant_grounding(
    grounded_matches,
    chunks,
    variant_map,
    client,
    config,
    call_collector,
    chunk_contexts=None,
    chunk_context_indices=None,
    max_workers=4,
    *,
    stage,
) -> tuple[list[RiskMatch], list[RemovedVariantGroup]]:
    """Replace parent-level matches with their grounded variants.

    ``stage`` labels the records this call emits (grounding or expansion);
    it is required so a new call site cannot silently mislabel them.

    Returns (matches, removed): the surviving matches (non-parent matches
    plus one match per grounded variant) and one RemovedVariantGroup record
    per distinct parent group that no chunk grounded a variant for.
    """
    parent_matches = []
    non_parent_matches = []
    for m in grounded_matches:
        if m.risk_id in variant_map:
            parent_matches.append(m)
        else:
            non_parent_matches.append(m)

    if not parent_matches:
        return grounded_matches, []

    logger.info(
        "Variant grounding: %d parent matches (%d non-parent matches kept as-is)",
        len(parent_matches),
        len(non_parent_matches),
    )

    variant_matches: list[RiskMatch] = []
    # One records list per parent match; the submit and merge loops walk the
    # same list-position pairing, so the merge order is the parent order.
    task_records: list[list[LLMCallRecord]] = [[] for _ in parent_matches]
    outcomes: dict[str, _ParentOutcome] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _ground_variants_one,
                pm,
                chunks,
                variant_map,
                client,
                config.model,
                task_records[pos],
                chunk_contexts,
                chunk_context_indices,
            ): pm
            for pos, pm in enumerate(parent_matches)
        }
        for future in as_completed(futures):
            parent_match, grounded, attempted = future.result()
            outcome = outcomes.setdefault(parent_match.risk_id, _ParentOutcome())
            outcome.record(grounded, attempted)
            for vid, (evidence, confidence) in grounded.items():
                vinfo = _variant_info(variant_map[parent_match.risk_id], vid)
                if not vinfo:
                    continue
                stub = _variant_stub(vid, vinfo, parent_match)
                variant_matches.append(
                    build_risk_match(
                        stub,
                        taxonomy=vinfo.get("taxonomy", ""),
                        accepted_by=parent_match.accepted_by,
                        grounding_confidence=confidence,
                        evidence=evidence,
                        scores_override=parent_match.scores.model_copy(deep=True),
                        # the variant's confidence follows the same path as
                        # its parent's: a measured score, or null when the
                        # parent was never measured; only an RRF-backed
                        # parent inherits the ranking score as confidence
                        use_cross_encoder=parent_match.confidence_source in CONFIDENCE_FOLLOWS_MEASUREMENT,
                    ).model_copy(update={"scores_inherited_from": parent_match.risk_id})
                )
    if call_collector is not None:
        for records in task_records:
            merge_call_records(call_collector, records, counts=_counts_variant)

    removed = _removed_group_records(outcomes, variant_map, stage)
    return non_parent_matches + variant_matches, removed


class _ParentOutcome:
    """Aggregate one parent group's variant grounding across its chunk matches.

    Tracks whether any chunk grounded a variant and whether any chunk sent
    variants to ground_variants at all, so the removal decision and its
    reason come from the union over chunks, not from whichever match
    finished first.
    """

    def __init__(self):
        self.grounded_variant_ids: list[str] = []
        self.attempted = False

    def record(self, grounded, attempted) -> None:
        """Fold one chunk's result into the parent's outcome."""
        if grounded:
            self.grounded_variant_ids.extend(grounded)
        if attempted:
            self.attempted = True

    @property
    def removed_reason(self) -> str | None:
        """The removal reason, or None when a variant survived somewhere."""
        if self.grounded_variant_ids:
            return None
        return "no_variant_grounded" if self.attempted else "no_parent_evidence"


def _removed_group_records(outcomes, variant_map, stage) -> list[RemovedVariantGroup]:
    """Build one removal record per parent group that grounded no variant.

    A group records ``no_variant_grounded`` with the ids that were sent to
    ground_variants when at least one chunk had evidence to ground against;
    otherwise it records ``no_parent_evidence`` with no attempted ids. The
    records sort by parent id so the serialized list never depends on
    future completion order.
    """
    removed = []
    for parent_id in sorted(outcomes):
        outcome = outcomes[parent_id]
        reason = outcome.removed_reason
        if reason is None:
            continue
        variants = variant_map[parent_id]
        attempted = [v["risk_id"] for v in variants] if reason == "no_variant_grounded" else []
        removed.append(
            RemovedVariantGroup(
                parent_id=parent_id,
                stage=stage,
                reason=reason,
                attempted_variant_ids=attempted,
            )
        )
    return removed


def _reconcile_removed_groups(removed, merged, variant_map):
    """Drop removal records whose variant ids are in the final output.

    Expansion can re-enter a variant the grounding stage removed: a found
    variant of another parent sharing the group proposes it as a sibling,
    and the group-grounding LLM — a different prompt from the variant
    grounder — accepts it. A record that says ``no_variant_grounded`` while
    the same document's risks carry one of the group's variant ids would
    contradict the output, so at result assembly each record is checked
    against the merged risks and dropped when they intersect. The stage
    outcome stays visible through the expansion counters (expanded risks
    and re-entered parents) when a dropped group came back.
    """
    if not removed:
        return removed
    output_ids = {m.risk_id for m in merged}
    kept = [r for r in removed if not ({v["risk_id"] for v in variant_map.get(r.parent_id, [])} & output_ids)]
    dropped = len(removed) - len(kept)
    if dropped:
        logger.info(
            "Removed-variant records reconciled against the output: %d dropped (variant re-entered via expansion)",
            dropped,
        )
    return kept


def _variant_info(variants, variant_id):
    """Look up one variant's metadata by risk id; None when absent."""
    for v in variants:
        if v["risk_id"] == variant_id:
            return v
    return None


def _run_judge(
    chunk_results,
    chunks,
    client,
    config,
    retrieval,
    call_collector,
    plan,
    chunk_contexts=None,
):
    """Settle borderline candidates per the plan's judge mode.

    ``auto_promote`` moves every borderline candidate into accepted,
    ``llm`` asks the judge over judge-only contexts, and ``none`` leaves
    them (no cross-encoder means no borderline candidates exist anyway).
    """
    max_workers = config.max_concurrent
    judge_tasks = [(i, cr) for i, cr in enumerate(chunk_results) if cr.borderline]
    logger.info("Judge: %d chunks with borderline candidates (judge=%s)", len(judge_tasks), plan.judge)
    if plan.judge == "auto_promote":
        for cr in chunk_results:
            if cr.borderline:
                # Promotion is an acceptance: stamp it the moment it
                # happens, the way retrieval stamps selection labels.
                cr.borderline_judged = [c.model_copy(update={"accepted_by": "auto_promoted"}) for c in cr.borderline]
                cr.accepted.extend(cr.borderline_judged)
    elif plan.judge == "llm":
        if judge_tasks:
            judge_contexts = _judge_contexts(chunks, retrieval, chunk_contexts, judge_tasks)
            # One records list per chunk; the submit and merge loops walk the
            # same list-position pairing, so the merge order is the chunk order.
            task_records: list[list[LLMCallRecord]] = [[] for _ in judge_tasks]
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _judge_one,
                        i,
                        cr,
                        client,
                        config.model,
                        task_records[pos],
                        retrieval.judge_prompt,
                        retrieval.grounding_batch_size,
                        judge_contexts[i] if judge_contexts else "",
                    ): i
                    for pos, (i, cr) in enumerate(judge_tasks)
                }
                for future in as_completed(futures):
                    idx, judged = future.result()
                    # Promotion is an acceptance: the label stamps when it
                    # happens, so determine_accepted_by needs no ambient
                    # state to rebuild it.
                    stamped = [c.model_copy(update={"accepted_by": "llm_judge"}) for c in judged]
                    chunk_results[idx].borderline_judged = stamped
                    chunk_results[idx].accepted.extend(stamped)
            if call_collector is not None:
                for records in task_records:
                    merge_call_records(call_collector, records, counts=_counts_judge)


def _judge_contexts(chunks, retrieval, chunk_contexts, judge_tasks):
    """Pick the context list the judge reads.

    A positive ``judge_context_tokens`` builds judge-only contexts with
    that budget, and only for the chunks that have borderline candidates
    to judge; the grounding and prefilter contexts are untouched. Chunks
    with nothing to judge get an empty string, which no judge task reads.
    """
    if retrieval.judge_context_tokens > 0:
        indexes = {i for i, _cr in judge_tasks}
        return build_chunk_contexts(chunks, max_context_tokens=retrieval.judge_context_tokens, only=indexes)
    return chunk_contexts


def _build_chunk_risk_map(chunk_results, all_matches, plan):
    """Map chunk index to the risk ids found there, per the plan's grounding mode.

    Grounded runs attribute risks through their evidence chunks; ungrounded
    runs have no evidence, so accepted candidates attribute to the chunk
    they were retrieved from.
    """
    chunk_risk_ids: dict[int, list[str]] = {}
    if not plan.grounding:
        for cr in chunk_results:
            for candidate in cr.accepted:
                chunk_risk_ids.setdefault(cr.chunk_index, []).append(candidate.risk_id)
    else:
        for m in all_matches:
            for ev in m.evidence:
                if ev.chunk_index is None:
                    continue
                chunk_risk_ids.setdefault(ev.chunk_index, []).append(m.risk_id)
    return chunk_risk_ids


def _build_found_risk_chunks(merged, chunk_results) -> dict[str, set[int]]:
    """Map each found risk id to the chunks that show it.

    Sources are the accepted retrieval candidates and the merged grounding
    evidence; both tell the expansion grouper where a found risk lives.
    """
    found_risk_chunks: dict[str, set[int]] = {}
    for cr in chunk_results:
        for candidate in cr.accepted:
            cid = candidate.risk_id.split(" ")[0].strip()
            found_risk_chunks.setdefault(cid, set()).add(cr.chunk_index)
    for m in merged:
        for ev in m.evidence:
            if ev.chunk_index is None:
                continue
            found_risk_chunks.setdefault(m.risk_id, set()).add(ev.chunk_index)
    return found_risk_chunks


def _ground_expansion_group(group, chunks, documents, client, config, records):
    """Ground one expansion group once against the group's chunk set."""
    return ground_risk_group(
        chunks=chunks,
        chunk_indices=group.chunk_indices,
        risks=list(group.risk_lookup.values()),
        client=client,
        model=config.model,
        document=str(documents[0]) if documents else "",
        records=records,
    )


def _ground_expansion_group_multi(group, chunks, documents, client, config, records, passes):
    """Ground one group across several passes; the first pass wins per risk id.

    A risk grounded by an earlier pass keeps its evidence and confidence even
    when a later pass grounds it differently.
    """
    merged_results: dict[str, tuple[list[EvidenceSpan], str]] = {}
    for _ in range(passes):
        grounded = _ground_expansion_group(group, chunks, documents, client, config, records)
        for rid, (evidence, confidence) in grounded.items():
            if rid not in merged_results:
                merged_results[rid] = (evidence, confidence)
    return merged_results


def _build_expansion_matches(grounded, index, risk_lookup) -> list[RiskMatch]:
    """Convert one group's grounded risks into expansion matches.

    Expansion matches carry no retrieval score — nothing was measured, so
    confidence is null with source none, never a made-up 0.0.
    """
    matches = []
    for rid, (evidence, confidence) in grounded.items():
        info = risk_lookup.get(rid, {})
        stub = ScoredCandidate(
            risk_id=rid,
            risk_name=info.get("name", ""),
            risk_description=info.get("description", ""),
        )
        matches.append(
            build_risk_match(
                stub,
                taxonomy=index.get_taxonomy(rid),
                accepted_by="expansion",
                grounding_confidence=confidence,
                evidence=evidence,
            )
        )
    return matches


def _ground_expansion_groups(
    groups, chunks, documents, index, client, config, max_workers, call_collector, expansion_passes, risk_lookup
) -> tuple[list[RiskMatch], int]:
    """Ground all expansion groups concurrently.

    With one pass each group grounds directly; with several, the per-group
    first-wins union runs inside the worker before any match is built.
    Returns the expansion matches and the grounded-risk count.
    """

    def _submit(pool, group, records):
        if expansion_passes > 1:
            return pool.submit(
                _ground_expansion_group_multi,
                group,
                chunks,
                documents,
                client,
                config,
                records,
                expansion_passes,
            )
        return pool.submit(_ground_expansion_group, group, chunks, documents, client, config, records)

    expansion_matches: list[RiskMatch] = []
    expanded_grounded = 0
    # One records list per group; the submit and merge loops walk the same
    # list-position pairing, so the merge order is the group order.
    group_records: list[list[LLMCallRecord]] = [[] for _ in groups]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {_submit(pool, g, group_records[pos]): g for pos, g in enumerate(groups)}
        for future in as_completed(futures):
            grounded = future.result()
            expanded_grounded += len(grounded)
            expansion_matches.extend(_build_expansion_matches(grounded, index, risk_lookup))
    if call_collector is not None:
        for records in group_records:
            merge_call_records(call_collector, records, counts=_counts_expansion)
    return expansion_matches, expanded_grounded


def _expand_siblings(risks, merged, chunk_results, chunks, stats, dropped_ids: set[str] | None = None) -> tuple:
    """Prepare sibling-expansion candidates and their grounding groups.

    Returns (expanded, groups, risk_lookup): the sibling candidates not yet
    found, their grounding groups (empty when there are no candidates), and
    the catalog lookup the matches are built from. Records the candidate and
    group counts in stats. When ``dropped_ids`` carries the prefilter's
    dropped catalog ids, ``stats["prefilter_reentered"]`` counts how many
    of them expansion proposes again.
    """
    from asago_policy_mapper.extract.expand import (
        build_expansion_graph,
        expand_with_siblings,
        group_for_grounding,
    )

    risk_lookup = {r.id: {"name": r.name or "", "description": r.description or ""} for r in risks}
    risk_to_parent = {}
    for r in risks:
        parent = getattr(r, "isPartOf", "") or ""
        if parent:
            risk_to_parent[r.id] = parent

    expansion_graph = build_expansion_graph(risks)
    merged_ids = {m.risk_id for m in merged}
    expanded = expand_with_siblings(merged_ids, expansion_graph, risk_lookup)
    stats["expanded_candidates"] = len(expanded)
    if dropped_ids:
        stats["prefilter_reentered"] = len({er.risk_id for er in expanded} & dropped_ids)
    if not expanded:
        return expanded, [], risk_lookup

    groups = group_for_grounding(
        expanded,
        _build_found_risk_chunks(merged, chunk_results),
        risk_to_parent,
        len(chunks),
    )
    stats["expansion_groups"] = len(groups)
    return expanded, groups, risk_lookup


def _prefilter_dropped_catalog(drops, variant_map) -> tuple[set[str], dict[str, list[str]]]:
    """The dropped ids at catalog granularity, plus each dropped parent's members.

    A dropped synthetic parent counts as its member variant ids, because
    expansion proposes and grounds at that granularity; the parents map to
    their member ids so the run can report how many whole groups came back.
    """
    dropped_ids: set[str] = set()
    parent_members: dict[str, list[str]] = {}
    for drop in drops:
        members = (variant_map or {}).get(drop.risk_id)
        if members:
            parent_members[drop.risk_id] = [m["risk_id"] for m in members]
            dropped_ids.update(parent_members[drop.risk_id])
        else:
            dropped_ids.add(drop.risk_id)
    return dropped_ids, parent_members


def _record_reentry_parents(stats, parent_members: dict[str, list[str]], expanded) -> None:
    """Count dropped parents whose member variants expansion proposed again."""
    if not parent_members:
        return
    proposed = {er.risk_id for er in expanded}
    stats["prefilter_reentered_parents"] = sum(1 for members in parent_members.values() if proposed & set(members))


def _run_expansion(
    risks,
    merged,
    chunk_results,
    chunks,
    documents,
    index,
    client,
    config,
    max_workers,
    call_collector,
    expansion_passes: int = 1,
    variant_map: dict | None = None,
    chunk_contexts=None,
    prefilter_drops: list[FilteredCandidate] | None = None,
    chunk_context_indices=None,
) -> tuple[list[RiskMatch], dict, list[RemovedVariantGroup]]:
    """Expand to sibling risks, ground them, and record removed variant groups.

    Returns (merged, stats, removed): the possibly expanded merged list, the
    expansion stats, and the removal records for variant groups the expansion
    path's variant grounding dropped.
    """
    stats = {"expanded_candidates": 0, "expanded_grounded": 0, "expansion_groups": 0}
    dropped_ids, parent_members = (
        _prefilter_dropped_catalog(prefilter_drops, variant_map)
        if prefilter_drops
        else (
            set(),
            {},
        )
    )
    expanded, groups, risk_lookup = _expand_siblings(
        risks, merged, chunk_results, chunks, stats, dropped_ids=dropped_ids or None
    )
    # Records 0 for dropped parents whose members expansion never proposed,
    # so the key appears whenever parents were dropped, even when expansion
    # proposes nothing at all.
    _record_reentry_parents(stats, parent_members, expanded)
    if not expanded:
        if dropped_ids:
            stats["prefilter_reentered_grounded"] = 0
        return merged, stats, []

    logger.info(
        "Expansion: %d sibling candidates in %d groups (%d passes)",
        stats["expanded_candidates"],
        stats["expansion_groups"],
        expansion_passes,
    )

    expansion_matches, stats["expanded_grounded"] = _ground_expansion_groups(
        groups, chunks, documents, index, client, config, max_workers, call_collector, expansion_passes, risk_lookup
    )
    if dropped_ids:
        stats["prefilter_reentered_grounded"] = len({m.risk_id for m in expansion_matches} & dropped_ids)

    removed: list[RemovedVariantGroup] = []
    if expansion_matches and variant_map:
        expansion_matches, removed = _run_variant_grounding(
            expansion_matches,
            chunks,
            variant_map,
            client,
            config,
            call_collector,
            chunk_contexts=chunk_contexts,
            chunk_context_indices=chunk_context_indices,
            max_workers=max_workers,
            stage="expansion",
        )

    if expansion_matches:
        merged = merge_matches(merged + expansion_matches)

    return merged, stats, removed


def _run_causal_synthesis(merged, chunks, client, config, max_workers, call_collector):
    chunk_texts = {i: chunks[i].text for i in range(len(chunks))}

    # One records list per risk; the submit and merge loops walk the same
    # list-position pairing, so the merge order is the merged order.
    task_records: list[list[LLMCallRecord]] = [[] for _ in merged]

    def _synthesize_one(risk_match, records):
        return risk_match.risk_id, synthesize_causal_chain_with_status(
            risk_match=risk_match,
            chunk_texts=chunk_texts,
            client=client,
            model=config.model,
            records=records,
        )

    outcomes: dict[str, CausalSynthesisResult] = {}
    logger.info("Causal synthesis: %d risks", len(merged))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_synthesize_one, m, task_records[pos]): m for pos, m in enumerate(merged)}
        for future in as_completed(futures):
            risk_id, outcome = future.result()
            outcomes[risk_id] = outcome

    if call_collector is not None:
        for records in task_records:
            merge_call_records(call_collector, records, counts=_counts_causal)
    for m in merged:
        # Every merged risk contributed a collected future, so its risk_id
        # is always present in outcomes.
        outcome = outcomes[m.risk_id]
        m.causal_status = outcome.status
        chain = outcome.chain
        if chain:
            # Assigned copy-on-write: a model_copy()'d risk shares its
            # provenance dict, so in-place writes would leak into the copy's
            # original.
            marks: dict[str, CausalProvenanceValue] = {}
            for field in _CAUSAL_FIELDS:
                value = getattr(chain, field) or None
                if value is not None:
                    setattr(m, field, value)
                    marks[field] = {"source": "generated"}
            if marks:
                m.causal_provenance = {**m.causal_provenance, **marks}

    return merged


def _filter_agentic_risks(risks, parsed):
    """Drop agentic risks when no document mentions agent terminology."""
    if _document_discusses_agents([p.content for p in parsed]):
        return risks
    original_count = len(risks)
    risks = [r for r in risks if getattr(r, "risk_type", None) != "agentic"]
    filtered = original_count - len(risks)
    if filtered:
        logger.info(
            "Filtered %d agentic risks (no agent terminology in documents)",
            filtered,
        )
    return risks


def _build_risk_index(risks, retrieval) -> RiskIndex:
    """Build the retrieval index over the risk catalog with the configured encoders."""
    return RiskIndex(
        risks,
        bi_encoder_model=retrieval.bi_encoder_model,
        cross_encoder_model=retrieval.effective_cross_encoder_model,
        colbert_model=retrieval.colbert_model,
        query_instruction=retrieval.query_instruction,
        cross_encoder_type=retrieval.cross_encoder_type,
        bi_encoder_api_key=retrieval.bi_encoder_api_key,
        bi_encoder_model_name=retrieval.bi_encoder_model_name,
        cross_encoder_model_name=retrieval.cross_encoder_model_name,
    )


def _run_query_gen(chunks, client, config, call_collector, max_workers, timing) -> tuple[list, set[int]]:
    """Generate search queries per section group for query-gen retrieval.

    Returns the generated queries and the chunk indices that got no query
    and fall back to per-chunk retrieval.
    """
    with timed(timing, "query_gen_ms"):
        groups = group_chunks(chunks)
        # Query-gen records are unnumbered (querygen-<chunk>) and
        # generate_queries appends them on the calling thread after its
        # group-order sort, so the run's collector takes them directly.
        qg_result = generate_queries(
            chunks,
            groups,
            client,
            config.model,
            records=call_collector,
            max_workers=max_workers,
            return_fallbacks=True,
        )
        assert isinstance(qg_result, tuple)
        query_results, fallback_list = qg_result
        fallback_chunk_indices = set(fallback_list)
        logger.info(
            "Query generation: %d queries from %d groups, %d fallback chunks",
            len(query_results),
            len(groups),
            len(fallback_chunk_indices),
        )
    return query_results, fallback_chunk_indices


def _merge_query_candidates(chunk_results, chunks, qr, candidates):
    """Add a query's candidates to every chunk the query covers.

    The first occurrence of a risk id wins; later queries only append risks
    the chunk has not seen yet. Every candidate is stamped query_gen via
    model_copy — the same candidate objects are shared between chunks, so
    nothing is mutated in place.
    """
    stamped = [c.model_copy(update={"accepted_by": "query_gen"}) for c in candidates]
    for ci in qr.chunk_indices:
        chunk = chunks[ci]
        if chunk_results[ci] is not None:
            existing_ids = {c.risk_id for c in chunk_results[ci].accepted}
            for c in stamped:
                if c.risk_id not in existing_ids:
                    chunk_results[ci].accepted.append(c)
                    existing_ids.add(c.risk_id)
        else:
            chunk_results[ci] = ChunkResult(
                chunk_index=ci,
                source=chunk.source,
                page=chunk.page,
                section=chunk.section,
                accepted=list(stamped),
                borderline=[],
                borderline_judged=[],
                stats={
                    "candidates_retrieved": len(stamped),
                    "auto_accepted": len(stamped),
                    "borderline": 0,
                    "discarded": 0,
                    "bm25_rescued": 0,
                },
            )


def _retrieve_chunk_result(chunks, i, index, retrieval, plan) -> ChunkResult:
    """Run per-chunk hybrid retrieval with the pipeline's retrieval settings.

    The RRF floor applies only on the unreranked path, so the plan's
    rerank decision routes it: a reranked (cross-encoder or ColBERT) run
    passes 0.0 and never lets the floor behave as a filter beside the
    reranker.
    """
    floor = retrieval.rrf_min_score if plan.rerank == "none" else 0.0
    return retrieve_chunk(
        chunks,
        i,
        index,
        top_n_accept=retrieval.top_n_accept,
        top_n_judge=retrieval.top_n_judge,
        min_score_floor=retrieval.min_score_floor,
        bm25_rescue_rank=retrieval.bm25_rescue_rank,
        use_cross_encoder=retrieval.use_cross_encoder,
        rrf_min_score=floor,
        threshold_high=retrieval.threshold_high,
        threshold_low=retrieval.threshold_low,
    )


def _run_retrieval(chunks, index, retrieval, query_results, fallback_chunk_indices, plan, timing) -> list[ChunkResult]:
    """Retrieve candidates per chunk: query-gen queries first, then per-chunk fallback.

    The plan's candidate source picks the retrieval mode: with query
    generation, each query's candidates are unioned into every
    chunk the query covers (capped per chunk), and chunks without a query —
    plus fallback chunks — go through per-chunk retrieval. Without it,
    every chunk goes through per-chunk retrieval.
    """
    with timed(timing, "retrieve_ms"):
        chunk_results: list[ChunkResult] = [None] * len(chunks)  # type: ignore[list-item]

        if plan.candidate_source == "query_gen":
            for qr in query_results:
                candidates = index.hybrid_search(
                    qr.query,
                    top_k=50,
                    bm25_rescue_rank=retrieval.bm25_rescue_rank,
                    rrf_min_score=retrieval.rrf_min_score or 0.015,
                    rerank=False,
                )
                _merge_query_candidates(chunk_results, chunks, qr, candidates)

            _cap_accepted_candidates(chunk_results, _QUERY_GEN_MAX_CANDIDATES)

            retrieval_indices = fallback_chunk_indices | {i for i in range(len(chunks)) if chunk_results[i] is None}
            for i in sorted(retrieval_indices):
                chunk_results[i] = _retrieve_chunk_result(chunks, i, index, retrieval, plan)
        else:
            for i in range(len(chunks)):
                chunk_results[i] = _retrieve_chunk_result(chunks, i, index, retrieval, plan)

    return chunk_results


def _build_chunk_summaries(chunk_results, chunks) -> list[ChunkSummary]:
    """Summarize per-chunk retrieval counters for the ExtractionResult."""
    return [
        ChunkSummary(
            index=cr.chunk_index,
            source=cr.source,
            page=cr.page,
            section=cr.section,
            text_preview=chunks[cr.chunk_index].text[:200],
            candidates_retrieved=cr.stats.get("candidates_retrieved", 0),
            auto_accepted=cr.stats.get("auto_accepted", 0),
            borderline=cr.stats.get("borderline", 0),
            discarded=cr.stats.get("discarded", 0),
            bm25_rescued=cr.stats.get("bm25_rescued", 0),
            rerank_unscored=cr.stats.get("rerank_unscored", 0),
        )
        for cr in chunk_results
    ]


def _prefilter_cases(cr: ChunkResult, index) -> list[DecisionCase]:
    """Build the assessor cases for one chunk's accepted candidates.

    A synthetic parent (a collapsed variant group, keyed in the index's
    ``variant_map``) is flagged with ``is_variant_parent=True``; its name
    and description are exactly what the grounder reads, because the
    candidate came from the same index collapse that produced them. The
    parent id itself carries no ``---`` — only its member variants do —
    so the variant map, not the id text, identifies it.
    """
    return [
        DecisionCase(
            risk_id=c.risk_id,
            name=c.risk_name,
            description=c.risk_description,
            is_variant_parent=c.risk_id in index.variant_map,
        )
        for c in cr.accepted
    ]


def _prefilter_request(
    context: str, cases: list[DecisionCase], assessment, assessor, question
) -> tuple[dict | list, bool]:
    """The request the prefilter recorded, and whether it was sent.

    Assessors that batch (the HTTP probability assessor splits at
    ``max_batch``) record every request body they actually sent, with
    the batch boundaries preserved; the returned flag is False. When
    the assessor sends nothing (replay) or fails before sending, the
    record carries the unbatched body the assessor's own request
    builder produces — the shape one batch's body would have, not
    necessarily what would have left the machine, since a batching
    assessor would split it at ``max_batch`` — and the flag is True, so
    the call record itself (not the body) carries the synthesized
    marker. Assessors without a request builder keep the generic
    fallback shape. The key travels in a header, so no body ever
    carries a credential.
    """
    exchange = getattr(assessment, "exchange", None)
    if isinstance(exchange, dict) and isinstance(exchange.get("requests"), list):
        # Deep-copy the request bodies: the exchange the assessor
        # recorded stays the assessor's own, so nothing the record's
        # consumers do to a request body — at any nesting depth — can
        # reach back into the assessor's recorded exchange (pydantic
        # does not copy plain dict fields on construction).
        return copy.deepcopy(exchange["requests"]), False
    builder = getattr(assessor, "_request_body", None)
    if callable(builder):
        body = dict(builder(question, context, cases))
    else:
        body = {
            "context": context,
            "items": [{"id": c.risk_id, "text": f"{c.name}: {c.description}"} for c in cases],
        }
    return body, True


def _prefilter_decision_log(decisions: list[Decision], assessment, rule, assessor_id: str) -> DecisionLog:
    """Assemble the prefilter's DecisionLog for one chunk."""
    items = [
        DecisionItem(
            risk_id=d.risk_id,
            answers=answer_payloads(assessment.answers.get(d.risk_id, ())),
            keep=d.keep,
            reason=d.reason,
        )
        for d in decisions
    ]
    return DecisionLog(
        question_id=GROUNDING_LIKELIHOOD_QUESTION_V1.question_id,
        question_version=GROUNDING_LIKELIHOOD_QUESTION_V1.version,
        prompt_name="prefilter",
        prompt_sha256="",
        assessor_id=assessor_id,
        rule_id=getattr(rule, "rule_id", ""),
        latency_ms=assessment.latency_ms,
        items=items,
        unmatched_ids=list(assessment.unmatched_ids),
    )


def _prefilter_dropped_score(answers) -> float | None:
    """The probability a drop decision rests on, or None when none exists.

    ``DropThreshold`` drops only on probabilities (a ``NoAnswer``
    bypasses), so a dropped candidate's score is its maximum probability;
    a rule that drops despite having no probability gets ``None``.
    """
    probabilities = [a.probability for a in answers if isinstance(a, ProbabilityAnswer)]
    return max(probabilities) if probabilities else None


def _prefilter_drop_record(candidate, decision, answers, taxonomy, chunk_index, accepted_by) -> FilteredCandidate:
    """Build the recorded drop for one filtered-out candidate.

    ``accepted_by`` keeps the retrieval provenance (threshold, rrf,
    llm_judge) the candidate earned before the prefilter dropped it;
    ``dropped_by`` names the prefilter separately. The score follows the
    same rule as the match boundary: a candidate no reranker measured
    records null, never its 0.0 placeholder.
    """
    return FilteredCandidate(
        risk_id=candidate.risk_id,
        risk_name=candidate.risk_name,
        taxonomy=taxonomy,
        cross_encoder_score=candidate.cross_encoder_score if candidate.rerank_source is not None else None,
        rrf_score=candidate.rrf_score,
        bm25_rank=candidate.bm25_rank,
        accepted_by=accepted_by,
        chunk_index=chunk_index,
        dropped_by="prefilter",
        score=_prefilter_dropped_score(answers),
        reason=decision.reason,
    )


def _prefilter_one_chunk(
    cr, context, prefilter, index, retrieval
) -> tuple[ChunkResult, list[FilteredCandidate], DecisionCallRecord]:
    """Assess and filter one chunk's accepted candidates.

    Returns (filtered copy, drop records, decision call record). The copy
    shares the original's borderline lists and stats; only ``accepted``
    shrinks, and only on the copy.
    """
    cases = _prefilter_cases(cr, index)
    try:
        assessment = prefilter.assessor.assess(GROUNDING_LIKELIHOOD_QUESTION_V1, context, cases)
    except Exception as e:
        # Fail-open: the filter must never lose risks to a broken
        # assessor. Every case keeps, with the failure recorded the same
        # way an unanswered case is. The pipeline holds no secret of its
        # own and does not re-redact: exceptions escaping a conforming
        # assessor arrive already boundary-redacted, with the type
        # preserved (the shared contract has assessors redact their own
        # escaping exceptions against the secret they captured), so the
        # detail is the exception text as it reached this handler and the
        # warning carries the exception type and that same detail rather
        # than the raw exception — a credential in an exception message
        # reaches neither the log nor the result. A test double that
        # injects raw key text models a non-conforming assessor; the
        # contract assigns that defense to the boundary, not here.
        detail = str(e) or type(e).__name__
        logger.warning(
            "GroundingPrefilter assessor %s failed on chunk %s; keeping all candidates: %s: %s",
            getattr(prefilter.assessor, "assessor_id", "<unknown>"),
            cr.chunk_index,
            type(e).__name__,
            detail,
        )
        assessment = AnswerBatch(
            answers={c.risk_id: (NoAnswer("error", detail),) for c in cases},
            unmatched_ids=(),
            latency_ms=0.0,
            exchange=None,
        )
    decisions = prefilter.rule.decide(cases, assessment)
    # A rule that omits a case keeps it: the kept set is the cases the
    # rule kept plus the cases it never mentioned, so a non-conforming
    # rule fails open exactly like a non-conforming assessor. Keep still
    # wins on conflicting decisions for one id, as it did before; only
    # an id whose every decision drops is dropped.
    decision_by_id = {d.risk_id: d for d in decisions}
    accepted_ids = {c.risk_id for c in cr.accepted}
    keep_ids = {d.risk_id for d in decisions if d.keep} | (accepted_ids - set(decision_by_id))
    request_body, synthesized = _prefilter_request(
        context, cases, assessment, prefilter.assessor, GROUNDING_LIKELIHOOD_QUESTION_V1
    )
    calls_record = DecisionCallRecord(
        stage="prefilter",
        chunk_index=cr.chunk_index,
        request=request_body,
        synthesized=synthesized,
        decision=_prefilter_decision_log(
            decisions, assessment, prefilter.rule, getattr(prefilter.assessor, "assessor_id", "")
        ),
    )
    drops = [
        _prefilter_drop_record(
            candidate,
            decision_by_id[candidate.risk_id],
            assessment.answers.get(candidate.risk_id, ()),
            index.get_taxonomy(candidate.risk_id),
            cr.chunk_index,
            determine_accepted_by(candidate),
        )
        for candidate in cr.accepted
        if candidate.risk_id not in keep_ids
    ]
    filtered_copy = ChunkResult(
        chunk_index=cr.chunk_index,
        source=cr.source,
        page=cr.page,
        section=cr.section,
        accepted=[c for c in cr.accepted if c.risk_id in keep_ids],
        borderline=cr.borderline,
        borderline_judged=cr.borderline_judged,
        stats=dict(cr.stats),
    )
    return filtered_copy, drops, calls_record


def _run_prefilter_stage(
    chunk_results: list[ChunkResult],
    chunk_contexts: list[str],
    prefilter: GroundingPrefilter,
    index,
    retrieval,
    chunks=(),
) -> tuple[list[ChunkResult], list[FilteredCandidate], list[DecisionCallRecord]]:
    """Filter accepted candidates per chunk before grounding.

    Returns (filtered copies, prefilter drops, decision call records).
    The copies carry every candidate the filter kept; the original
    ``accepted`` lists stay untouched, because ``_build_found_risk_chunks``
    and the expansion-context choice read them. Candidates the filter
    dropped never reach grounding, and a dropped synthetic parent means
    its variants never reach variant grounding either (they enter the
    pipeline only through the parent's grounded match).

    Each chunk's context comes from the same selection rule grounding
    uses (``_stage_context``), so the filter scores exactly the text
    grounding grounds against — including the chunk-text fallback —
    rather than agreeing with grounding only when both stages happen to
    see dense, aligned lists.
    """
    filtered_copies: list[ChunkResult] = []
    drops: list[FilteredCandidate] = []
    calls: list[DecisionCallRecord] = []
    for i, cr in enumerate(chunk_results):
        if cr is None or not cr.accepted:
            filtered_copies.append(cr)
            continue
        context = _stage_context(chunk_contexts, cr.chunk_index, chunks)
        copy, chunk_drops, call = _prefilter_one_chunk(cr, context, prefilter, index, retrieval)
        filtered_copies.append(copy)
        drops.extend(chunk_drops)
        calls.append(call)
    return filtered_copies, drops, calls


def _assessor_endpoint_host(assessor) -> str | None:
    """The hostname of a probability assessor's endpoint, when it has one."""
    endpoint = getattr(assessor, "endpoint", None)
    base_url = getattr(endpoint, "base_url", None)
    if isinstance(base_url, str) and base_url:
        return urlsplit(base_url).hostname
    return None


def _prefilter_metadata(prefilter: GroundingPrefilter) -> dict:
    """The metadata.prefilter block recorded when the filter is on."""
    assessor = prefilter.assessor
    rule = prefilter.rule
    return {
        "assessor_id": getattr(assessor, "assessor_id", ""),
        "rule_id": getattr(rule, "rule_id", ""),
        "threshold": getattr(rule, "threshold", None),
        "endpoint_host": _assessor_endpoint_host(assessor),
        "sends_offsite": bool(getattr(assessor, "sends_offsite", False)),
        "fields_sent": _prefilter_fields_sent(assessor),
    }


def _prefilter_fields_sent(assessor) -> list[str]:
    """The policy-text fields the assessor's request actually carries.

    The default covers the seam's generic request: the chunk context plus
    each case's name and description. A mapping that sends more (Jev also
    sends the fixed question and the question's criterion inside each
    question's instructions) names what it sends, so the metadata block
    reports the real exchange rather than the generic shape.
    """
    fields = getattr(assessor, "fields_sent", None)
    if fields is not None:
        return list(fields)
    return _PREFILTER_FIELDS_SENT.copy()


def _run_matches_stage(
    chunk_results,
    chunks,
    client,
    config,
    retrieval,
    plan,
    index,
    call_collector,
    chunk_contexts,
    chunk_context_indices,
    timing,
) -> tuple[list[RiskMatch], list[FilteredCandidate], int]:
    """Ground accepted candidates, or collect them ungrounded when grounding is disabled."""
    if not plan.grounding or client is None:
        all_matches = _collect_ungrounded(chunk_results, index, retrieval)
        all_filtered: list[FilteredCandidate] = []
        timing["grounding_ms"] = 0.0
        grounding_filtered = 0
    else:
        with timed(timing, "grounding_ms"):
            all_matches, all_filtered, grounding_filtered = _run_grounding(
                chunk_results,
                chunks,
                client,
                config,
                retrieval,
                index,
                call_collector,
                grounding_passes=retrieval.grounding_passes,
                grounding_batch_size=retrieval.grounding_batch_size,
                chunk_contexts=chunk_contexts,
                chunk_context_indices=chunk_context_indices,
            )
    return all_matches, all_filtered, grounding_filtered


def _build_retrieval_stats(
    chunk_results, chunks, timing, grounding_filtered, plan, prefilter_dropped=None
) -> RetrievalStats:
    """Aggregate per-chunk counters into the run's retrieval stats.

    With the judge auto-promoting (no LLM judge), the borderline candidates
    are promoted, not judged, so their count moves from llm_judged to
    auto_promoted. ``prefilter_dropped`` counts the candidates the
    pre-grounding filter removed; ``grounding_filtered`` counts only what
    grounding itself rejected, so the two never overlap. Pass ``None``
    (the default) when the filter did not run: the field then stays out
    of the serialized output.
    """
    auto_promote = plan.judge != "llm"
    judged = sum(len(cr.borderline_judged) for cr in chunk_results)
    return RetrievalStats(
        total_chunks=len(chunks),
        total_candidates_retrieved=sum(cr.stats.get("candidates_retrieved", 0) for cr in chunk_results),
        auto_accepted=sum(cr.stats.get("auto_accepted", 0) for cr in chunk_results),
        llm_judged=0 if auto_promote else judged,
        auto_promoted=judged if auto_promote else 0,
        grounding_filtered=grounding_filtered,
        prefilter_dropped=prefilter_dropped,
        timing_ms=timing,
    )


def _build_result_metadata(
    config,
    retrieval,
    expansion_stats,
    query_results,
    fallback_chunk_indices,
    client=None,
    prefilter=None,
    plan=None,
) -> dict:
    """Assemble the run metadata recorded alongside the extraction result."""
    if plan is None:
        plan = resolve_plan(retrieval, prefilter=prefilter is not None, explicit=frozenset())
    metadata = {
        "model": config.model,
        **({"service_tier": config.service_tier} if config.service_tier is not None else {}),
        **retrieval.to_metadata(),
        "run_plan": _run_plan_metadata(retrieval, plan, prefilter),
        "expansion_stats": expansion_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if prefilter is not None:
        metadata["prefilter"] = _prefilter_metadata(prefilter)
    mode = getattr(client, "mode", None)
    if isinstance(mode, StructuredOutputMode):
        metadata["structured_output_mode"] = mode.value
    if plan.candidate_source == "query_gen":
        metadata["query_gen_queries"] = [
            {"query": qr.query, "chunk_indices": qr.chunk_indices, "section": qr.section} for qr in query_results
        ]
        metadata["query_gen_fallback_chunks"] = sorted(fallback_chunk_indices)
    return metadata


def _run_plan_metadata(retrieval, plan, prefilter) -> dict:
    """Record the requested settings next to the effective plan.

    Planned is not executed: the execution counters live elsewhere in the
    result, and a stage the plan includes may still be skipped at runtime
    (no client, a catalog without variants). ``effective`` is the plan the
    pipeline ran with; when no plan was resolved, one is derived here so
    the metadata always carries it.
    """
    if plan is None:
        plan = resolve_plan(retrieval, prefilter=prefilter is not None, explicit=frozenset())
    return {
        "requested": retrieval.to_metadata(),
        "effective": asdict(plan),
        "warnings": list(plan.warnings),
    }


def _empty_result_metadata(retrieval, prefilter, plan) -> dict:
    """The metadata a short-circuited run records: the run plan, nothing else.

    Empty-document and no-risk runs exit before any stage runs, so the
    execution counters the full metadata carries do not exist yet; the
    plan still describes what the run would have done. The base recorded
    no metadata at all here, so only the run_plan block is new.
    """
    return {
        "run_plan": _run_plan_metadata(retrieval, plan, prefilter),
    }


_EMPTY_EXPANSION_STATS = {"expanded_candidates": 0, "expanded_grounded": 0, "expansion_groups": 0}


def _run_variant_grounding_stage(
    all_matches,
    chunks,
    index,
    client,
    config,
    plan,
    call_collector,
    chunk_contexts,
    chunk_context_indices,
    max_workers,
    timing,
) -> tuple[list[RiskMatch], list[RemovedVariantGroup]]:
    """Variant-ground grounded matches when variants and a client are present.

    Returns the (possibly variant-grounded) matches and the removal records.
    """
    if index.variant_map and plan.grounding and client is not None:
        with timed(timing, "variant_grounding_ms"):
            return _run_variant_grounding(
                all_matches,
                chunks,
                index.variant_map,
                client,
                config,
                call_collector,
                chunk_contexts=chunk_contexts,
                chunk_context_indices=chunk_context_indices,
                max_workers=max_workers,
                stage="grounding",
            )
    return all_matches, []


def _run_expansion_stage(
    merged,
    risks,
    chunk_results,
    chunks,
    documents,
    index,
    client,
    config,
    retrieval,
    plan,
    call_collector,
    chunk_contexts,
    chunk_context_indices,
    max_workers,
    timing,
    prefilter_drops=None,
) -> tuple[list, dict, list[RemovedVariantGroup]]:
    """Expand to sibling risks and ground them; grounded runs only.

    Returns the (possibly expanded) merged list, the expansion stats and the
    removed-variant-group records; the counters stay at zero and the records
    empty when the stage is skipped. ``prefilter_drops`` carries the
    prefilter's drop records, whose ids the expansion counts when it proposes
    them again.
    """
    if plan.expansion and client is not None:
        with timed(timing, "expansion_ms"):
            merged, expansion_stats, removed = _run_expansion(
                risks,
                merged,
                chunk_results,
                chunks,
                documents,
                index,
                client,
                config,
                max_workers,
                call_collector,
                expansion_passes=retrieval.expansion_passes,
                variant_map=index.variant_map if index.variant_map else None,
                chunk_contexts=chunk_contexts,
                prefilter_drops=prefilter_drops,
                chunk_context_indices=chunk_context_indices,
            )
            return merged, expansion_stats, removed
    return merged, _EMPTY_EXPANSION_STATS.copy(), []


def _run_causal_synthesis_stage(merged, chunks, client, config, plan, call_collector, max_workers, timing):
    """Synthesize causal chains for each merged risk when the plan runs the stage.

    The plan's truth table gates causal synthesis on grounding: an ungrounded
    run has no verified evidence to synthesize from, so every match records
    why the stage skipped it instead.
    """
    if plan.causal_synthesis and client is not None:
        with timed(timing, "causal_synthesis_ms"):
            return _run_causal_synthesis(
                merged,
                chunks,
                client,
                config,
                max_workers,
                call_collector,
            )
    # Grounding off wins over the synthesis flag: the plan gates causal
    # synthesis on grounding, so an ungrounded run names the grounding
    # reason even when synthesis was on.
    if not plan.causal_synthesis:
        status = "disabled_no_grounding" if not plan.grounding else "disabled_synthesis_off"
    else:
        status = "disabled_no_client"
    for m in merged:
        m.causal_status = status
    return merged


def _expand_chunk_variants(chunk_results, index):
    """Expand collapsed parent candidates back to their variants in place.

    Ungrounded runs keep the variant granularity at match time, so the
    accepted and borderline lists are re-expanded per chunk.
    """
    for cr in chunk_results:
        cr.accepted = index.expand_variants(cr.accepted)
        cr.borderline = index.expand_variants(cr.borderline)


def _parse_and_chunk(documents, ocr, retrieval, timing) -> tuple | None:
    """Parse documents and chunk them; returns (parsed, chunks, contexts, context indices) or None.

    None means the run short-circuits: every document parsed empty or the
    documents produced no chunks. The parse and chunk stages are timed.
    """
    with timed(timing, "parse_ms"):
        parsed = [parse_document(doc, ocr=ocr) for doc in documents]
        parsed = [p for p in parsed if p.content.strip()]
        if not parsed:
            logger.warning("All documents are empty after parsing")
            return None

    with timed(timing, "chunk_ms"):
        chunks = chunk_documents(parsed, max_tokens=retrieval.chunk_max_tokens)
        if not chunks:
            return None
        chunk_contexts = build_chunk_contexts(chunks)
        chunk_context_indices = build_chunk_context_indices(chunks)
    logger.info("Chunked %d document(s) into %d chunks", len(parsed), len(chunks))
    return parsed, chunks, chunk_contexts, chunk_context_indices


def run_extraction(
    documents: list[Path],
    client,
    config: LLMConfig,
    risks: list,
    retrieval: RetrievalConfig | None = None,
    *,
    ocr: bool = False,
    prefilter: GroundingPrefilter | None = None,
    plan: RunPlan | None = None,
) -> ExtractionResult:
    retrieval = retrieval or RetrievalConfig()
    if prefilter is not None and retrieval.no_grounding:
        raise ValueError("prefilter requires grounding; it filters what grounding sees, so --no-grounding refuses it")
    validate_rrf_min_score(retrieval.rrf_min_score)
    if plan is None:
        plan = resolve_plan(retrieval, prefilter=prefilter is not None, explicit=frozenset())
    else:
        # Execution must follow the retrieval settings it was handed, so a
        # caller-supplied plan has to resolve the same way. Warnings sit
        # outside this check: they depend on which options the caller's
        # CLI layer saw passed explicitly, which the pipeline cannot know.
        derived = resolve_plan(retrieval, prefilter=prefilter is not None, explicit=frozenset())
        if replace(plan, warnings=()) != replace(derived, warnings=()):
            raise ValueError(
                "the supplied run plan does not match the retrieval settings; "
                "pass the settings the plan was resolved from, or omit the plan and let the run resolve it"
            )
    timing: dict[str, float] = {}
    max_workers = config.max_concurrent

    prepared = _parse_and_chunk(documents, ocr, retrieval, timing)
    if prepared is None:
        return _empty_result(documents, _empty_result_metadata(retrieval, prefilter, plan))
    parsed, chunks, chunk_contexts, chunk_context_indices = prepared

    risks = _filter_agentic_risks(risks, parsed)

    with timed(timing, "index_ms"):
        if not risks:
            logger.error("No risks loaded from Nexus")
            return _empty_result(documents, _empty_result_metadata(retrieval, prefilter, plan))
        index = _build_risk_index(risks, retrieval)

    call_collector: list[LLMCallRecord] = []
    fallback_chunk_indices: set[int] = set()
    query_results: list = []

    if plan.candidate_source == "query_gen":
        query_results, fallback_chunk_indices = _run_query_gen(
            chunks, client, config, call_collector, max_workers, timing
        )

    chunk_results = _run_retrieval(chunks, index, retrieval, query_results, fallback_chunk_indices, plan, timing)
    logger.info(
        "Retrieval complete: %d chunks, %d accepted candidates",
        len(chunks),
        sum(len(cr.accepted) for cr in chunk_results if cr is not None),
    )

    if index.variant_map and not plan.grounding:
        _expand_chunk_variants(chunk_results, index)

    chunk_summaries = _build_chunk_summaries(chunk_results, chunks)

    with timed(timing, "judge_ms"):
        _run_judge(
            chunk_results,
            chunks,
            client,
            config,
            retrieval,
            call_collector,
            plan,
            chunk_contexts=chunk_contexts,
        )

    prefilter_drops: list[FilteredCandidate] = []
    decision_calls: list[DecisionCallRecord] = []
    if plan.prefilter and prefilter is not None:
        with timed(timing, "prefilter_ms"):
            grounding_inputs, prefilter_drops, decision_calls = _run_prefilter_stage(
                chunk_results, chunk_contexts, prefilter, index, retrieval, chunks
            )
    else:
        grounding_inputs = chunk_results

    all_matches, all_filtered, grounding_filtered = _run_matches_stage(
        grounding_inputs,
        chunks,
        client,
        config,
        retrieval,
        plan,
        index,
        call_collector,
        chunk_contexts,
        chunk_context_indices,
        timing,
    )
    all_filtered = prefilter_drops + all_filtered

    all_matches, removed_variant_groups = _run_variant_grounding_stage(
        all_matches,
        chunks,
        index,
        client,
        config,
        plan,
        call_collector,
        chunk_contexts,
        chunk_context_indices,
        max_workers,
        timing,
    )

    chunk_risk_ids = _build_chunk_risk_map(chunk_results, all_matches, plan)
    for cs in chunk_summaries:
        cs.accepted_risk_ids = sorted(set(chunk_risk_ids.get(cs.index, [])))

    with timed(timing, "merge_ms"):
        merged = merge_matches(all_matches)

    merged, expansion_stats, expansion_removed = _run_expansion_stage(
        merged,
        risks,
        chunk_results,
        chunks,
        documents,
        index,
        client,
        config,
        retrieval,
        plan,
        call_collector,
        chunk_contexts,
        chunk_context_indices,
        max_workers,
        timing,
        prefilter_drops=prefilter_drops or None,
    )
    removed_variant_groups.extend(expansion_removed)
    removed_variant_groups = _reconcile_removed_groups(removed_variant_groups, merged, index.variant_map)
    # the combined list sorts so a record's position never depends on which
    # stage produced it first or on future completion order
    removed_variant_groups.sort(key=lambda r: (r.parent_id, r.stage))

    merged = _run_causal_synthesis_stage(merged, chunks, client, config, plan, call_collector, max_workers, timing)

    # the invariant holds the run to the stage's own gate: exactly the runs
    # whose variant-grounding stage could replace parents with variants
    _assert_no_synthetic_parents(
        merged,
        index.variant_map,
        check=bool(index.variant_map and plan.grounding and client is not None),
    )

    total_stats = _build_retrieval_stats(
        chunk_results,
        chunks,
        timing,
        grounding_filtered,
        plan,
        prefilter_dropped=len(prefilter_drops) if prefilter is not None else None,
    )

    return ExtractionResult(
        risks=merged,
        source_documents=[str(d) for d in documents],
        retrieval_stats=total_stats,
        metadata=_build_result_metadata(
            config, retrieval, expansion_stats, query_results, fallback_chunk_indices, client, prefilter, plan
        ),
        chunks=chunk_summaries,
        llm_calls=call_collector,
        grounding_filtered_candidates=all_filtered,
        decision_calls=decision_calls,
        prefilter_ran=prefilter is not None,
        removed_variant_groups=removed_variant_groups,
    )


def _empty_result(documents: list[Path], metadata: dict | None = None) -> ExtractionResult:
    return ExtractionResult(
        risks=[],
        source_documents=[str(d) for d in documents],
        retrieval_stats=RetrievalStats(
            total_chunks=0,
            total_candidates_retrieved=0,
            auto_accepted=0,
            llm_judged=0,
            grounding_filtered=0,
        ),
        metadata=metadata or {},
    )
