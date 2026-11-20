from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_serializer, model_validator


def _reject_nan(value: float | None) -> float | None:
    """Reject NaN scores: they break the merge ordering keys.

    NaN compares unequal to everything, including itself, so max() and
    sort() become input-order dependent wherever a NaN reaches them. A NaN
    confidence or cross-encoder score is always a bug — a score that was
    never computed or failed to parse — so the models refuse it up front.
    A None is "not calculated" and passes through untouched.
    """
    if value is not None and value != value:
        raise ValueError("score must not be NaN")
    return value


class ScoredCandidate(BaseModel):
    risk_id: str
    risk_name: str
    risk_description: str
    bm25_rank: int = 0
    embedding_distance: float = 0.0
    cross_encoder_score: float = 0.0
    rrf_score: float = 0.0
    # Which reranker produced cross_encoder_score; None means "never
    # calculated" (the candidate was outside the rerank window or its only
    # measurement was invalid) — the 0.0 default is a placeholder, not a
    # measurement.
    rerank_source: Literal["cross_encoder", "colbert"] | None = None
    accepted_by: str | None = None


DPV_RISK_CONTROLS = Literal[
    "AvoidanceControl",
    "MitigationControl",
    "ModificationControl",
    "MonitorControl",
    "InterruptionControl",
    "InterventionControl",
    "ReductionControl",
    "ResolutionControl",
    "InvestigationControl",
    "OverrideControl",
    "TransferControl",
]


class MitigationRef(BaseModel):
    action_id: str
    action_name: str | None = None
    description: str | None = None
    source: str
    category: Literal["technical", "operational", "governance"] | None = None
    risk_control: DPV_RISK_CONTROLS | None = None


class EvidenceSpan(BaseModel):
    text: str
    document: str
    page: int | None = None
    section: str | None = None
    chunk_index: int | None
    sentence_index: int = 0
    cross_encoder_score: float = 0.0
    verification: Literal["verified", "normalized", "not_found"] | None = None
    found_chunk_index: int | None = None
    # Length of the quote in characters, recorded so consumers can weigh
    # short verified quotes (a one-word quote verifies trivially) without
    # re-deriving it from the serialized text. Constructed spans fill it
    # automatically when not set explicitly.
    text_length: int | None = None

    @model_validator(mode="after")
    def _fill_text_length(self) -> "EvidenceSpan":
        if self.text_length is None:
            self.text_length = len(self.text)
        return self

    _reject_nan_score = field_validator("cross_encoder_score")(_reject_nan)


class RetrievalScores(BaseModel):
    bm25_rank: int
    embedding_distance: float
    cross_encoder_score: float | None
    rrf_score: float


# How a risk candidate was accepted into the result set.
ACCEPTED_BY = Literal[
    "rank",
    "threshold",
    "rrf",
    "query_gen",
    "llm_judge",
    "auto_promoted",
    "expansion",
]


class RiskMatch(BaseModel):
    risk_id: str
    risk_name: str
    risk_description: str
    taxonomy: str = ""
    confidence: float | None
    confidence_source: Literal["cross_encoder", "colbert", "rrf", "none"] = "none"
    grounding_confidence: str
    accepted_by: ACCEPTED_BY
    evidence: list[EvidenceSpan]
    scores: RetrievalScores
    mitigations: list[MitigationRef] = []
    threat: str | None = None
    threat_source: str | None = None
    vulnerability: str | None = None
    consequence: str | None = None
    impact: str | None = None
    scores_inherited_from: str | None = None
    # Why there is no causal chain: "disabled" covers synthesis turned off,
    # no client or no grounding; the other values come from
    # CausalSynthesisStatus.
    causal_status: Literal["generated", "no_evidence", "empty_answer", "validation_exhausted", "disabled"] | None = None
    # Which source supplied each causal field: LLM synthesis or the built-in
    # catalog tables.
    causal_provenance: dict[str, Literal["generated", "catalog"]] = {}

    _reject_nan_confidence = field_validator("confidence")(_reject_nan)

    @model_validator(mode="after")
    def _confidence_needs_a_source(self) -> "RiskMatch":
        """A numeric confidence must name where it came from.

        Null confidence means "not calculated"; a number without a source
        would be exactly the unsourced score this schema exists to prevent.
        """
        if self.confidence is not None and self.confidence_source == "none":
            raise ValueError(
                "numeric confidence requires a confidence_source other than 'none' "
                f"(risk {self.risk_id} has confidence={self.confidence})"
            )
        return self


class RetrievalStats(BaseModel):
    total_chunks: int
    total_candidates_retrieved: int
    auto_accepted: int
    llm_judged: int
    auto_promoted: int = 0
    grounding_filtered: int
    # None means the pre-grounding filter did not run; the count is
    # omitted then so a filter-off run serializes byte-identically to a
    # pre-filter run.
    prefilter_dropped: int | None = None
    timing_ms: dict[str, float] = {}

    @model_serializer(mode="wrap")
    def _omit_prefilter_dropped_when_off(self, handler) -> dict[str, Any]:
        """Leave ``prefilter_dropped`` out unless the prefilter ran.

        ``grounding_filtered`` counts only candidates grounding itself
        rejected; the prefilter's drops are counted separately here and
        never overlap with it.
        """
        data: dict[str, Any] = handler(self)
        if data.get("prefilter_dropped") is None:
            data.pop("prefilter_dropped", None)
        return data


class ChunkSummary(BaseModel):
    index: int
    source: str
    page: int | None = None
    section: str | None = None
    text_preview: str
    candidates_retrieved: int
    auto_accepted: int
    borderline: int
    discarded: int
    bm25_rescued: int = 0
    accepted_risk_ids: list[str] = []


class DecisionItem(BaseModel):
    """One risk's raw assessor answers and reduced keep/drop verdict."""

    risk_id: str
    answers: list[dict]
    keep: bool
    reason: str


class DecisionLog(BaseModel):
    """The decision seam's record of one keep/drop reduction.

    Identifies the question, prompt, assessor and rule that produced the
    decisions, so a run's borderline judgements can be replayed and audited.
    ``prompt_sha256`` hashes the rendered system template, telling alternate
    ``--judge-prompt`` templates apart.
    """

    question_id: str
    question_version: str
    prompt_name: str
    prompt_sha256: str
    assessor_id: str
    rule_id: str
    latency_ms: float
    items: list[DecisionItem]
    unmatched_ids: list[str] = []


class LLMCallRecord(BaseModel):
    call_id: str
    stage: Literal["judge", "grounding", "variant_grounding", "causal_synthesis", "query_gen"]
    chunk_index: int = -1
    risk_ids: list[str]
    messages: list[dict]
    response: dict | str | list[dict]
    duration_ms: float
    result_summary: str
    prompt_shortenings: list[dict] = []
    decision: DecisionLog | None = None


class FilteredCandidate(BaseModel):
    risk_id: str
    risk_name: str
    taxonomy: str = ""
    cross_encoder_score: float
    rrf_score: float = 0.0
    bm25_rank: int = 0
    accepted_by: str
    chunk_index: int
    dropped_by: str | None = None
    score: float | None = None
    reason: str | None = None

    @model_serializer(mode="wrap")
    def _omit_prefilter_fields(self, handler) -> dict[str, Any]:
        """Leave the prefilter provenance fields out unless the prefilter set them.

        Grounding rejections keep all three ``None``, and serializing them
        would change the output of every filter-off run; only a prefilter
        drop carries ``dropped_by``/``score``/``reason``.
        """
        data: dict[str, Any] = handler(self)
        for key in ("dropped_by", "score", "reason"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class DecisionCallRecord(BaseModel):
    """One decision-stage call: what was sent and what was decided.

    ``request`` holds the exact request the stage sent, with no
    credentials (the prefilter's key travels in a header the body never
    sees). Assessors that batch report one body per batch as a list;
    single-shot assessors report one body as a dict. ``decision`` is the
    ``DecisionLog`` the seam produced.
    """

    stage: str
    chunk_index: int
    request: dict | list
    decision: DecisionLog | None = None


class ExtractionResult(BaseModel):
    version: str = "0.4"
    risks: list[RiskMatch]
    source_documents: list[str]
    token_usage: dict = {}
    retrieval_stats: RetrievalStats
    metadata: dict = {}
    chunks: list[ChunkSummary] = []
    llm_calls: list[LLMCallRecord] = []
    grounding_filtered_candidates: list[FilteredCandidate] = []
    decision_calls: list[DecisionCallRecord] = []
    eval: dict | None = None

    @model_serializer(mode="wrap")
    def _omit_decision_calls_when_off(self, handler) -> dict[str, Any]:
        """Leave ``decision_calls`` out unless the prefilter recorded any.

        With the filter off the list stays empty, and the serialized result
        must stay byte-identical to the pre-filter output.
        """
        data: dict[str, Any] = handler(self)
        if not data.get("decision_calls"):
            data.pop("decision_calls", None)
        return data


class _JudgeVerdict(BaseModel):
    risk_id: str
    relevant: bool
    justification: str


class _JudgeVerdicts(BaseModel):
    items: list[_JudgeVerdict]


class _GroundingVerdict(BaseModel):
    risk_id: str
    grounded: bool
    confidence: Literal["high", "medium", "low"]


class _RiskEvidence(BaseModel):
    risk_id: str
    grounded: bool
    confidence: Literal["high", "medium", "low"]
    quotes: list[str]


class _RiskEvidenceList(BaseModel):
    items: list[_RiskEvidence]


class _CausalChain(BaseModel):
    threat: str
    threat_source: str
    vulnerability: str
    consequence: str
    impact: str


class _CausalChains(BaseModel):
    items: list[_CausalChain]


CausalSynthesisStatus = Literal["generated", "no_evidence", "empty_answer", "validation_exhausted"]


@dataclass
class CausalSynthesisResult:
    """The result of one causal-synthesis attempt, with the reason for no chain.

    ``status`` keeps the three "no chain" reasons apart: ``no_evidence``
    (the risk's evidence points at no known chunk), ``empty_answer`` (the
    model returned no chain content) and ``validation_exhausted`` (the
    model output failed parsing or validation until retries ran out).
    """

    status: CausalSynthesisStatus
    chain: _CausalChain | None = None


@dataclass
class RetrievalConfig:
    bi_encoder_model: str = "all-mpnet-base-v2"
    bi_encoder_api_key: str = "none"
    bi_encoder_model_name: str | None = None
    query_instruction: str = (
        "Given a text passage from an AI governance policy document, retrieve AI risk"
        " descriptions that are relevant to the concepts, requirements, or concerns"
        " discussed in the passage"
    )
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_type: str = "score"
    cross_encoder_model_name: str | None = None
    colbert_model: str | None = None
    chunk_max_tokens: int = 512
    top_n_accept: int = 10
    top_n_judge: int = 10
    min_score_floor: float = 0.70
    threshold_high: float | None = None
    threshold_low: float | None = None
    bm25_rescue_rank: int = 0
    rrf_min_score: float = 0.01
    use_cross_encoder: bool = True
    no_judge: bool = False
    no_grounding: bool = False
    judge_prompt: str = "judge_risk"
    judge_context_tokens: int = 0
    expand_siblings: bool = True
    grounding_passes: int = 3
    expansion_passes: int = 3
    no_causal_synthesis: bool = False
    grounding_batch_size: int = 15
    query_gen: bool = True

    @property
    def effective_cross_encoder_model(self) -> str | None:
        if not self.use_cross_encoder or self.colbert_model:
            return None
        return self.cross_encoder_model

    @property
    def effective_rrf_min_score(self) -> float:
        return self.rrf_min_score if not self.use_cross_encoder else 0.0

    def to_metadata(self) -> dict:
        return {
            "bi_encoder_model": self.bi_encoder_model,
            "cross_encoder_model": self.effective_cross_encoder_model,
            "cross_encoder_type": self.cross_encoder_type,
            "cross_encoder_model_name": self.cross_encoder_model_name,
            "use_cross_encoder": self.use_cross_encoder,
            "colbert_model": self.colbert_model,
            "chunk_max_tokens": self.chunk_max_tokens,
            "top_n_accept": self.top_n_accept,
            "top_n_judge": self.top_n_judge,
            "min_score_floor": self.min_score_floor,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "bm25_rescue_rank": self.bm25_rescue_rank,
            "rrf_min_score": self.rrf_min_score,
            "judge_prompt": self.judge_prompt,
            "no_judge": self.no_judge,
            "no_grounding": self.no_grounding,
            "judge_context_tokens": self.judge_context_tokens,
            "expand_siblings": self.expand_siblings,
            "grounding_passes": self.grounding_passes,
            "expansion_passes": self.expansion_passes,
            "no_causal_synthesis": self.no_causal_synthesis,
            "grounding_batch_size": self.grounding_batch_size,
            "query_gen": self.query_gen,
        }
