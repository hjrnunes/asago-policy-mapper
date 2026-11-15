from __future__ import annotations

import math
from collections import defaultdict

from pydantic import BaseModel

from asago_policy_mapper.extract.models import EvidenceSpan, RiskMatch


def _confidence_order_key(confidence: float | None) -> tuple[bool, float]:
    """Order confidences where None is lower than any number.

    The first component separates null from numeric so no numeric value —
    however extreme — can compare equal to None; False sorts below True,
    so None takes the lowest slot. The second orders the numeric values
    themselves.
    """
    return (confidence is not None, confidence or 0.0)


def _tiebreak_key(model: BaseModel) -> str:
    """Total-order tie-break for models the ordering keys cannot separate.

    Serializes every field deterministically — fields sorted by name — so
    content-different models compare differently and content-identical
    models compare equal. The convention is fixed rather than semantic: it
    exists so the winner never depends on input order when the ordering
    keys tie.
    """
    return str(sorted(model.model_dump().items()))


def _best_match_key(m: RiskMatch) -> tuple:
    """Ordering key that picks a risk group's representative match.

    Confidence decides first. Ties prefer the match whose evidence reaches
    the lowest chunk index, then the lexicographically largest tuple of
    evidence texts. The serialization tail makes the order total, so ties
    past those components still pick a fixed representative. A match
    without evidence ranks last on the chunk component. The models reject
    NaN confidence and scores at construction, so every comparison here is
    between real numbers and the order is total.
    """
    min_chunk = min((e.chunk_index for e in m.evidence), default=math.inf)
    return (
        _confidence_order_key(m.confidence),
        -min_chunk,
        tuple(e.text for e in m.evidence),
        _tiebreak_key(m),
    )


def _confidence_desc_key(confidence: float | None) -> tuple[bool, float]:
    """Order confidences descending for output sorting, None last.

    The nullness flag sends None below every number; the negated numeric
    value sorts higher confidences first.
    """
    is_none = confidence is None
    numeric = confidence if confidence is not None else 0.0
    return (is_none, -numeric)


def _max_confidence(confidences) -> float | None:
    """Max over confidences where None counts as lower than any number."""
    return max((c for c in confidences if c is not None), default=None)


def _evidence_sort_key(s: EvidenceSpan) -> tuple:
    """Ordering key for a merged risk's evidence spans.

    Cross-encoder score decides first (descending); ties break on chunk
    index, then sentence index, then text, then the serialization tail, so
    the order never depends on the order the group's matches arrived in.
    """
    return (-s.cross_encoder_score, s.chunk_index, s.sentence_index, s.text, _tiebreak_key(s))


def _merge_evidence(group: list[RiskMatch], max_evidence: int) -> list[EvidenceSpan]:
    """Collect, order, deduplicate and cap a group's evidence spans.

    Duplicate texts keep the representative the ordering selects (not the
    first copy seen), and the cap applies after deduplication.
    """
    spans = [span for m in group for span in m.evidence]
    spans.sort(key=_evidence_sort_key)
    seen_texts: set[str] = set()
    unique: list[EvidenceSpan] = []
    for span in spans:
        if span.text not in seen_texts:
            seen_texts.add(span.text)
            unique.append(span)
    return unique[:max_evidence]


def merge_matches(
    matches: list[RiskMatch],
    max_evidence: int = 3,
) -> list[RiskMatch]:
    if not matches:
        return []

    groups: dict[str, list[RiskMatch]] = defaultdict(list)
    for m in matches:
        groups[m.risk_id].append(m)

    merged = []
    for risk_id, group in groups.items():
        best = max(group, key=_best_match_key)
        merged.append(
            best.model_copy(
                update={
                    "confidence": _max_confidence(m.confidence for m in group),
                    "evidence": _merge_evidence(group, max_evidence),
                }
            )
        )

    merged.sort(key=lambda m: (_confidence_desc_key(m.confidence), m.risk_id))
    return merged
