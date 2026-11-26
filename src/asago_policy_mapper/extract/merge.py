from __future__ import annotations

import json
import math
from collections import defaultdict

from pydantic import BaseModel

from asago_policy_mapper.extract.models import EvidenceSpan, RiskMatch


def _tiebreak_key(model: BaseModel) -> str:
    """Total-order tie-break for models the ordering keys cannot separate.

    Serializes every field with sorted keys, so content-different models
    compare differently and content-identical models compare equal. The
    JSON encoding is injective by construction for these models' fields:
    a str field round-trips as one JSON string, so two dumps that differ
    have different field values. The convention is fixed rather than
    semantic: it exists so the winner never depends on input order when
    the ordering keys tie.
    """
    return json.dumps(model.model_dump(), sort_keys=True)


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
    return (m.confidence, -min_chunk, tuple(e.text for e in m.evidence), _tiebreak_key(m))


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
                    "confidence": max(m.confidence for m in group),
                    "evidence": _merge_evidence(group, max_evidence),
                }
            )
        )

    merged.sort(key=lambda m: (-m.confidence, m.risk_id))
    return merged
