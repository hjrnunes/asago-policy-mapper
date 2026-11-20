"""Deterministic merging and evidence selection (issue #150 part 1).

merge_matches must produce the same output for every ordering of its input:
the best match per risk and the surviving evidence spans may not depend on
the order parallel stages happened to finish in. These tests pin the
tie-break keys: confidence first, then lowest evidence chunk index, then
evidence texts; evidence spans order by cross-encoder score, then chunk,
sentence and text; duplicate texts keep the representative that ordering
selects; the output sorts by (-confidence, risk_id).
"""

import itertools

import pytest
from pydantic import ValidationError

from asago_policy_mapper.extract.merge import merge_matches
from asago_policy_mapper.extract.models import (
    EvidenceSpan,
    RetrievalScores,
    RiskMatch,
)


def _span(text, chunk_index, sentence_index=0, score=0.5, document="policy.pdf"):
    return EvidenceSpan(
        text=text,
        document=document,
        chunk_index=chunk_index,
        sentence_index=sentence_index,
        cross_encoder_score=score,
    )


def _scores(score):
    return RetrievalScores(
        bm25_rank=1,
        embedding_distance=0.2,
        cross_encoder_score=score,
        rrf_score=0.03,
    )


def _match(risk_id, confidence, spans, grounding_confidence="high"):
    return RiskMatch(
        risk_id=risk_id,
        risk_name=f"Risk {risk_id}",
        risk_description=f"Description of {risk_id}.",
        confidence=confidence,
        confidence_source="cross_encoder" if confidence is not None else "none",
        grounding_confidence=grounding_confidence,
        accepted_by="threshold",
        evidence=list(spans),
        scores=_scores(confidence),
    )


def _tie_group():
    """Four same-risk matches with tied confidence and distinct evidence.

    The spans were chosen so the old order-sensitive code (max by confidence
    with no tie-break; spans appended in input order, then stably sorted by
    score only) picks different winners under different permutations, while
    the deterministic keys pick the same winner every time:

    - best match: all four tie on confidence, so the minimum evidence chunk
      index decides — match 1 (quote-d, chunk 1) beats the chunk-2 and
      chunk-3 matches; its grounding_confidence tag identifies it.
    - evidence cap: every top span ties on cross-encoder score, so chunk
      index, sentence index and text decide the order; quote-a appears in
      chunks 2 and 3, and the chunk-2 copy is the representative that
      survives deduplication.
    """
    return [
        _match("R-TIE", 0.8, [_span("quote-b", 2, score=0.9), _span("quote-a", 2, score=0.9)], "g0"),
        _match("R-TIE", 0.8, [_span("quote-d", 1, score=0.9)], "g1"),
        _match("R-TIE", 0.8, [_span("quote-c", 3, score=0.9), _span("quote-e", 3, score=0.8)], "g2"),
        _match("R-TIE", 0.8, [_span("quote-a", 3, score=0.9)], "g3"),
    ]


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_merge_is_permutation_invariant(order):
    """Every ordering of tied-confidence matches merges to the same result."""
    group = _tie_group()
    matches = [group[i] for i in order]

    merged = merge_matches(matches, max_evidence=3)

    assert len(merged) == 1
    m = merged[0]
    # best match: all confidences tie; the lowest evidence chunk index wins,
    # so the representative is the match carrying quote-d from chunk 1
    assert m.confidence == 0.8
    assert m.grounding_confidence == "g1"
    # evidence: the top spans tie on score, so chunk index and text order
    # them; duplicate quote-a keeps its chunk-2 copy; the cap keeps three
    assert [(e.text, e.chunk_index) for e in m.evidence] == [
        ("quote-d", 1),
        ("quote-a", 2),
        ("quote-b", 2),
    ]


def test_evidence_cap_deterministic():
    """The cap keeps the same spans regardless of input order.

    Six spans tie on cross-encoder score, so the sort key (-score,
    chunk_index, sentence_index, text) fixes the order: quote-d (chunk 1),
    then quote-a (chunk 2), then quote-b (chunk 2). The cap of three keeps
    exactly those, and the duplicate quote-a in chunk 3 never survives
    deduplication to compete for a slot.
    """
    spans = [
        _span("quote-e", 3, score=0.8),
        _span("quote-a", 3, score=0.9),
        _span("quote-b", 2, score=0.9),
        _span("quote-c", 3, score=0.9),
        _span("quote-a", 2, score=0.9),
        _span("quote-d", 1, score=0.9),
    ]

    forward = merge_matches([_match("R-CAP", 0.7, spans)], max_evidence=3)
    backward = merge_matches([_match("R-CAP", 0.7, list(reversed(spans)))], max_evidence=3)

    assert [(e.text, e.chunk_index) for e in forward[0].evidence] == [
        ("quote-d", 1),
        ("quote-a", 2),
        ("quote-b", 2),
    ]
    assert forward[0].evidence == backward[0].evidence


def test_evidence_order_breaks_ties_by_sentence_then_text():
    """Within one chunk, sentence index then text order tied-score spans."""
    spans = [
        _span("b-text", 1, sentence_index=1, score=0.9),
        _span("a-text", 1, sentence_index=1, score=0.9),
        _span("c-text", 1, sentence_index=0, score=0.9),
    ]

    merged = merge_matches([_match("R-ORDER", 0.7, spans)], max_evidence=3)

    assert [e.text for e in merged[0].evidence] == ["c-text", "a-text", "b-text"]


def test_merge_representative_chosen_after_evidence_ordering():
    """Duplicate texts keep the span the evidence ordering selects.

    The same text appears in chunks 1 and 4. After sorting by (-score,
    chunk, sentence, text) the chunk-1 copy comes first, so that is the
    representative that survives deduplication, wherever it appeared in the
    input.
    """
    early = _match("R-DUP", 0.8, [_span("same text", 4, score=0.5)])
    late = _match("R-DUP", 0.8, [_span("same text", 1, score=0.5), _span("other text", 0, score=0.1)])

    for matches in ([early, late], [late, early]):
        merged = merge_matches(matches, max_evidence=3)
        assert len(merged[0].evidence) == 2
        assert merged[0].evidence[0].text == "same text"
        assert merged[0].evidence[0].chunk_index == 1
        assert merged[0].evidence[1].text == "other text"


def test_merge_best_tie_breaks_on_lowest_evidence_chunk():
    """Tied confidence and tied evidence texts fall back to lowest chunk."""
    late_chunk = _match("R-CHUNK", 0.8, [_span("z-text", 5, score=0.5)], "late")
    early_chunk = _match("R-CHUNK", 0.8, [_span("a-text", 2, score=0.5), _span("b-text", 9, score=0.5)], "early")

    for matches in ([late_chunk, early_chunk], [early_chunk, late_chunk]):
        merged = merge_matches(matches, max_evidence=3)
        assert merged[0].grounding_confidence == "early"


def test_merge_best_tie_breaks_on_evidence_texts():
    """Tied confidence and tied lowest chunk fall back to evidence texts."""
    low_texts = _match("R-TEXT", 0.8, [_span("a-text", 1, score=0.5)], "low-texts")
    high_texts = _match("R-TEXT", 0.8, [_span("b-text", 1, score=0.5)], "high-texts")

    for matches in ([low_texts, high_texts], [high_texts, low_texts]):
        merged = merge_matches(matches, max_evidence=3)
        assert merged[0].grounding_confidence == "high-texts"


def test_merge_best_tie_prefers_match_with_evidence():
    """A tied match without evidence ranks below any evidence-bearing match."""
    with_evidence = _match("R-EMPTY", 0.8, [_span("quote", 4, score=0.5)], "has-evidence")
    without_evidence = _match("R-EMPTY", 0.8, [], "no-evidence")

    for matches in ([with_evidence, without_evidence], [without_evidence, with_evidence]):
        merged = merge_matches(matches, max_evidence=3)
        assert merged[0].grounding_confidence == "has-evidence"
        assert [e.text for e in merged[0].evidence] == ["quote"]


def test_merge_best_uses_minimum_chunk_not_first_span_chunk():
    """The best-match key reads min(evidence chunk_index), not the first span's.

    The early-chunk match lists its lowest chunk last in its own evidence
    list, so only the minimum — not the first span — can pick it.
    """
    ordered = _match("R-MIN", 0.8, [_span("x-text", 5, score=0.5), _span("y-text", 2, score=0.5)], "min")
    first_low = _match("R-MIN", 0.8, [_span("z-text", 3, score=0.5)], "first")

    for matches in ([ordered, first_low], [first_low, ordered]):
        merged = merge_matches(matches, max_evidence=3)
        assert merged[0].grounding_confidence == "min"


def test_merge_best_tie_breaks_on_evidence_texts_as_tuple():
    """The evidence-texts component compares whole tuples, not max text.

    Match A holds texts ("a-text", "z-text") and match B ("b-text",
    "c-text"); both tie on confidence and lowest chunk. Tuple comparison
    decides on the first element, so B wins even though A carries the
    larger single text.
    """
    wide_tail = _match("R-TUP", 0.8, [_span("a-text", 1, score=0.5), _span("z-text", 1, score=0.5)], "wide-tail")
    high_head = _match("R-TUP", 0.8, [_span("b-text", 1, score=0.5), _span("c-text", 1, score=0.5)], "high-head")

    for matches in ([wide_tail, high_head], [high_head, wide_tail]):
        merged = merge_matches(matches, max_evidence=3)
        assert merged[0].grounding_confidence == "high-head"


def test_merge_full_key_tie_still_deterministic():
    """A tie on every ordering component still picks a fixed representative.

    Both matches share confidence, evidence chunk indexes and evidence
    texts, so only the retained fields (here grounding_confidence) separate
    them. The winner must not depend on input order.
    """
    first = _match("R-FULL", 0.8, [_span("same quote", 2, score=0.5)], "alpha")
    second = _match("R-FULL", 0.8, [_span("same quote", 2, score=0.5)], "beta")

    for matches in ([first, second], [second, first]):
        merged = merge_matches(matches, max_evidence=3)
        assert merged[0].grounding_confidence == "beta"


def test_merge_evidence_tie_still_deterministic():
    """Equal-key duplicate spans with different metadata keep a fixed one.

    The same text appears in two documents with identical score, chunk and
    sentence indexes. The surviving span must be the same whichever input
    order the matches arrive in; the serialization tail keeps the span with
    the smallest full content (document "a.pdf").
    """
    from_doc_b = _span("tied quote", 1, score=0.5, document="b.pdf")
    from_doc_a = _span("tied quote", 1, score=0.5, document="a.pdf")
    other = _span("other quote", 0, score=0.4, document="a.pdf")

    for spans in ([from_doc_b, other, from_doc_a], [from_doc_a, other, from_doc_b]):
        merged = merge_matches([_match("R-EV", 0.7, spans)], max_evidence=3)
        assert len(merged[0].evidence) == 2
        assert merged[0].evidence[0].text == "tied quote"
        assert merged[0].evidence[0].document == "a.pdf"


def test_merge_duplicate_quote_keeps_highest_scoring_copy():
    """Duplicate texts keep their highest-scoring copy as the representative.

    The same quote text appears twice with different cross-encoder scores.
    The evidence ordering puts the higher-scoring copy first, so that copy
    survives deduplication — regardless of input order, and even when the
    lower-scoring copy sits in an earlier chunk.
    """
    high_score_late_chunk = _span("shared quote", 4, score=0.9, document="high.pdf")
    low_score_early_chunk = _span("shared quote", 1, score=0.3, document="low.pdf")

    for spans in (
        [high_score_late_chunk, low_score_early_chunk],
        [low_score_early_chunk, high_score_late_chunk],
    ):
        merged = merge_matches([_match("R-DUPSCORE", 0.7, spans)], max_evidence=3)
        assert len(merged[0].evidence) == 1
        assert merged[0].evidence[0].text == "shared quote"
        assert merged[0].evidence[0].cross_encoder_score == 0.9
        assert merged[0].evidence[0].chunk_index == 4
        assert merged[0].evidence[0].document == "high.pdf"


def test_merge_cap_applies_after_dedup():
    """The cap counts unique texts, not raw spans.

    Four distinct texts appear as five spans (one duplicated). Capping at
    three after deduplication keeps three unique quotes; capping before
    dedup would leave only two.
    """
    spans = [
        _span("quote-1", 1, score=0.9),
        _span("quote-1", 2, score=0.9),
        _span("quote-2", 3, score=0.9),
        _span("quote-3", 4, score=0.9),
        _span("quote-4", 5, score=0.8),
    ]

    merged = merge_matches([_match("R-CAP2", 0.7, spans)], max_evidence=3)

    assert [e.text for e in merged[0].evidence] == ["quote-1", "quote-2", "quote-3"]
    assert merged[0].evidence[0].chunk_index == 1


def test_merge_output_sorted_by_confidence_then_risk_id():
    """The output order is (-confidence, risk_id), not input or dict order."""
    matches = [
        _match("R-B", 0.7, [_span("b", 0)]),
        _match("R-C", 0.9, [_span("c", 0)]),
        _match("R-A", 0.7, [_span("a", 0)]),
        _match("R-D", 0.9, [_span("d", 0)]),
    ]

    merged = merge_matches(matches, max_evidence=3)

    assert [m.risk_id for m in merged] == ["R-C", "R-D", "R-A", "R-B"]


def test_merge_characterization_highest_confidence_wins():
    """Characterization: unequal confidences keep today's choice.

    With distinct confidences the maximum decides the representative match
    exactly as before; the deterministic tie-breaks only matter when
    confidences tie. The surviving match carries the high-confidence match's
    identity (here its grounding_confidence tag), and the merged evidence
    still orders by cross-encoder score first.
    """
    low = _match("R-CHAR", 0.5, [_span("from low", 0, score=0.99)], "from-low")
    high = _match("R-CHAR", 0.9, [_span("from high", 5, score=0.01)], "from-high")

    for matches in ([low, high], [high, low]):
        merged = merge_matches(matches, max_evidence=3)

        assert len(merged) == 1
        assert merged[0].confidence == 0.9
        assert merged[0].grounding_confidence == "from-high"
        assert merged[0].evidence[0].text == "from low"
        assert merged[0].evidence[0].cross_encoder_score == 0.99


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_merge_rejects_nan_confidence_in_any_position(order):
    """NaN confidence fails validation in every input position.

    NaN compares unequal to everything, so a NaN reaching max() or sort()
    makes the winner depend on input order. The models reject NaN
    confidence at construction instead, which keeps the ordering total;
    the rejection itself must not depend on where the NaN sits.
    """
    confidences = [0.5, float("nan"), 0.7]
    values = [confidences[i] for i in order]

    with pytest.raises(ValidationError, match="must not be NaN"):
        RiskMatch(
            risk_id="R-NAN",
            risk_name="Risk R-NAN",
            risk_description="Description.",
            confidence=values[0],
            confidence_source="cross_encoder",
            grounding_confidence="high",
            accepted_by="threshold",
            evidence=[_span("quote", 0)],
            scores=_scores(0.5),
        )
        _match("R-NAN", values[1], [_span("quote", 1)])
        _match("R-NAN", values[2], [_span("quote", 2)])


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_merge_rejects_nan_evidence_score_in_any_position(order):
    """NaN cross-encoder scores fail validation in every input position.

    Evidence ordering starts at the cross-encoder score, so a NaN span
    score would make the surviving evidence order depend on input order;
    the models reject NaN scores at construction instead.
    """
    scores = [0.9, float("nan"), 0.5]
    values = [scores[i] for i in order]

    with pytest.raises(ValidationError, match="must not be NaN"):
        _span("nan quote", 0, score=values[0])
        _span("nan quote", 1, score=values[1])
        _span("nan quote", 2, score=values[2])
