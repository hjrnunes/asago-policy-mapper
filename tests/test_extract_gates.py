"""Tests for the pre-grounding probability threshold rule in extract/screening.py.

The rule (``ThresholdDrop``) drops a candidate only when every
probability answer is at or below the threshold; anything the assessor
could not answer (``NoAnswer``) keeps the candidate with a
``no_answer:<reason>`` reason, so a broken provider never silently loses
risks.
"""

from asago_policy_mapper.extract.gates import ThresholdDrop
from asago_policy_mapper.extract.screening import (
    AnswerBatch,
    BooleanAnswer,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
)


def _cases(*risk_ids: str) -> list[DecisionCase]:
    return [
        DecisionCase(
            risk_id=risk_id,
            name=f"Risk {risk_id}",
            description=f"Description for {risk_id}.",
        )
        for risk_id in risk_ids
    ]


def _batch(
    answers: dict[str, tuple],
    unmatched_ids: tuple[str, ...] = (),
    latency_ms: float = 1.0,
    exchange: dict | None = None,
) -> AnswerBatch:
    return AnswerBatch(
        answers=answers,
        unmatched_ids=unmatched_ids,
        latency_ms=latency_ms,
        exchange=exchange,
    )


class _ProbabilityAssessor:
    """Stub assessor that answers only with ProbabilityAnswer objects."""

    assessor_id = "stub-probability"
    answer_types = frozenset({ProbabilityAnswer})
    sends_offsite = False
    prompt_name = "stub-probability"

    def __init__(self, answers: dict[str, tuple], unmatched_ids: tuple[str, ...] = ()):
        self.answers = answers
        self.unmatched_ids = unmatched_ids


class _MixedAssessor:
    """Stub assessor that advertises ProbabilityAnswer and BooleanAnswer (reduction edge)."""

    assessor_id = "stub-mixed"
    answer_types = frozenset({ProbabilityAnswer, BooleanAnswer})
    sends_offsite = False
    prompt_name = "stub-mixed"


class _BooleanAssessor:
    """Stub assessor that answers only with BooleanAnswer objects (rejection case)."""

    assessor_id = "stub-boolean"
    answer_types = frozenset({BooleanAnswer})
    sends_offsite = False
    prompt_name = "stub-boolean"


def test_threshold_exactly_0_10_drops():
    """A probability exactly at the threshold drops; just above keeps.

    ThresholdDrop(threshold=0.10) drops on p <= threshold, so 0.10 drops and
    0.10 + epsilon keeps, pinning the boundary on both sides.
    """
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    assert rule.rule_id == "p-le-drop/1"
    assert rule.threshold == 0.10

    batch = _batch({"R-001": (ProbabilityAnswer(0.10),), "R-002": (ProbabilityAnswer(0.11),)})
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].risk_id == "R-001"
    assert decisions[0].keep is False
    assert decisions[0].reason == "p<=0.10"
    assert decisions[1].risk_id == "R-002"
    assert decisions[1].keep is True


def test_unavailable_keeps_with_no_answer_reason():
    """Every NoAnswer answer keeps the candidate with reason no_answer:<reason>."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch(
        {
            "R-001": (NoAnswer("timeout"),),
            "R-002": (NoAnswer("error"),),
            "R-003": (NoAnswer("invalid"),),
            "R-004": (NoAnswer("missing"),),
        }
    )
    decisions = rule.decide(_cases("R-001", "R-002", "R-003", "R-004"), batch)

    assert len(decisions) == 4
    assert [d.risk_id for d in decisions] == ["R-001", "R-002", "R-003", "R-004"]
    for decision, reason in zip(decisions, ("timeout", "error", "invalid", "missing")):
        assert decision.keep is True
        assert decision.reason == f"no_answer:{reason}"


def test_rule_rejects_categorical_assessor():
    """Constructing the rule with an assessor whose answer_types lacks ProbabilityAnswer raises TypeError."""
    assessor = _BooleanAssessor()
    try:
        ThresholdDrop(assessor)
    except TypeError as e:
        message = str(e)
        assert "ProbabilityAnswer" in message
        assert "BooleanAnswer" in message
    else:
        raise AssertionError("DropThreshold must reject an assessor that does not answer with ProbabilityAnswer")


def test_rule_rejects_assessor_without_answer_types():
    """An assessor with no answer_types attribute at all also raises TypeError (not AttributeError)."""

    class _Bare:
        assessor_id = "stub-bare"
        sends_offsite = False

    try:
        ThresholdDrop(_Bare())
    except TypeError as e:
        assert "got answer_types=none" in str(e)
    else:
        raise AssertionError("DropThreshold must reject an assessor without an emits declaration")


def test_rule_keeps_its_assessor():
    """The rule retains the assessor it was constructed with, for the pipeline to drive."""
    assessor = _ProbabilityAssessor({})
    rule = ThresholdDrop(assessor)

    assert rule.assessor is assessor


def test_duplicate_probabilities_reduce_to_most_conservative():
    """Duplicate ProbabilityAnswer objects for one id drop only when all are low.

    The judge seam keeps duplicate answers; the probability rule drops only
    when every answer is at or below the threshold (the conservative
    reduction: any high probability keeps, mirroring the any-yes keeps
    rule).
    """
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch(
        {
            "R-001": (ProbabilityAnswer(0.05), ProbabilityAnswer(0.50)),
            "R-002": (ProbabilityAnswer(0.02), ProbabilityAnswer(0.03)),
        }
    )
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].keep is True
    assert decisions[1].keep is False
    assert decisions[1].reason == "p<=0.10"


def test_empty_answers_keep_with_missing_reason():
    """An omitted risk (empty outcome tuple) keeps with a no_answer:missing reason.

    The assessor omitted the id from the response; keeping it safeguards
    the pipeline against partial answers losing risks.
    """
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    decisions = rule.decide(_cases("R-001"), _batch({}))

    assert decisions[0].keep is True
    assert decisions[0].reason == "no_answer:missing"


def test_keep_reason_names_probability():
    """A kept candidate's reason carries its probability for the run report."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch({"R-001": (ProbabilityAnswer(0.42),)})
    decisions = rule.decide(_cases("R-001"), batch)

    assert decisions[0].keep is True
    assert decisions[0].reason == "p=0.42"


def test_threshold_default_is_0_10():
    """The rule's default threshold is 0.10, with the drop reason naming it."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))

    assert rule.threshold == 0.10

    batch = _batch({"R-001": (ProbabilityAnswer(0.10),)})
    decisions = rule.decide(_cases("R-001"), batch)
    assert decisions[0].keep is False


def test_threshold_is_configurable():
    """A stricter threshold drops more candidates and updates the reason."""
    rule = ThresholdDrop(_ProbabilityAssessor({}), threshold=0.25)

    batch = _batch({"R-001": (ProbabilityAnswer(0.20),), "R-002": (ProbabilityAnswer(0.30),)})
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].keep is False
    assert decisions[0].reason == "p<=0.25"
    assert decisions[1].keep is True


def test_threshold_reason_keeps_full_precision():
    """A threshold that two-decimal formatting would round keeps its exact value in the reason."""
    rule = ThresholdDrop(_ProbabilityAssessor({}), threshold=0.105)

    batch = _batch({"R-001": (ProbabilityAnswer(0.104),), "R-002": (ProbabilityAnswer(0.11),)})
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].keep is False
    assert decisions[0].reason == "p<=0.105"
    assert decisions[1].keep is True


def test_unsupported_answer_keeps_with_unsupported_reason():
    """A non-empty answer the rule cannot reduce keeps with no_answer:unsupported."""
    rule = ThresholdDrop(_MixedAssessor())
    batch = _batch({"R-001": (BooleanAnswer(True, "rationale"),)})

    decisions = rule.decide(_cases("R-001"), batch)

    assert decisions[0].risk_id == "R-001"
    assert decisions[0].keep is True
    assert decisions[0].reason == "no_answer:unsupported"


def test_answers_for_ids_not_in_cases_are_ignored():
    """Answers and unmatched ids for risks outside the requested cases appear in no decision."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch(
        {"R-001": (ProbabilityAnswer(0.01),), "R-999": (ProbabilityAnswer(0.99),)},
        unmatched_ids=("R-888",),
    )

    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert [d.risk_id for d in decisions] == ["R-001", "R-002"]
    assert decisions[0].keep is False
    assert decisions[1].reason == "no_answer:missing"


def test_no_answer_dominates_keeps():
    """A mix of NoAnswer and ProbabilityAnswer keeps the candidate (NoAnswer wins)."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch({"R-001": (ProbabilityAnswer(0.01), NoAnswer("timeout"))})
    decisions = rule.decide(_cases("R-001"), batch)

    assert decisions[0].keep is True
    assert decisions[0].reason == "no_answer:timeout"


def test_keep_reason_uses_repr():
    """A kept probability that :g formatting would round keeps its exact value."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch({"R-001": (ProbabilityAnswer(0.10000001),)})
    decisions = rule.decide(_cases("R-001"), batch)

    assert decisions[0].keep is True
    assert decisions[0].reason == "p=0.10000001"
