"""Tests for the pre-grounding probability threshold rule in extract/gates.py.

The rule drops a candidate only when a probability outcome is at or below
the threshold; anything the assessor could not answer (``Unavailable``)
keeps the candidate with a bypass reason, so a broken provider never
silently loses risks.
"""

from asago_policy_mapper.extract.decide import (
    AssessmentBatch,
    Categorical,
    DecisionCase,
    Probability,
    Unavailable,
)
from asago_policy_mapper.extract.gates import ThresholdDrop


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
    outcomes: dict[str, tuple],
    unmatched_ids: tuple[str, ...] = (),
    latency_ms: float = 1.0,
    wire: dict | None = None,
) -> AssessmentBatch:
    return AssessmentBatch(
        outcomes=outcomes,
        unmatched_ids=unmatched_ids,
        latency_ms=latency_ms,
        wire=wire,
    )


class _ProbabilityAssessor:
    """Stub assessor that emits only Probability outcomes."""

    assessor_id = "stub-probability"
    emits = frozenset({Probability})
    offsite = False

    def __init__(self, outcomes: dict[str, tuple], unmatched_ids: tuple[str, ...] = ()):
        self.outcomes = outcomes
        self.unmatched_ids = unmatched_ids


class _MixedAssessor:
    """Stub assessor that advertises Probability and Categorical (reduction edge)."""

    assessor_id = "stub-mixed"
    emits = frozenset({Probability, Categorical})
    offsite = False


class _CategoricalAssessor:
    """Stub assessor that emits only Categorical outcomes (rejection case)."""

    assessor_id = "stub-categorical"
    emits = frozenset({Categorical})
    offsite = False


def test_threshold_exactly_0_10_drops():
    """A probability exactly at the threshold drops; just above keeps.

    ThresholdDrop(threshold=0.10) drops on p <= threshold, so 0.10 drops and
    0.10 + epsilon keeps, pinning the boundary on both sides.
    """
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    assert rule.rule_id == "p-le-drop/1"
    assert rule.threshold == 0.10

    batch = _batch({"R-001": (Probability(0.10),), "R-002": (Probability(0.11),)})
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].risk_id == "R-001"
    assert decisions[0].keep is False
    assert decisions[0].reason == "p<=0.10"
    assert decisions[1].risk_id == "R-002"
    assert decisions[1].keep is True


def test_unavailable_keeps_with_bypass_reason():
    """Every Unavailable outcome keeps the candidate with reason bypass:<reason>."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch(
        {
            "R-001": (Unavailable("timeout"),),
            "R-002": (Unavailable("error"),),
            "R-003": (Unavailable("invalid"),),
            "R-004": (Unavailable("missing"),),
        }
    )
    decisions = rule.decide(_cases("R-001", "R-002", "R-003", "R-004"), batch)

    assert len(decisions) == 4
    assert [d.risk_id for d in decisions] == ["R-001", "R-002", "R-003", "R-004"]
    for decision, reason in zip(decisions, ("timeout", "error", "invalid", "missing")):
        assert decision.keep is True
        assert decision.reason == f"bypass:{reason}"


def test_rule_rejects_categorical_assessor():
    """Constructing the rule with an assessor whose emits lacks Probability raises TypeError."""
    assessor = _CategoricalAssessor()
    try:
        ThresholdDrop(assessor)
    except TypeError as e:
        message = str(e)
        assert "Probability" in message
        assert "Categorical" in message
    else:
        raise AssertionError("ThresholdDrop must reject an assessor that does not emit Probability")


def test_rule_rejects_assessor_without_emits():
    """An assessor with no emits attribute at all also raises TypeError (not AttributeError)."""

    class _Bare:
        assessor_id = "stub-bare"
        offsite = False

    try:
        ThresholdDrop(_Bare())
    except TypeError as e:
        assert "got emits=none" in str(e)
    else:
        raise AssertionError("ThresholdDrop must reject an assessor without an emits declaration")


def test_rule_keeps_its_assessor():
    """The rule retains the assessor it was constructed with, for the pipeline to drive."""
    assessor = _ProbabilityAssessor({})
    rule = ThresholdDrop(assessor)

    assert rule.assessor is assessor


def test_duplicate_probabilities_reduce_to_most_conservative():
    """Duplicate Probability outcomes for one id drop only when all are low.

    The judge seam keeps duplicate outcomes; the probability rule drops only
    when every outcome is at or below the threshold (the conservative
    reduction: any high probability keeps, mirroring any-yes-keeps).
    """
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch(
        {
            "R-001": (Probability(0.05), Probability(0.50)),
            "R-002": (Probability(0.02), Probability(0.03)),
        }
    )
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].keep is True
    assert decisions[1].keep is False
    assert decisions[1].reason == "p<=0.10"


def test_empty_outcomes_keep_with_missing_reason():
    """An omitted risk (empty outcome tuple) keeps with a bypass:missing reason.

    The assessor omitted the id from the response; keeping it safe-guards
    the pipeline against partial answers losing risks.
    """
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    decisions = rule.decide(_cases("R-001"), _batch({}))

    assert decisions[0].keep is True
    assert decisions[0].reason == "bypass:missing"


def test_keep_reason_names_probability():
    """A kept candidate's reason carries its probability for the run report."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch({"R-001": (Probability(0.42),)})
    decisions = rule.decide(_cases("R-001"), batch)

    assert decisions[0].keep is True
    assert decisions[0].reason == "p=0.42"


def test_threshold_default_is_0_10():
    """The rule's default threshold is 0.10, with the drop reason naming it."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))

    assert rule.threshold == 0.10

    batch = _batch({"R-001": (Probability(0.10),)})
    decisions = rule.decide(_cases("R-001"), batch)
    assert decisions[0].keep is False


def test_threshold_is_configurable():
    """A stricter threshold drops more candidates and updates the reason."""
    rule = ThresholdDrop(_ProbabilityAssessor({}), threshold=0.25)

    batch = _batch({"R-001": (Probability(0.20),), "R-002": (Probability(0.30),)})
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].keep is False
    assert decisions[0].reason == "p<=0.25"
    assert decisions[1].keep is True


def test_threshold_reason_keeps_full_precision():
    """A threshold that two-decimal formatting would round keeps its exact value in the reason."""
    rule = ThresholdDrop(_ProbabilityAssessor({}), threshold=0.105)

    batch = _batch({"R-001": (Probability(0.104),), "R-002": (Probability(0.11),)})
    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert decisions[0].keep is False
    assert decisions[0].reason == "p<=0.105"
    assert decisions[1].keep is True


def test_unsupported_outcome_keeps_with_unsupported_reason():
    """A non-empty outcome the rule cannot reduce keeps with bypass:unsupported."""
    rule = ThresholdDrop(_MixedAssessor())
    batch = _batch({"R-001": (Categorical(True),)})

    decisions = rule.decide(_cases("R-001"), batch)

    assert decisions[0].risk_id == "R-001"
    assert decisions[0].keep is True
    assert decisions[0].reason == "bypass:unsupported"


def test_outcomes_for_ids_not_in_cases_are_ignored():
    """Outcomes and unmatched ids for risks outside the requested cases appear in no decision."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch(
        {"R-001": (Probability(0.01),), "R-999": (Probability(0.99),)},
        unmatched_ids=("R-888",),
    )

    decisions = rule.decide(_cases("R-001", "R-002"), batch)

    assert [d.risk_id for d in decisions] == ["R-001", "R-002"]
    assert decisions[0].keep is False
    assert decisions[1].reason == "bypass:missing"


def test_unavailable_outcome_dominates_keeps():
    """A mix of Unavailable and Probability keeps the candidate (Unavailable wins)."""
    rule = ThresholdDrop(_ProbabilityAssessor({}))
    batch = _batch({"R-001": (Probability(0.01), Unavailable("timeout"))})
    decisions = rule.decide(_cases("R-001"), batch)

    assert decisions[0].keep is True
    assert decisions[0].reason == "bypass:timeout"
