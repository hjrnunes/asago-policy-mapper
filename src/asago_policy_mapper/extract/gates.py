"""Keep/drop rules that sit outside the judge seam's own module.

The probability threshold rule reduces ``Probability`` and ``Unavailable``
outcomes to ``Decision`` objects for the pre-grounding filter. It lives
here rather than in ``extract/decide.py`` so adding it does not touch the
judge-seam types module.
"""

from __future__ import annotations

from collections.abc import Sequence

from asago_policy_mapper.extract.decide import (
    AssessmentBatch,
    Decision,
    DecisionCase,
    Outcome,
    Probability,
    Unavailable,
)

_UNAVAILABLE_DEFAULT = "missing"
"""Reason for a risk the assessor omitted entirely (empty outcome tuple)."""

_UNSUPPORTED = "unsupported"
"""Reason for outcomes this rule cannot reduce (e.g. a lone ``Categorical``)."""


def _format_threshold(threshold: float) -> str:
    """Format the threshold exactly: two decimals when lossless, else full precision."""
    two = f"{threshold:.2f}"
    return two if float(two) == threshold else f"{threshold:g}"


class ThresholdDrop:
    """Keep/drop rule "p-le-drop/1": drop only on a low enough probability.

    A case is dropped when every ``Probability`` outcome for it is at or
    below ``threshold``; a single higher probability keeps it. An
    ``Unavailable`` outcome (or an omitted risk) keeps the case with a
    ``bypass:<reason>`` reason, so an unreachable or broken assessor never
    loses risks. Constructing the rule with an assessor whose ``emits``
    lacks ``Probability`` raises ``TypeError``.
    """

    rule_id = "p-le-drop/1"

    def __init__(self, assessor, threshold: float = 0.10):
        emits = getattr(assessor, "emits", None) or frozenset()
        if Probability not in emits:
            raise TypeError(
                f"ThresholdDrop requires an assessor that emits Probability; "
                f"got emits={sorted(t.__name__ for t in emits) or 'none'}"
            )
        self.assessor = assessor
        self.threshold = threshold

    def decide(self, cases: Sequence[DecisionCase], batch: AssessmentBatch) -> list[Decision]:
        return [self._decide_case(case, batch.outcomes.get(case.risk_id, ())) for case in cases]

    def _decide_case(self, case: DecisionCase, outcomes: tuple[Outcome, ...]) -> Decision:
        for outcome in outcomes:
            if isinstance(outcome, Unavailable):
                return Decision(case.risk_id, True, f"bypass:{outcome.reason}")
        probabilities = [o.p for o in outcomes if isinstance(o, Probability)]
        if not outcomes:
            return Decision(case.risk_id, True, f"bypass:{_UNAVAILABLE_DEFAULT}")
        if not probabilities:
            return Decision(case.risk_id, True, f"bypass:{_UNSUPPORTED}")
        if max(probabilities) <= self.threshold:
            return Decision(case.risk_id, False, f"p<={_format_threshold(self.threshold)}")
        return Decision(case.risk_id, True, f"p={max(probabilities):g}")
