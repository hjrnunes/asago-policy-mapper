"""Keep/drop rules that sit outside the judge seam's own module.

The probability threshold rule reduces ``Probability`` and ``Unavailable``
outcomes to ``Decision`` objects for the pre-grounding filter. It lives
here rather than in ``extract/decide.py`` so adding it does not touch the
judge-seam types module.
"""

from __future__ import annotations

import math
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

_INVALID = "invalid"
"""Reason for a probability the rule cannot compare to the threshold (NaN)."""


def _format_threshold(threshold: float) -> str:
    """Format the threshold for the drop reason, two decimals when lossless.

    The two-decimal form is a readability shortcut, not a precision claim:
    a threshold it would round (0.105) falls back to ``repr``-style
    formatting so the reason keeps the exact value.
    """
    two = f"{threshold:.2f}"
    return two if float(two) == threshold else repr(threshold)


class ThresholdDrop:
    """Keep/drop rule "p-le-drop/1": drop only on a low enough probability.

    A case is dropped when every ``Probability`` outcome for it is at or
    below ``threshold``; a single higher probability keeps it. An
    ``Unavailable`` outcome (or an omitted risk) keeps the case with a
    ``bypass:<reason>`` reason, so an unreachable or broken assessor never
    loses risks. A probability the rule cannot compare to the threshold
    (NaN) keeps the case the same way. Constructing the rule with an
    assessor whose ``emits`` lacks ``Probability``, or with a threshold
    that is NaN or outside [0, 1], raises ``TypeError`` and ``ValueError``
    respectively.
    """

    rule_id = "p-le-drop/1"

    def __init__(self, assessor, threshold: float = 0.10):
        emits = getattr(assessor, "emits", None) or frozenset()
        if Probability not in emits:
            raise TypeError(
                f"ThresholdDrop requires an assessor that emits Probability; "
                f"got emits={sorted(t.__name__ for t in emits) or 'none'}"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(f"ThresholdDrop threshold must be a number; got {threshold!r}")
        if not 0.0 <= threshold <= 1.0:
            # Value comparison first, so a huge int compares by value instead of
            # overflowing float(); NaN fails the comparison and lands here too.
            raise ValueError(f"ThresholdDrop threshold must be in [0, 1]; got {threshold!r}")
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
        comparable = [p for p in probabilities if not math.isnan(p)]
        if len(comparable) < len(probabilities):
            # A NaN cannot be compared to the threshold, and max() with one
            # is order-dependent; treat it as an answer the rule cannot use.
            return Decision(case.risk_id, True, f"bypass:{_INVALID}")
        if max(comparable) <= self.threshold:
            return Decision(case.risk_id, False, f"p<={_format_threshold(self.threshold)}")
        return Decision(case.risk_id, True, f"p={max(comparable)!r}")
