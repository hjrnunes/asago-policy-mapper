"""Keep/drop rules that sit outside the screening seam's own module.

The probability threshold rule reduces ``ProbabilityAnswer`` and
``NoAnswer`` answers to ``Decision`` objects for the pre-grounding
filter. It lives here rather than in ``extract/screening.py`` so adding
it does not touch the screening-seam types module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from asago_policy_mapper.extract.screening import (
    Answer,
    AnswerBatch,
    BooleanAnswer,
    Decision,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
)

_MISSING = "missing"
"""Reason for a risk the assessor omitted entirely (empty answer tuple)."""

_UNSUPPORTED = "unsupported"
"""Reason for answers this rule cannot reduce (e.g. a lone ``BooleanAnswer``)."""

_NON_COMPARABLE = "invalid"
"""Keep reason for a probability the rule cannot compare to the threshold (NaN)."""


def _format_threshold(threshold: float) -> str:
    """Format the threshold exactly: two decimals when lossless, else full precision."""
    two = f"{threshold:.2f}"
    return two if float(two) == threshold else repr(threshold)


class ThresholdDrop:
    """Keep/drop rule "p-le-drop/1": drop only on a low enough probability.

    A case is dropped when every ``ProbabilityAnswer`` for it is at or
    below ``threshold``; a single higher probability keeps it. Any answer
    the assessor could not interpret keeps the case (fail-open): a
    ``NoAnswer`` or an omitted risk keeps with a ``no_answer:<reason>``
    reason, so an unreachable or broken assessor never loses risks. A
    probability the rule cannot compare to the threshold (NaN) keeps the
    case the same way. Constructing the rule with an assessor whose
    ``answer_types`` lacks ``ProbabilityAnswer``, or with a threshold
    that is NaN or outside [0, 1], raises ``TypeError`` and
    ``ValueError`` respectively.
    """

    rule_id = "p-le-drop/1"

    def __init__(self, assessor, threshold: float = 0.10):
        answer_types = getattr(assessor, "answer_types", None) or frozenset()
        if ProbabilityAnswer not in answer_types:
            raise TypeError(
                f"ThresholdDrop requires an assessor that emits ProbabilityAnswer; "
                f"got answer_types={sorted(t.__name__ for t in answer_types) or 'none'}"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(f"ThresholdDrop threshold must be a number; got {threshold!r}")
        if not 0.0 <= threshold <= 1.0:
            # Value comparison first, so a huge int compares by value instead of
            # overflowing float(); NaN fails the comparison and lands here too.
            raise ValueError(f"ThresholdDrop threshold must be in [0, 1]; got {threshold!r}")
        self.assessor = assessor
        self.threshold = threshold

    def decide(self, cases: Sequence[DecisionCase], batch: AnswerBatch) -> list[Decision]:
        return [self._decide_case(case, batch.answers.get(case.risk_id, ())) for case in cases]

    def _decide_case(self, case: DecisionCase, answers: tuple[Answer, ...]) -> Decision:
        for answer in answers:
            if isinstance(answer, NoAnswer):
                return Decision(case.risk_id, True, f"no_answer:{answer.reason}")
        if any(isinstance(answer, BooleanAnswer) for answer in answers):
            # Fail-open on answers of a kind this rule does not reduce: a
            # boolean beside the probabilities (affirmative or not) is not a
            # probability, so the case keeps rather than dropping on the
            # probabilities alone.
            return Decision(case.risk_id, True, f"no_answer:{_UNSUPPORTED}")
        probabilities = [a.probability for a in answers if isinstance(a, ProbabilityAnswer)]
        if not answers:
            return Decision(case.risk_id, True, f"no_answer:{_MISSING}")
        if not probabilities:
            return Decision(case.risk_id, True, f"no_answer:{_UNSUPPORTED}")
        comparable = [p for p in probabilities if not math.isnan(p)]
        if len(comparable) < len(probabilities):
            # A NaN cannot be compared to the threshold, and max() with one
            # is order-dependent; treat it as an answer the rule cannot use.
            return Decision(case.risk_id, True, f"no_answer:{_NON_COMPARABLE}")
        if max(comparable) <= self.threshold:
            return Decision(case.risk_id, False, f"p<={_format_threshold(self.threshold)}")
        return Decision(case.risk_id, True, f"p={max(comparable)!r}")
