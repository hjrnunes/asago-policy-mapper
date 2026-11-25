"""Keep/drop rules that sit outside the screening seam's own module.

The probability threshold rule reduces ``ProbabilityAnswer`` and
``NoAnswer`` answers to ``Decision`` objects for the pre-grounding
filter. It lives here rather than in ``extract/screening.py`` so adding
it does not touch the screening-seam types module.
"""

from __future__ import annotations

from collections.abc import Sequence

from asago_policy_mapper.extract.screening import (
    Answer,
    AnswerBatch,
    Decision,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
)

_MISSING = "missing"
"""Reason for a risk the assessor omitted entirely (empty answer tuple)."""

_UNSUPPORTED = "unsupported"
"""Reason for answers this rule cannot reduce (e.g. a lone ``BooleanAnswer``)."""


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
    reason, so an unreachable or broken assessor never loses risks.
    Constructing the rule with an assessor whose ``answer_types`` lacks
    ``ProbabilityAnswer`` raises ``TypeError``.
    """

    rule_id = "p-le-drop/1"

    def __init__(self, assessor, threshold: float = 0.10):
        answer_types = getattr(assessor, "answer_types", None) or frozenset()
        if ProbabilityAnswer not in answer_types:
            raise TypeError(
                f"ThresholdDrop requires an assessor that emits ProbabilityAnswer; "
                f"got answer_types={sorted(t.__name__ for t in answer_types) or 'none'}"
            )
        self.assessor = assessor
        self.threshold = threshold

    def decide(self, cases: Sequence[DecisionCase], batch: AnswerBatch) -> list[Decision]:
        return [self._decide_case(case, batch.answers.get(case.risk_id, ())) for case in cases]

    def _decide_case(self, case: DecisionCase, answers: tuple[Answer, ...]) -> Decision:
        for answer in answers:
            if isinstance(answer, NoAnswer):
                return Decision(case.risk_id, True, f"no_answer:{answer.reason}")
        probabilities = [a.probability for a in answers if isinstance(a, ProbabilityAnswer)]
        if not answers:
            return Decision(case.risk_id, True, f"no_answer:{_MISSING}")
        if not probabilities:
            return Decision(case.risk_id, True, f"no_answer:{_UNSUPPORTED}")
        if max(probabilities) <= self.threshold:
            return Decision(case.risk_id, False, f"p<={_format_threshold(self.threshold)}")
        return Decision(case.risk_id, True, f"p={max(probabilities)!r}")
