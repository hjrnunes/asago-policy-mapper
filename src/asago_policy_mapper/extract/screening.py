"""The decision seam: a versioned question, an assessor, and rules.

The borderline judge and the pre-grounding filter both put a question
about each candidate risk to a decision model and reduce the answers to a
keep-or-drop decision. A ``Question`` states the question put to an
assessor. An ``Assessor`` (see ``assessors/``) answers it for a batch of
``DecisionCase`` objects, returning raw answers per risk id. A rule
reduces those answers to ``Decision`` objects: ``Keep`` keeps a case when
any answer is a yes; ``DropThreshold`` drops a case when every
probability is at or below a threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Question:
    """The question an assessor answers, with an id and version for logging."""

    question_id: str
    version: str
    criterion: str


RELEVANCE_QUESTION_V1 = Question(
    question_id="risk-relevance",
    version="1",
    criterion=(
        "A risk is relevant to a policy passage when the passage directly names "
        "the risk or its specific technical concept, describes its mitigation, "
        "management or oversight, or establishes a governance framework that "
        "inherently covers it."
    ),
)
"""The borderline-judge question, mirroring templates/prompts/judge_risk_system.j2.

The template counts three routes to relevance: direct mention of the risk or
its technical concept, description of its mitigation/oversight, and
governance that only implies coverage of the risk. ``version`` stays at "1"
while those semantics hold; a template that changes them bumps the version.
"""


@dataclass(frozen=True)
class DecisionCase:
    """One risk put to an assessor, decoupled from the retrieval candidate.

    ``is_variant_parent`` is set exactly when the risk id names a collapsed
    variant parent (an id containing ``---``), so assessors can treat
    synthetic parent entries differently from real risks.
    """

    risk_id: str
    name: str
    description: str
    is_variant_parent: bool = False


@dataclass(frozen=True)
class BooleanAnswer:
    """A yes/no answer with an optional rationale (the judge justification)."""

    holds: bool
    rationale: str | None = None


@dataclass(frozen=True)
class ProbabilityAnswer:
    """A probability answer for the yes/no question."""

    probability: float


@dataclass(frozen=True)
class NoAnswer:
    """The assessor could not answer; ``detail`` says why."""

    reason: Literal["missing", "invalid", "timeout", "error"]
    detail: str = ""


Answer = BooleanAnswer | ProbabilityAnswer | NoAnswer
"""One raw answer for one risk id, before a rule reduces it."""


@dataclass(frozen=True)
class AnswerBatch:
    """An assessor's raw answers for one batch of ``DecisionCase`` objects.

    ``answers`` stays raw: an absent key or an empty tuple means the
    assessor omitted the risk, and more than one entry means duplicate
    answers (the rule reduces them). ``unmatched_ids`` carries returned ids
    that were not in the batch, which the pipeline ignores. ``latency_ms``
    measures the invocation, never multiplied per candidate. ``exchange``
    holds the raw requests and responses the assessor sent and received
    (``{"messages", "response"}`` for the chat judge).
    """

    answers: dict[str, tuple[Answer, ...]]
    unmatched_ids: tuple[str, ...]
    latency_ms: float
    exchange: dict | None


class Assessor(Protocol):
    """An assessor that answers a ``Question`` for a batch of cases.

    ``answer_types`` declares the answer types the assessor can produce, so
    a rule can reject an assessor whose answers it cannot reduce.
    ``sends_offsite`` states whether answers leave the machine, for runs
    that must stay local. ``prompt_name`` names the prompt the answers came
    from, so the decision log can tell prompt variants apart.
    """

    assessor_id: str
    answer_types: frozenset[type]
    sends_offsite: bool
    prompt_name: str

    def assess(self, question: Question, context: str, cases: Sequence[DecisionCase]) -> AnswerBatch: ...


class Rule(Protocol):
    """A rule that reduces a batch of answers to decisions.

    ``rule_id`` versions the reduction, so a decision log names the exact
    rule semantics that produced each decision.
    """

    rule_id: str

    def decide(self, cases: Sequence[DecisionCase], batch: AnswerBatch) -> list[Decision]: ...


@dataclass(frozen=True)
class Decision:
    """The keep or drop verdict for one risk id, with the rule's reason."""

    risk_id: str
    keep: bool
    reason: str


class Keep:
    """Keep/drop rule "keep/1": keep when any answer is a yes.

    A case is kept when any of its answers is a ``BooleanAnswer`` that
    holds, regardless of what the other answers are. An empty answer tuple
    (the assessor omitted the risk) drops with reason "missing". Otherwise
    the drop reason names the first answer: "no" for a ``BooleanAnswer``
    that does not hold, "no_answer:<reason>" for a ``NoAnswer``,
    "probability:<p>" for a ``ProbabilityAnswer``.
    """

    rule_id = "keep/1"

    def decide(self, cases: Sequence[DecisionCase], batch: AnswerBatch) -> list[Decision]:
        decisions: list[Decision] = []
        for case in cases:
            decisions.append(self._decide_case(case, batch.answers.get(case.risk_id, ())))
        return decisions

    def _decide_case(self, case: DecisionCase, answers: tuple[Answer, ...]) -> Decision:
        if any(isinstance(a, BooleanAnswer) and a.holds for a in answers):
            return Decision(case.risk_id, True, "yes")
        if not answers:
            return Decision(case.risk_id, False, "missing")
        return Decision(case.risk_id, False, self._drop_reason(answers[0]))

    @staticmethod
    def _drop_reason(answer: Answer) -> str:
        if isinstance(answer, BooleanAnswer):
            return "no"
        if isinstance(answer, NoAnswer):
            return f"no_answer:{answer.reason}"
        return f"probability:{answer.probability!r}"


_MISSING = "missing"
"""Keep reason for a risk the assessor omitted entirely (empty answer tuple)."""

_UNSUPPORTED = "unsupported"
"""Keep reason for answers this rule cannot reduce (e.g. a lone ``BooleanAnswer``)."""

_NON_COMPARABLE = "invalid"
"""Keep reason for a probability the rule cannot compare to the threshold (NaN)."""


def _format_threshold(threshold: float) -> str:
    """Format the threshold for the drop reason, two decimals when lossless.

    The two-decimal form is a readability shortcut, not a precision claim:
    a threshold it would round (0.105) falls back to ``repr``-style
    formatting so the reason keeps the exact value.
    """
    two = f"{threshold:.2f}"
    return two if float(two) == threshold else repr(threshold)


class DropThreshold:
    """Keep/drop rule "drop-threshold/1": drop when every probability is at or below the threshold.

    A case is dropped when every ``ProbabilityAnswer`` for it is at or
    below ``threshold``; a single higher probability keeps it. A
    ``NoAnswer`` (or an omitted risk) keeps the case with a
    ``no_answer:<reason>`` reason, so an unreachable or broken assessor
    never loses risks. A probability the rule cannot compare to the
    threshold (NaN) keeps the case the same way. Constructing the rule
    with an assessor whose ``answer_types`` lacks ``ProbabilityAnswer``,
    or with a threshold that is NaN or outside [0, 1], raises
    ``TypeError`` and ``ValueError`` respectively.
    """

    rule_id = "drop-threshold/1"

    def __init__(self, assessor, threshold: float = 0.10):
        answer_types = getattr(assessor, "answer_types", None) or frozenset()
        if ProbabilityAnswer not in answer_types:
            raise TypeError(
                f"DropThreshold requires an assessor that emits ProbabilityAnswer; "
                f"got answer_types={sorted(t.__name__ for t in answer_types) or 'none'}"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(f"DropThreshold threshold must be a number; got {threshold!r}")
        if not 0.0 <= threshold <= 1.0:
            # Value comparison first, so a huge int compares by value instead of
            # overflowing float(); NaN fails the comparison and lands here too.
            raise ValueError(f"DropThreshold threshold must be in [0, 1]; got {threshold!r}")
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
        comparable = [p for p in probabilities if not math.isnan(p)]
        if len(comparable) < len(probabilities):
            # A NaN cannot be compared to the threshold, and max() with one
            # is order-dependent; treat it as an answer the rule cannot use.
            return Decision(case.risk_id, True, f"no_answer:{_NON_COMPARABLE}")
        if max(comparable) <= self.threshold:
            return Decision(case.risk_id, False, f"p<={_format_threshold(self.threshold)}")
        return Decision(case.risk_id, True, f"p={max(comparable)!r}")


def answer_payloads(answers: Sequence[Answer]) -> list[dict]:
    """Serialize answers for a ``DecisionItem``, keeping duplicates raw.

    Each answer carries an explicit stable ``kind`` ("boolean",
    "probability" or "no_answer"), so the output JSON does not depend on
    the Python class names.
    """
    return [{"kind": _answer_kind(a), **_answer_fields(a)} for a in answers]


def _answer_kind(answer: Answer) -> str:
    if isinstance(answer, BooleanAnswer):
        return "boolean"
    if isinstance(answer, ProbabilityAnswer):
        return "probability"
    return "no_answer"


def _answer_fields(answer: Answer) -> dict:
    if isinstance(answer, BooleanAnswer):
        return {"holds": answer.holds, "rationale": answer.rationale}
    if isinstance(answer, ProbabilityAnswer):
        return {"probability": answer.probability}
    return {"reason": answer.reason, "detail": answer.detail}
