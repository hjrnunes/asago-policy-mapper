"""The Jev mapping for the decision seam's probability assessor.

Jev is TypeSafe's System One decision model, served at
``POST /v1/systemone`` on ``https://api.typesafe.ai`` with a ``Bearer``
API key in the ``Authorization`` header (the env var named by
``api_key_env``, never the key itself). This module is a thin subclass of
the generic HTTP probability adapter: it translates the seam's
``{context, cases}`` input onto Jev's exchange and Jev's answers back onto
the generic ``{id, p}`` scores, inheriting the generic adapter's
semantics unchanged (``NoAnswer("timeout"/"error"/"invalid"/"missing")``,
the sends-offsite determination, credentials never in
repr/serialization/exchange logs).

The Jev wire shape:

- Request: ``{"model", "state", "questions": {qid: {"type": "noul",
  "instructions", ...}}}``. The mapping sends the chunk context as
  ``state`` and one Noul question per case, keyed by the risk id (Jev
  question ids are code-side keys, not sent to the model). The
  instructions object carries the fixed question, the question's
  criterion and the case's name/description, mirroring the docs'
  structured instructions pattern.
- Response: ``{"model", "answers": {qid: {"type": "noul", "noul": p}},
  "usage"}``. Each answer's ``noul`` is the probability of yes in
  [0, 1]; the mapping reads it as the case's score.

Batching maps directly: Jev evaluates a request's questions in parallel
and caps requests by token context (64k per request, 32k per question),
not by question count, so the adapter's ``max_batch`` (default 32) is the
client-side batch knob. Rate limits (429) and overload (529) surface as
HTTP errors, which the base maps to ``NoAnswer("error")``, fail-open.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from asago_policy_mapper.assessors.http_probability import (
    _REDACTED,
    HttpProbabilityAssessor,
    ProbabilityEndpoint,
    _as_float,
)
from asago_policy_mapper.extract.screening import (
    Answer,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
    Question,
)

_JEV_BASE_URL = "https://api.typesafe.ai"
"""The Jev (TypeSafe System One) API host."""

_JEV_MODEL = "jev-latest"
"""The Jev model alias: the newest stable release at any point in time."""

_JEV_API_KEY_ENV = "POLICY_MAPPER_JEV_API_KEY"
"""The env var carrying the Jev API key; a profile can name another."""

_JEV_TIMEOUT_S = 30.0
"""A generous default: Jev answers fast, but the wire can queue."""

_QUESTION = "Given the policy passage in `state`, does the proposition in `proposition` hold for the risk in `risk`?"


@dataclass(frozen=True)
class JevProbabilityEndpoint(ProbabilityEndpoint):
    """The Jev endpoint: the generic fields with Jev's URL and model.

    Every field keeps its override, and the API key still travels by env
    var name only: ``api_key_env`` names the variable, so ``repr`` and
    ``to_dict`` stay credential-free.
    """

    base_url: str = _JEV_BASE_URL
    model: str = _JEV_MODEL
    api_key_env: str = _JEV_API_KEY_ENV
    timeout_s: float = _JEV_TIMEOUT_S
    max_batch: int = 32


class JevProbabilityAssessor(HttpProbabilityAssessor):
    """An assessor that scores cases through Jev Noul questions.

    The exchange translation only: everything else (client lifecycle,
    timeout/error mapping, batching, exchange recording with redaction)
    comes from the generic adapter. The assessor id names Jev, so runs
    record which mapping produced their probabilities.
    """

    _PATH = "/v1/systemone"

    #: What the Jev request carries beyond the generic shape: each
    #: question's instructions embed the fixed question text and the
    #: question's criterion (the ``Question`` grounding criterion) alongside
    #: the case's name and description.
    fields_sent = ("chunk_context", "question", "criterion", "risk_name", "risk_description")

    def __init__(self, endpoint: ProbabilityEndpoint, transport=None):
        super().__init__(endpoint, transport=transport)
        self.assessor_id = f"jev:{endpoint.model}"

    def _request_body(self, question: Question, context: str, cases: Sequence[DecisionCase]) -> dict:
        """Build Jev's SystemOne request: state plus one Noul question per case."""
        return {
            "model": self.endpoint.model,
            "state": context,
            "questions": {c.risk_id: self._question(question, c) for c in cases},
        }

    def _batches(self, cases: Sequence[DecisionCase]) -> list[list[DecisionCase]]:
        """Split cases so no request carries two questions with the same key.

        Jev keys questions by the caller's id, so two cases sharing a
        risk id must travel in separate requests; a single dict
        comprehension would silently keep only the last one, and which
        cases survive would then depend on ``max_batch``. A batch closes
        when it is full or when the next case's id already appears in it,
        so duplicate ids each get their own question and their outcomes
        accumulate across requests like the generic adapter's. For
        distinct ids the split is exactly the base adapter's contiguous
        slicing; the input order is preserved throughout.
        """
        size = max(1, self.endpoint.max_batch)
        batches: list[list[DecisionCase]] = []
        current: list[DecisionCase] = []
        keys: set[str] = set()
        for case in cases:
            if current and (len(current) >= size or case.risk_id in keys):
                batches.append(current)
                current = []
                keys = set()
            current.append(case)
            keys.add(case.risk_id)
        if current:
            batches.append(current)
        return batches

    def _question(self, question: Question, case: DecisionCase) -> dict:
        """One Noul question: the fixed question, the task's proposition and the case.

        Jev question ids are code-side keys and are not sent to the model,
        so the risk id travels only as the question key; the instructions
        carry the question's complete meaning on their own.
        """
        return {
            "type": "noul",
            "instructions": {
                "question": _QUESTION,
                "proposition": question.criterion,
                "risk": {"name": case.name, "description": case.description},
            },
        }

    def _answers_from_payload(
        self, payload: object, cases: Sequence[DecisionCase], secret: str
    ) -> tuple[dict[str, Answer], list[str]]:
        """Read Jev's answers object into answers, keyed by risk id.

        Jev's ``answers`` is a JSON object keyed by the question ids sent,
        so ids are unique per response. An answer for a case that is not a
        Noul (wrong or missing ``type``, unreadable ``noul``) is invalid;
        a case whose question the response omitted is missing; ids outside
        the batch go to ``unmatched``.

        ``secret`` is the credential captured once per ``assess`` call;
        every redaction in the batch uses that captured value, so
        redaction stays anchored to the secret that was actually sent
        even when the environment changes mid-call.
        """
        if not isinstance(payload, dict):
            return self._mark_all(cases, "error"), []
        answers = payload.get("answers")
        if answers is None:
            # Jev reports a request-level failure as a 200 body carrying
            # an error object instead of answers; that is an error for
            # every case, not a missing answer. An absent and an explicit
            # null answers key mean the same thing (the null-equals-
            # missing reading the generic adapter takes).
            if "error" in payload:
                detail = _error_detail(payload["error"], secret)
                return {c.risk_id: NoAnswer("error", detail) for c in cases}, []
            return self._mark_all(cases, "missing"), []
        if not isinstance(answers, dict):
            # Answers present but unreadable is a payload problem, even
            # when an error object rides along: the answers won.
            return self._mark_all(cases, "invalid"), []
        case_ids = frozenset({c.risk_id for c in cases})
        answers_out: dict[str, Answer] = {}
        unmatched: list[str] = []
        for qid, answer in answers.items():
            if qid not in case_ids:
                # The id leaves the payload and lands in the recorded
                # batch (and so the run output), so it goes through the
                # same secret replacement the recorded response does —
                # the base adapter's assembly-point redaction covers it.
                unmatched.append(qid)
                continue
            number = _noul_probability(answer)
            answers_out[qid] = ProbabilityAnswer(number) if number is not None else NoAnswer("invalid")
        missing = {c.risk_id: NoAnswer("missing") for c in cases if c.risk_id not in answers_out}
        answers_out.update(missing)
        return answers_out, unmatched


def _noul_probability(answer: object) -> float | None:
    """Read a Noul answer's probability, or None when it cannot be read.

    A non-object answer, an answer whose type is not ``noul``, or a
    ``noul`` value outside [0, 1] (or not a number at all) reads as None.
    """
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        return None
    return _as_float(answer.get("noul"))


def _error_detail(error: object, secret: str) -> str:
    """A short summary of a request-level error object, redacted.

    Jev's error body is free-form, so a string is taken as-is (truncated
    to 200) and anything else collapses to a stable marker. The text
    comes from the endpoint, which also saw the credential in its
    Authorization header, so it is redacted against that secret first —
    replacing before truncating, so a secret straddling the cut is fully
    replaced rather than partially kept. An empty secret (no key
    configured) is a no-op: there is nothing to replace. The detail
    lands in ``NoAnswer.detail`` and so in the run output.
    """
    if not isinstance(error, str):
        return "endpoint returned an error object"
    if not secret:
        return error[:200]
    return error.replace(secret, _REDACTED)[:200]
