"""The Laya local-arm assessor for the decision seam.

Laya (laya-mlx, Apache-2.0) is a typed-decision model running natively on
Apple Silicon via MLX: its ``noul`` question type returns a calibrated
P(true) from one bidirectional forward pass — the same semantic Jev reads
from ``answers[qid].noul``. This module maps the seam's ``{context,
cases}`` input onto one in-process ``agent.predict(state, questions)``
call per batch and reads each answer's ``noul`` back as the probability.

This is NOT an ``HttpProbabilityAssessor`` subclass: there is no exchange,
no transport and no client lifecycle. The seam semantics are mirrored
directly — ``NoAnswer("error")`` for laya exceptions (fail-open),
``NoAnswer("invalid")`` for unreadable answers, ``NoAnswer("missing")``
for questions the result omitted, unmatched ids for extras — and the
duplicate-id guard (no two same-id cases in one predict call, since laya
keys answers by question id) replicates the Jev mapping's ``_batches``
logic.

**Context length.** Laya's context counts state, instructions and options
together, and our grounding contexts (~1.2k tokens) exceed the checkpoints'
limits (512 for the English 421M, 1024 for the multilingual 322M — the
default). ``predict`` head-truncates an oversized state and reports it in
``usage.state_tokens_dropped`` / ``usage.truncated``, which this assessor
records in its exchange so runs can measure the loss. Note that laya budgets
each question row independently: ``state_tokens_dropped`` depends on the
state length and that question's own instruction length, not on how many
questions share the pass. The default ``max_batch`` is 16 to bound each
forward pass's question count (and so per-pass latency), not to buy state
room.

**Dependency.** laya-mlx is installed ad hoc into the venv
(``VIRTUAL_ENV=<venv> UV_NO_CONFIG=1 uv pip install laya-mlx``) and never
enters ``pyproject.toml`` or ``uv.lock`` — the mutmut precedent. It is
imported lazily inside the constructor (like httpx in the HTTP adapters),
so imports of this module never require the install; ``uv sync`` removes
it, so reinstall after syncing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from asago_policy_mapper.assessors.http_probability import ProbabilityEndpoint, _as_float
from asago_policy_mapper.assessors.replay import context_sha256
from asago_policy_mapper.extract.screening import (
    AnswerBatch,
    Assessor,
    DecisionCase,
    Answer,
    NoAnswer,
    ProbabilityAnswer,
    Question,
)

_LAYA_BASE_URL = "http://localhost/laya"
"""A placeholder kept for ``ProbabilityEndpoint`` field-shape compatibility
only: the assessor never opens a client, and metadata records no host for
an in-process run."""

_LAYA_MODEL = "aac6fef/laya-multilingual-mlx"
"""The default checkpoint: 1024-token context (the English 421M has 512,
which our grounding contexts overflow far harder)."""

_LAYA_API_KEY_ENV = "POLICY_MAPPER_LAYA_API_KEY"
"""No key exists for a local model; kept for profile-shape compatibility."""

_LAYA_TIMEOUT_S = 30.0
"""Generous: measured 0.1–2.5s per predict; laya raises rather than hangs."""

_LAYA_MAX_BATCH = 16
"""Bounds the question count per forward pass (and so per-pass latency).
Laya budgets each question row independently — state_tokens_dropped does
not shrink with fewer questions per pass."""

_LAYA_INSTALL_HINT = (
    "laya-mlx is an ad-hoc, out-of-band install (never in pyproject/uv.lock); "
    "reinstall with: VIRTUAL_ENV=<venv> UV_NO_CONFIG=1 uv pip install laya-mlx"
)


@dataclass(frozen=True)
class LayaProbabilityEndpoint(ProbabilityEndpoint):
    """The Laya endpoint: the generic fields with the local arm's defaults.

    ``base_url`` is a placeholder (in-process), ``api_key_env`` names an
    unset variable (no key), and ``model`` is the laya checkpoint id. Every
    field keeps its override; ``repr`` and ``to_dict`` stay
    credential-free by construction.
    """

    base_url: str = _LAYA_BASE_URL
    model: str = _LAYA_MODEL
    api_key_env: str = _LAYA_API_KEY_ENV
    timeout_s: float = _LAYA_TIMEOUT_S
    max_batch: int = _LAYA_MAX_BATCH


class LayaProbabilityAssessor:
    """An ``Assessor`` that scores cases through local laya noul questions.

    The model loads once, in the constructor (the first ever load downloads
    the checkpoint; later loads use the local cache), and the agent is
    reused across ``assess`` calls. A missing laya-mlx install raises
    ``ImportError`` with the install hint at construction, not mid-run.
    """

    answer_types = frozenset({ProbabilityAnswer})
    in_process = True
    """Marks the assessor as in-process: no URL, no endpoint host."""

    def __init__(self, endpoint: LayaProbabilityEndpoint, agent=None):
        try:
            import laya_mlx
        except ImportError as e:
            raise ImportError(f"{e}; {self._install_hint()}") from e
        self.endpoint = endpoint
        self.assessor_id = f"laya:{endpoint.model}"
        self.sends_offsite = False
        self.prompt_name = "laya-noul"
        # The import must succeed even when an agent is injected (the stub
        # tests inject one; the real constructor always loads the model).
        self._agent = agent if agent is not None else laya_mlx.load(endpoint.model)

    @staticmethod
    def _install_hint() -> str:
        return _LAYA_INSTALL_HINT

    def assess(self, question: Question, context: str, cases: Sequence[DecisionCase]) -> AnswerBatch:
        if not cases:
            # Built inline, never shared: a caller mutating one result's
            # lists must not change what a later assessment records.
            return AnswerBatch(
                answers={}, unmatched_ids=(), latency_ms=0.0, exchange={"requests": [], "responses": []}
            )
        answers_out: dict[str, tuple[Answer, ...]] = {}
        unmatched: list[str] = []
        exchanges: list[dict] = []
        latency_ms = 0.0
        for batch in self._batches(cases):
            t0 = time.perf_counter()
            exchange, batch_answers, batch_unmatched = self._assess_batch(question, context, batch)
            latency_ms += (time.perf_counter() - t0) * 1000
            exchanges.append(exchange)
            unmatched.extend(batch_unmatched)
            for rid, answer in batch_answers.items():
                entry = answers_out.get(rid)
                answers_out[rid] = (entry or ()) + (answer,)
        return AnswerBatch(
            answers=answers_out,
            unmatched_ids=tuple(unmatched),
            latency_ms=latency_ms,
            exchange={
                "requests": [w["request"] for w in exchanges],
                "responses": [w["response"] for w in exchanges],
            },
        )

    def _batches(self, cases: Sequence[DecisionCase]) -> list[list[DecisionCase]]:
        """Split cases so no predict call carries two questions with the same key.

        Laya keys answers by the caller's question id, so two cases sharing
        a risk id must travel in separate predict calls; a single dict
        comprehension would silently keep only the last one. This
        replicates the Jev mapping's ``_batches`` (which lives on the HTTP
        adapter this class deliberately does not subclass).
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

    def _assess_batch(
        self, question: Question, context: str, cases: Sequence[DecisionCase]
    ) -> tuple[dict, dict[str, Answer], list[str]]:
        questions = {c.risk_id: self._question(question, c) for c in cases}
        # The state is not recorded verbatim (it is the whole chunk context);
        # its digest and length let a run match the recorded request against
        # the grounding context it filtered, the same key replay uses.
        request_body = {
            "model": self.endpoint.model,
            "state_sha256": context_sha256(context),
            "state_chars": len(context),
            "summary": True,
            "questions": questions,
        }
        try:
            result = self._agent.predict(context, questions)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            error_record = {"error": "error", "status": None, "detail": str(e)}
            return (
                {"request": request_body, "response": error_record},
                self._mark_all(cases, "error", detail=str(e)),
                [],
            )
        answers, unmatched = self._answers_from_result(result, cases)
        return {"request": request_body, "response": result}, answers, unmatched

    def _question(self, question: Question, case: DecisionCase) -> dict:
        """One noul question with compact string instructions.

        Laya's context counts instructions against the state, so the
        Jev-style structured instructions (~150 tokens) are too costly at
        batch sizes we need. The string carries the proposition and the
        risk's name and description — the question's complete meaning.
        """
        return {
            "type": "noul",
            "instructions": f"{question.criterion} Risk: {case.name} — {case.description}",
        }

    def _answers_from_result(
        self, result: object, cases: Sequence[DecisionCase]
    ) -> tuple[dict[str, Answer], list[str]]:
        """Read laya's result into answers, keyed by risk id.

        An answer that is not a readable noul is invalid; a question the
        result omitted is missing; ids outside the batch go to
        ``unmatched``. A non-dict result is an error for the whole batch.
        """
        if not isinstance(result, dict):
            return self._mark_all(cases, "error"), []
        answers = result.get("answers")
        if not isinstance(answers, dict):
            if "answers" not in result:
                return self._mark_all(cases, "missing"), []
            return self._mark_all(cases, "invalid"), []
        case_ids = frozenset({c.risk_id for c in cases})
        matched: dict[str, Answer] = {}
        unmatched: list[str] = []
        for qid, answer in answers.items():
            if qid not in case_ids:
                unmatched.append(qid)
                continue
            number = _noul_probability(answer)
            matched[qid] = ProbabilityAnswer(number) if number is not None else NoAnswer("invalid")
        missing = {c.risk_id: NoAnswer("missing") for c in cases if c.risk_id not in matched}
        matched.update(missing)
        return matched, unmatched

    def _mark_all(self, cases: Sequence[DecisionCase], reason, detail: str = "") -> dict[str, Answer]:
        return {c.risk_id: NoAnswer(reason, detail) for c in cases}


def _noul_probability(answer: object) -> float | None:
    """Read a laya noul answer's probability, or None when it cannot be read.

    A non-object answer, an answer whose type is not ``noul``, or a
    ``noul`` value outside [0, 1] (or not a number at all) reads as None.
    """
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        return None
    return _as_float(answer.get("noul"))
