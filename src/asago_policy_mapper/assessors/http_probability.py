"""A generic HTTP probability assessor for the decision seam.

The exchange shape is deliberately generic: the request is
``{"context": str, "items": [{"id": str, "text": str}]}`` and the response
is ``{"scores": [{"id": str, "p": float}]}``. The ``Question`` the
assessor receives is not part of the request body; the endpoint is
configured with the question it scores, and provider-specific request
shapes arrive as subclasses.

The assessor does not go through the chat client's retry and truncation
path. It owns its timeout, never catches ``KeyboardInterrupt`` or
``SystemExit``, and records the exact request body it sent in its
``exchange`` without credentials: the request carries none by
construction (the key travels in a header), and the recorded response is
redacted against the configured credential.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from asago_policy_mapper.extract.screening import (
    Answer,
    AnswerBatch,
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
    Question,
)

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
"""Hosts that count as on this machine for the ``sends_offsite`` flag."""

_PATH = "/assess"

_NoAnswerReason = Literal["missing", "invalid", "timeout", "error"]

_REDACTED = "[redacted]"


@dataclass(frozen=True)
class ProbabilityEndpoint:
    """Where a probability assessor runs, with the credential by env-var name only.

    ``api_key_env`` holds the NAME of the environment variable carrying the
    API key, never the key itself; ``repr`` and ``to_dict`` therefore never
    include a credential.
    """

    base_url: str
    model: str
    api_key_env: str
    timeout_s: float = 10.0
    max_batch: int = 32

    def to_dict(self) -> dict:
        """Serialize without credentials: every field is non-secret by construction."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "timeout_s": self.timeout_s,
            "max_batch": self.max_batch,
        }


def _is_local_host(host: str | None) -> bool:
    """Tell whether a host names this machine (loopback or localhost)."""
    return host in _LOCAL_HOSTS


class HttpProbabilityAssessor:
    """An ``Assessor`` that scores cases against a probability endpoint.

    Emits ``ProbabilityAnswer`` answers for scored ids and ``NoAnswer`` for
    timeouts (``timeout``), transport or HTTP-status errors (``error``),
    malformed or conflicting scores (``invalid``) and ids the response
    omitted (``missing``). Ids outside the batch land in ``unmatched_ids``.

    The client (and its connection pool) is created per ``assess`` call and
    closed at the end, unless a transport was injected, which stays under
    the caller's ownership and is never closed here. Constructing an
    offsite assessor whose ``api_key_env`` variable is unset warns once:
    policy text would leave this machine unauthenticated.
    """

    answer_types = frozenset({ProbabilityAnswer})

    def __init__(self, endpoint: ProbabilityEndpoint, transport=None):
        import httpx

        self.endpoint = endpoint
        self.assessor_id = f"http-probability:{endpoint.model}"
        self.sends_offsite = not _is_local_host(urlsplit(endpoint.base_url).hostname)
        self.prompt_name = "http-probability"
        self._httpx = httpx
        self._transport = transport
        if self.sends_offsite and not os.environ.get(endpoint.api_key_env):
            warnings.warn(
                f"{endpoint.api_key_env} is unset; {endpoint.base_url} is offsite"
                " and will receive unauthenticated requests",
                stacklevel=2,
            )

    def assess(self, question: Question, context: str, cases: Sequence[DecisionCase]) -> AnswerBatch:
        if not cases:
            # Built inline: a shared dict would let one caller's append leak
            # into the next empty batch's exchange.
            return AnswerBatch(answers={}, unmatched_ids=(), latency_ms=0.0, exchange={"requests": [], "responses": []})
        answers: dict[str, tuple[Answer, ...]] = {}
        unmatched: list[str] = []
        exchanges: list[dict] = []
        latency_ms = 0.0
        client = self._open_client()
        try:
            for batch in self._batches(cases):
                t0 = time.perf_counter()
                exchange, batch_answers, batch_unmatched = self._assess_batch(context, batch, client)
                latency_ms += (time.perf_counter() - t0) * 1000
                exchanges.append(exchange)
                unmatched.extend(batch_unmatched)
                for rid, answer in batch_answers.items():
                    entry = answers.get(rid)
                    answers[rid] = (entry or ()) + (answer,)
        finally:
            if self._transport is None:
                client.close()
        return AnswerBatch(
            answers=answers,
            unmatched_ids=tuple(unmatched),
            latency_ms=latency_ms,
            exchange={
                "requests": [w["request"] for w in exchanges],
                "responses": [w["response"] for w in exchanges],
            },
        )

    def _open_client(self):
        httpx = self._httpx
        kwargs: dict = {"base_url": self.endpoint.base_url, "timeout": self.endpoint.timeout_s}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _batches(self, cases: Sequence[DecisionCase]) -> list[list[DecisionCase]]:
        size = max(1, self.endpoint.max_batch)
        return [list(cases[i : i + size]) for i in range(0, len(cases), size)]

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        api_key = os.environ.get(self.endpoint.api_key_env) or ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _assess_batch(
        self, context: str, cases: Sequence[DecisionCase], client
    ) -> tuple[dict, dict[str, Answer], list[str]]:
        request_body = {
            "context": context,
            "items": [{"id": c.risk_id, "text": f"{c.name}: {c.description}"} for c in cases],
        }
        httpx = self._httpx
        try:
            response = client.post(_PATH, headers=self._headers(), json=request_body)
        except httpx.TimeoutException:
            return self._exchange(request_body, None), self._mark_all(cases, "timeout"), []
        except (KeyboardInterrupt, SystemExit):
            raise
        except (httpx.HTTPError, OSError) as error:
            return self._exchange(request_body, None), self._mark_all(cases, "error", detail=str(error)), []

        try:
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            detail = f"HTTP {error.response.status_code}"
            return self._exchange(request_body, None), self._mark_all(cases, "error", detail=detail), []
        except ValueError as error:
            return self._exchange(request_body, None), self._mark_all(cases, "error", detail=f"JSON: {error}"), []
        answers, unmatched = self._answers_from_payload(payload, cases)
        return self._exchange(request_body, payload), answers, unmatched

    def _answers_from_payload(
        self, payload: object, cases: Sequence[DecisionCase]
    ) -> tuple[dict[str, Answer], list[str]]:
        if not isinstance(payload, dict):
            return self._mark_all(cases, "error"), []
        scores = payload.get("scores")
        if scores is None:
            # A missing scores key and an explicit null mean the same thing:
            # the endpoint answered but scored nothing, so every case is
            # missing rather than the whole payload being invalid.
            return self._mark_all(cases, "missing"), []
        if not isinstance(scores, list):
            return self._mark_all(cases, "invalid"), []
        answers, unmatched = _reduce_scores(scores, frozenset({c.risk_id for c in cases}))
        missing = {c.risk_id: NoAnswer("missing") for c in cases if c.risk_id not in answers}
        answers.update(missing)
        return answers, unmatched

    def _exchange(self, request_body: dict, response_payload: object) -> dict:
        """Record the exact request body and the redacted response; never credentials.

        The request never carries the key (it travels in a header), so it
        stays exact. The response is redacted against the configured
        credential in case the endpoint echoes it back.
        """
        secret = os.environ.get(self.endpoint.api_key_env) or ""
        return {"request": request_body, "response": _redact(response_payload, secret)}

    def _mark_all(self, cases: Sequence[DecisionCase], reason: _NoAnswerReason, detail: str = "") -> dict[str, Answer]:
        return {c.risk_id: NoAnswer(reason, detail) for c in cases}


def _valid_probability(value: object) -> bool:
    """Accept an int/float in [0, 1]; reject bools, non-finite and out-of-range.

    The range check runs before any float conversion so huge integers
    (which overflow ``float``) compare by value instead of raising.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and 0.0 <= value <= 1.0 and math.isfinite(value)


def _as_float(value: object) -> float | None:
    """Coerce a validated value to float, or None when it was never valid."""
    if not _valid_probability(value):
        return None
    assert isinstance(value, (int, float))
    return float(value)


def _reduce_scores(scores: Sequence[object], case_ids: frozenset[str]) -> tuple[dict[str, Answer], list[str]]:
    """Turn ``{"scores": [{"id", "p"}]}`` rows into answers plus unmatched ids.

    An invalid or conflicting score for an id is final: later rows for that
    id cannot restore it. Identical duplicates collapse to one answer. Ids
    outside the batch go to ``unmatched`` instead of answers.
    """
    answers: dict[str, Answer] = {}
    seen: dict[str, float] = {}
    invalid: set[str] = set()
    unmatched: list[str] = []
    for row in scores:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if not isinstance(rid, str):
            continue
        if rid not in case_ids:
            unmatched.append(rid)
            continue
        if rid in invalid:
            continue
        number = _as_float(row.get("p"))
        if number is None:
            invalid.add(rid)
            answers[rid] = NoAnswer("invalid")
            continue
        prior = seen.get(rid)
        if prior is not None and prior != number:
            invalid.add(rid)
            answers[rid] = NoAnswer("invalid")
            continue
        seen[rid] = number
        answers[rid] = ProbabilityAnswer(number)
    return answers, unmatched


def _redact(value: object, secret: str) -> object:
    """Replace every occurrence of ``secret`` in a decoded response with a marker.

    An empty secret (no key configured) returns the value unchanged. The
    walk covers dict keys as well as values, because an endpoint can echo
    the credential back as a key.
    """
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, _REDACTED)
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value
