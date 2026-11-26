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
redacted against the captured credential. The credential is read from the
environment once per ``assess`` call, and every use in that call — the
header, the exchange, the unmatched ids, the error details, and the
adapter's own escaping exception texts — consumes the captured value, so
redaction is anchored to the secret that was actually sent.
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

    Everything the response contributes to the batch is redacted against
    the configured credential before it leaves this adapter: the recorded
    exchange, the unmatched ids, the error details and the adapter's own
    exception texts alike, so an endpoint echoing the key back cannot leak
    it into a run's serialized output. The credential is read from the
    environment once, at the top of ``assess()``; every downstream use in
    that call consumes the captured value, so redaction stays anchored to
    the secret that was actually sent even if the environment changes
    mid-call.
    """

    # NoAnswer is declared beside ProbabilityAnswer because the timeout,
    # error, invalid and missing paths provably emit it; a rule rejecting
    # assessors that can answer with NoAnswer would wrongly reject this one.
    answer_types = frozenset({ProbabilityAnswer, NoAnswer})

    _PATH = "/assess"
    """The path requests go to; provider mappings override this."""

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
        # Single capture for the whole call. The headers this assessor sends
        # and every redaction it applies downstream — the recorded exchange,
        # the unmatched ids, the error details and its own exception texts —
        # consume this captured value, never a fresh environment read.
        # Redaction is anchored to the secret that was actually sent, so a
        # mid-call environment change cannot make the raw secret reappear.
        secret = os.environ.get(self.endpoint.api_key_env) or ""
        answers: dict[str, tuple[Answer, ...]] = {}
        unmatched: list[str] = []
        exchanges: list[dict] = []
        latency_ms = 0.0
        client = None
        try:
            # Client construction sits inside the try with the requests:
            # a construction failure (bad URL, TLS setup) propagates to
            # the caller, which fails the batch open instead of losing
            # the run, and there is nothing to close when it never
            # opened.
            client = self._open_client()
            for batch in self._batches(cases):
                t0 = time.perf_counter()
                try:
                    exchange, batch_answers, batch_unmatched = self._assess_batch(
                        question, context, batch, client, secret
                    )
                except (KeyboardInterrupt, SystemExit):
                    # Control-flow exceptions keep their exact form: callers
                    # rely on them, and they carry no endpoint-derived text.
                    raise
                except Exception as error:
                    # The boundary guarantee: whatever escapes this adapter
                    # carries no raw credential in its text. The type stays
                    # intact (the pipeline's fail-open branch keys on it);
                    # only the message is redacted against the captured
                    # secret, so the pipeline never needs to read the
                    # environment itself.
                    raise _redacted_exception(error, secret) from error
                latency_ms += (time.perf_counter() - t0) * 1000
                exchanges.append(exchange)
                unmatched.extend(batch_unmatched)
                for rid, answer in batch_answers.items():
                    entry = answers.get(rid)
                    answers[rid] = (entry or ()) + (answer,)
        finally:
            if client is not None and self._transport is None:
                client.close()
        return AnswerBatch(
            answers=answers,
            unmatched_ids=tuple(_redact_id(rid, secret) for rid in unmatched),
            latency_ms=latency_ms,
            latency_recorded=True,
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

    def _headers(self, secret: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    def _request_body(self, question: Question, context: str, cases: Sequence[DecisionCase]) -> dict:
        """Build the request body for one batch; provider mappings override this hook.

        The generic shape is ``{"context": str, "items": [{"id", "text"}]}``.
        The ``question`` rides along for mappings whose wire carries the
        question being asked; the generic endpoint is configured with the
        question it scores, so it ignores the parameter.
        """
        return {
            "context": context,
            "items": [{"id": c.risk_id, "text": f"{c.name}: {c.description}"} for c in cases],
        }

    def _assess_batch(
        self, question: Question, context: str, cases: Sequence[DecisionCase], client, secret: str
    ) -> tuple[dict, dict[str, Answer], list[str]]:
        request_body = self._request_body(question, context, cases)
        httpx = self._httpx
        try:
            response = client.post(self._PATH, headers=self._headers(secret), json=request_body)
        except httpx.TimeoutException:
            return (
                self._exchange(request_body, self._error_record("timeout"), secret),
                self._mark_all(cases, "timeout"),
                [],
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except (httpx.HTTPError, OSError) as error:
            return (
                self._exchange(request_body, self._error_record("error"), secret),
                self._mark_all(cases, "error", detail=_redact_text(str(error), secret)),
                [],
            )

        try:
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            detail = f"HTTP {error.response.status_code}"
            return (
                self._exchange(request_body, self._error_record("error", status=error.response.status_code), secret),
                self._mark_all(cases, "error", detail=detail),
                [],
            )
        except ValueError as error:
            # A 2xx whose body is not JSON: the status was fine, the
            # payload was not.
            return (
                self._exchange(request_body, self._error_record("error", status=response.status_code), secret),
                self._mark_all(cases, "error", detail=_redact_text(f"JSON: {error}", secret)),
                [],
            )
        answers, unmatched = self._answers_from_payload(payload, cases, secret)
        return self._exchange(request_body, payload, secret), answers, unmatched

    @staticmethod
    def _error_record(reason: str, status: int | None = None) -> dict:
        """The explicit record for a failed batch: never a null response.

        Persisted call records carry what was received; a batch that never
        got a payload records the reason it failed and the HTTP status
        when there was one.
        """
        return {"error": reason, "status": status}

    def _answers_from_payload(
        self, payload: object, cases: Sequence[DecisionCase], secret: str
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

    def _exchange(self, request_body: dict, response_payload: object, secret: str) -> dict:
        """Record the exact request body and the redacted response; never credentials.

        The request never carries the key (it travels in a header), so it
        stays exact. The response is redacted against the captured
        credential in case the endpoint echoes it back.
        """
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


def _redact_id(rid: str, secret: str) -> str:
    """Redact one response-echoed id against the credential, like the exchange.

    Ids outside the batch come verbatim from the response, so an endpoint
    that echoes the API key back as an id would carry it into
    ``unmatched_ids`` (and from there the decision log) unredacted. This
    closes that path the same way ``_redact`` closes the exchange.
    """
    if not secret:
        return rid
    return rid.replace(secret, _REDACTED)


def _redact_text(text: str, secret: str) -> str:
    """Replace every occurrence of the captured secret in an error detail.

    An empty secret (no key configured) returns the text unchanged. Error
    details carry endpoint-derived text (an exception message, a response
    fragment), so they are redacted like any other response content.
    """
    if not secret:
        return text
    return text.replace(secret, _REDACTED)


def _redacted_exception(error: Exception, secret: str) -> Exception:
    """Rebuild ``error`` with its message redacted against the captured secret.

    The type is preserved exactly: the pipeline's fail-open handling keys
    on the exception type, so only the message changes. When the message
    carries no secret (the common case) the original exception object is
    returned untouched, keeping its traceback identity.
    """
    message = str(error)
    if not secret or secret not in message:
        return error
    try:
        return type(error)(_redact_text(message, secret))
    except Exception:
        # A type whose constructor rejects a plain message cannot be rebuilt
        # with redacted text; the pipeline's fail-open branch still needs the
        # type, so the original escapes rather than a mangled rebuild.
        return error


def _redact(value: object, secret: str) -> object:
    """Replace every occurrence of ``secret`` in a decoded response with a marker.

    <<<<<<< HEAD
        An empty secret (no key configured) returns the value unchanged. The
        walk covers dict keys as well as values, because an endpoint can echo
        the credential back as a key.
    =======
        The walk covers strings, values AND object keys: a hostile or broken
        endpoint that echoes the credential in a key position must not land it
        in the recorded wire (which persists into ``decision_calls`` in the
        run output). Two keys that redact to the same marker would silently
        drop one entry, so collided markers are numbered to keep both. An
        empty secret (no key configured) returns the value unchanged.
    >>>>>>> 38fe623 (fix(assessors): keep both values when redaction markers collide)
    """
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, _REDACTED)
    if isinstance(value, dict):
        redacted: dict = {}
        for key, item in value.items():
            red_key = _redact(key, secret)
            if red_key in redacted:
                # Keep both values: number this marker beyond the first
                # collision-free candidate.
                n = 2
                while f"{red_key}({n})" in redacted:
                    n += 1
                red_key = f"{red_key}({n})"
            redacted[red_key] = _redact(item, secret)
        return redacted
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value
