"""Tests for the Jev mapping over the HTTP probability assessor.

All network interaction goes through ``httpx.MockTransport``, so no real
Jev endpoint is contacted and no real key is needed. The mapping is a
thin subclass of the generic adapter: it translates the decision seam's
``{context, cases}`` input onto Jev's ``/v1/systemone`` wire (one Noul
question per case, keyed by risk id) and reads each answer's ``noul``
probability back as the generic ``{id, p}`` score, keeping 149a's
semantics (no-answer reasons, sends-offsite determination, credentials never
in repr/serialization/wire logs).
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from asago_policy_mapper.assessors.http_probability import (
    _REDACTED,
    HttpProbabilityAssessor,
    ProbabilityEndpoint,
)
from asago_policy_mapper.assessors.jev import JevProbabilityAssessor, JevProbabilityEndpoint
from asago_policy_mapper.extract.screening import (
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
    Question,
)

QUESTION_V1 = Question(question_id="grounding-probability", version="1", criterion="Is the risk worth grounding?")

QUESTION = "Given the policy passage in `state`, does the proposition in `proposition` hold for the risk in `risk`?"

_MODULE = Path("src/asago_policy_mapper/assessors/jev.py")


def _cases(*risk_ids: str) -> list[DecisionCase]:
    return [DecisionCase(risk_id=rid, name=f"Risk {rid}", description=f"Description for {rid}.") for rid in risk_ids]


def _endpoint(**overrides) -> JevProbabilityEndpoint:
    fields: dict = {
        "base_url": "https://api.typesafe.ai",
        "model": "jev-latest",
        "api_key_env": "POLICY_MAPPER_JEV_API_KEY",
        "timeout_s": 30.0,
        "max_batch": 32,
    }
    fields.update(overrides)
    return JevProbabilityEndpoint(**fields)


def _jev_body(*entries: tuple[str, object]) -> dict:
    """A Jev SystemOne response body with one Noul answer per entry."""
    return {
        "model": "jev-1.14.0",
        "answers": {rid: {"type": "noul", "noul": p} for rid, p in entries},
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }


def _assessor(handler, endpoint: JevProbabilityEndpoint | None = None) -> JevProbabilityAssessor:
    return JevProbabilityAssessor(endpoint or _endpoint(), transport=httpx.MockTransport(handler))


def _p_value(answer) -> float | None:
    return answer.probability if isinstance(answer, ProbabilityAnswer) else None


def test_endpoint_defaults_point_at_jev():
    """The endpoint defaults to the Jev URL, model, env var and a 30s timeout."""
    endpoint = JevProbabilityEndpoint()

    assert endpoint.base_url == "https://api.typesafe.ai"
    assert endpoint.model == "jev-latest"
    assert endpoint.api_key_env == "POLICY_MAPPER_JEV_API_KEY"
    assert endpoint.timeout_s == 30.0
    assert endpoint.max_batch == 32


def test_endpoint_overrides_still_work():
    """Every field stays overridable, one at a time or together."""
    endpoint = JevProbabilityEndpoint(model="jev-1.13.0", timeout_s=5.0, max_batch=8)

    assert endpoint.model == "jev-1.13.0"
    assert endpoint.timeout_s == 5.0
    assert endpoint.max_batch == 8
    assert endpoint.base_url == "https://api.typesafe.ai"


def test_endpoint_serialization_excludes_credentials(monkeypatch):
    """The inherited to_dict never reads the env var or carries a key."""
    monkeypatch.setenv("POLICY_MAPPER_JEV_API_KEY", "secret-value-123")
    endpoint = _endpoint()

    text = repr(endpoint)
    dumped = endpoint.to_dict()

    assert "secret-value-123" not in text
    assert "secret-value-123" not in json.dumps(dumped)
    assert dumped["api_key_env"] == "POLICY_MAPPER_JEV_API_KEY"
    assert dumped["base_url"] == "https://api.typesafe.ai"


def test_endpoint_is_a_probability_endpoint():
    """The Jev endpoint subclasses the generic one, so profile wiring keeps working."""
    endpoint = _endpoint()

    assert isinstance(endpoint, ProbabilityEndpoint)


def test_assessor_id_and_offsite():
    """The assessor id names Jev; the real API host is offsite, loopback is not."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_jev_body()))

    assert assessor.assessor_id == "jev:jev-latest"
    assert assessor.sends_offsite is True

    local = _assessor(
        lambda request: httpx.Response(200, json=_jev_body()),
        _endpoint(base_url="http://localhost:9100"),
    )
    assert local.sends_offsite is False


def test_assessor_subclasses_generic():
    """The mapping is a thin subclass of the generic adapter."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_jev_body()))

    assert isinstance(assessor, HttpProbabilityAssessor)
    assert ProbabilityAnswer in assessor.answer_types


def test_posts_to_the_systemone_path():
    """The request goes to Jev's /v1/systemone on the endpoint host."""
    urls: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(request.url)
        return httpx.Response(200, json=_jev_body(("R-001", 0.5)))

    assessor = _assessor(handler)
    assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert str(urls[0]) == "https://api.typesafe.ai/v1/systemone"


def test_request_body_carries_jev_shape():
    """The request is exactly Jev's SystemOne shape: model, state, one Noul per case."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_jev_body(("R-001", 0.5)))

    assessor = _assessor(handler)
    assessor.assess(QUESTION_V1, "policy context", _cases("R-001"))

    assert seen["body"] == {
        "model": "jev-latest",
        "state": "policy context",
        "questions": {
            "R-001": {
                "type": "noul",
                "instructions": {
                    "question": QUESTION,
                    "proposition": "Is the risk worth grounding?",
                    "risk": {"name": "Risk R-001", "description": "Description for R-001."},
                },
            },
        },
    }


def test_one_question_per_case_keyed_by_risk_id():
    """Each case becomes one Noul question keyed by its risk id; ids stay out of the question."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["questions"] = body["questions"]
        return httpx.Response(200, json=_jev_body(("R-001", 0.5), ("R-002", 0.2)))

    assessor = _assessor(handler)
    assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002"))

    assert set(seen["questions"]) == {"R-001", "R-002"}
    for question in seen["questions"].values():
        assert question["type"] == "noul"
        assert set(question["instructions"]) == {"question", "proposition", "risk"}
        assert question["instructions"]["proposition"] == "Is the risk worth grounding?"
    assert seen["questions"]["R-002"]["instructions"]["risk"] == {
        "name": "Risk R-002",
        "description": "Description for R-002.",
    }


def test_task_proposition_travels_per_assess_call():
    """The question's criterion is read per call, so a different question changes the exchange."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode())["questions"])
        return httpx.Response(200, json=_jev_body(("R-001", 0.5)))

    assessor = _assessor(handler)
    other = Question(question_id="other", version="1", criterion="A different proposition.")
    assessor.assess(QUESTION_V1, "context", _cases("R-001"))
    assessor.assess(other, "context", _cases("R-001"))

    assert seen[0]["R-001"]["instructions"]["proposition"] == "Is the risk worth grounding?"
    assert seen[1]["R-001"]["instructions"]["proposition"] == "A different proposition."


def test_is_variant_parent_case_keeps_the_same_shape():
    """A synthetic parent case produces the same question shape; the flag itself is not sent."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_jev_body(("parent-1", 0.5)))

    assessor = _assessor(handler)
    case = DecisionCase(
        risk_id="parent-1", name="Parent risk", description="Parent description.", is_variant_parent=True
    )
    assessor.assess(QUESTION_V1, "context", [case])

    question = seen["body"]["questions"]["parent-1"]
    assert question["instructions"]["risk"] == {"name": "Parent risk", "description": "Parent description."}
    assert "synthetic" not in json.dumps(question)


def test_noul_answers_become_probabilities():
    """Each answer's noul probability maps to a ProbabilityAnswer keyed by risk id."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_jev_body(("R-001", 0.0), ("R-002", 1.0))))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002"))

    assert _p_value(batch.answers["R-001"][0]) == 0.0
    assert _p_value(batch.answers["R-002"][0]) == 1.0


def test_noul_answers_out_of_range_are_invalid():
    """A noul outside [0, 1] is an invalid answer, not a probability."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_jev_body(("R-001", 1.5))))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert isinstance(batch.answers["R-001"][0], NoAnswer)
    assert batch.answers["R-001"][0].reason == "invalid"


def test_negative_noul_is_invalid():
    """A negative noul is invalid."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_jev_body(("R-001", -0.1))))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"


def test_answers_key_missing_means_all_missing():
    """No answers object means no question was answered: every case is missing."""
    assessor = _assessor(lambda request: httpx.Response(200, json={"model": "jev-1.14.0", "usage": {}}))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002"))

    assert all(a.reason == "missing" for answers in batch.answers.values() for a in answers)
    assert set(batch.answers) == {"R-001", "R-002"}


def test_partial_answers_mark_missing():
    """A question the response omitted stays missing, fail-open."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_jev_body(("R-001", 0.9))))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002"))

    assert _p_value(batch.answers["R-001"][0]) == 0.9
    assert batch.answers["R-002"][0].reason == "missing"


def test_extra_answer_ids_go_to_unmatched():
    """Answers for ids outside the batch land in unmatched_ids, ignored elsewhere."""
    body = _jev_body(("R-001", 0.9))
    body["answers"]["R-999"] = {"type": "noul", "noul": 0.4}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.unmatched_ids == ("R-999",)
    assert set(batch.answers) == {"R-001"}


def test_unmatched_answer_before_a_matched_one_does_not_stop_reading():
    """An unmatched id first in the answers object does not stop later ids from scoring."""
    body = {
        "model": "jev-1.14.0",
        "answers": {
            "R-999": {"type": "noul", "noul": 0.4},
            "R-001": {"type": "noul", "noul": 0.7},
            "R-002": {"type": "noul", "noul": 0.2},
        },
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002"))

    assert batch.unmatched_ids == ("R-999",)
    assert _p_value(batch.answers["R-001"][0]) == 0.7
    assert _p_value(batch.answers["R-002"][0]) == 0.2


def test_answer_of_wrong_type_is_invalid():
    """An answer whose type is not noul cannot be read as a probability: invalid."""
    body = _jev_body(("R-001", 0.9))
    body["answers"]["R-001"] = {"type": "choice", "choice": "yes", "confidence": 0.9, "probabilities": {}}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"


def test_answer_without_type_is_invalid():
    """An answer missing its type field cannot be confirmed a noul: invalid."""
    body = _jev_body(("R-001", 0.9))
    body["answers"]["R-001"] = {"noul": 0.5}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"


def test_non_numeric_noul_is_invalid():
    """A noul that is not a number is invalid."""
    body = _jev_body(("R-001", 0.9))
    body["answers"]["R-001"] = {"type": "noul", "noul": "high"}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"


def test_boolean_noul_is_invalid():
    """A noul of true/false is invalid, mirroring the generic score rule."""
    body = _jev_body(("R-001", 0.9))
    body["answers"]["R-001"] = {"type": "noul", "noul": True}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"


def test_non_object_answer_is_invalid():
    """An answer that is not an object cannot be read: invalid for that id."""
    body = _jev_body(("R-001", 0.9))
    body["answers"]["R-001"] = 5

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"


def test_non_object_answers_is_invalid():
    """A non-object answers value cannot be read: every case is invalid."""
    assessor = _assessor(lambda request: httpx.Response(200, json={"model": "m", "answers": [1, 2]}))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"


def test_non_object_payload_is_error():
    """A non-object JSON payload is an error, mirroring the generic adapter."""
    assessor = _assessor(lambda request: httpx.Response(200, json=[1, 2, 3]))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "error"


def test_timeout_is_unavailable_timeout():
    """A timeout marks the batch unavailable with reason timeout."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    endpoint = _endpoint(timeout_s=0.05)
    assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "timeout"
    assert batch.exchange["responses"] == [None]


def test_transport_error_is_unavailable_error():
    """A transport error marks the batch unavailable with reason error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assessor = _assessor(handler)

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "error"
    assert batch.exchange["responses"] == [None]


def test_http_error_status_is_unavailable_error():
    """A non-2xx status (Jev: 401 bad key, 422 validation, 429 rate limit) is an error."""
    for status in (401, 422, 429):
        assessor = _assessor(lambda request, s=status: httpx.Response(s, json={"detail": "no"}))

        batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

        assert batch.answers["R-001"][0].reason == "error", status
        assert batch.exchange["responses"] == [None], status


def test_wire_excludes_api_key(monkeypatch):
    """The wire record carries the Jev request bodies and redacted responses, never the key."""
    monkeypatch.setenv("POLICY_MAPPER_JEV_API_KEY", "secret-value-123")

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_jev_body(("R-001", 0.9)))

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION_V1, "policy context", _cases("R-001"))

    assert seen["headers"] == "Bearer secret-value-123"
    assert "secret-value-123" not in json.dumps(batch.exchange)
    assert batch.exchange["requests"][0]["state"] == "policy context"
    assert batch.exchange["responses"][0]["answers"]["R-001"]["noul"] == 0.9


def test_no_authorization_header_when_env_unset(monkeypatch):
    """With the env var unset, no Authorization header goes out at all."""
    monkeypatch.delenv("POLICY_MAPPER_JEV_API_KEY", raising=False)
    auth_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth_headers.append(request.headers.get("Authorization"))
        return httpx.Response(200, json=_jev_body(("R-001", 0.5)))

    assessor = _assessor(handler)

    batch = assessor.assess(QUESTION_V1, "", _cases("R-001"))

    assert auth_headers == [None]
    assert batch.exchange["requests"][0]["state"] == ""


def test_response_echoing_the_api_key_is_redacted(monkeypatch):
    """A response echoing the key gets it redacted in the wire record."""
    monkeypatch.setenv("POLICY_MAPPER_JEV_API_KEY", "secret-value-123")

    body = _jev_body(("R-001", 0.9))
    body["echo"] = "key=secret-value-123"

    assessor = _assessor(lambda request: httpx.Response(200, json=body))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert "secret-value-123" not in json.dumps(batch.exchange)
    assert batch.exchange["responses"][0]["echo"] == "key=[redacted]"



def test_unmatched_id_echoing_the_api_key_is_redacted(monkeypatch):
    """An unmatched question id carrying the key is redacted, not recorded.

    Unmatched ids leave the response payload and land in the batch's
    ``unmatched_ids`` — which persists into the run output — so a
    hostile or broken echo keyed by the secret must not survive there,
    the same way the recorded response's keys do not.
    """
    monkeypatch.setenv("POLICY_MAPPER_JEV_API_KEY", "secret-value-123")

    body = _jev_body(("R-001", 0.9))
    body["answers"]["secret-value-123"] = {"type": "noul", "noul": 0.4}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert "secret-value-123" not in json.dumps(batch.unmatched_ids)
    assert batch.unmatched_ids == ("[redacted]",)
    # The matched answer is unaffected: the redaction touches only the
    # unmatched id, never the scores.
    assert _p_value(batch.answers["R-001"][0]) == 0.9




def test_unmatched_id_partial_key_echo_is_redacted(monkeypatch):
    """A key embedded inside an unmatched id is replaced in place."""
    monkeypatch.setenv("POLICY_MAPPER_JEV_API_KEY", "secret-value-123")

    body = _jev_body(("R-001", 0.9))
    body["answers"]["prefix-secret-value-123-suffix"] = {"type": "noul", "noul": 0.4}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.unmatched_ids == ("prefix-[redacted]-suffix",)




def test_unmatched_id_with_no_key_configured_stays_exact(monkeypatch):
    """With no key configured there is no secret to redact; ids stay exact."""
    monkeypatch.delenv("POLICY_MAPPER_JEV_API_KEY", raising=False)

    body = _jev_body(("R-001", 0.9))
    body["answers"]["R-999"] = {"type": "noul", "noul": 0.4}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.unmatched_ids == ("R-999",)


def test_wire_keeps_request_on_timeout_and_error():
    """Timeout and error batches still record the exact Jev request body in the wire."""
    endpoint = _endpoint(timeout_s=0.05)

    def hang_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(hang_handler))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))
    assert batch.exchange["requests"][0]["questions"]["R-001"]["type"] == "noul"
    assert batch.exchange["responses"] == [None]

    def error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(error_handler))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))
    assert batch.exchange["requests"][0]["questions"]["R-001"]["type"] == "noul"
    assert batch.exchange["responses"] == [None]


def test_batches_respect_max_batch():
    """Cases split into separate Jev requests at max_batch, boundaries preserved."""
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_bodies.append(body)
        return httpx.Response(200, json=_jev_body(*[(rid, 0.5) for rid in body["questions"]]))

    endpoint = _endpoint(max_batch=2)
    assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002", "R-003"))

    assert len(seen_bodies) == 2
    assert list(seen_bodies[0]["questions"]) == ["R-001", "R-002"]
    assert list(seen_bodies[1]["questions"]) == ["R-003"]
    assert set(batch.answers) == {"R-001", "R-002", "R-003"}
    assert all(_p_value(outcomes[0]) == 0.5 for outcomes in batch.answers.values())
    assert batch.latency_ms >= 0.0


def test_duplicate_risk_ids_never_share_a_request():
    """Two cases with the same risk id travel in separate requests, outcomes accumulate.

    Jev keys questions by the caller's id, so a shared key in one request
    would silently collapse the cases (only one question survives the
    dict). The mapping must keep every case assessable, and the decision
    must not depend on max_batch.
    """
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_bodies.append(body)
        # Score by which description the surviving question carries.
        question = next(iter(body["questions"].values()))
        p = 0.9 if "first" in question["instructions"]["risk"]["description"] else 0.01
        return httpx.Response(200, json=_jev_body((next(iter(body["questions"])), p)))

    cases = [
        DecisionCase(risk_id="R-1", name="Risk one", description="the first case"),
        DecisionCase(risk_id="R-2", name="Risk two", description="unrelated"),
        DecisionCase(risk_id="R-1", name="Risk one again", description="the second case"),
    ]
    for max_batch in (1, 2, 32):
        seen_bodies.clear()
        endpoint = _endpoint(max_batch=max_batch)
        assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
        batch = assessor.assess(QUESTION_V1, "context", cases)

        # Every request holds at most one question per key.
        assert all(len(body["questions"]) == len(set(body["questions"])) for body in seen_bodies)
        # Both R-1 answers come back, whatever the batch size.
        assert len(batch.answers["R-1"]) == 2, f"max_batch={max_batch}"
        assert sorted(p for o in batch.answers["R-1"] for p in [_p_value(o)]) == [0.01, 0.9]


def test_duplicate_risk_ids_across_batches_keep_outcomes():
    """Duplicate ids accumulate one outcome per case across requests, like the generic adapter."""
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        keys = list(body["questions"])
        seen.append(keys)
        return httpx.Response(200, json=_jev_body((keys[0], 0.7)))

    endpoint = _endpoint(max_batch=1)
    assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-001"))

    assert seen == [["R-001"], ["R-001"]]
    assert len(batch.answers["R-001"]) == 2
    assert all(_p_value(o) == 0.7 for o in batch.answers["R-001"])


def test_batching_preserves_order_for_distinct_ids():
    """With distinct ids the split is exactly the base adapter's contiguous slicing."""
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        keys = list(body["questions"])
        seen.append(keys)
        return httpx.Response(200, json=_jev_body(*[(k, 0.5) for k in keys]))

    endpoint = _endpoint(max_batch=2)
    assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002", "R-003", "R-004"))

    assert seen == [["R-001", "R-002"], ["R-003", "R-004"]]


@pytest.mark.parametrize("max_batch", [0, -3, 1])
def test_min_batch_size_is_one(max_batch):
    """A max_batch below 1 still yields one case per request, mirroring the base adapter."""
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        keys = list(body["questions"])
        seen.append(keys)
        return httpx.Response(200, json=_jev_body(*[(k, 0.5) for k in keys]))

    endpoint = _endpoint(max_batch=max_batch)
    assessor = JevProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    assessor.assess(QUESTION_V1, "context", _cases("R-001", "R-002"))

    assert seen == [["R-001"], ["R-002"]]


def test_response_echoing_the_key_in_a_dict_key_is_redacted(monkeypatch):
    """A response echoing the key as an object KEY gets it redacted in the wire record."""
    monkeypatch.setenv("POLICY_MAPPER_JEV_API_KEY", "secret-value-123")

    body = _jev_body(("R-001", 0.9))
    body["debug"] = {"secret-value-123": "echo"}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert "secret-value-123" not in json.dumps(batch.exchange)
    assert batch.exchange["responses"][0]["debug"] == {_REDACTED: "echo"}


def test_response_echoing_the_key_in_a_list_is_redacted(monkeypatch):
    """A response echoing the key inside a list value gets it redacted."""
    monkeypatch.setenv("POLICY_MAPPER_JEV_API_KEY", "secret-value-123")

    body = _jev_body(("R-001", 0.9))
    body["traces"] = ["before", "key=secret-value-123", {"secret-value-123": "nested"}]

    assessor = _assessor(lambda request: httpx.Response(200, json=body))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert "secret-value-123" not in json.dumps(batch.exchange)
    assert batch.exchange["responses"][0]["traces"] == ["before", f"key={_REDACTED}", {_REDACTED: "nested"}]


def test_response_unchanged_when_no_key_configured(monkeypatch):
    """With no key configured, the recorded response is the raw payload, redaction a no-op."""
    monkeypatch.delenv("POLICY_MAPPER_JEV_API_KEY", raising=False)

    body = _jev_body(("R-001", 0.9))
    body["debug"] = {"XXXX": "placeholder", "traces": ["XXXX"]}

    assessor = _assessor(lambda request: httpx.Response(200, json=body))
    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.exchange["responses"][0] == body


def test_empty_cases_return_empty_batch():
    """No cases, no requests: an empty batch with an empty wire."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_jev_body())

    assessor = _assessor(handler)

    batch = assessor.assess(QUESTION_V1, "context", [])

    assert batch.answers == {}
    assert seen == []
    assert batch.exchange == {"requests": [], "responses": []}


def test_prefilter_composition():
    """The Jev assessor works through the prefilter seam: DropThreshold over its answers."""
    from asago_policy_mapper.extract.pipeline import GROUNDING_LIKELIHOOD_QUESTION_V1, GroundingPrefilter
    from asago_policy_mapper.extract.screening import DropThreshold

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_jev_body(("R-001", 0.02), ("R-002", 0.9)))

    assessor = _assessor(handler)
    rule = DropThreshold(assessor, threshold=0.10)
    prefilter = GroundingPrefilter(assessor=assessor, rule=rule)
    cases = _cases("R-001", "R-002")
    assessment = prefilter.assessor.assess(GROUNDING_LIKELIHOOD_QUESTION_V1, "context", cases)
    decisions = prefilter.rule.decide(cases, assessment)

    kept = {d.risk_id for d in decisions if d.keep}
    dropped = {d.risk_id for d in decisions if not d.keep}
    assert kept == {"R-002"}
    assert dropped == {"R-001"}


def test_jev_module_has_no_module_level_httpx_import():
    """The Jev module's own import graph adds no httpx, mirroring the base contract."""
    tree = ast.parse(_MODULE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            assert "httpx" not in names, f"module-level import of httpx: {names}"
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "httpx":
            raise AssertionError(f"module-level import from httpx: {node.module}")


def test_jev_import_stays_lazy():
    """Importing the assessors package pulls in neither assessor module."""
    code = (
        "import sys\n"
        "import asago_policy_mapper.assessors\n"
        "assert 'asago_policy_mapper.assessors.jev' not in sys.modules\n"
        "assert 'asago_policy_mapper.assessors.http_probability' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("missing", ["noul", "type"])
def test_answer_missing_fields_is_invalid(missing):
    """An answer missing either its type or its noul value is invalid."""
    body = _jev_body(("R-001", 0.9))
    del body["answers"]["R-001"][missing]

    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION_V1, "context", _cases("R-001"))

    assert batch.answers["R-001"][0].reason == "invalid"
