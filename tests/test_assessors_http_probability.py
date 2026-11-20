"""Tests for the generic HTTP probability assessor.

All network interaction goes through ``httpx.MockTransport``, so no real
endpoint is contacted. The assessor implements the ``Assessor``
protocol from ``extract/screening.py`` and emits ``ProbabilityAnswer`` and
``NoAnswer`` answers.
"""

import json
import time
import warnings

import httpx
import pytest

from asago_policy_mapper.assessors.http_probability import (
    HttpProbabilityAssessor,
    ProbabilityEndpoint,
)
from asago_policy_mapper.extract.screening import (
    DecisionCase,
    NoAnswer,
    ProbabilityAnswer,
    Question,
)

QUESTION = Question(question_id="risk-relevance", version="1", criterion="Is the risk relevant?")


def _cases(*risk_ids: str) -> list[DecisionCase]:
    return [DecisionCase(risk_id=rid, name=f"Risk {rid}", description=f"Description for {rid}.") for rid in risk_ids]


def _endpoint(base_url: str = "http://localhost:9100/v1") -> ProbabilityEndpoint:
    return ProbabilityEndpoint(
        base_url=base_url,
        model="prob-model",
        api_key_env="PROB_API_KEY",
    )


def _scores_body(*entries: tuple[str, object]) -> dict:
    return {"scores": [{"id": rid, "p": p} for rid, p in entries]}


def _assessor(
    handler,
    endpoint: ProbabilityEndpoint | None = None,
) -> HttpProbabilityAssessor:
    transport = httpx.MockTransport(handler)
    return HttpProbabilityAssessor(endpoint or _endpoint(), transport=transport)


def _p_value(answer) -> float | None:
    return answer.probability if isinstance(answer, ProbabilityAnswer) else None


def test_endpoint_repr_and_serialization_exclude_credentials(monkeypatch):
    """The endpoint's repr and dict forms never carry the API key value.

    ``api_key_env`` stores the environment variable NAME, and neither repr
    nor to_dict() reads the variable or includes a key.
    """
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")
    endpoint = _endpoint()
    text = repr(endpoint)
    dumped = endpoint.to_dict()

    assert "secret-value-123" not in text
    assert "secret-value-123" not in json.dumps(dumped)
    assert endpoint.api_key_env == "PROB_API_KEY"
    assert dumped["api_key_env"] == "PROB_API_KEY"
    assert dumped["base_url"] == "http://localhost:9100/v1"
    assert dumped["model"] == "prob-model"
    assert dumped["timeout_s"] == 10.0
    assert dumped["max_batch"] == 32


def test_wire_excludes_api_key(monkeypatch):
    """The wire record carries the exact request body, never credentials.

    The header check reads the lowercased key (httpx.Headers lowercases on
    dict conversion), so it cannot pass by accident of case.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=_scores_body(("R-001", 0.9)))

    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")
    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "policy context", _cases("R-001"))

    assert seen["body"]["context"] == "policy context"
    assert seen["body"]["items"] == [{"id": "R-001", "text": "Risk R-001: Description for R-001."}]
    assert "secret-value-123" not in json.dumps(batch.exchange)
    # The recorded exchange is what was sent, and the header that carried
    # the key went out on the wire but never into the record.
    assert batch.exchange["requests"][0] == seen["body"]
    assert seen["headers"].get("authorization") == "Bearer secret-value-123"
    assert "authorization" not in json.dumps(batch.exchange).lower()
    assert _p_value(batch.answers["R-001"][0]) == 0.9


def test_offsite_flag_follows_host(monkeypatch):
    """sends_offsite is False for loopback hosts and True for anything else."""
    monkeypatch.setenv("PROB_API_KEY", "k")
    assert _assessor(lambda request: httpx.Response(200, json=_scores_body())).sends_offsite is False

    for base in ("http://example.com/v1", "http://10.0.0.5/v1", "http://[::1]:9100/v1"):
        endpoint = _endpoint(base_url=base)
        assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body()), endpoint)
        assert assessor.sends_offsite is (base != "http://[::1]:9100/v1"), base


def test_offsite_without_key_warns(monkeypatch):
    """An offsite endpoint with an unset key variable warns at construction."""
    monkeypatch.delenv("PROB_API_KEY", raising=False)

    with pytest.warns(UserWarning, match="PROB_API_KEY is unset"):
        _assessor(lambda request: httpx.Response(200, json=_scores_body()), _endpoint(base_url="http://example.com/v1"))


def test_offsite_with_key_does_not_warn(monkeypatch):
    """An offsite endpoint with a set key variable constructs silently."""
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _assessor(lambda request: httpx.Response(200, json=_scores_body()), _endpoint(base_url="http://example.com/v1"))


def test_local_without_key_does_not_warn(monkeypatch):
    """A loopback endpoint needs no key and constructs silently."""
    monkeypatch.delenv("PROB_API_KEY", raising=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _assessor(lambda request: httpx.Response(200, json=_scores_body()))


def test_assessor_emits_probability():
    """The assessor declares {Probability} and a stable id naming model and base URL."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body()))

    assert ProbabilityAnswer in assessor.answer_types
    assert "prob-model" in assessor.assessor_id


def test_valid_scores_become_probabilities():
    """Scores within [0, 1] map to ProbabilityAnswer answers, keyed by risk id."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body(("R-001", 0.0), ("R-002", 1.0))))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert _p_value(batch.answers["R-001"][0]) == 0.0
    assert _p_value(batch.answers["R-002"][0]) == 1.0
    assert batch.unmatched_ids == ()
    assert batch.latency_ms >= 0.0


def test_nan_out_of_range_and_bool_are_invalid():
    """Non-finite, out-of-range and boolean values all become NoAnswer("invalid")."""
    for bad, raw in (
        ("NaN", True),
        ("Infinity", True),
        (-0.1, False),
        (1.1, False),
        (True, False),
        (False, False),
    ):

        def handler(request: httpx.Request, raw=raw, bad=bad) -> httpx.Response:
            if raw:
                # NaN and Infinity are valid JSON tokens that Python's json.loads
                # accepts but json.dumps (httpx's json=) refuses, so send raw text.
                text = '{"scores": [{"id": "R-001", "p": ' + bad + "} ]}"
                return httpx.Response(200, content=text.encode(), headers={"Content-Type": "application/json"})
            return httpx.Response(200, json=_scores_body(("R-001", bad)))

        assessor = _assessor(handler)
        batch = assessor.assess(QUESTION, "context", _cases("R-001"))

        answer = batch.answers["R-001"][0]
        assert isinstance(answer, NoAnswer), f"p={bad!r} must be invalid"
        assert answer.reason == "invalid"


def test_conflicting_duplicate_scores_invalid():
    """Two different scores for one id become NoAnswer("invalid")."""
    body = {"scores": [{"id": "R-001", "p": 0.2}, {"id": "R-001", "p": 0.8}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "invalid"


def test_invalid_is_final_for_an_id():
    """Once a row for an id is invalid or conflicting, later rows cannot restore it."""
    for body in (
        {"scores": [{"id": "R-001", "p": 0.2}, {"id": "R-001", "p": 0.8}, {"id": "R-001", "p": 0.2}]},
        {"scores": [{"id": "R-001", "p": 7.5}, {"id": "R-001", "p": 0.5}]},
    ):
        assessor = _assessor(lambda request, body=body: httpx.Response(200, json=body))

        batch = assessor.assess(QUESTION, "context", _cases("R-001"))

        answer = batch.answers["R-001"][0]
        assert isinstance(answer, NoAnswer), body
        assert answer.reason == "invalid"


def test_huge_integer_score_is_invalid():
    """A JSON integer too large for float becomes invalid, not an OverflowError."""
    body = {"scores": [{"id": "R-001", "p": 10**400}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "invalid"


def test_agreeing_duplicate_scores_are_valid():
    """Two identical scores for one id collapse to one ProbabilityAnswer."""
    body = {"scores": [{"id": "R-001", "p": 0.3}, {"id": "R-001", "p": 0.3}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert _p_value(batch.answers["R-001"][0]) == 0.3


def test_partial_response_marks_missing():
    """An id absent from the response becomes NoAnswer("missing") for that candidate only."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body(("R-001", 0.5))))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert _p_value(batch.answers["R-001"][0]) == 0.5
    missing = batch.answers["R-002"][0]
    assert isinstance(missing, NoAnswer)
    assert missing.reason == "missing"


def test_extra_ids_go_to_unmatched():
    """Ids in the response but not in the batch land in unmatched_ids, not outcomes."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body(("R-001", 0.5), ("R-999", 0.9))))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert "R-999" not in batch.answers
    assert batch.unmatched_ids == ("R-999",)


def test_timeout_is_unavailable():
    """A request timeout becomes NoAnswer("timeout") for every candidate."""
    from tests.test_assessors_http_probability import _hang_handler

    assessor = _assessor(_hang_handler)
    endpoint = _endpoint()
    endpoint = ProbabilityEndpoint(
        base_url=endpoint.base_url, model=endpoint.model, api_key_env=endpoint.api_key_env, timeout_s=0.05
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(_hang_handler))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    for rid in ("R-001", "R-002"):
        answer = batch.answers[rid][0]
        assert isinstance(answer, NoAnswer)
        assert answer.reason == "timeout"


def _hang_handler(request: httpx.Request) -> httpx.Response:
    """Simulate a server that never answers by raising a read timeout."""
    raise httpx.ReadTimeout("simulated read timeout", request=request)


def test_transport_error_is_unavailable():
    """A transport error (connection refused) becomes NoAnswer("error")."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assessor = _assessor(handler)

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "error"


def test_http_error_status_is_unavailable():
    """A non-2xx status becomes NoAnswer("error") carrying the HTTP status."""
    assessor = _assessor(lambda request: httpx.Response(500, json={"error": "boom"}))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "error"
    assert answer.detail == "HTTP 500"


def test_http_error_detail_names_the_json_failure():
    """A 200 whose body is not JSON becomes NoAnswer("error") with a JSON detail."""
    assessor = _assessor(
        lambda request: httpx.Response(200, content=b"not json", headers={"Content-Type": "application/json"})
    )

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "error"
    assert answer.detail.startswith("JSON:")


def test_transport_error_detail_names_the_failure():
    """A transport error keeps the exception text in NoAnswer.detail."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assessor = HttpProbabilityAssessor(_endpoint(), transport=httpx.MockTransport(handler))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "error"
    assert "connection refused" in answer.detail


def test_keyboard_interrupt_is_not_caught():
    """KeyboardInterrupt and SystemExit propagate; only operational errors become NoAnswer."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt

    assessor = _assessor(handler)

    with pytest.raises(KeyboardInterrupt):
        assessor.assess(QUESTION, "context", _cases("R-001"))


def test_batches_respect_max_batch():
    """Candidates split into batches of at most max_batch, preserving order."""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        requests.append(body)
        return httpx.Response(200, json=_scores_body(*[(item["id"], 0.5) for item in body["items"]]))

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", max_batch=2
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))

    cases = _cases("R-001", "R-002", "R-003", "R-004", "R-005")
    batch = assessor.assess(QUESTION, "context", cases)

    assert [len(r["items"]) for r in requests] == [2, 2, 1]
    assert [c.risk_id for c in cases] == [f"R-{i:03d}" for i in range(1, 6)]
    assert all(_p_value(batch.answers[f"R-{i:03d}"][0]) == 0.5 for i in range(1, 6))
    assert len(batch.exchange["requests"]) == 3


def test_wire_records_response_and_instructions():
    """The wire record keeps the raw response payload and the context it sent."""
    response_body = _scores_body(("R-001", 0.7))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "policy context", _cases("R-001"))

    assert batch.exchange["responses"][0] == response_body
    assert "policy context" in json.dumps(batch.exchange)


def test_request_body_carries_task_and_items():
    """The request body carries exactly the generic shape: context and items."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_scores_body(("R-001", 0.5)))

    assessor = _assessor(handler)
    assessor.assess(QUESTION, "policy context", _cases("R-001"))

    assert seen["body"] == {
        "context": "policy context",
        "items": [{"id": "R-001", "text": "Risk R-001: Description for R-001."}],
    }


def test_request_sends_json_content_type(monkeypatch):
    """The request goes out with an application/json Content-Type header."""
    content_types: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        content_types.append(request.headers.get("Content-Type", ""))
        return httpx.Response(200, json=_scores_body(("R-001", 0.5)))

    assessor = _assessor(handler)
    assessor.assess(QUESTION, "context", _cases("R-001"))

    assert content_types == ["application/json"]


def test_api_key_sent_as_bearer_when_env_set(monkeypatch):
    """A set env var becomes a Bearer Authorization header on the wire only, never in the record."""
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")
    auth_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth_headers.append(request.headers.get("Authorization"))
        return httpx.Response(200, json=_scores_body(("R-001", 0.5)))

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert auth_headers == ["Bearer secret-value-123"]
    assert "secret-value-123" not in json.dumps(batch.exchange)


def test_no_authorization_header_when_env_unset(monkeypatch):
    """With the env var unset, no Authorization header goes out at all."""
    monkeypatch.delenv("PROB_API_KEY", raising=False)
    auth_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth_headers.append(request.headers.get("Authorization"))
        return httpx.Response(200, json=_scores_body(("R-001", 0.5)))

    assessor = _assessor(handler)
    assessor.assess(QUESTION, "context", _cases("R-001"))

    assert auth_headers == [None]


def test_wire_keeps_request_on_timeout_and_error(monkeypatch):
    """Timeout and error batches record the request and an explicit error object.

    A persisted ``null`` response says nothing; the error object carries
    the reason and, for HTTP statuses, the status code.
    """
    from tests.test_assessors_http_probability import _hang_handler

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", timeout_s=0.05
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(_hang_handler))
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))
    assert batch.exchange["requests"][0]["items"] == [{"id": "R-001", "text": "Risk R-001: Description for R-001."}]
    assert batch.exchange["responses"] == [{"error": "timeout", "status": None}]

    def error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(error_handler))
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))
    assert batch.exchange["requests"][0]["context"] == "context"
    assert batch.exchange["responses"] == [{"error": "error", "status": None}]

    # A non-2xx status is the other "error" path; the error object carries
    # the status so a run can tell a 401 from a 503, and the batch's
    # outcomes read the error reason.
    assessor = HttpProbabilityAssessor(
        endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(503, json={"e": 1}))
    )
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))
    assert batch.exchange["requests"][0]["context"] == "context"
    assert batch.exchange["responses"] == [{"error": "error", "status": 503}]
    assert batch.answers["R-001"][0].reason == "error"

    # A 200 whose body is not JSON is the ValueError path; it records the
    # request and the same error object.
    assessor = HttpProbabilityAssessor(
        endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>not json</html>"))
    )
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))
    assert batch.exchange["requests"][0]["context"] == "context"
    assert batch.exchange["responses"] == [{"error": "error", "status": 200}]
    assert batch.answers["R-001"][0].reason == "error"


def test_wire_keeps_request_and_response_on_non_object_payload():
    """A non-object JSON payload still records both the request and the raw payload."""
    assessor = _assessor(lambda request: httpx.Response(200, json=[1, 2, 3]))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert batch.exchange["requests"][0]["context"] == "context"
    assert batch.exchange["responses"] == [[1, 2, 3]]


def test_invalid_row_does_not_stop_later_rows():
    """An invalid score for one id does not stop the loop; later ids still score."""
    body = {"scores": [{"id": "R-001", "p": 7.5}, {"id": "R-002", "p": 0.5}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert isinstance(batch.answers["R-001"][0], NoAnswer)
    assert _p_value(batch.answers["R-002"][0]) == 0.5


def test_conflict_for_one_id_does_not_stop_later_rows():
    """A conflicting duplicate for one id does not stop the loop; later ids still score."""
    body = {"scores": [{"id": "R-001", "p": 0.2}, {"id": "R-001", "p": 0.8}, {"id": "R-002", "p": 0.4}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert isinstance(batch.answers["R-001"][0], NoAnswer)
    assert _p_value(batch.answers["R-002"][0]) == 0.4


def test_latency_accumulates_across_batches():
    """latency_ms sums across sub-batch requests, and is zero for no cases."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body(("R-001", 0.5))))
    empty = assessor.assess(QUESTION, "context", [])
    assert empty.latency_ms == 0.0

    sleeps = [0.01, 0.02]

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(sleeps.pop(0))
        body = json.loads(request.content.decode())
        return httpx.Response(200, json=_scores_body(*[(item["id"], 0.5) for item in body["items"]]))

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", max_batch=1
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    # The two batches slept 10 ms and 20 ms; accumulated latency stays near that
    # sum (upper bound generous: this pins summation, not wall-clock precision).
    assert 29.0 <= batch.latency_ms <= 2000.0


def test_latency_sums_request_durations(monkeypatch):
    """latency_ms is the sum of per-request durations in milliseconds, exactly."""
    import asago_policy_mapper.assessors.http_probability as module

    class _Clock:
        """Serves the module a scripted clock while other callers get the real one."""

        def __init__(self):
            self.values = iter([0.0, 0.5, 0.0, 0.25])

        def perf_counter(self):
            return next(self.values)

    monkeypatch.setattr(module, "time", _Clock())

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        return httpx.Response(200, json=_scores_body(*[(item["id"], 0.5) for item in body["items"]]))

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", max_batch=1
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert batch.latency_ms == 750.0


def test_outcomes_accumulate_across_batches():
    """Outcomes from later sub-batches merge with earlier ones under their ids."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        scores = [(item["id"], 0.2 if item["id"] == "R-001" else 0.9) for item in body["items"]]
        return httpx.Response(200, json=_scores_body(*scores))

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", max_batch=1
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert _p_value(batch.answers["R-001"][0]) == 0.2
    assert _p_value(batch.answers["R-002"][0]) == 0.9


def test_duplicate_ids_across_batches_keep_all_outcomes():
    """The same id in two sub-batches yields both outcomes, not just the latest."""
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        ids = [item["id"] for item in body["items"]]
        calls.append(ids)
        return httpx.Response(200, json=_scores_body(*[(i, 0.3 if len(calls) == 1 else 0.6) for i in ids]))

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", max_batch=1
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    batch = assessor.assess(
        QUESTION,
        "context",
        [
            DecisionCase(risk_id="R-001", name="Risk R-001", description="d1."),
            DecisionCase(risk_id="R-001", name="Risk R-001", description="d2."),
        ],
    )

    assert calls == [["R-001"], ["R-001"]]
    assert len(batch.answers["R-001"]) == 2
    assert _p_value(batch.answers["R-001"][0]) == 0.3
    assert _p_value(batch.answers["R-001"][1]) == 0.6


def test_min_batch_size_is_one():
    """A max_batch below 1 still sends every case, one per request at least."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        return httpx.Response(200, json=_scores_body(*[(item["id"], 0.5) for item in body["items"]]))

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", max_batch=0
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert set(batch.answers) == {"R-001", "R-002"}


def test_empty_cases_return_empty_batch():
    """No cases means an empty batch with empty wire, not a request."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body()))

    batch = assessor.assess(QUESTION, "context", [])

    assert batch.answers == {}
    assert batch.unmatched_ids == ()
    assert batch.exchange == {"requests": [], "responses": []}


def test_empty_batch_exchange_is_not_shared_between_results():
    """Appending to one empty batch's exchange leaves the next empty batch clean."""
    assessor = _assessor(lambda request: httpx.Response(200, json=_scores_body()))

    first = assessor.assess(QUESTION, "context", [])
    first.exchange["requests"].append("not a real request")
    first.exchange["responses"].append("not a real response")

    second = assessor.assess(QUESTION, "context", [])

    assert second.exchange == {"requests": [], "responses": []}


def test_scores_after_unmatched_row_still_apply():
    """A non-dict row in the scores list is skipped, and later rows still score."""
    body = {"scores": ["garbage", {"id": "R-001", "p": 0.6}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert _p_value(batch.answers["R-001"][0]) == 0.6


def test_row_with_non_string_id_is_ignored():
    """A row whose id is not a string is dropped, not sent to unmatched_ids."""
    body = {"scores": [{"id": 42, "p": 0.6}, {"id": "R-001", "p": 0.6}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert batch.unmatched_ids == ()
    assert _p_value(batch.answers["R-001"][0]) == 0.6


def test_scores_key_missing_means_all_missing():
    """A response without a scores key marks every candidate missing."""
    assessor = _assessor(lambda request: httpx.Response(200, json={"unexpected": True}))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "missing"


def test_non_object_payload_is_error():
    """A JSON body that is not an object becomes NoAnswer("error")."""
    assessor = _assessor(lambda request: httpx.Response(200, json=[1, 2, 3]))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "error"


def test_scores_null_is_missing():
    """A scores key of null means the same as no scores key: every candidate is missing."""
    text = '{"scores": null}'
    assessor = _assessor(
        lambda request: httpx.Response(200, content=text.encode(), headers={"Content-Type": "application/json"})
    )

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "missing"


def test_scores_object_is_invalid():
    """A scores key holding an object (not a list) marks every candidate invalid."""
    assessor = _assessor(lambda request: httpx.Response(200, json={"scores": {"R-001": 0.5}}))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    answer = batch.answers["R-001"][0]
    assert isinstance(answer, NoAnswer)
    assert answer.reason == "invalid"


def test_unhashable_row_id_is_ignored():
    """A row whose id is a list is dropped without raising."""
    body = {"scores": [{"id": ["x"], "p": 0.6}, {"id": "R-001", "p": 0.6}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert _p_value(batch.answers["R-001"][0]) == 0.6


def test_response_echoing_the_api_key_is_redacted(monkeypatch):
    """An endpoint that echoes the credential back gets it redacted from the wire."""
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"debug": {"authorization": "Bearer secret-value-123"}, "scores": [{"id": "R-001", "p": 0.5}]},
        )

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert "secret-value-123" not in json.dumps(batch.exchange)
    assert "[redacted]" in json.dumps(batch.exchange)
    assert _p_value(batch.answers["R-001"][0]) == 0.5


def test_response_echoing_the_key_in_a_dict_key_is_redacted(monkeypatch):
    """An endpoint that echoes the credential as an object KEY gets it redacted.

    The wire's responses persist into ``decision_calls`` in the run
    output, so no position in a recorded response — value or key — may
    carry the credential.
    """
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"echo": {"secret-value-123": "x"}, "scores": [{"id": "R-001", "p": 0.5}]},
        )

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert "secret-value-123" not in json.dumps(batch.exchange)
    assert batch.exchange["responses"][0]["echo"] == {"[redacted]": "x"}
    assert _p_value(batch.answers["R-001"][0]) == 0.5


def test_response_redaction_keeps_both_values_on_key_collision(monkeypatch):
    """A redacted key colliding with an existing key keeps both entries.

    An endpoint that echoes the credential as one key and also carries
    literal ``[redacted]`` keys makes several keys redact to the same
    marker; a plain dict comprehension would silently drop all but the
    last entry's value. The numbered markers keep every value so no data
    is lost while no credential survives.
    """
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "echo": {
                    "secret-value-123": "first",
                    "[redacted]": "second",
                    "[redacted](2)": "third",
                },
                "scores": [{"id": "R-001", "p": 0.5}],
            },
        )

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    echo = batch.exchange["responses"][0]["echo"]
    assert sorted(echo.values()) == ["first", "second", "third"]
    assert len(echo) == 3
    assert "secret-value-123" not in json.dumps(batch.exchange)


def test_response_redaction_collision_numbering_is_stable(monkeypatch):
    """The collision markers number deterministically from the first use.

    Three colliding keys land on distinct markers in insertion order, so
    repeated runs of the same response record identically; a marker's own
    collision numbers within that marker (the second use of
    ``[redacted](2)`` becomes ``[redacted](2)(2)``).
    """
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"echo": {"secret-value-123": "a", "[redacted]": "b", "[redacted](2)": "c"}},
        )

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    echo = batch.exchange["responses"][0]["echo"]
    assert echo == {"[redacted]": "a", "[redacted](2)": "b", "[redacted](2)(2)": "c"}


def test_response_redaction_absent_when_no_key(monkeypatch):
    """With no key configured the response is recorded unchanged, byte for byte."""
    monkeypatch.delenv("PROB_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"id": "R-001", "p": 0.5}], "filler": "XXXX"})

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert batch.exchange["responses"] == [{"scores": [{"id": "R-001", "p": 0.5}], "filler": "XXXX"}]


def test_redaction_walks_lists_and_dicts(monkeypatch):
    """Redaction replaces the secret inside lists, not only top-level strings and dicts."""
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"trail": ["secret-value-123", {"inner": "secret-value-123"}]})

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    dumped = json.dumps(batch.exchange)
    assert "secret-value-123" not in dumped
    assert dumped.count("[redacted]") == 2


def test_redaction_covers_dict_keys(monkeypatch):
    """An endpoint that echoes the credential as a dict KEY gets that key redacted too."""
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"secret-value-123": "value", "scores": [{"id": "R-001", "p": 0.5}]})

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    dumped = json.dumps(batch.exchange)
    assert "secret-value-123" not in dumped
    assert "[redacted]" in dumped
    assert _p_value(batch.answers["R-001"][0]) == 0.5


def test_unmatched_row_after_invalid_id_is_still_unmatched():
    """Skipping an invalid id's later rows does not skip later unmatched rows."""
    body = {
        "scores": [
            {"id": "R-001", "p": 7.5},
            {"id": "R-001", "p": 0.9},
            {"id": "R-999", "p": 0.4},
            {"id": "R-002", "p": 0.4},
        ]
    }
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert isinstance(batch.answers["R-001"][0], NoAnswer)
    assert _p_value(batch.answers["R-002"][0]) == 0.4
    assert batch.unmatched_ids == ("R-999",)


def test_invalid_id_does_not_stop_later_ids():
    """A row for an already-invalid id is skipped, and later ids still score."""
    body = {"scores": [{"id": "R-001", "p": 7.5}, {"id": "R-001", "p": 0.9}, {"id": "R-002", "p": 0.4}]}
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert isinstance(batch.answers["R-001"][0], NoAnswer)
    assert _p_value(batch.answers["R-002"][0]) == 0.4


def test_unmatched_ids_redact_the_api_key(monkeypatch):
    """An id echoed back that matches the credential never reaches unmatched_ids raw.

    Unmatched ids come verbatim from the response, so an endpoint echoing
    the API key (alone or inside a longer id) would otherwise carry it
    through ``AnswerBatch.unmatched_ids`` into the decision log's
    serialized output. The adapter redacts them against the same secret
    the exchange is redacted with.
    """
    monkeypatch.setenv("PROB_API_KEY", "secret-value-123")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _scores_body(("R-001", 0.9))
        body["scores"].insert(0, {"id": "secret-value-123", "p": 0.4})
        body["scores"].append({"id": "prefix-secret-value-123-suffix", "p": 0.2})
        return httpx.Response(200, json=body)

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert "secret-value-123" not in json.dumps(batch.unmatched_ids)
    assert batch.unmatched_ids == ("[redacted]", "prefix-[redacted]-suffix")
    assert _p_value(batch.answers["R-001"][0]) == 0.9
    # The serialized batch as a whole carries no credential.
    assert "secret-value-123" not in json.dumps(
        {
            "unmatched_ids": list(batch.unmatched_ids),
            "exchange": batch.exchange,
            "answers": [a for a in batch.answers.values()],
        },
        default=repr,
    )


def test_unmatched_ids_without_key_pass_through_unchanged(monkeypatch):
    """With no key configured, unmatched ids keep their exact raw value."""
    monkeypatch.delenv("PROB_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_scores_body(("R-999", 0.4), ("R-001", 0.9)))

    assessor = _assessor(handler)
    batch = assessor.assess(QUESTION, "context", _cases("R-001"))

    assert batch.unmatched_ids == ("R-999",)


def test_conflicting_id_does_not_stop_later_ids():
    """A row for an already-conflicting id is skipped, and later ids still score."""
    body = {
        "scores": [
            {"id": "R-001", "p": 0.2},
            {"id": "R-001", "p": 0.8},
            {"id": "R-001", "p": 0.2},
            {"id": "R-002", "p": 0.4},
        ]
    }
    assessor = _assessor(lambda request: httpx.Response(200, json=body))

    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert isinstance(batch.answers["R-001"][0], NoAnswer)
    assert _p_value(batch.answers["R-002"][0]) == 0.4


class _ClosableTransport(httpx.BaseTransport):
    """A transport that records close() calls, for ownership checks."""

    def __init__(self):
        self.closed = False
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(200, json=_scores_body(("R-001", 0.5)))

    def close(self) -> None:
        self.closed = True


def test_injected_transport_is_never_closed():
    """A caller-supplied transport serves every sub-batch and is not closed."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        return httpx.Response(200, json=_scores_body(*[(item["id"], 0.5) for item in body["items"]]))

    transport = _ClosableTransport()
    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", max_batch=1
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=transport)
    batch = assessor.assess(QUESTION, "context", _cases("R-001", "R-002"))

    assert transport.calls == 2
    assert transport.closed is False
    assert set(batch.answers) == {"R-001", "R-002"}
