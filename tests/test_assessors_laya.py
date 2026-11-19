"""Tests for the Laya local-arm assessor.

All tests stub the ``laya_mlx`` module in ``sys.modules`` — no checkpoint
download, no MLX import at collection time. The assessor implements the
``BinaryAssessor`` protocol from ``extract/decide.py`` and emits
``Probability`` and ``Unavailable`` outcomes, mirroring the Jev mapping's
semantics: one noul question per case keyed by risk id, ``answers[qid]``
read as the probability, fail-open on errors.
"""

import json
import sys
import types

import pytest

from asago_policy_mapper.extract.decide import (
    BinaryTask,
    DecisionCase,
    Probability,
)

TASK = BinaryTask(task_id="grounding-probability", version="1", proposition="Is the risk worth grounding?")


def _cases(*risk_ids: str) -> list[DecisionCase]:
    return [DecisionCase(risk_id=rid, name=f"Risk {rid}", description=f"Description for {rid}.") for rid in risk_ids]


class _StubAnswer(dict):
    """A laya noul answer with attribute-free dict access, like the real one."""


def _laya_result(*entries: tuple[str, object], truncated: bool = False, dropped: int = 0) -> dict:
    """A laya predict-style result with one noul answer per entry."""
    return {
        "model": "laya-rl-agent",
        "answers": {rid: {"type": "noul", "noul": p, "confidence": p, "answer_confidence": p} for rid, p in entries},
        "usage": {
            "input_tokens": 85,
            "output_tokens": 0,
            "state_tokens": 10,
            "state_tokens_dropped": dropped,
            "truncated": truncated,
            "truncated_questions": [],
        },
    }


class _StubAgent:
    """A stand-in for laya_mlx's Agent, recording predicts."""

    def __init__(self, results=None, error=None):
        self.predicts: list[tuple[object, dict]] = []
        self._results = results or []
        self._error = error

    def predict(self, state, questions):
        self.predicts.append((state, dict(questions)))
        if self._error is not None:
            raise self._error
        if len(self._results) == 1:
            return self._results[0]
        return self._results.pop(0) if self._results else _laya_result()


def _install_stub(monkeypatch, agent: _StubAgent | None = None, load_error: Exception | None = None) -> _StubAgent:
    """Install a stub laya_mlx module; the assessor's lazy import picks it up."""
    module = types.ModuleType("laya_mlx")

    def _load(model_id_or_path="convaiinnovations/laya", **kwargs):
        if load_error is not None:
            raise load_error
        if agent is None:
            return _StubAgent()
        return agent

    module.load = _load
    monkeypatch.setitem(sys.modules, "laya_mlx", module)
    return agent or _StubAgent()


def _endpoint(**overrides):
    from asago_policy_mapper.assessors.laya import LayaProbabilityEndpoint

    fields: dict = {}
    fields.update(overrides)
    return LayaProbabilityEndpoint(**fields)


def _assessor(agent: _StubAgent, monkeypatch, **overrides):
    from asago_policy_mapper.assessors.laya import LayaProbabilityAssessor

    _install_stub(monkeypatch, agent)
    return LayaProbabilityAssessor(_endpoint(**overrides))


def _p_value(outcome) -> float | None:
    return outcome.p if isinstance(outcome, Probability) else None


# --- endpoint ---


def test_endpoint_defaults():
    """The endpoint defaults to the multilingual checkpoint and in-process shape."""
    endpoint = _endpoint()

    assert endpoint.model == "aac6fef/laya-multilingual-mlx"
    assert endpoint.base_url == "http://localhost/laya"
    assert endpoint.api_key_env == "POLICY_MAPPER_LAYA_API_KEY"
    assert endpoint.timeout_s == 30.0
    assert endpoint.max_batch == 16


def test_endpoint_is_a_probability_endpoint():
    """The Laya endpoint subclasses the generic one for metadata compatibility."""
    from asago_policy_mapper.assessors.http_probability import ProbabilityEndpoint

    assert isinstance(_endpoint(), ProbabilityEndpoint)


def test_endpoint_overrides():
    """Every field stays overridable."""
    endpoint = _endpoint(model="aac6fef/laya-mlx", max_batch=8, timeout_s=5.0)

    assert endpoint.model == "aac6fef/laya-mlx"
    assert endpoint.max_batch == 8
    assert endpoint.timeout_s == 5.0


def test_endpoint_serialization_excludes_credentials(monkeypatch):
    """repr and to_dict never carry a key value (there is none to carry)."""
    monkeypatch.setenv("POLICY_MAPPER_LAYA_API_KEY", "secret-value-123")
    endpoint = _endpoint()

    assert "secret-value-123" not in repr(endpoint)
    assert "secret-value-123" not in json.dumps(endpoint.to_dict())


# --- assessor basics ---


def test_assessor_id_and_offsite(monkeypatch):
    """The id names laya and the checkpoint; in-process is never offsite."""
    agent = _StubAgent()
    assessor = _assessor(agent, monkeypatch)

    assert assessor.assessor_id == "laya:aac6fef/laya-multilingual-mlx"
    assert assessor.offsite is False
    assert Probability in assessor.emits


def test_injected_agent_is_the_one_used(monkeypatch):
    """An explicitly injected agent serves predict calls; the default load is bypassed."""
    agent = _StubAgent([_laya_result(("R-001", 0.5))])

    from asago_policy_mapper.assessors.laya import LayaProbabilityAssessor

    def _no_load(*args, **kwargs):
        raise AssertionError("laya_mlx.load must not run when an agent is injected")

    module = types.ModuleType("laya_mlx")
    module.load = _no_load
    monkeypatch.setitem(sys.modules, "laya_mlx", module)

    assessor = LayaProbabilityAssessor(_endpoint(), agent=agent)
    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert len(agent.predicts) == 1
    assert _p_value(batch.outcomes["R-001"][0]) == 0.5


def test_default_load_uses_the_endpoint_model(monkeypatch):
    """Without an injected agent, laya_mlx.load runs with the endpoint's checkpoint."""
    from asago_policy_mapper.assessors.laya import LayaProbabilityAssessor

    loaded: list = []
    module = types.ModuleType("laya_mlx")
    module.load = lambda model_id_or_path, **kwargs: loaded.append(model_id_or_path) or _StubAgent()
    monkeypatch.setitem(sys.modules, "laya_mlx", module)

    endpoint = _endpoint(model="aac6fef/laya-mlx")
    LayaProbabilityAssessor(endpoint)

    assert loaded == ["aac6fef/laya-mlx"]


def test_import_error_names_the_install(monkeypatch):
    """A missing laya_mlx raises ImportError with the ad-hoc install hint."""
    import builtins

    real_import = builtins.__import__

    def _no_laya(name, *args, **kwargs):
        if name == "laya_mlx":
            raise ImportError("No module named 'laya_mlx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_laya)

    from asago_policy_mapper.assessors.laya import LayaProbabilityAssessor

    with pytest.raises(ImportError, match="laya-mlx"):
        LayaProbabilityAssessor(_endpoint())


def test_empty_cases_return_empty_batch(monkeypatch):
    """No cases, no predict calls: an empty batch with an empty wire."""
    agent = _StubAgent()
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", [])

    assert batch.outcomes == {}
    assert batch.unmatched_ids == ()
    assert batch.latency_ms == 0.0
    assert agent.predicts == []
    assert batch.wire == {"requests": [], "responses": []}


def test_latency_positive_after_a_predict(monkeypatch):
    """A completed assess records a positive latency (perf_counter deltas)."""
    agent = _StubAgent([_laya_result(("R-001", 0.5))])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.latency_ms > 0.0


def test_latency_accumulates_across_batches(monkeypatch):
    """Latency sums across predict calls under a fake clock: 2 batches = 2 deltas."""
    import asago_policy_mapper.assessors.laya as laya_module

    class _Clock:
        def __init__(self):
            self.now = 100.0

        def perf_counter(self):
            self.now += 1.0
            return self.now

    clock = _Clock()
    monkeypatch.setattr(laya_module.time, "perf_counter", clock.perf_counter)
    agent = _StubAgent([_laya_result(("R-001", 0.1)), _laya_result(("R-002", 0.2))])
    assessor = _assessor(agent, monkeypatch, max_batch=1)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    # Each batch consumes two clock ticks (start and end): 1000 ms each.
    assert batch.latency_ms == 2000.0


def test_duplicate_ids_accumulate_outcomes_across_batches(monkeypatch):
    """Two cases with one id each get their own outcome, accumulated in order."""
    agent = _StubAgent([_laya_result(("R-1", 0.1)), _laya_result(("R-1", 0.9))])
    assessor = _assessor(agent, monkeypatch, max_batch=32)

    cases = [
        DecisionCase(risk_id="R-1", name="One", description="first"),
        DecisionCase(risk_id="R-1", name="Two", description="second"),
    ]
    batch = assessor.assess(TASK, "context", cases)

    assert [_p_value(o) for o in batch.outcomes["R-1"]] == [0.1, 0.9]


def test_one_predict_per_batch_with_noul_questions(monkeypatch):
    """Each batch is one predict call: state verbatim, one noul per case keyed by risk id."""
    agent = _StubAgent([_laya_result(("R-001", 0.5))])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "policy context", _cases("R-001"))

    assert len(agent.predicts) == 1
    state, questions = agent.predicts[0]
    assert state == "policy context"
    assert set(questions) == {"R-001"}
    question = questions["R-001"]
    assert question["type"] == "noul"
    instructions = question["instructions"]
    assert isinstance(instructions, str)
    assert "Is the risk worth grounding?" in instructions
    assert "Risk R-001" in instructions
    assert "Description for R-001." in instructions
    assert _p_value(batch.outcomes["R-001"][0]) == 0.5


def test_batching_respects_max_batch(monkeypatch):
    """Cases split into separate predict calls at max_batch, boundaries preserved."""
    agent = _StubAgent()
    assessor = _assessor(agent, monkeypatch, max_batch=2)

    assessor.assess(TASK, "context", _cases("R-001", "R-002", "R-003"))

    assert [sorted(q) for _, q in agent.predicts] == [["R-001", "R-002"], ["R-003"]]


def test_duplicate_risk_ids_never_share_a_request(monkeypatch):
    """Two cases with the same risk id travel in separate predict calls.

    Laya keys answers by question id, so a shared key in one predict would
    silently collapse the cases (only one question survives the dict).
    """
    agent = _StubAgent()
    assessor = _assessor(agent, monkeypatch)

    cases = [
        DecisionCase(risk_id="R-1", name="One", description="first"),
        DecisionCase(risk_id="R-2", name="Two", description="other"),
        DecisionCase(risk_id="R-1", name="One again", description="second"),
    ]
    assessor.assess(TASK, "context", cases)

    for _, questions in agent.predicts:
        assert len(questions) == len(set(questions))
    assert sum(len(q) for _, q in agent.predicts) == 3


def test_batching_preserves_order_for_distinct_ids(monkeypatch):
    """With distinct ids the split is contiguous slicing in input order."""
    agent = _StubAgent()
    assessor = _assessor(agent, monkeypatch, max_batch=2)

    assessor.assess(TASK, "context", _cases("R-001", "R-002", "R-003", "R-004"))

    assert [list(q) for _, q in agent.predicts] == [["R-001", "R-002"], ["R-003", "R-004"]]


def test_min_batch_size_is_one(monkeypatch):
    """A max_batch below 1 still yields one case per predict call."""
    agent = _StubAgent()
    assessor = _assessor(agent, monkeypatch, max_batch=0)

    assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    assert [list(q) for _, q in agent.predicts] == [["R-001"], ["R-002"]]


# --- answer parsing ---


def test_valid_noul_answers_become_probabilities(monkeypatch):
    """noul values in [0, 1] map to Probability outcomes keyed by risk id."""
    agent = _StubAgent([_laya_result(("R-001", 0.0), ("R-002", 1.0))])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    assert _p_value(batch.outcomes["R-001"][0]) == 0.0
    assert _p_value(batch.outcomes["R-002"][0]) == 1.0


def test_out_of_range_noul_is_invalid(monkeypatch):
    """A noul outside [0, 1] is invalid, not a probability."""
    agent = _StubAgent([_laya_result(("R-001", 1.5))])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.outcomes["R-001"][0].reason == "invalid"
    # Readability outcomes carry no detail unless there is one to carry.
    assert batch.outcomes["R-001"][0].detail == ""


def test_result_without_answers_marks_missing_with_empty_detail(monkeypatch):
    """A result with no answers key marks every case missing, detail empty.

    ``_mark_all`` defaults its detail to the empty string: readability
    outcomes say only their reason unless there is a detail to carry.
    """
    agent = _StubAgent([{"model": "laya-rl-agent"}])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    assert batch.outcomes["R-001"][0].reason == "missing"
    assert batch.outcomes["R-001"][0].detail == ""
    assert batch.outcomes["R-002"][0].reason == "missing"
    assert batch.outcomes["R-002"][0].detail == ""


def test_boolean_and_non_numeric_noul_are_invalid(monkeypatch):
    """bools, strings and missing noul fields read as invalid."""
    for bad in (True, "high", None):
        result = _laya_result()
        result["answers"]["R-001"] = {"type": "noul", "noul": bad}
        agent = _StubAgent([result])
        assessor = _assessor(agent, monkeypatch)

        batch = assessor.assess(TASK, "context", _cases("R-001"))

        assert batch.outcomes["R-001"][0].reason == "invalid", repr(bad)


def test_wrong_answer_type_is_invalid(monkeypatch):
    """An answer whose type is not noul cannot be read as a probability."""
    result = _laya_result()
    result["answers"]["R-001"] = {"type": "choice", "choice": "yes", "probabilities": {}}
    agent = _StubAgent([result])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.outcomes["R-001"][0].reason == "invalid"


def test_partial_answers_mark_missing(monkeypatch):
    """A question the result omitted stays missing, fail-open."""
    agent = _StubAgent([_laya_result(("R-001", 0.9))])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    assert _p_value(batch.outcomes["R-001"][0]) == 0.9
    assert batch.outcomes["R-002"][0].reason == "missing"


def test_missing_answers_key_means_all_missing(monkeypatch):
    """No answers object at all: every case is missing."""
    agent = _StubAgent([{"model": "laya-rl-agent", "usage": {}}])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    assert all(o.reason == "missing" for outcomes in batch.outcomes.values() for o in outcomes)


def test_extra_answer_ids_go_to_unmatched(monkeypatch):
    """Answers for ids outside the batch land in unmatched_ids, ignored elsewhere."""
    result = _laya_result(("R-001", 0.9))
    result["answers"]["R-999"] = {"type": "noul", "noul": 0.4}
    agent = _StubAgent([result])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.unmatched_ids == ("R-999",)
    assert set(batch.outcomes) == {"R-001"}


def test_unmatched_answer_before_a_matched_one_does_not_stop_reading(monkeypatch):
    """An unmatched id first in the answers does not stop later ids from scoring."""
    result = {
        "model": "laya-rl-agent",
        "answers": {
            "R-999": {"type": "noul", "noul": 0.4},
            "R-001": {"type": "noul", "noul": 0.7},
            "R-002": {"type": "noul", "noul": 0.2},
        },
        "usage": {},
    }
    agent = _StubAgent([result])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    assert batch.unmatched_ids == ("R-999",)
    assert _p_value(batch.outcomes["R-001"][0]) == 0.7
    assert _p_value(batch.outcomes["R-002"][0]) == 0.2


def test_non_dict_answer_is_invalid(monkeypatch):
    """An answer that is not an object cannot be read: invalid for that id."""
    result = _laya_result()
    result["answers"]["R-001"] = 5
    agent = _StubAgent([result])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.outcomes["R-001"][0].reason == "invalid"


def test_laya_exception_is_error_fail_open(monkeypatch):
    """Any laya exception marks the batch Unavailable('error'); nothing raises."""
    agent = _StubAgent(error=RuntimeError("mlx blew up"))
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-002"))

    assert all(o.reason == "error" for outcomes in batch.outcomes.values() for o in outcomes)


def test_non_dict_result_is_error(monkeypatch):
    """A result that is not a dict cannot be read: error for the batch."""
    agent = _StubAgent([None])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.outcomes["R-001"][0].reason == "error"


def test_non_dict_answers_is_invalid(monkeypatch):
    """A non-dict answers value: every case is invalid."""
    agent = _StubAgent([{"model": "m", "answers": [1, 2], "usage": {}}])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.outcomes["R-001"][0].reason == "invalid"


# --- wire record ---


def test_wire_records_request_answers_and_usage(monkeypatch):
    """The wire record carries the request shape, the answers and the full usage block."""
    from asago_policy_mapper.assessors.replay import context_sha256

    result = _laya_result(("R-001", 0.7), truncated=True, dropped=263)
    agent = _StubAgent([result])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "policy context", _cases("R-001"))

    request = batch.wire["requests"][0]
    assert request["model"] == "aac6fef/laya-multilingual-mlx"
    # The state travels as a digest, never verbatim: the request record
    # identifies the grounding context without copying its whole text.
    assert request["state_sha256"] == context_sha256("policy context")
    assert request["state_chars"] == len("policy context")
    assert request["summary"] is True
    assert request["questions"]["R-001"]["type"] == "noul"
    response = batch.wire["responses"][0]
    assert response["answers"]["R-001"]["noul"] == 0.7
    assert response["usage"]["state_tokens_dropped"] == 263
    assert response["usage"]["truncated"] is True


def test_wire_keeps_request_on_error(monkeypatch):
    """A failed predict records the request and an explicit error object.

    A persisted ``null`` response says nothing; the error object carries
    the reason and the exception text so a run can explain its bypasses.
    """
    agent = _StubAgent(error=RuntimeError("boom"))
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.wire["requests"][0]["questions"]["R-001"]["type"] == "noul"
    assert batch.wire["responses"] == [{"error": "error", "status": None, "detail": "boom"}]


def test_unavailable_error_carries_the_exception_text(monkeypatch):
    """A laya exception lands in Unavailable.detail, not just the reason."""
    agent = _StubAgent(error=RuntimeError("weight load failed"))
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    outcome = batch.outcomes["R-001"][0]
    assert outcome.reason == "error"
    assert outcome.detail == "weight load failed"


def test_wire_accumulates_across_batches(monkeypatch):
    """Multiple predict calls record one request and one response each, in order."""
    agent = _StubAgent([_laya_result(("R-001", 0.1)), _laya_result(("R-003", 0.9))])
    assessor = _assessor(agent, monkeypatch, max_batch=1)

    batch = assessor.assess(TASK, "context", _cases("R-001", "R-003"))

    assert [list(r["questions"]) for r in batch.wire["requests"]] == [["R-001"], ["R-003"]]
    assert [
        r["answers"]["R-001"]["noul"] if "R-001" in r["answers"] else r["answers"]["R-003"]["noul"]
        for r in batch.wire["responses"]
    ] == [0.1, 0.9]
    assert batch.latency_ms >= 0.0


def test_truncation_visible_in_wire(monkeypatch):
    """state_tokens_dropped lands in the wire so the pilot can measure context loss."""
    result = _laya_result(("R-001", 0.5), truncated=True, dropped=831)
    agent = _StubAgent([result])
    assessor = _assessor(agent, monkeypatch)

    batch = assessor.assess(TASK, "context", _cases("R-001"))

    assert batch.wire["responses"][0]["usage"]["state_tokens_dropped"] == 831


# --- composition ---


def test_prefilter_composition(monkeypatch):
    """The Laya assessor works through the prefilter seam: ThresholdDrop over its outcomes."""
    from asago_policy_mapper.extract.gates import ThresholdDrop
    from asago_policy_mapper.extract.pipeline import PREFILTER_TASK, Prefilter

    agent = _StubAgent([_laya_result(("R-001", 0.02), ("R-002", 0.9))])
    assessor = _assessor(agent, monkeypatch)
    rule = ThresholdDrop(assessor, threshold=0.10)
    prefilter = Prefilter(assessor=assessor, rule=rule)
    cases = _cases("R-001", "R-002")
    assessment = prefilter.assessor.assess(PREFILTER_TASK, "context", cases)
    decisions = prefilter.rule.decide(cases, assessment)

    kept = {d.risk_id for d in decisions if d.keep}
    dropped = {d.risk_id for d in decisions if not d.keep}
    assert kept == {"R-002"}
    assert dropped == {"R-001"}


def test_module_has_no_module_level_laya_import():
    """The assessor module's own import graph adds no laya_mlx (lazy import contract)."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/asago_policy_mapper/assessors/laya.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            assert "laya_mlx" not in names, f"module-level import of laya_mlx: {names}"
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "laya_mlx":
            raise AssertionError(f"module-level import from laya_mlx: {node.module}")


def test_package_import_stays_lazy():
    """Importing the assessors package pulls in no assessor module."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import asago_policy_mapper.assessors\n"
        "assert 'asago_policy_mapper.assessors.laya' not in sys.modules\n"
        "assert 'asago_policy_mapper.assessors.jev' not in sys.modules\n"
        "assert 'asago_policy_mapper.assessors.http_probability' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.slow
def test_live_laya_smoke():
    """A real checkpoint load and one assess call — the pilot's smoke path.

    Skipped unless laya-mlx is installed (it never is in CI: the package
    is an out-of-band install). Downloads the checkpoint on first ever
    run; later runs use the local cache. The response must also serialize
    as plain JSON: a run persists it in ``decision_calls`` verbatim, so a
    numpy scalar (or anything JSON-hostile) surfacing in the recorded
    response would break the run's output write. A source read of
    laya-mlx (v0.7.2 e2bfb94, byte-identical to v0.7.0 in the relevant
    function) shows predict() builds its output with Python builtins, so
    this is a guard, not a known bug.
    """
    import json

    pytest.importorskip("laya_mlx")
    from asago_policy_mapper.assessors.laya import LayaProbabilityAssessor, LayaProbabilityEndpoint

    assessor = LayaProbabilityAssessor(LayaProbabilityEndpoint())
    cases = [
        DecisionCase(
            risk_id="smoke-1",
            name="Human oversight",
            description="The AI system may act without adequate human oversight or control.",
        )
    ]
    batch = assessor.assess(TASK, "The policy requires human oversight of all AI systems in production.", cases)

    assert batch.outcomes["smoke-1"], "no outcome came back"
    outcome = batch.outcomes["smoke-1"][0]
    assert isinstance(outcome, Probability), f"live laya returned {outcome!r}"
    assert 0.0 <= outcome.p <= 1.0
    # The recorded response must survive the run's JSON output write.
    json.dumps(batch.wire["responses"])
