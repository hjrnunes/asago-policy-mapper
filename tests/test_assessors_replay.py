"""Tests for the replay assessor that serves stored outcomes from a file.

The replay assessor lets the offline evaluation (``scripts/eval_prefilter.py``)
compare rules and thresholds on saved outcomes without contacting any
provider.
"""

import json

import pytest

from asago_policy_mapper.assessors.replay import ReplayAssessor
from asago_policy_mapper.extract.decide import (
    BinaryTask,
    DecisionCase,
    Probability,
    Unavailable,
)

TASK = BinaryTask(task_id="risk-relevance", version="1", proposition="Is the risk relevant?")


def _cases(*risk_ids: str) -> list[DecisionCase]:
    return [DecisionCase(risk_id=rid, name=f"Risk {rid}", description=f"Description for {rid}.") for rid in risk_ids]


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def test_replay_roundtrip(tmp_path):
    """A stored probability keyed by (context_sha256, risk_id) is served back unchanged."""
    context = "chunk context text"
    rows = [
        {"context_sha256": _sha256(context), "risk_id": "R-001", "p": 0.42},
        {"context_sha256": _sha256(context), "risk_id": "R-002", "p": 0.9},
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-002"))

    assert batch.outcomes["R-001"] == (Probability(0.42),)
    assert batch.outcomes["R-002"] == (Probability(0.9),)
    assert batch.unmatched_ids == ()
    assert batch.latency_ms >= 0.0


def test_replay_serves_unavailable_outcomes(tmp_path):
    """Stored unavailable rows are served as Unavailable outcomes with their reason."""
    context = "ctx"
    rows = [
        {"context_sha256": _sha256(context), "risk_id": "R-001", "unavailable": "timeout"},
        {"context_sha256": _sha256(context), "risk_id": "R-002", "unavailable": "invalid"},
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-002"))

    assert batch.outcomes["R-001"] == (Unavailable("timeout"),)
    assert batch.outcomes["R-002"] == (Unavailable("invalid"),)


def test_replay_unknown_key_is_missing(tmp_path):
    """A (context, risk_id) with no stored row yields Unavailable("missing")."""
    path = tmp_path / "outcomes.jsonl"
    path.write_text(json.dumps({"context_sha256": _sha256("other"), "risk_id": "R-001", "p": 0.5}) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, "this context", _cases("R-001"))

    assert batch.outcomes["R-001"] == (Unavailable("missing"),)


def test_replay_duplicate_rows_keep_all_outcomes(tmp_path):
    """Duplicate stored rows for one key serve as duplicate outcomes, not one."""
    context = "ctx"
    row = {"context_sha256": _sha256(context), "risk_id": "R-001", "p": 0.5}
    path = tmp_path / "outcomes.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001"))

    assert batch.outcomes["R-001"] == (Probability(0.5), Probability(0.5))


def test_replay_unavailable_missing_row_is_not_droppable(tmp_path):
    """A stored unavailable=missing row survives as an outcome even beside a p row.

    Dropping the row (as a parse failure would) would leave only the p
    outcome, so this pins that "missing" is a recognized reason.
    """
    context = "ctx"
    key = _sha256(context)
    rows = [
        {"context_sha256": key, "risk_id": "R-001", "unavailable": "missing"},
        {"context_sha256": key, "risk_id": "R-001", "p": 0.5},
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001"))

    assert batch.outcomes["R-001"] == (Unavailable("missing"), Probability(0.5))


def test_replay_reads_only_once(tmp_path):
    """The file is read at construction, not on every assess call."""
    context = "ctx"
    path = tmp_path / "outcomes.jsonl"
    path.write_text(json.dumps({"context_sha256": _sha256(context), "risk_id": "R-001", "p": 0.5}) + "\n")

    assessor = ReplayAssessor(path)
    path.write_text("")  # truncate after construction

    batch = assessor.assess(TASK, context, _cases("R-001"))
    assert batch.outcomes["R-001"] == (Probability(0.5),)


def test_replay_properties(tmp_path):
    """The replay assessor is local, offline, and emits Probability and Unavailable."""
    path = tmp_path / "o.jsonl"
    path.write_text(json.dumps({"context_sha256": _sha256("c"), "risk_id": "R-001", "p": 0.5}) + "\n")
    assessor = ReplayAssessor(path)

    assert assessor.offsite is False
    assert Probability in assessor.emits
    assert Unavailable in assessor.emits
    assert "replay" in assessor.assessor_id


def test_replay_missing_file_is_an_error(tmp_path):
    """A missing answers file raises ValueError instead of answering missing for everything."""
    with pytest.raises(ValueError, match="answers file not found"):
        ReplayAssessor(tmp_path / "absent.jsonl")


def test_replay_empty_file_is_an_error(tmp_path):
    """An empty answers file raises ValueError: nothing to replay."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    with pytest.raises(ValueError, match="no usable rows"):
        ReplayAssessor(empty)


def test_replay_all_malformed_file_is_an_error(tmp_path):
    """A file whose every line is unusable raises ValueError, same as an empty file."""
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n" + json.dumps({"no": "keys"}) + "\n")

    with pytest.raises(ValueError, match="no usable rows"):
        ReplayAssessor(path)


def test_replay_wire_is_none(tmp_path):
    """Replay makes no wire request; wire stays None like other offline assessors."""
    path = tmp_path / "outcomes.jsonl"
    path.write_text(json.dumps({"context_sha256": _sha256("ctx"), "risk_id": "R-001", "p": 0.5}) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, "ctx", _cases("R-001"))

    assert batch.wire is None


def test_replay_bad_rows_are_counted_not_fatal(tmp_path, capsys):
    """Unparseable lines and unusable rows are counted and reported; good rows still load."""
    context = "ctx"
    key = _sha256(context)
    lines = [
        "not json at all",
        json.dumps({"context_sha256": key, "risk_id": 42, "p": 0.5}),  # non-string risk_id
        json.dumps({"context_sha256": 42, "risk_id": "R-001", "p": 0.5}),  # non-string hash
        json.dumps({"context_sha256": key, "risk_id": "R-003", "p": True}),  # boolean p
        json.dumps({"context_sha256": key, "risk_id": "R-004", "p": "high"}),  # non-numeric p
        json.dumps({"context_sha256": key, "risk_id": "R-005", "unavailable": "kaput"}),  # unknown reason
        json.dumps(["a", "list"]),  # not an object
        "",  # blank line
        json.dumps({"context_sha256": key, "risk_id": "R-001", "p": 0.5}),  # the good row, last
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(lines) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-003", "R-004", "R-005"))

    assert batch.outcomes["R-001"] == (Probability(0.5),)
    for rid in ("R-003", "R-004", "R-005"):
        assert batch.outcomes[rid] == (Unavailable("missing"),), rid
    assert assessor.malformed_lines == 7  # every unusable line; blank lines excluded
    assert "malformed line" in capsys.readouterr().err


def test_replay_out_of_range_p_is_skipped(tmp_path):
    """Out-of-range, non-finite and oversized p values skip the row instead of loading."""
    context = "ctx"
    key = _sha256(context)
    raws = [
        '{"context_sha256": "%s", "risk_id": "R-001", "p": -0.1}',
        '{"context_sha256": "%s", "risk_id": "R-002", "p": 1.5}',
        '{"context_sha256": "%s", "risk_id": "R-003", "p": NaN}',
        '{"context_sha256": "%s", "risk_id": "R-004", "p": Infinity}',
        '{"context_sha256": "%s", "risk_id": "R-005", "p": ' + "1" + "0" * 400 + "}",
        '{"context_sha256": "%s", "risk_id": "R-006", "p": 0.5}',
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(r % key for r in raws) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-002", "R-003", "R-004", "R-005", "R-006"))

    for rid in ("R-001", "R-002", "R-003", "R-004", "R-005"):
        assert batch.outcomes[rid] == (Unavailable("missing"),), rid
    assert batch.outcomes["R-006"] == (Probability(0.5),)


def test_replay_boundary_p_values_load(tmp_path):
    """p values of exactly 0 and exactly 1 load as valid probabilities."""
    context = "ctx"
    key = _sha256(context)
    rows = [
        {"context_sha256": key, "risk_id": "R-001", "p": 0},
        {"context_sha256": key, "risk_id": "R-002", "p": 1},
        {"context_sha256": key, "risk_id": "R-003", "p": 0.0},
        {"context_sha256": key, "risk_id": "R-004", "p": 1.0},
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-002", "R-003", "R-004"))

    assert batch.outcomes["R-001"] == (Probability(0.0),)
    assert batch.outcomes["R-002"] == (Probability(1.0),)
    assert batch.outcomes["R-003"] == (Probability(0.0),)
    assert batch.outcomes["R-004"] == (Probability(1.0),)


def test_replay_unavailable_never_falls_through_to_p(tmp_path):
    """A row with an unknown or non-string unavailable reason never serves its p."""
    context = "ctx"
    key = _sha256(context)
    rows = [
        {"context_sha256": key, "risk_id": "R-001", "unavailable": "kaput", "p": 0.9},
        {"context_sha256": key, "risk_id": "R-002", "unavailable": 42, "p": 0.9},
        {"context_sha256": key, "risk_id": "R-003", "unavailable": None, "p": 0.9},
        {"context_sha256": key, "risk_id": "R-004", "p": 0.5},  # keeps the file usable
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-002", "R-003"))

    for rid in ("R-001", "R-002", "R-003"):
        assert batch.outcomes[rid] == (Unavailable("missing"),), rid


def test_replay_unavailable_reasons_are_exact(tmp_path):
    """Each stored Unavailable reason round-trips exactly; an unknown reason drops the row."""
    context = "ctx"
    key = _sha256(context)
    rows = [
        {"context_sha256": key, "risk_id": "R-001", "unavailable": "missing", "p": 0.9},
        {"context_sha256": key, "risk_id": "R-002", "unavailable": "invalid", "p": 0.9},
        {"context_sha256": key, "risk_id": "R-003", "unavailable": "timeout", "p": 0.9},
        {"context_sha256": key, "risk_id": "R-004", "unavailable": "error", "p": 0.9},
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-002", "R-003", "R-004"))

    # Each row carries both a reason and a p; the reason wins for every
    # recognized reason, including "missing".
    assert batch.outcomes["R-001"] == (Unavailable("missing"),)
    assert batch.outcomes["R-002"] == (Unavailable("invalid"),)
    assert batch.outcomes["R-003"] == (Unavailable("timeout"),)
    assert batch.outcomes["R-004"] == (Unavailable("error"),)


def test_replay_latency_is_milliseconds(monkeypatch, tmp_path):
    """The reported latency is in milliseconds, scaled from seconds."""
    import asago_policy_mapper.assessors.replay as module

    class _Clock:
        def __init__(self):
            self.values = iter([1.5, 2.0])

        def perf_counter(self):
            return next(self.values)

    monkeypatch.setattr(module, "time", _Clock())
    path = tmp_path / "o.jsonl"
    path.write_text(json.dumps({"context_sha256": _sha256("ctx"), "risk_id": "R-001", "p": 0.5}) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, "ctx", _cases("R-001"))

    assert batch.latency_ms == 500.0


def test_replay_rejects_one_bad_key_field_at_a_time(tmp_path):
    """A row is unusable when EITHER key field is not a string, not only both."""
    context = "ctx"
    key = _sha256(context)
    rows = [
        {"context_sha256": 42, "risk_id": "R-001", "p": 0.5},  # bad hash alone
        {"context_sha256": key, "risk_id": 42, "p": 0.5},  # bad risk_id alone
        {"context_sha256": key, "risk_id": "R-003", "p": 0.5},  # the good row
    ]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-001", "R-003"))

    assert batch.outcomes["R-001"] == (Unavailable("missing"),)
    assert batch.outcomes["R-003"] == (Probability(0.5),)


def test_replay_reads_utf8(tmp_path):
    """Rows with non-ASCII text load correctly under the explicit utf-8 encoding."""
    context = "café ☕ ctx"
    rows = [{"context_sha256": _sha256(context), "risk_id": "R-é", "p": 0.5}]
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    assessor = ReplayAssessor(path)
    batch = assessor.assess(TASK, context, _cases("R-é"))

    assert batch.outcomes["R-é"] == (Probability(0.5),)
