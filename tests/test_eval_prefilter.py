"""Tests for the prefilter evaluation script.

The evaluation sweeps the drop threshold on a tuning split of the fixture
policies and reports metrics on a held-out split, using the replay
assessor so no provider is contacted.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# The mutmut sandbox and the sandbox-isolation canary copy tests/ without
# scripts/. Fall back to the enclosing checkout's scripts/ so ordinary
# mutmut runs keep collecting this module, which covers mutated src/ code
# through the replay assessor. Skip only when scripts/ exists nowhere, as
# in the canary's scratch copy. Appending (not inserting) keeps any path
# mutmut itself puts ahead, such as mutants/src, in charge of imports.
for _scripts_dir in (
    Path(__file__).resolve().parent.parent / "scripts",
    Path(__file__).resolve().parent.parent.parent / "scripts",
):
    if (_scripts_dir / "eval_prefilter.py").is_file():
        if str(_scripts_dir.parent) not in sys.path:
            sys.path.append(str(_scripts_dir.parent))
        SCRIPT = _scripts_dir / "eval_prefilter.py"
        break
else:
    pytest.skip("scripts/ is not present in this checkout", allow_module_level=True)

TUNING_POLICIES = {"tune-a", "tune-b"}
HELD_OUT_POLICIES = {"hold-a", "hold-b"}


def _outcome_rows(policy: str, context: str, entries: list[tuple[str, float]]) -> list[dict]:
    """Outcome rows for one (policy, context) keyed by risk id and p."""
    return [{"context_sha256": _sha(context), "risk_id": rid, "p": p} for rid, p in entries]


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _fixture_row(policy: str, chunk_index: int, context: str, candidates: list[dict], grounded: list[str]) -> dict:
    return {
        "policy": policy,
        "chunk_index": chunk_index,
        "context": context,
        "context_sha256": _sha(context),
        "candidates": candidates,
        "grounded_ids": grounded,
    }


def _candidate(rid: str) -> dict:
    return {"risk_id": rid, "name": f"Risk {rid}", "description": f"Description for {rid}.", "synthetic": False}


def _make_fixtures(tmp_path: Path) -> Path:
    """Write a fixtures file covering all four split policies."""
    rows = [
        # tune-a: one low (dropped), one high (kept), one grounded-low (recall loss when dropped)
        _fixture_row("tune-a", 0, "ctx-ta-0", [_candidate("R-1"), _candidate("R-2"), _candidate("R-3")], ["R-3"]),
        # tune-b: a grounded high candidate (never dropped) and an ungrounded low one
        _fixture_row("tune-b", 0, "ctx-tb-0", [_candidate("R-4"), _candidate("R-5")], ["R-4"]),
        _fixture_row("hold-a", 0, "ctx-ha-0", [_candidate("R-6"), _candidate("R-7")], ["R-6"]),
        _fixture_row("hold-b", 0, "ctx-hb-0", [_candidate("R-8")], ["R-8"]),
    ]
    fixtures = tmp_path / "fixtures.jsonl"
    fixtures.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    outcomes = tmp_path / "outcomes.jsonl"
    outcome_rows = [
        *_outcome_rows("tune-a", "ctx-ta-0", [("R-1", 0.02), ("R-2", 0.90), ("R-3", 0.05)]),
        *_outcome_rows("tune-b", "ctx-tb-0", [("R-4", 0.80), ("R-5", 0.03)]),
        *_outcome_rows("hold-a", "ctx-ha-0", [("R-6", 0.95), ("R-7", 0.01)]),
        *_outcome_rows("hold-b", "ctx-hb-0", [("R-8", 0.04)]),
    ]
    outcomes.write_text("\n".join(json.dumps(r) for r in outcome_rows) + "\n")
    return fixtures


BATTERY_POLICIES = {
    "amadeus",
    "ars",
    "aus-gov",
    "camden-borough-work",
    "cisco-supplier",
    "dhs-gov",
    "ebay",
    "eu-com",
    "firstsource",
    "fs-isac",
    "gray",
    "guy-nhs",
    "icrc",
    "leicestershire_police",
    "lenovo",
    "llvm",
    "lse-legreg",
    "lse-marking",
    "new-york-state",
    "npcc",
    "ovic",
    "penn",
    "prosus",
    "rdash-nhs",
    "sap",
    "st-johns",
    "vps",
}


def test_split_lists_are_disjoint_and_fixed():
    """The checked-in split lists name distinct policies and together cover the battery exactly."""
    from scripts.eval_prefilter import HELD_OUT_POLICIES, TUNING_POLICIES

    assert TUNING_POLICIES & HELD_OUT_POLICIES == set()
    assert TUNING_POLICIES | HELD_OUT_POLICIES == BATTERY_POLICIES


def test_split_assignment(tmp_path):
    """Rows partition by their policy's split; unknown policies land in neither."""
    from scripts.eval_prefilter import split_rows

    rows = [
        _fixture_row("tune-a", 0, "c0", [_candidate("R-1")], []),
        _fixture_row("hold-a", 0, "c1", [_candidate("R-2")], []),
        _fixture_row("unknown", 0, "c2", [_candidate("R-3")], []),
    ]

    tuning, held_out = split_rows(rows, TUNING_POLICIES, HELD_OUT_POLICIES)

    assert [r["policy"] for r in tuning] == ["tune-a"]
    assert [r["policy"] for r in held_out] == ["hold-a"]


def test_threshold_metrics():
    """Metrics at one threshold: drop rate, recall loss, bypass rate over candidates."""
    from scripts.eval_prefilter import threshold_metrics

    context = "ctx"
    sha = _sha(context)
    rows = [
        _fixture_row("p", 0, context, [_candidate("R-1"), _candidate("R-2"), _candidate("R-3")], ["R-3"]),
    ]
    outcomes = {(sha, "R-1"): 0.02, (sha, "R-2"): 0.90, (sha, "R-3"): 0.05}
    unavailable = {(sha, "R-9")}  # not in this row's candidates

    metrics = threshold_metrics(rows, outcomes, unavailable, threshold=0.10)

    # R-1 (0.02) and R-3 (0.05) drop; R-2 (0.90) keeps. R-3 is grounded -> recall loss.
    assert metrics["candidates"] == 3
    assert metrics["dropped"] == 2
    assert metrics["drop_rate"] == 2 / 3
    assert metrics["grounded_total"] == 1
    assert metrics["grounded_dropped"] == 1
    assert metrics["recall_loss"] == 1.0
    assert metrics["bypassed"] == 0


def test_threshold_metrics_bypass():
    """Unavailable outcomes keep the candidate and count as bypassed."""
    from scripts.eval_prefilter import threshold_metrics

    context = "ctx"
    sha = _sha(context)
    rows = [_fixture_row("p", 0, context, [_candidate("R-1")], ["R-1"])]
    outcomes = {}
    unavailable = {(sha, "R-1")}

    metrics = threshold_metrics(rows, outcomes, unavailable, threshold=0.10)

    assert metrics["dropped"] == 0
    assert metrics["bypassed"] == 1
    assert metrics["bypass_rate"] == 1.0
    assert metrics["recall_loss"] == 0.0


def test_threshold_metrics_absent_outcome_bypasses():
    """An outcome key with no row at all also bypasses (replay answers missing)."""
    from scripts.eval_prefilter import threshold_metrics

    rows = [_fixture_row("p", 0, "ctx", [_candidate("R-1")], [])]

    metrics = threshold_metrics(rows, {}, set(), threshold=0.10)

    assert metrics["dropped"] == 0
    assert metrics["bypassed"] == 1


def test_latency_percentiles():
    """p50 and p95 come from the per-row assessment latencies, nearest-rank."""
    from scripts.eval_prefilter import percentile

    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.5) == 5.0
    # Odd-sized samples: nearest-rank p50 of five values is the third.
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    # Thirty values: p95 is the 29th (ceil(0.95 * 30)).
    assert percentile([float(i) for i in range(1, 31)], 0.95) == 29.0


def test_effective_p_reduces_like_the_rule():
    """Duplicate probabilities reduce to the maximum; any Unavailable bypasses."""
    from asago_policy_mapper.extract.decide import Probability, Unavailable
    from scripts.eval_prefilter import _effective_p

    assert _effective_p([Probability(0.90), Probability(0.02)]) == 0.90
    assert _effective_p([Probability(0.02), Probability(0.03)]) == 0.03
    assert _effective_p([Probability(0.5), Unavailable("timeout")]) is None
    assert _effective_p([]) is None


def test_duplicate_probabilities_use_max_not_last(tmp_path):
    """A candidate stored with 0.90 then 0.02 keeps at threshold 0.10, like the rule."""
    from asago_policy_mapper.assessors.replay import ReplayAssessor
    from scripts.eval_prefilter import _outcome_lookups

    context = "ctx"
    sha = _sha(context)
    rows = [  # grounded candidate with conflicting stored scores
        _fixture_row("p", 0, context, [_candidate("R-1")], ["R-1"]),
    ]
    outcome_rows = [
        {"context_sha256": sha, "risk_id": "R-1", "p": 0.90},
        {"context_sha256": sha, "risk_id": "R-1", "p": 0.02},
    ]
    outcomes_file = tmp_path / "outcomes.jsonl"
    outcomes_file.write_text("\n".join(json.dumps(r) for r in outcome_rows) + "\n")

    outcomes, unavailable = _outcome_lookups(ReplayAssessor(outcomes_file))
    from scripts.eval_prefilter import threshold_metrics

    metrics = threshold_metrics(rows, outcomes, unavailable, threshold=0.10)

    assert metrics["dropped"] == 0
    assert metrics["recall_loss"] == 0.0
    assert metrics["bypassed"] == 0


def test_choose_threshold_minimizes_recall_loss_then_maximizes_drops():
    """Among thresholds with zero recall loss, the one dropping most wins."""
    from scripts.eval_prefilter import choose_threshold

    sweep = [
        {"threshold": 0.05, "recall_loss": 0.1, "dropped": 10},
        {"threshold": 0.10, "recall_loss": 0.0, "dropped": 8},
        {"threshold": 0.20, "recall_loss": 0.0, "dropped": 12},
        {"threshold": 0.30, "recall_loss": 0.05, "dropped": 20},
    ]

    best = choose_threshold(sweep)

    assert best["threshold"] == 0.20


def test_choose_threshold_breaks_full_tie_by_smallest_threshold():
    """Equal recall loss and equal drops pick the smaller threshold."""
    from scripts.eval_prefilter import choose_threshold

    sweep = [
        {"threshold": 0.30, "recall_loss": 0.0, "dropped": 5},
        {"threshold": 0.10, "recall_loss": 0.0, "dropped": 5},
        {"threshold": 0.20, "recall_loss": 0.0, "dropped": 5},
    ]

    best = choose_threshold(sweep)

    assert best["threshold"] == 0.10


def test_eval_script_runs_end_to_end(tmp_path):
    """The script sweeps on tuning, reports on held-out, and writes a report."""
    from scripts.eval_prefilter import HELD_OUT_POLICIES, TUNING_POLICIES

    tune_policy = sorted(TUNING_POLICIES)[0]
    hold_policy = sorted(HELD_OUT_POLICIES)[0]
    rows = [
        # tuning row: low drops, grounded low loses recall, high keeps
        _fixture_row(tune_policy, 0, "ctx-ta-0", [_candidate("R-1"), _candidate("R-2"), _candidate("R-3")], ["R-3"]),
        # held-out rows
        _fixture_row(hold_policy, 0, "ctx-ha-0", [_candidate("R-6"), _candidate("R-7")], ["R-6"]),
        _fixture_row(hold_policy, 1, "ctx-hb-0", [_candidate("R-8")], ["R-8"]),
    ]
    fixtures = tmp_path / "fixtures.jsonl"
    fixtures.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    outcome_rows = [
        *_outcome_rows(tune_policy, "ctx-ta-0", [("R-1", 0.02), ("R-2", 0.90), ("R-3", 0.05)]),
        *_outcome_rows(hold_policy, "ctx-ha-0", [("R-6", 0.95), ("R-7", 0.01)]),
        *_outcome_rows(hold_policy, "ctx-hb-0", [("R-8", 0.04)]),
    ]
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("\n".join(json.dumps(r) for r in outcome_rows) + "\n")

    report = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(fixtures),
            str(outcomes),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text())
    assert "sweep" in payload
    assert "held_out" in payload
    # Tuning: t=0.01 drops nothing, t=0.02 drops only R-1 (0.02) at zero recall
    # loss; t>=0.05 drops the grounded R-3. Zero-loss max-drops picks 0.02.
    assert payload["chosen_threshold"] == 0.02
    held = payload["held_out"]
    for field in ("drop_rate", "recall_loss", "bypass_rate", "latency_p50_ms", "latency_p95_ms"):
        assert field in held, field
    assert held["candidates"] == 3  # the two held-out rows' candidates
    # At 0.02 only R-7 (0.01) drops; R-8 (0.04) and grounded R-6 (0.95) keep.
    assert held["dropped"] == 1
    assert held["grounded_dropped"] == 0
    assert held["recall_loss"] == 0.0
    assert held["drop_rate"] == 1 / 3
    assert held["latency_p50_ms"] >= 0.0


def test_eval_script_rejects_empty_split(tmp_path):
    """A fixtures file whose policies land in only one split exits non-zero with a message."""
    from scripts.eval_prefilter import TUNING_POLICIES

    tune_policy = sorted(TUNING_POLICIES)[0]
    rows = [_fixture_row(tune_policy, 0, "ctx", [_candidate("R-1")], [])]
    fixtures = tmp_path / "fixtures.jsonl"
    fixtures.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(fixtures), str(outcomes)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "both splits must be non-empty" in completed.stderr


def test_eval_main_in_process(tmp_path, capsys):
    """main() runs the sweep and writes the report; latency fields present."""
    from scripts.eval_prefilter import HELD_OUT_POLICIES, TUNING_POLICIES, main

    tune_policy = sorted(TUNING_POLICIES)[0]
    hold_policy = sorted(HELD_OUT_POLICIES)[0]
    rows = [
        _fixture_row(tune_policy, 0, "ctx-t", [_candidate("R-1"), _candidate("R-2")], ["R-1"]),
        _fixture_row(hold_policy, 0, "ctx-h", [_candidate("R-3")], []),
    ]
    fixtures = tmp_path / "fixtures.jsonl"
    fixtures.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    outcomes = tmp_path / "outcomes.jsonl"
    outcome_rows = [
        *_outcome_rows(tune_policy, "ctx-t", [("R-1", 0.90), ("R-2", 0.02)]),
        *_outcome_rows(hold_policy, "ctx-h", [("R-3", 0.50)]),
    ]
    outcomes.write_text("\n".join(json.dumps(r) for r in outcome_rows) + "\n")

    report = tmp_path / "report.json"
    code = main([str(fixtures), str(outcomes), "--report", str(report)])

    assert code == 0
    payload = json.loads(report.read_text())
    assert len(payload["sweep"]) >= 1
    assert payload["chosen_threshold"] in {m["threshold"] for m in payload["sweep"]}
    held = payload["held_out"]
    assert held["candidates"] == 1
    assert held["latency_p50_ms"] >= 0.0
    assert held["latency_p95_ms"] >= held["latency_p50_ms"]
    out = capsys.readouterr().out
    assert "chosen_threshold" in out
