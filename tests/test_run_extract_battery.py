"""Tests for the battery runner's extract-command construction and endpoint decision.

The battery runner shells out to the CLI, so its two behaviors matter here:
the flags it passes (never a redundant or warning-triggering one) and when
it decides a run needs no LLM endpoint at all.
"""

from pathlib import Path

from run_extract_battery import _battery_run_plan, _build_extract_command


def _command(**overrides) -> list[str]:
    kwargs = dict(
        policies=[Path("p.md")],
        out=Path("out/run1"),
        base_url="http://localhost:8000/v1",
        model="test-model",
    )
    kwargs.update(overrides)
    return _build_extract_command(**kwargs)


def test_expand_siblings_not_passed_when_it_matches_the_cli_default():
    """The positive --expand-siblings form never appears on the command line.

    Expansion defaults to on in the CLI, so passing the positive form adds
    nothing — and on --no-grounding runs it would print an ignored-option
    warning. Only a declined expansion passes a flag.
    """
    assert "--expand-siblings" not in _command()
    assert "--expand-siblings" not in _command(no_grounding=True)

    cmd = _command(expand_siblings=False)
    assert "--no-expand-siblings" in cmd


def test_battery_endpoint_decision_uses_the_run_plan():
    """The battery runner asks the plan whether an endpoint is needed.

    A pure IR run (no judge, no grounding, no query generation, no
    cross-encoder) needs no endpoint. The old raw-flag check missed the
    no-cross-encoder case: without the cross-encoder no borderline
    candidates exist, so even a run without --no-judge needs no LLM once
    grounding and query generation are off.
    """

    def _needs_llm(**flags):
        plan = _battery_run_plan(
            query_gen=flags.get("query_gen", True),
            no_judge=flags.get("no_judge", False),
            no_grounding=flags.get("no_grounding", False),
            expand_siblings=flags.get("expand_siblings", True),
            colbert_model=flags.get("colbert_model"),
            no_cross_encoder=flags.get("no_cross_encoder", False),
            rrf_min_score=flags.get("rrf_min_score", 0.015),
        )
        return plan.needs_llm

    # The pure IR run needs no endpoint.
    assert _needs_llm(no_judge=True, no_grounding=True, query_gen=False) is False
    # Without the cross-encoder the judge is unreachable, so the same run
    # needs no endpoint even without --no-judge.
    assert _needs_llm(no_cross_encoder=True, no_grounding=True, query_gen=False) is False
    # Query generation alone is an LLM stage.
    assert _needs_llm(no_judge=True, no_grounding=True, query_gen=True) is True
    # Grounding alone is an LLM stage.
    assert _needs_llm(no_judge=True, no_grounding=False, query_gen=False) is True
    # ColBERT replaces the reranker; the judge stays reachable, so a
    # ColBERT run with grounding keeps needing the endpoint.
    assert _needs_llm(colbert_model="colbert-ir", no_grounding=False, query_gen=False) is True


def test_ir_battery_command_carries_no_endpoint_dependence():
    """IR battery runs pass exactly the flags that switch each stage off."""
    cmd = _command(no_judge=True, no_grounding=True, query_gen=False, no_cross_encoder=True)
    assert "--no-judge" in cmd
    assert "--no-grounding" in cmd
    assert "--no-query-gen" in cmd
    assert "--no-cross-encoder" in cmd
