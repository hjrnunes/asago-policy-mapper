"""Tests for the battery runner's extract-command construction and endpoint decision.

The battery runner shells out to the CLI, so its two behaviors matter here:
the flags it passes (never a redundant or warning-triggering one) and when
it decides a run needs no LLM endpoint at all.
"""

import sys
from pathlib import Path

import pytest

# The mutmut sandbox and the sandbox-isolation canary copy tests/ without
# the repository root, where run_extract_battery.py lives. Fall back to the
# enclosing checkout's root so ordinary mutmut runs keep collecting this
# module, which covers the battery's plan-based endpoint decision. Skip
# only when the script exists nowhere, as in the canary's scratch copy.
# Appending (not inserting) keeps any path mutmut itself puts ahead, such
# as mutants/src, in charge of imports.
for _root in (
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent,
):
    if (_root / "run_extract_battery.py").is_file():
        if str(_root) not in sys.path:
            sys.path.append(str(_root))
        break
else:
    pytest.skip("run_extract_battery.py is not present in this checkout", allow_module_level=True)

from run_extract_battery import _battery_run_plan, _build_extract_command  # noqa: E402


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


def test_battery_plan_builds_retrieval_through_the_cli_builder():
    """The battery's plan resolves from the CLI's own config builder.

    The endpoint decision must follow the same flag-to-RetrievalConfig
    mapping the extract command uses, so a change to that mapping (for
    example a new coupling between the cross-encoder and another flag)
    reaches the battery without a second hand-synced copy. The resolved
    plan matches what the CLI would resolve from the same flags.
    """
    from asago_policy_mapper.cli import _build_retrieval_config
    from asago_policy_mapper.extract.models import RetrievalConfig
    from asago_policy_mapper.extract.plan import resolve_plan

    flags = dict(
        query_gen=False,
        no_judge=True,
        no_grounding=False,
        expand_siblings=True,
        colbert_model="colbert-ir",
        no_cross_encoder=False,
        rrf_min_score=0.02,
    )
    battery_plan = _battery_run_plan(**flags)

    defaults = RetrievalConfig()
    cli_retrieval = _build_retrieval_config(
        bi_encoder_model=defaults.bi_encoder_model,
        bi_encoder_api_key=defaults.bi_encoder_api_key,
        bi_encoder_model_name=None,
        query_instruction=None,
        cross_encoder_model=defaults.cross_encoder_model,
        cross_encoder_type=defaults.cross_encoder_type,
        cross_encoder_model_name=None,
        colbert_model=flags["colbert_model"],
        chunk_max_tokens=defaults.chunk_max_tokens,
        top_n_accept=defaults.top_n_accept,
        top_n_judge=defaults.top_n_judge,
        min_score_floor=defaults.min_score_floor,
        threshold_high=None,
        threshold_low=None,
        bm25_rescue_rank=defaults.bm25_rescue_rank,
        rrf_min_score=flags["rrf_min_score"],
        no_cross_encoder=flags["no_cross_encoder"],
        no_judge=flags["no_judge"],
        no_grounding=flags["no_grounding"],
        judge_prompt=defaults.judge_prompt,
        judge_context_tokens=defaults.judge_context_tokens,
        expand_siblings=flags["expand_siblings"],
        grounding_passes=defaults.grounding_passes,
        grounding_batch_size=defaults.grounding_batch_size,
        expansion_passes=defaults.expansion_passes,
        no_causal_synthesis=False,
        query_gen=flags["query_gen"],
    )
    assert battery_plan == resolve_plan(cli_retrieval, prefilter=False, explicit=frozenset())
