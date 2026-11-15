"""Wiring tests for the run plan (asago-uzfb1.12, step order per bead).

One ``RunPlan`` (``extract/plan.py::resolve_plan``) decides the CLI's
LLM check, the saved settings, the CLI summary and the pipeline's stage
gates. The reranker gets an explicit switch on every ``hybrid_search``
call site, and ``--judge-context-tokens`` builds judge-only contexts.

These tests pin the wiring, not the resolver's truth table (that is
``test_extract_plan.py``) or the prefilter's own behaviour.
"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

from asago_policy_mapper.cli import app
from asago_policy_mapper.extract.models import RetrievalConfig, ScoredCandidate
from asago_policy_mapper.extract.plan import resolve_plan

runner = CliRunner()


@pytest.fixture
def mock_config():
    from asago_policy_mapper.llm import LLMConfig

    return LLMConfig(base_url="http://localhost:8000/v1", model="test-model", max_context=0)


def _policy(tmp_path) -> Path:
    policy = tmp_path / "policy.txt"
    policy.write_text("test policy")
    return policy


# --- the CLI check and error (step 1) ---


def _validate_kwargs(tmp_path, retrieval=None, **overrides):
    kwargs = dict(
        output_format="json",
        policy_files=[_policy(tmp_path)],
        base_url="http://localhost:8000/v1",
        model="test-model",
        retrieval=retrieval if retrieval is not None else RetrievalConfig(query_gen=False),
        prefilter=False,
        explicit=frozenset(),
    )
    kwargs.update(overrides)
    return kwargs


def test_error_lists_llm_stages(tmp_path, capsys):
    """A missing endpoint names the stages that need one, not the flags that imply them.

    The old error explained the escape hatch ("unless both --no-judge and
    --no-grounding are set"); the plan-based error says which stages call
    an LLM for this exact combination of options.
    """
    from asago_policy_mapper.cli import _validate_extract_options

    # Judge + grounding on: both stages need the endpoint.
    with pytest.raises(typer.Exit):
        _validate_extract_options(
            **_validate_kwargs(tmp_path, retrieval=RetrievalConfig(query_gen=False), base_url=None, model=None)
        )
    err = capsys.readouterr().err
    assert err.strip() == (
        "Error: --base-url/--model required: stages judge, grounding, variant_grounding, expansion, "
        "causal_synthesis call an LLM"
    )

    # Query generation adds itself to the front of the list.
    with pytest.raises(typer.Exit):
        _validate_extract_options(
            **_validate_kwargs(
                tmp_path,
                retrieval=RetrievalConfig(query_gen=True, no_judge=True, no_grounding=True, expand_siblings=False),
                base_url=None,
                model=None,
            )
        )
    err = capsys.readouterr().err
    assert err.strip() == "Error: --base-url/--model required: stages query_gen call an LLM"

    # A plan with no LLM stages never errors, even with no endpoint.
    plan = _validate_extract_options(
        **_validate_kwargs(
            tmp_path,
            retrieval=RetrievalConfig(
                query_gen=False, no_judge=True, no_grounding=True, expand_siblings=False, use_cross_encoder=False
            ),
            base_url=None,
            model=None,
        )
    )
    assert plan.needs_llm is False


def test_ir_summary_does_not_hide_query_generation(tmp_path, capsys):
    """The CLI summary reports query generation even on IR-only runs.

    The old summary keyed off ``no_judge and no_grounding`` and printed
    "IR-only, no LLM stages" for query-gen runs, hiding the one LLM stage
    that still runs.
    """
    from asago_policy_mapper.cli import _write_extraction_outputs

    result = SimpleNamespace(
        risks=[], retrieval_stats=SimpleNamespace(auto_accepted=3, llm_judged=0, grounding_filtered=0)
    )
    retrieval = RetrievalConfig(query_gen=True, no_judge=True, no_grounding=True, expand_siblings=False)
    plan = resolve_plan(retrieval, prefilter=False, explicit=frozenset())

    _write_extraction_outputs(
        result,
        {"risks": []},
        tmp_path,
        output_format="json",
        plan=plan,
        tracker=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0),
    )

    out = capsys.readouterr().out
    assert "IR-only, no LLM stages" not in out
    assert "query generation" in out


def test_llm_judge_summary_without_grounding(tmp_path, capsys):
    """A judge-only run (no grounding) reports the LLM-judged line.

    The judge runs as an LLM stage here, so the summary must not call the
    run IR-only: it names the LLM-judged count and the missing grounding.
    """
    from asago_policy_mapper.cli import _write_extraction_outputs

    result = SimpleNamespace(
        risks=[], retrieval_stats=SimpleNamespace(auto_accepted=2, llm_judged=5, grounding_filtered=0)
    )
    retrieval = RetrievalConfig(query_gen=False, no_grounding=True, expand_siblings=False)
    plan = resolve_plan(retrieval, prefilter=False, explicit=frozenset())
    assert plan.judge == "llm" and plan.grounding is False and plan.needs_llm

    _write_extraction_outputs(
        result,
        {"risks": []},
        tmp_path,
        output_format="json",
        plan=plan,
        tracker=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0),
    )

    out = capsys.readouterr().out
    assert "IR-only" not in out
    assert "5 LLM-judged (no grounding)" in out


def test_existing_command_lines_accepted(tmp_path, monkeypatch):
    """The CLI invocations used across the test suite still run.

    Replays the command lines from ``tests/test_extract_cli.py``: the
    flags themselves, their combinations, and the env-var-driven forms
    must all keep working after the plan resolves them.
    """
    from asago_policy_mapper.extract import pipeline

    captured = {}

    def fake_run_extraction(**kwargs):
        captured["retrieval"] = kwargs["retrieval"]
        return pipeline._empty_result(kwargs["documents"])

    class _Risk:
        id = "r1"
        name = "r1"
        description = "A risk."
        isDefinedByTaxonomy = "test"
        isPartOf = ""

    class FakeNexus:
        def __init__(self, base_dir=None):
            pass

        def get_all_risks(self):
            return [_Risk()]

    import asago_policy_mapper.cli as cli

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)
    monkeypatch.setattr(cli, "_attach_mitigations", lambda result, nexus_base_dir: None)

    policy = _policy(tmp_path)
    out = tmp_path / "out"
    common = ["extract", str(policy), "-o", str(out), "--base-url", "http://localhost:8000/v1", "--model", "test-model"]

    # From test_extract_cli.py: the plain invocation, the pure IR run and
    # the expand-siblings-needs-LLM combination. The IR lines pair
    # --no-query-gen with --no-judge/--no-grounding, exactly as the suite
    # writes them: query generation defaults on and calls an LLM, so an
    # endpoint-less run must disable it (the old CLI refused the
    # endpoint-less query-gen form with the same exit code).
    command_lines = [
        list(common),
        ["extract", str(policy), "-o", str(out), "--no-judge", "--no-grounding", "--no-query-gen"],
        ["extract", str(policy), "-o", str(out), "--no-judge", "--no-grounding", "--no-query-gen", "-f", "yaml"],
        [
            "extract",
            str(policy),
            "-o",
            str(out),
            "--no-judge",
            "--no-grounding",
            "--no-query-gen",
            "--no-expand-siblings",
            "-f",
            "both",
        ],
        [
            "extract",
            str(policy),
            "-o",
            str(out),
            "--no-judge",
            "--no-grounding",
            "--no-query-gen",
            "--no-expand-siblings",
            "--nexus-base-dir",
            "",
        ],
        [
            "extract",
            str(policy),
            "-o",
            str(out),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test-model",
            "--no-judge",
            "--no-grounding",
            "--no-expand-siblings",
        ],
    ]
    for args in command_lines:
        result = runner.invoke(app, args, env={"POLICY_MAPPER_BASE_URL": "", "POLICY_MAPPER_MODEL": ""})
        assert result.exit_code == 0, (args, result.output, result.exception)

    # The env-var form reaches the pipeline with the endpoint values.
    result = runner.invoke(
        app,
        ["extract", str(policy), "-o", str(out), "--no-judge", "--no-grounding", "--no-expand-siblings"],
        env={
            "POLICY_MAPPER_BASE_URL": "http://localhost:8000/v1",
            "POLICY_MAPPER_MODEL": "test-model",
            "POLICY_MAPPER_MAX_OUTPUT_TOKENS": "4096",
        },
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["retrieval"].no_judge is True


def test_explicit_option_names_reads_the_real_click_context():
    """_explicit_option_names maps passed flags to their dashed names.

    Store-true flags map directly (``no_judge`` → ``--no-judge``), the
    boolean pairs record the polarity actually passed (``--no-query-gen``
    when the positive form was declined), and options left at their
    defaults or fed through environment variables never make the set.
    """
    import click
    import click.testing

    from asago_policy_mapper.cli import _explicit_option_names

    seen = {}

    @click.command()
    @click.option("--no-judge", is_flag=True)
    @click.option("--query-gen/--no-query-gen", default=True)
    @click.option("--rrf-min-score", type=float, default=0.015)
    def probe(no_judge, query_gen, rrf_min_score):
        seen["result"] = _explicit_option_names(("no_judge", "query_gen", "rrf_min_score"))

    # Only --no-judge and the declined boolean pair were passed on the
    # command line; rrf_min_score keeps its default.
    click.testing.CliRunner().invoke(probe, ["--no-judge", "--no-query-gen"], env={"POLICY_MAPPER_RRF_MIN_SCORE": ""})
    assert seen["result"] == frozenset({"--no-judge", "--no-query-gen"})

    # The positive forms and an explicit value all land in the set.
    click.testing.CliRunner().invoke(probe, ["--rrf-min-score", "0.02"])
    assert seen["result"] == frozenset({"--rrf-min-score"})

    # Nothing passed: the set is empty even with defaults resolved.
    click.testing.CliRunner().invoke(probe, [])
    assert seen["result"] == frozenset()


def test_cli_forwards_the_resolved_plan(tmp_path, monkeypatch):
    """The pipeline runs with the plan the CLI resolved, not a fresh one.

    The CLI resolves the plan once (from the explicit flags) and forwards
    it: the pipeline's metadata then records the plan whose warnings the
    CLI echoed, instead of re-resolving one without the explicit set and
    silently dropping them.
    """
    from asago_policy_mapper.extract import pipeline

    captured = {}

    def fake_run_extraction(**kwargs):
        captured["plan"] = kwargs.get("plan")
        return pipeline._empty_result(kwargs["documents"])

    class _Risk:
        id = "r1"
        name = "r1"
        description = "A risk."
        isDefinedByTaxonomy = "test"
        isPartOf = ""

    class FakeNexus:
        def __init__(self, base_dir=None):
            pass

        def get_all_risks(self):
            return [_Risk()]

    import asago_policy_mapper.cli as cli

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)
    monkeypatch.setattr(cli, "_attach_mitigations", lambda result, nexus_base_dir: None)
    monkeypatch.setattr("ai_atlas_nexus.AIAtlasNexus", FakeNexus)

    policy = _policy(tmp_path)
    out = tmp_path / "out"
    # --no-judge is redundant with --no-cross-encoder, so the forwarded
    # plan must carry that warning: a re-resolved plan (explicit empty)
    # would carry none.
    result = runner.invoke(
        app,
        [
            "extract",
            str(policy),
            "-o",
            str(out),
            "--no-judge",
            "--no-cross-encoder",
            "--no-grounding",
            "--no-query-gen",
            "--no-expand-siblings",
        ],
        env={"POLICY_MAPPER_BASE_URL": "http://localhost:8000/v1", "POLICY_MAPPER_MODEL": "test-model"},
    )
    assert result.exit_code == 0, (result.output, result.exception)

    forwarded = captured["plan"]
    assert forwarded is not None, "the CLI must forward the resolved plan"
    retrieval = RetrievalConfig(
        query_gen=False,
        use_cross_encoder=False,
        no_judge=True,
        no_grounding=True,
        expand_siblings=False,
    )
    expected = resolve_plan(retrieval, prefilter=False, explicit=frozenset({"--no-judge", "--no-cross-encoder"}))
    assert asdict(forwarded) == asdict(expected)
    assert any(warning.startswith("--no-judge is redundant with --no-cross-encoder") for warning in forwarded.warnings)


# --- the metadata (step 2) ---


def test_metadata_records_requested_and_effective(mock_config, tmp_path, monkeypatch):
    """metadata.run_plan holds the requested retrieval settings and the effective plan.

    Planned is not executed: the execution counters (llm_calls, stats)
    stay next to the plan, and the effective plan reflects what the flags
    resolve to (not what was asked for).
    """
    from asago_policy_mapper.extract import pipeline

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return []

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())

    doc = tmp_path / "p.md"
    doc.write_text(
        "AI policy text about data governance and risk management. "
        "The organization commits to bias detection and mitigation strategies. "
        "Training data integrity is critical for model reliability and safety."
    )
    retrieval = RetrievalConfig(query_gen=False, no_judge=True)
    risks = [
        SimpleNamespace(
            id="R-001",
            name="n",
            description="d",
            concern="",
            risk_type="",
            isDefinedByTaxonomy="t",
            isPartOf="",
            exact_mappings=[],
            close_mappings=[],
            broad_mappings=[],
            narrow_mappings=[],
            related_mappings=[],
        )
    ]

    result = pipeline.run_extraction(
        documents=[doc],
        client=None,
        config=mock_config,
        risks=risks,
        retrieval=retrieval,
        ocr=False,
    )

    run_plan = result.metadata["run_plan"]
    assert run_plan["requested"] == retrieval.to_metadata()
    assert run_plan["effective"] == asdict(resolve_plan(retrieval, prefilter=False, explicit=frozenset()))
    assert run_plan["warnings"] == []
    # The execution counters stay next to the plan.
    assert result.retrieval_stats.total_chunks >= 0
    assert "query_gen_queries" not in result.metadata or result.metadata["query_gen_queries"] == []


def test_empty_run_records_the_plan(tmp_path, mock_config):
    """A short-circuited run still records the run plan under its key.

    An empty document exits before any stage runs; the metadata carries
    only ``run_plan``, and it describes the plan the caller passed (or
    derives one). A derived plan on a prefilter run records the prefilter
    stage, because the empty-exit path forwards the prefilter flag.
    """
    from asago_policy_mapper.extract import pipeline

    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    retrieval = RetrievalConfig(query_gen=False, no_judge=True, no_grounding=True, expand_siblings=False)
    plan = resolve_plan(retrieval, prefilter=True, explicit=frozenset())

    result = pipeline.run_extraction(
        documents=[empty],
        client=None,
        config=mock_config,
        risks=[],
        retrieval=retrieval,
        ocr=False,
        plan=plan,
    )

    assert set(result.metadata) == {"run_plan"}
    assert result.metadata["run_plan"]["effective"] == asdict(plan)

    # Without a plan, the empty run derives one from the same inputs:
    # the prefilter flag survives the short circuit. (The prefilter needs
    # grounding, so this retrieval keeps grounding on.)
    prefilter_retrieval = RetrievalConfig(query_gen=False, no_judge=True)
    prefilter = SimpleNamespace(profile="stub")
    result = pipeline.run_extraction(
        documents=[empty],
        client=None,
        config=mock_config,
        risks=[],
        retrieval=prefilter_retrieval,
        ocr=False,
        prefilter=prefilter,
    )
    derived = resolve_plan(prefilter_retrieval, prefilter=True, explicit=frozenset())
    assert result.metadata["run_plan"]["effective"] == asdict(derived)
    assert result.metadata["run_plan"]["effective"]["prefilter"] is True


# --- the explicit reranker argument (step 3) ---


class _RecordingIndex:
    """Index stand-in that records every hybrid_search call's rerank argument."""

    variant_map = {}
    risks = []

    def __init__(self):
        self.calls = []

    def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
        self.calls.append({"query": query, "rerank": rerank, "rrf_min_score": rrf_min_score})
        return []

    def get_taxonomy(self, rid):
        return "test-taxonomy"

    def expand_variants(self, candidates):
        return candidates


def test_hybrid_search_explicit_rerank_overrides_floor(monkeypatch, tmp_path):
    """rerank=True forces the rerank path even with a positive rrf_min_score floor.

    Every hybrid_search call site passes rerank explicitly, so the floor
    only filters; it never decides whether the reranker runs.
    """
    from asago_policy_mapper.extract.index import RiskIndex

    index = RiskIndex.__new__(RiskIndex)
    index._colbert = None
    index._cross_encoder = None
    index._remote_cross_encoder = None
    index._bm25 = None
    index._bi_encoder = None
    index._remote_bi_encoder = None
    index._embeddings = None
    index._risk_ids = ["R-001"]
    index._risk_meta = {"R-001": {"name": "n", "description": "d", "concern": "", "taxonomy": "t"}}
    index._variant_map = {}
    index._variant_taxonomy = {}

    candidates = [ScoredCandidate(risk_id="R-001", risk_name="n", risk_description="d", rrf_score=0.03)]

    def fake_rerank(text, rrf_candidates, top_k=50):
        return [c.model_copy(update={"cross_encoder_score": 0.5}) for c in rrf_candidates]

    monkeypatch.setattr(index, "rerank", fake_rerank)
    monkeypatch.setattr("asago_policy_mapper.extract.index.RiskIndex.search_bm25", lambda self, text, top_k=100: [])
    monkeypatch.setattr(
        "asago_policy_mapper.extract.index.RiskIndex.search_semantic", lambda self, text, top_k=100: candidates
    )

    # Explicit rerank=True with a positive floor: reranked, not floored.
    results = index.hybrid_search("query", top_k=5, rrf_min_score=0.02, rerank=True)
    assert [c.cross_encoder_score for c in results] == [0.5]

    # Explicit rerank=False with no floor: never reranked (the fused RRF
    # score replaces the raw one; no cross-encoder score is set).
    results = index.hybrid_search("query", top_k=5, rrf_min_score=0.0, rerank=False)
    assert [c.risk_id for c in results] == ["R-001"]
    assert all(c.cross_encoder_score == 0.0 for c in results)

    # rerank=None keeps the old rule: rerank iff rrf_min_score <= 0.
    results = index.hybrid_search("query", top_k=5, rrf_min_score=0.0, rerank=None)
    assert [c.cross_encoder_score for c in results] == [0.5]

    # ...and a positive floor with rerank=None stays on the floor path
    # (the fused candidate survives, no cross-encoder score is set).
    results = index.hybrid_search("query", top_k=5, rrf_min_score=0.01, rerank=None)
    assert [c.risk_id for c in results] == ["R-001"]
    assert all(c.cross_encoder_score == 0.0 for c in results)


def test_rrf_floor_does_not_toggle_reranking(mock_config, tmp_path, monkeypatch):
    """The per-chunk path reranks based on the plan, not on the RRF floor.

    With the cross-encoder on, a positive --rrf-min-score must not switch
    per-chunk retrieval to floor-only output; with it off, retrieval
    passes rerank=False and the floor filters (the old default rule).
    """
    from asago_policy_mapper.extract import pipeline

    index = _RecordingIndex()
    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: index)

    doc = tmp_path / "p.md"
    doc.write_text(
        "AI policy text about data governance and risk management. "
        "The organization commits to bias detection and mitigation strategies. "
        "Training data integrity is critical for model reliability and safety."
    )
    risks = [
        SimpleNamespace(
            id="R-001",
            name="n",
            description="d",
            concern="",
            risk_type="",
            isDefinedByTaxonomy="t",
            isPartOf="",
            exact_mappings=[],
            close_mappings=[],
            broad_mappings=[],
            narrow_mappings=[],
            related_mappings=[],
        )
    ]

    # Cross-encoder on: the per-chunk call passes rerank=True.
    pipeline.run_extraction(
        documents=[doc],
        client=None,
        config=mock_config,
        risks=risks,
        retrieval=RetrievalConfig(query_gen=False, rrf_min_score=0.02),
        ocr=False,
    )
    assert index.calls, "per-chunk retrieval must call hybrid_search"
    assert all(call["rerank"] is True for call in index.calls)

    # Query generation on: every query-gen search passes rerank=False.
    index.calls.clear()
    from asago_policy_mapper.extract.querygen import QueryResult

    monkeypatch.setattr(
        pipeline,
        "generate_queries",
        lambda *args, **kwargs: ([QueryResult(query="q", chunk_indices=[0], section=None)], set()),
    )
    pipeline.run_extraction(
        documents=[doc],
        client=None,
        config=mock_config,
        risks=risks,
        retrieval=RetrievalConfig(query_gen=True),
        ocr=False,
    )
    assert any(call["rerank"] is False for call in index.calls), "query-gen searches pass rerank=False"

    # Cross-encoder off: the per-chunk call passes rerank=False.
    index.calls.clear()
    pipeline.run_extraction(
        documents=[doc],
        client=None,
        config=mock_config,
        risks=risks,
        retrieval=RetrievalConfig(query_gen=False, use_cross_encoder=False, rrf_min_score=0.01),
        ocr=False,
    )
    assert index.calls
    assert all(call["rerank"] is False for call in index.calls)


# --- judge contexts (step 4) ---


def test_judge_context_tokens_used_when_positive(mock_config, monkeypatch, tmp_path):
    """A positive --judge-context-tokens builds judge-only contexts.

    ``_run_judge`` passes each borderline chunk a context built with
    ``build_chunk_contexts(chunks, n)`` instead of the grounding context,
    under the requested budget (the neighbor chunk stays out).
    """
    from asago_policy_mapper.extract import pipeline
    from asago_policy_mapper.extract.parse import Chunk
    from asago_policy_mapper.extract.retrieve import ChunkResult, build_chunk_contexts

    seen = {}

    def fake_judge_one(i, cr, client, model, call_collector, judge_prompt, max_batch_size, context_text):
        seen[i] = context_text
        return i, []

    monkeypatch.setattr(pipeline, "_judge_one", fake_judge_one)

    chunks = [Chunk(text="word " * 40, source="p.md", index=0), Chunk(text="more words " * 40, source="p.md", index=1)]
    cr = ChunkResult(
        chunk_index=0,
        source="p.md",
        page=None,
        section=None,
        accepted=[],
        borderline=[ScoredCandidate(risk_id="R-001", risk_name="n", risk_description="d", cross_encoder_score=0.5)],
        borderline_judged=[],
        stats={},
    )
    retrieval = RetrievalConfig(query_gen=False, judge_context_tokens=30)

    pipeline._run_judge([cr], chunks, MagicMock(), mock_config, retrieval, [], chunk_contexts=["grounding context"])

    # The judge context is built from the chunks under the token budget,
    # not the grounding context: the 30-token budget keeps the neighbor
    # chunk out where the default 1024-token budget would include it.
    expected = build_chunk_contexts(chunks, max_context_tokens=30)[0]
    assert seen[0] == expected
    assert seen[0] != "grounding context"
    assert "more words" not in seen[0]

    # A budget of exactly 1 still builds judge contexts (the smallest
    # positive budget never expands to neighbors); 0 shares the grounding
    # context instead.
    seen.clear()
    retrieval = RetrievalConfig(query_gen=False, judge_context_tokens=1)
    pipeline._run_judge([cr], chunks, MagicMock(), mock_config, retrieval, [], chunk_contexts=["grounding context"])
    assert seen[0] == build_chunk_contexts(chunks, max_context_tokens=1)[0]
    assert seen[0] != "grounding context"

    seen.clear()
    retrieval = RetrievalConfig(query_gen=False, judge_context_tokens=0)
    pipeline._run_judge([cr], chunks, MagicMock(), mock_config, retrieval, [], chunk_contexts=["grounding context"])
    assert seen[0] == "grounding context"


def test_judge_context_budget_does_not_change_grounding_context(mock_config, tmp_path, monkeypatch):
    """The judge-only contexts never leak into grounding or the prefilter.

    The run has borderline candidates (so the judge builds its own
    contexts) and two chunks (so a judge-sized budget differs from the
    grounding budget). Grounding still receives exactly
    ``build_chunk_contexts(chunks)`` — the default 1024-token budget.
    """
    from asago_policy_mapper.extract import pipeline

    grounding_contexts = []
    judge_contexts = []

    def fake_ground(**kwargs):
        grounding_contexts.append(kwargs["chunk_text"])
        return {"R-001": ([], "high")}

    def fake_judge_one(i, cr, client, model, call_collector, judge_prompt, max_batch_size, context_text):
        judge_contexts.append(context_text)
        return i, [cr.borderline[0]]

    monkeypatch.setattr(pipeline, "ground_and_extract_evidence", fake_ground)
    monkeypatch.setattr(pipeline, "_judge_one", fake_judge_one)

    class _StubIndex:
        variant_map = {}
        risks = []

        def hybrid_search(self, query, top_k=50, bm25_rescue_rank=0, rrf_min_score=0.0, rerank=None):
            return [ScoredCandidate(risk_id="R-001", risk_name="n", risk_description="d", cross_encoder_score=0.5)]

        def get_taxonomy(self, rid):
            return "test-taxonomy"

    monkeypatch.setattr(pipeline, "_build_risk_index", lambda risks, retrieval: _StubIndex())

    doc = tmp_path / "p.md"
    doc.write_text("AI policy text.\n" + "filler " * 300 + "\nsecond chunk words here.\n" + "filler " * 300)
    # threshold_high=0.9, threshold_low=0.1: the 0.5-scored candidate is
    # borderline, so the judge runs and builds its 10-token contexts; the
    # judged candidate joins accepted and grounding runs over it.
    retrieval = RetrievalConfig(
        query_gen=False,
        judge_context_tokens=10,
        grounding_passes=1,
        threshold_high=0.9,
        threshold_low=0.1,
    )

    pipeline.run_extraction(
        documents=[doc],
        client=MagicMock(),
        config=mock_config,
        risks=[
            SimpleNamespace(
                id="R-001",
                name="n",
                description="d",
                concern="",
                risk_type="",
                isDefinedByTaxonomy="t",
                isPartOf="",
                exact_mappings=[],
                close_mappings=[],
                broad_mappings=[],
                narrow_mappings=[],
                related_mappings=[],
            )
        ],
        retrieval=retrieval,
        ocr=False,
    )

    assert judge_contexts, "the judge must run over borderline candidates"
    assert grounding_contexts, "grounding must still run with its own contexts"
    from asago_policy_mapper.extract.parse import chunk_documents, parse_document
    from asago_policy_mapper.extract.retrieve import build_chunk_contexts

    parsed = [parse_document(doc, ocr=False)]
    chunks = chunk_documents(parsed, max_tokens=retrieval.chunk_max_tokens)
    expected = build_chunk_contexts(chunks)
    assert grounding_contexts == expected
    # The judge context is the budget-constrained one, not the grounding
    # context: a leak would show as grounding receiving judge_contexts.
    assert judge_contexts[0] != expected[0]


# --- the warnings (step 6) ---


def test_overlap_warns_not_errors(tmp_path, capsys, monkeypatch):
    """Overlapping flags warn on stderr and the run proceeds.

    ``--no-judge`` with ``--no-cross-encoder`` is redundant and
    ``--judge-context-tokens`` is unused without the LLM judge, but both
    combinations run to completion (exit 0) with a warning, not a
    validation error.
    """
    from asago_policy_mapper.extract import pipeline

    class _Risk:
        id = "r1"
        name = "r1"
        description = "A risk."
        isDefinedByTaxonomy = "test"
        isPartOf = ""

    class FakeNexus:
        def __init__(self, base_dir=None):
            pass

        def get_all_risks(self):
            return [_Risk()]

    import asago_policy_mapper.cli as cli

    captured = {}

    def fake_run_extraction_capture(**kwargs):
        captured["retrieval"] = kwargs["retrieval"]
        return pipeline._empty_result(kwargs["documents"])

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction_capture)
    monkeypatch.setattr(cli, "_attach_mitigations", lambda result, nexus_base_dir: None)
    monkeypatch.setattr("ai_atlas_nexus.AIAtlasNexus", FakeNexus)

    policy = _policy(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "extract",
            str(policy),
            "-o",
            str(out),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test-model",
            "--no-judge",
            "--no-cross-encoder",
            "--no-grounding",
            "--no-query-gen",
            "--judge-context-tokens",
            "512",
        ],
        env={"POLICY_MAPPER_BASE_URL": "", "POLICY_MAPPER_MODEL": ""},
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "--no-judge is redundant with --no-cross-encoder" in result.output
    assert "--judge-context-tokens is unused" in result.output
