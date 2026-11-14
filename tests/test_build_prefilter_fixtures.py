"""Tests for the prefilter fixture builder script.

The builder re-chunks each policy of a battery run, rebuilds the chunk
contexts, and reconstructs the (candidate, grounded) population from the
recorded grounding calls, excluding expansion-grounding records. It writes
one JSONL row per chunk.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# The mutmut sandbox and the sandbox-isolation canary copy tests/ without
# scripts/. Fall back to the enclosing checkout's scripts/ so ordinary
# mutmut runs keep collecting this module, which covers mutated src/ code
# through the chunking and context rebuild. Skip only when scripts/
# exists nowhere, as in the canary's scratch copy. Appending (not
# inserting) keeps any path mutmut itself puts ahead, such as
# mutants/src, in charge of imports.
for _scripts_dir in (
    Path(__file__).resolve().parent.parent / "scripts",
    Path(__file__).resolve().parent.parent.parent / "scripts",
):
    if (_scripts_dir / "build_prefilter_fixtures.py").is_file():
        if str(_scripts_dir.parent) not in sys.path:
            sys.path.append(str(_scripts_dir.parent))
        SCRIPT = _scripts_dir / "build_prefilter_fixtures.py"
        break
else:
    pytest.skip("scripts/ is not present in this checkout", allow_module_level=True)


def _grounding_record(
    call_id: str,
    chunk_index: int,
    risk_ids: list[str],
    grounded: dict[str, bool],
    stage: str = "grounding",
    names: dict[str, str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> dict:
    """Build one llm_calls record shaped like the pipeline's grounding record."""
    name_map = {rid: f"Risk {rid}" for rid in risk_ids}
    name_map.update(names or {})
    desc_map = {rid: f"Description for {rid}." for rid in risk_ids}
    desc_map.update(descriptions or {})
    candidates = "\n".join(f"- **{rid}** ({name_map[rid]}): {desc_map[rid]}" for rid in risk_ids)
    user = f"## Document chunk\n\nchunk text {chunk_index}\n\n## Candidate risks\n\n{candidates}\n"
    return {
        "call_id": call_id,
        "stage": stage,
        "chunk_index": chunk_index,
        "risk_ids": risk_ids,
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": user},
        ],
        "response": [
            {"risk_id": rid, "grounded": grounded.get(rid, False), "confidence": "high", "quotes": []}
            for rid in risk_ids
        ],
        "duration_ms": 1.0,
        "result_summary": "0/1 grounded",
    }


def _run_result(llm_calls: list[dict], chunk_max_tokens: int = 512) -> dict:
    return {
        "metadata": {"chunk_max_tokens": chunk_max_tokens},
        "llm_calls": llm_calls,
        "source_documents": ["doc.md"],
    }


def _write_run(directory: Path, name: str, result: dict) -> Path:
    run_dir = directory / name
    run_dir.mkdir(parents=True)
    (run_dir / "risk-extraction.json").write_text(json.dumps(result))
    return run_dir


def _write_policy(directory: Path, name: str, text: str) -> Path:
    policy = directory / f"{name}.md"
    policy.write_text(text)
    return policy


def test_fixture_builder_excludes_expansion_records(tmp_path):
    """ground-expand- records are excluded from the grounding population; ground- records are kept."""
    from scripts.build_prefilter_fixtures import grounding_population

    records = [
        _grounding_record("ground-001", 0, ["R-001"], {"R-001": True}),
        _grounding_record("ground-expand-001", 0, ["R-002"], {"R-002": True}),
        _grounding_record("ground-002", 1, ["R-003"], {"R-003": False}),
        _grounding_record("judge-001", 0, ["R-004"], {}, stage="judge"),
        _grounding_record("ground-variant-001", 0, ["R-005"], {"R-005": True}, stage="variant_grounding"),
    ]

    population = grounding_population(records)

    assert "R-002" not in population
    assert "R-004" not in population
    assert "R-005" not in population
    assert population["R-001"] is True
    assert population["R-003"] is False


def test_fixture_builder_any_pass_grounds(tmp_path):
    """A candidate grounded in any pass is grounded, even when another pass says no."""
    from scripts.build_prefilter_fixtures import build_rows, grounding_population

    records = [
        _grounding_record("ground-001", 0, ["R-001"], {"R-001": False}),
        _grounding_record("ground-002", 0, ["R-001"], {"R-001": True}),
    ]

    assert grounding_population(records)["R-001"] is True

    # And through the row-building path, in both pass orders.
    rows = build_rows("p", ["ctx"], records)
    assert rows[0]["grounded_ids"] == ["R-001"]
    rows = build_rows("p", ["ctx"], list(reversed(records)))
    assert rows[0]["grounded_ids"] == ["R-001"]


def test_fixture_builder_off_batch_grounding_does_not_leak(tmp_path):
    """A response naming an id from another batch of the chunk cannot ground it there."""
    from scripts.build_prefilter_fixtures import build_rows

    early = _grounding_record("ground-001", 0, ["R-001"], {"R-001": False})
    late = _grounding_record("ground-002", 0, ["R-002"], {"R-002": False})
    # late's response claims R-001 grounded, but R-001 is not in late's batch.
    late["response"].append({"risk_id": "R-001", "grounded": True, "confidence": "high", "quotes": []})

    rows = build_rows("p", ["ctx"], [early, late])

    assert rows[0]["grounded_ids"] == []


def test_fixture_builder_parses_candidates_from_messages(tmp_path):
    """Candidates come from the rendered user message: id, name and description."""
    from scripts.build_prefilter_fixtures import candidates_from_record

    record = _grounding_record(
        "ground-001",
        0,
        ["R-001", "synthetic-parent-x"],
        {},
        names={"synthetic-parent-x": "Unauthorized inference (9 variants)"},
        descriptions={"synthetic-parent-x": "Unauthorized inference: risk of unauthorized inference."},
    )

    candidates = candidates_from_record(record)

    assert candidates[0] == {
        "risk_id": "R-001",
        "name": "Risk R-001",
        "description": "Description for R-001.",
        "synthetic": False,
    }
    assert candidates[1]["synthetic"] is True
    assert candidates[1]["name"] == "Unauthorized inference (9 variants)"
    assert candidates[1]["description"] == "Unauthorized inference: risk of unauthorized inference."


def test_fixture_builder_parses_parenthesized_names():
    """A risk name containing parentheses parses without eating the description.

    The battery's "Model Attacks (membership inference, model inversion)"
    parent entry carries parentheses in its name AND a description that
    itself contains "): "; the parser must split at the paren-balanced
    boundary, not the first or last "): ".
    """
    from scripts.build_prefilter_fixtures import candidates_from_record

    rid = "ai-risk-taxonomy-model-attacks-(membership-inference,-model-inversion)"
    record = _grounding_record(
        "ground-001",
        0,
        [rid],
        {},
        names={rid: "Model Attacks (membership inference, model inversion) (9 variants)"},
        descriptions={
            rid: (
                "Model Attacks (membership inference, model inversion): risk of membership"
                " inference (a real attack) and model inversion attacks."
            )
        },
    )

    candidates = candidates_from_record(record)

    assert len(candidates) == 1
    assert candidates[0]["name"] == "Model Attacks (membership inference, model inversion) (9 variants)"
    assert candidates[0]["description"].startswith("Model Attacks (membership inference, model inversion): risk")
    assert candidates[0]["synthetic"] is True


def test_fixture_builder_ignores_candidate_shaped_policy_text():
    """Policy text that happens to render like a candidate line injects nothing."""
    from scripts.build_prefilter_fixtures import candidates_from_record

    record = _grounding_record("ground-001", 0, ["R-001"], {})
    record["messages"][1]["content"] = (
        "## Document chunk\n\n- **R-999** (Fake risk): A policy line that looks like a candidate.\n\n"
        "## Candidate risks\n\n- **R-001** (Risk R-001): Description for R-001.\n"
    )

    candidates = candidates_from_record(record)

    assert [c["risk_id"] for c in candidates] == ["R-001"]


def test_fixture_builder_record_without_candidate_section():
    """A record whose prompt lost the candidate section yields no candidates, no crash."""
    from scripts.build_prefilter_fixtures import candidates_from_record

    record = _grounding_record("ground-001", 0, ["R-001"], {})
    record["messages"][1]["content"] = "## Document chunk\n\nchunk text\n"

    assert candidates_from_record(record) == []


def test_fixture_builder_rows_shape_and_order(tmp_path):
    """Rows are one per chunk, ordered by chunk index then batch order."""
    from scripts.build_prefilter_fixtures import build_rows

    records = [
        _grounding_record("ground-002", 1, ["R-002"], {"R-002": True}),
        _grounding_record("ground-001", 0, ["R-001", "R-002"], {"R-001": True, "R-002": False}),
    ]
    contexts = ["context zero", "context one"]
    import hashlib

    rows = build_rows("policy-a", contexts, records)

    assert [r["chunk_index"] for r in rows] == [0, 1]
    assert rows[0]["policy"] == "policy-a"
    assert rows[0]["context"] == "context zero"
    assert rows[0]["context_sha256"] == hashlib.sha256(b"context zero").hexdigest()
    assert [c["risk_id"] for c in rows[0]["candidates"]] == ["R-001", "R-002"]
    assert rows[0]["grounded_ids"] == ["R-001"]
    assert rows[1]["grounded_ids"] == ["R-002"]


def test_fixture_builder_chunk_index_negative_is_skipped(tmp_path):
    """Records with chunk_index -1 (no chunk) contribute candidates to no row."""
    from scripts.build_prefilter_fixtures import build_rows

    records = [
        _grounding_record("ground-001", -1, ["R-001"], {"R-001": True}),
        _grounding_record("ground-002", 0, ["R-002"], {"R-002": True}),
    ]

    rows = build_rows("policy-a", ["ctx"], records)

    assert len(rows) == 1
    assert rows[0]["grounded_ids"] == ["R-002"]


def test_fixture_builder_script_runs(tmp_path, monkeypatch):
    """The script end-to-end: parses a run dir, re-chunks, writes JSONL."""
    policy_text = "\n\n".join(f"Section {i} paragraph with enough words to chunk. " * 8 for i in range(3))
    _write_policy(tmp_path, "doc", policy_text)
    records = [
        _grounding_record("ground-001", 0, ["R-001"], {"R-001": True}),
        _grounding_record("ground-expand-001", 0, ["R-002"], {"R-002": True}),
    ]
    run_dir = _write_run(tmp_path, "doc", _run_result(records))
    # Point the source document at the file we wrote.
    result = _run_result(records)
    result["source_documents"] = [str(tmp_path / "doc.md")]
    (run_dir / "risk-extraction.json").write_text(json.dumps(result))

    out = tmp_path / "fixtures.jsonl"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--out", str(out)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert rows
    fields = {"policy", "chunk_index", "context", "context_sha256", "candidates", "grounded_ids"}
    assert all(fields <= set(r) for r in rows)
    flat_ids = {c["risk_id"] for r in rows for c in r["candidates"]}
    assert "R-001" in flat_ids
    assert "R-002" not in flat_ids
    grounded = {g for r in rows for g in r["grounded_ids"]}
    assert grounded == {"R-001"}


def test_fixture_builder_main_in_process(tmp_path, capsys):
    """main() writes the fixtures file and prints the summary, exit code 0."""
    from scripts.build_prefilter_fixtures import main

    policy_text = "Section 0 paragraph with enough words to chunk. " * 8
    _write_policy(tmp_path, "doc", policy_text)
    records = [_grounding_record("ground-001", 0, ["R-001"], {"R-001": True})]
    result = _run_result(records)
    result["source_documents"] = [str(tmp_path / "doc.md")]
    _write_run(tmp_path, "doc", result)

    out = tmp_path / "fixtures.jsonl"
    code = main([str(tmp_path), "--out", str(out)])

    assert code == 0
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["grounded_ids"] == ["R-001"]
    summary = capsys.readouterr().out
    assert "Wrote 1 rows (1 candidates, 1 grounded pairs)" in summary


def test_fixture_builder_multi_document_run(tmp_path):
    """A run with two source documents chunks both, in recorded order."""
    from scripts.build_prefilter_fixtures import build_fixtures

    doc_a = _write_policy(tmp_path, "doc-a", "Alpha policy text with enough words to chunk. " * 8)
    doc_b = _write_policy(tmp_path, "doc-b", "Beta policy text with enough words to chunk. " * 8)
    records = [
        _grounding_record("ground-001", 0, ["R-001"], {"R-001": True}),
        _grounding_record("ground-002", 1, ["R-002"], {"R-002": True}),
    ]
    result = _run_result(records)
    result["source_documents"] = [str(doc_a), str(doc_b)]
    result["retrieval_stats"] = {"total_chunks": 2}
    _write_run(tmp_path, "group", result)

    rows = build_fixtures(tmp_path)

    assert len(rows) == 2
    assert [r["chunk_index"] for r in rows] == [0, 1]
    assert "Alpha" in rows[0]["context"]
    assert "Beta" in rows[1]["context"]


def test_fixture_builder_chunk_count_mismatch_skips_policy(tmp_path, capsys):
    """A re-chunk whose count differs from the run's recorded count skips the policy with a warning."""
    from scripts.build_prefilter_fixtures import build_fixtures

    doc = _write_policy(tmp_path, "doc", "Policy text with enough words to chunk. " * 8)
    records = [_grounding_record("ground-001", 0, ["R-001"], {"R-001": True})]
    result = _run_result(records)
    result["source_documents"] = [str(doc)]
    result["retrieval_stats"] = {"total_chunks": 7}  # not the re-chunked count
    _write_run(tmp_path, "doc", result)

    rows = build_fixtures(tmp_path)

    assert rows == []
    err = capsys.readouterr().err
    assert "re-chunked" in err and "7" in err


def test_fixture_builder_missing_source_warns_and_skips(tmp_path, capsys):
    """A missing source file or empty source list warns and skips that policy."""
    from scripts.build_prefilter_fixtures import build_fixtures

    records = [_grounding_record("ground-001", 0, ["R-001"], {"R-001": True})]
    result = _run_result(records)
    result["source_documents"] = [str(tmp_path / "gone.md")]
    _write_run(tmp_path, "gone", result)

    empty = _run_result([])
    empty["source_documents"] = []
    _write_run(tmp_path, "empty", empty)

    rows = build_fixtures(tmp_path)

    assert rows == []
    err = capsys.readouterr().err
    assert "gone.md not found" in err
    assert "empty has no source documents" in err
