"""Tests for embedding optimization dataset loader."""

from pathlib import Path

import pytest

_GT_DIR = Path(__file__).resolve().parent.parent / "evals" / "ground_truth"
_POLICY_DIR = Path(__file__).resolve().parent.parent / "policy_examples"


@pytest.mark.skipif(
    not (_GT_DIR / "sap.yaml").exists() or not any(_POLICY_DIR.glob("sap.*")),
    reason="Ground truth or policy file for 'sap' not available",
)
@pytest.mark.slow
def test_build_example_sap():
    from experiments.dspy_embedding.dataset import _build_example

    example = _build_example("sap", chunk_max_tokens=512)
    assert example is not None
    assert len(example.chunk_texts) > 0
    assert len(example.risk_ids) > 0
    assert example.policy_name == "sap"
    # All risk IDs should be risk-level, not category-level
    for rid in example.risk_ids:
        assert not rid.startswith("nist-ai-rmf")
        assert not rid.startswith("owasp-llm-2.0")


def test_build_example_nonexistent():
    from experiments.dspy_embedding.dataset import _build_example

    example = _build_example("nonexistent-policy-xyz")
    assert example is None


def test_build_threshold_examples_accepts_rank_label(tmp_path, monkeypatch):
    """Auto-accepted candidates include both rank and threshold labels."""
    import json as _json

    from asago_policy_mapper.extract.parse import Chunk
    from experiments.dspy_judge import dataset as dspy_judge_dataset

    # a ground-truth file with one risk id, so the rank-labelled risk is a positive
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "sap.yaml").write_text("risks:\n  - id: risk-a\n")
    monkeypatch.setattr(dspy_judge_dataset, "_GT_DIR", gt_dir)

    run_dir = tmp_path / "run"
    (run_dir / "sap").mkdir(parents=True)
    (run_dir / "sap" / "risk-extraction.json").write_text(
        _json.dumps(
            {
                "risks": [
                    {"risk_id": "risk-a", "risk_name": "A", "accepted_by": "rank"},
                    {"risk_id": "risk-b", "risk_name": "B", "accepted_by": "llm_judge"},
                    {"risk_id": "risk-c", "risk_name": "C", "accepted_by": "threshold"},
                ],
                "chunks": [{"index": 0, "accepted_risk_ids": ["risk-a", "risk-b", "risk-c"]}],
            }
        )
    )

    # the function skips chunk text shorter than 50 chars, so use a long one
    chunk = Chunk(index=0, text="x" * 80, source="doc.txt")
    examples = dspy_judge_dataset._build_threshold_examples_from_run(
        "sap", run_dir, {}, chunks=[chunk], max_per_chunk=10
    )
    # rank (default-run label) and threshold both auto-accepted; llm_judge excluded
    assert len(examples) == 1
    assert "risk-a" in examples[0].candidate_risks
    assert "risk-c" in examples[0].candidate_risks
    assert "risk-b" not in examples[0].candidate_risks
