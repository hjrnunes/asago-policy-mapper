"""Completion-order independence of the variant-grounding pool (issue #150 part 1).

The snapshot harness runs the pipeline's thread pools serially, so it
proves orchestration only: it cannot exercise completion order. These
tests drive _run_variant_grounding with executors whose futures complete
in different orders — the serial submission order the harness uses, and
the reverse — and assert the merged variant matches, quotes included,
stay identical with the same stubbed answers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.extract.merge import merge_matches
from asago_policy_mapper.extract.models import EvidenceSpan, RetrievalScores, RiskMatch
from asago_policy_mapper.extract.pipeline import _run_variant_grounding

PARENT_ID = "ai-risk-taxonomy-unauthorized-processing"
VARIANTS = [
    {
        "risk_id": f"{PARENT_ID}---health-data",
        "name": "Health data",
        "description": "Unauthorized processing of health records.",
        "taxonomy": "ai-risk-taxonomy",
    },
    {
        "risk_id": f"{PARENT_ID}---financial-data",
        "name": "Financial data",
        "description": "Unauthorized processing of financial records.",
        "taxonomy": "ai-risk-taxonomy",
    },
]
VARIANT_MAP = {PARENT_ID: VARIANTS}


def _evidence(text, chunk_index, score=0.5):
    return EvidenceSpan(
        text=text,
        document="policy.pdf",
        chunk_index=chunk_index,
        cross_encoder_score=score,
    )


def _parent_match(spans):
    return RiskMatch(
        risk_id=PARENT_ID,
        risk_name="Unauthorized processing",
        risk_description="Processing of personal data without legal basis or consent.",
        taxonomy="ai-risk-taxonomy",
        confidence=0.85,
        confidence_source="cross_encoder",
        grounding_confidence="high",
        accepted_by="threshold",
        evidence=list(spans),
        scores=RetrievalScores(
            bm25_rank=3,
            embedding_distance=0.15,
            cross_encoder_score=0.85,
            rrf_score=0.04,
        ),
    )


def _chunks(texts):
    return [SimpleNamespace(text=t, page=i, section=f"Section {i}") for i, t in enumerate(texts)]


class _CompletedFuture:
    """Future whose work already ran; result() replays it."""

    def __init__(self, result):
        self._result = result

    def result(self, timeout=None):
        return self._result


class _SubmissionOrderExecutor:
    """Executor that completes every task at submission, in submission order.

    The consumer's as_completed loop (patched to identity) then observes
    completions in submission order — the serial behavior the snapshot
    harness pins.
    """

    def __init__(self, max_workers=None):
        pass

    def submit(self, fn, *args, **kwargs):
        return _CompletedFuture(fn(*args, **kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ReverseCompletionExecutor(_SubmissionOrderExecutor):
    """Same work, same answers; only the completion order differs.

    Submissions still run in order, but the consumer's as_completed loop
    (patched to reverse) observes the futures completing in reverse
    submission order — the opposite of the serial executor. Every result
    flows through future.result(), so this exercises completion-order
    independence without threads.
    """


def _stub_ground_variants(**kwargs):
    """Deterministic answers keyed only by the chunk being grounded.

    Returns both variants grounded with quotes naming the chunk, so the
    assertion sees which chunk each parent's evidence selection picked.
    """
    ci = kwargs["chunk_index"]
    return {
        v["risk_id"]: ([_evidence(f"quote for {v['name']} from chunk {ci}", ci)], "high") for v in kwargs["variants"]
    }


def _run(parents, chunks, monkeypatch, executor_cls, as_completed):
    from asago_policy_mapper.extract import pipeline

    monkeypatch.setattr(pipeline, "ground_variants", _stub_ground_variants)
    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", executor_cls)
    monkeypatch.setattr(pipeline, "as_completed", as_completed)
    return _run_variant_grounding(
        parents,
        chunks,
        VARIANT_MAP,
        MagicMock(),
        SimpleNamespace(model="test-model"),
        None,
        max_workers=4,
    )


def _quotes(result):
    """Merged variant quotes by risk id, the pipeline's own dedup applied.

    The pipeline feeds _run_variant_grounding's output through
    merge_matches, which collapses same-id variant matches from parallel
    parents into one; the same merge here makes the comparison observe
    what a run actually reports.
    """
    merged = merge_matches(list(result))
    return {m.risk_id: [e.text for e in m.evidence] for m in sorted(merged, key=lambda m: m.risk_id)}


@pytest.fixture
def two_parents():
    """Two parents whose evidence orders differ; all scores tie at 0.5.

    Parent 1's spans list the late chunk first; parent 2's list the mid
    chunk first. With tied scores the deterministic key picks parent 1's
    chunk 1 and parent 2's chunk 0 regardless of list order.
    """
    return [
        _parent_match([_evidence("p1 late chunk", 2), _evidence("p1 early chunk", 1)]),
        _parent_match([_evidence("p2 mid chunk", 1), _evidence("p2 early chunk", 0)]),
    ]


def test_pipeline_reverse_completion_same_quotes(two_parents, monkeypatch):
    """Futures completing in reverse order keep the same variant quotes."""
    chunks = _chunks(["chunk zero", "chunk one", "chunk two"])

    forward = _run(
        two_parents,
        chunks,
        monkeypatch,
        _SubmissionOrderExecutor,
        lambda fs, timeout=None: list(fs),
    )
    reverse = _run(
        two_parents,
        chunks,
        monkeypatch,
        _ReverseCompletionExecutor,
        lambda fs, timeout=None: list(reversed(list(fs))),
    )

    assert _quotes(forward) == _quotes(reverse)
    quotes = _quotes(forward)
    assert sorted(quotes) == sorted(v["risk_id"] for v in VARIANTS), "both variants survive with quotes"
    # every variant keeps at least one quote; a variant that lost all its
    # evidence in both runs would otherwise pass the id checks above
    assert all(quotes.values()), "each variant carries at least one quote"
    # the deterministic evidence key picked chunk 1 for parent 1 and
    # chunk 0 for parent 2; the merged evidence therefore carries quotes
    # from both chunks, and the completion order changes neither pick
    all_quote_texts = [q for qs in quotes.values() for q in qs]
    assert sorted(q.rsplit(" ", 1)[-1] for q in all_quote_texts) == ["0", "0", "1", "1"]
    assert all(q.startswith("quote for ") for q in all_quote_texts)
    # every variant carries a quote from each grounded chunk
    for quote_texts in quotes.values():
        assert sorted(q.rsplit(" ", 1)[-1] for q in quote_texts) == ["0", "1"]


def test_pipeline_reverse_completion_same_risk_set(two_parents, monkeypatch):
    """Completion order changes neither which variants survive nor their scores."""
    chunks = _chunks(["chunk zero", "chunk one", "chunk two"])

    forward = _run(two_parents, chunks, monkeypatch, _SubmissionOrderExecutor, lambda fs, timeout=None: list(fs))
    reverse = _run(
        two_parents,
        chunks,
        monkeypatch,
        _ReverseCompletionExecutor,
        lambda fs, timeout=None: list(reversed(list(fs))),
    )

    forward_ids = sorted(m.risk_id for m in forward)
    reverse_ids = sorted(m.risk_id for m in reverse)
    assert forward_ids == reverse_ids
    forward_scores = {m.risk_id: (m.confidence, m.accepted_by) for m in forward}
    reverse_scores = {m.risk_id: (m.confidence, m.accepted_by) for m in reverse}
    assert forward_scores == reverse_scores
