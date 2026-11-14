"""Tests for the thread-local pipeline-stage marker in llm.py.

Stage workers run in parallel, so the marker must be thread-local: one
worker's stage must not leak into another's token totals or incidents.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from asago_policy_mapper.extract.attribute import ground_and_extract_evidence
from asago_policy_mapper.extract.models import ScoredCandidate, _JudgeVerdict, _JudgeVerdicts
from asago_policy_mapper.extract.querygen import ChunkGroup, generate_queries
from asago_policy_mapper.extract.retrieve import judge_borderline
from asago_policy_mapper.llm import (
    LLMConfig,
    TokenTracker,
    _track_completion,
    _wrap_with_tracking,
    current_llm_stage,
    llm_stage,
)


def _usage(prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


class TestLlmStage:
    def test_llm_stage_attributes_tokens(self):
        """Completions made inside llm_stage land in that stage's per-stage totals."""
        tracker = TokenTracker()
        config = LLMConfig(base_url="http://test", model="test-model", max_context=0)
        client = MagicMock()
        completion = SimpleNamespace(usage=_usage())
        client.create_with_completion.return_value = ("parsed", completion)
        _wrap_with_tracking(client, tracker, config)

        with llm_stage("judge"):
            client.chat.completions.create(model="test-model", messages=[{"role": "user", "content": "hi"}])

        assert tracker.per_stage["judge"]["calls"] == 1
        assert tracker.per_stage["judge"]["total_tokens"] == 15
        assert tracker.total_tokens == 15

    def test_llm_stage_is_thread_local(self):
        """Parallel workers in different stages keep their own totals.

        Both threads barrier inside their stage contexts; a module-global
        marker would let one worker's stage win and put the other's tokens in
        the wrong bucket.
        """
        tracker = TokenTracker()
        config = LLMConfig(base_url="http://test", model="test-model", max_context=0)
        client = MagicMock()
        client.create_with_completion.return_value = ("parsed", SimpleNamespace(usage=_usage()))
        _wrap_with_tracking(client, tracker, config)
        barrier = threading.Barrier(2)
        assignments = {}

        def worker(stage: str) -> None:
            with llm_stage(stage):
                barrier.wait()
                client.chat.completions.create(model="test-model", messages=[{"role": "user", "content": "hi"}])
                assignments[stage] = current_llm_stage()

        threads = [threading.Thread(target=worker, args=(s,)) for s in ("judge", "grounding")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert assignments == {"judge": "judge", "grounding": "grounding"}
        assert tracker.per_stage["judge"]["calls"] == 1
        assert tracker.per_stage["grounding"]["calls"] == 1
        assert set(tracker.per_stage) == {"judge", "grounding"}

    def test_llm_stage_restores_previous_stage_on_exit(self):
        assert current_llm_stage() is None
        with llm_stage("outer"):
            with llm_stage("inner"):
                assert current_llm_stage() == "inner"
            assert current_llm_stage() == "outer"
        assert current_llm_stage() is None

    def test_current_llm_stage_outside_context_is_none(self):
        assert current_llm_stage() is None

    def test_llm_stage_resets_on_exception(self):
        with pytest.raises(RuntimeError):
            with llm_stage("judge"):
                raise RuntimeError("boom")
        assert current_llm_stage() is None


class TestIncidentCarriesThreadStage:
    def test_incident_carries_thread_stage(self):
        """record_incident uses the calling thread's stage, not another worker's."""
        tracker = TokenTracker()
        tracker.set_stage("expansion_grounding")

        def record(stage: str, error: RuntimeError) -> None:
            with llm_stage(stage):
                try:
                    raise error
                except RuntimeError:
                    tracker.record_incident("validation_exhausted", str(error))

        t1 = threading.Thread(target=record, args=("judge", RuntimeError("judge error")))
        t2 = threading.Thread(target=record, args=("causal_synthesis", RuntimeError("causal error")))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        stages = {i["detail"]: i.get("stage") for i in tracker.incidents}
        assert stages == {
            "judge error": "judge",
            "causal error": "causal_synthesis",
        }

    def test_incident_falls_back_to_set_stage(self):
        """Without a thread stage, an incident keeps the explicit set_stage value."""
        tracker = TokenTracker()
        tracker.set_stage("grounding")
        tracker.record_incident("context_overflow", "reduced max_tokens")
        assert tracker.incidents[0]["stage"] == "grounding"

    def test_track_completion_uses_thread_stage(self):
        """_track_completion routes usage to the thread-local stage, not set_stage's."""
        tracker = TokenTracker()
        tracker.set_stage("expansion_grounding")
        with llm_stage("grounding"):
            _track_completion(tracker, SimpleNamespace(usage=_usage()))
        assert tracker.per_stage["grounding"]["calls"] == 1
        assert "expansion_grounding" not in tracker.per_stage


class TestCallSiteStages:
    """Each production call site wraps its client call in its named stage."""

    @staticmethod
    def _tracked_client() -> tuple[MagicMock, TokenTracker]:
        tracker = TokenTracker()
        config = LLMConfig(base_url="http://test", model="test-model", max_context=0)
        client = MagicMock()
        client.create_with_completion.return_value = ("parsed", SimpleNamespace(usage=_usage()))
        _wrap_with_tracking(client, tracker, config)
        return client, tracker

    def test_judge_batch_sets_judge_stage(self):
        client, tracker = self._tracked_client()
        client.create_with_completion.return_value = (
            _JudgeVerdicts(items=[_JudgeVerdict(risk_id="R-001", relevant=True, justification="ok")]),
            SimpleNamespace(usage=_usage()),
        )
        judge_borderline(
            [ScoredCandidate(risk_id="R-001", risk_name="Bias", risk_description="d", cross_encoder_score=0.9)],
            "chunk text",
            client,
            "test-model",
        )
        assert "judge" in tracker.per_stage

    def test_ground_and_extract_evidence_sets_grounding_stage(self):
        client, tracker = self._tracked_client()
        from asago_policy_mapper.extract.models import _RiskEvidence, _RiskEvidenceList

        client.create_with_completion.return_value = (
            _RiskEvidenceList(
                items=[_RiskEvidence(risk_id="R-001", grounded=True, confidence="high", quotes=["text"])]
            ),
            SimpleNamespace(usage=_usage()),
        )
        ground_and_extract_evidence(
            chunk_text="text",
            candidates=[
                ScoredCandidate(risk_id="R-001", risk_name="Bias", risk_description="d", cross_encoder_score=0.9)
            ],
            client=client,
            model="test-model",
            document="doc.pdf",
            chunk_index=0,
        )
        assert "grounding" in tracker.per_stage

    def test_generate_queries_sets_query_gen_stage(self):
        client, tracker = self._tracked_client()
        from types import SimpleNamespace as NS

        client.create_with_completion.return_value = (
            NS(queries=["q1"]),
            SimpleNamespace(usage=_usage()),
        )
        chunks = [NS(text="some text", source="doc.md", page=None, section=None, index=0)]
        generate_queries(chunks, [ChunkGroup(chunk_indices=[0], section=None)], client, "test-model")
        assert "query_gen" in tracker.per_stage


class TestStageNamesMatchSpec:
    @pytest.mark.parametrize(
        "stage",
        ["query_gen", "judge", "grounding", "variant_grounding", "expansion_grounding", "causal_synthesis"],
    )
    def test_spec_stage_names_are_accepted(self, stage):
        with llm_stage(stage):
            assert current_llm_stage() == stage
