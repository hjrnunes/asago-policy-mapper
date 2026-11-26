"""Tests for the prefilter answers recorder script.

The recorder runs an HTTP probability assessor over saved fixture rows
and writes the answers file the replay assessor and the offline
evaluation read. The tests cover the row-shape mapping and the full
save-then-record-then-replay chain against a mock transport, so no real
endpoint is contacted.
"""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

# The mutmut sandbox and the sandbox-isolation canary copy tests/ without
# scripts/. Fall back to the enclosing checkout's scripts/ so ordinary
# mutmut runs keep collecting this module, which covers mutated src/ code
# through the HTTP assessor. Skip only when scripts/ exists nowhere, as
# in the canary's scratch copy. Appending (not inserting) keeps any path
# mutmut itself puts ahead, such as mutants/src, in charge of imports.
for _scripts_dir in (
    Path(__file__).resolve().parent.parent / "scripts",
    Path(__file__).resolve().parent.parent.parent / "scripts",
):
    if (_scripts_dir / "record_prefilter_answers.py").is_file():
        if str(_scripts_dir.parent) not in sys.path:
            sys.path.append(str(_scripts_dir.parent))
        SCRIPT = _scripts_dir / "record_prefilter_answers.py"
        break
else:
    pytest.skip("scripts/ is not present in this checkout", allow_module_level=True)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _fixture_row(policy: str, context: str, candidates: list[dict], grounded: list[str]) -> dict:
    return {
        "policy": policy,
        "chunk_index": 0,
        "context": context,
        "context_sha256": _sha(context),
        "candidates": candidates,
        "grounded_ids": grounded,
    }


def _candidate(rid: str) -> dict:
    return {"risk_id": rid, "name": f"Risk {rid}", "description": f"Description for {rid}.", "synthetic": False}


def _write_fixtures(tmp_path: Path, rows: list[dict]) -> Path:
    fixtures = tmp_path / "fixtures.jsonl"
    fixtures.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return fixtures


def _scores_handler(request: httpx.Request) -> httpx.Response:
    """Serve a p for every requested item: 0.9 for R-x, 0.05 for R-y."""
    body = json.loads(request.content.decode())
    scores = [{"id": item["id"], "p": 0.9 if item["id"].startswith("R-x") else 0.05} for item in body["items"]]
    return httpx.Response(200, json={"scores": scores})


def test_record_answers_maps_outcomes_to_rows(tmp_path):
    """Scored candidates become p rows; missing ones become unavailable rows, each with latency."""
    from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
    from scripts.record_prefilter_answers import record_answers

    # The mock scores R-x1 and R-y1 but omits R-z1, which must become missing.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        scores = [{"id": item["id"], "p": 0.9} for item in body["items"] if item["id"] != "R-z1"]
        return httpx.Response(200, json={"scores": scores})

    endpoint = ProbabilityEndpoint(base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY")
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    rows = [_fixture_row("p", "ctx", [_candidate("R-x1"), _candidate("R-y1"), _candidate("R-z1")], [])]

    answers = record_answers(rows, assessor)

    by_rid = {a["risk_id"]: a for a in answers}
    assert by_rid["R-x1"]["p"] == 0.9
    assert by_rid["R-y1"]["p"] == 0.9
    assert by_rid["R-z1"]["unavailable"] == "missing"
    for answer in answers:
        assert answer["context_sha256"] == _sha("ctx")
        assert answer["latency_ms"] >= 0.0


def test_record_answers_covers_timeout_rows(tmp_path):
    """A timed-out batch records the reason, so replay keeps every candidate."""
    from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
    from scripts.record_prefilter_answers import record_answers

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    endpoint = ProbabilityEndpoint(
        base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY", timeout_s=0.05
    )
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    rows = [_fixture_row("p", "ctx", [_candidate("R-x1")], [])]

    answers = record_answers(rows, assessor)

    assert answers[0]["unavailable"] == "timeout"
    assert answers[0]["latency_ms"] >= 0.0


def test_record_then_replay_roundtrip(tmp_path):
    """The recorded answers file replays: scores, missing and latency all serve back."""
    from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
    from asago_policy_mapper.assessors.replay import ReplayAssessor
    from asago_policy_mapper.extract.screening import RELEVANCE_QUESTION_V1, DecisionCase, ProbabilityAnswer
    from scripts.record_prefilter_answers import record_answers

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        scores = [{"id": item["id"], "p": 0.9 if item["id"] == "R-x1" else 0.05} for item in body["items"]]
        return httpx.Response(200, json={"scores": scores})

    endpoint = ProbabilityEndpoint(base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY")
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    rows = [_fixture_row("p", "ctx", [_candidate("R-x1"), _candidate("R-y1")], [])]

    answers = record_answers(rows, assessor)
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text("\n".join(json.dumps(a) for a in answers) + "\n")

    replay = ReplayAssessor(answers_path)
    cases = [
        DecisionCase(risk_id=c["risk_id"], name=c["name"], description=c["description"]) for c in rows[0]["candidates"]
    ]
    batch = replay.assess(RELEVANCE_QUESTION_V1, "ctx", cases)

    assert batch.answers["R-x1"] == (ProbabilityAnswer(0.9),)
    assert batch.answers["R-y1"] == (ProbabilityAnswer(0.05),)
    assert replay.recorded_latencies is True
    # The replayed latency is the provider's recorded round-trip, not lookup time.
    recorded = {a["risk_id"]: a["latency_ms"] for a in answers}
    assert batch.latency_ms == max(recorded.values())


def test_record_answers_script_runs(tmp_path):
    """The CLI writes an answers file the replay assessor loads, against a local server."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's naming
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            scores = [{"id": item["id"], "p": 0.9} for item in body["items"]]
            payload = json.dumps({"scores": scores}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # silence the default stderr logging
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = [_fixture_row("p", "ctx", [_candidate("R-x1"), _candidate("R-y1")], [])]
        fixtures = _write_fixtures(tmp_path, rows)
        answers = tmp_path / "answers.jsonl"

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(fixtures),
                str(answers),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "prob-model",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert completed.returncode == 0, completed.stderr
    assert answers.exists()
    written = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()]
    assert len(written) == 2
    assert {a["risk_id"] for a in written} == {"R-x1", "R-y1"}
    assert all(a["p"] == 0.9 for a in written)
    assert all(a["latency_ms"] >= 0.0 for a in written)


def test_record_to_file_appends_per_row_and_flushes(tmp_path):
    """Each fixture row's answers are written before the next row is assessed.

    A recording that buffers everything and writes once loses the whole
    run on a mid-run crash; a per-row append keeps every row already
    assessed. The order of requests proves the writes interleave with the
    assessments: the second request arrives only after the first row's
    rows are on disk.
    """
    from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
    from scripts.record_prefilter_answers import record_to_file

    on_disk_after_first: list[int] = []
    contexts = ["ctx-a", "ctx-b"]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body["context"] == "ctx-b":
            # The second row's assessment starts only after the first row's
            # answers hit the file.
            on_disk_after_first.append(len(answers_path.read_text().splitlines()))
        scores = [{"id": item["id"], "p": 0.5} for item in body["items"]]
        return httpx.Response(200, json={"scores": scores})

    answers_path = tmp_path / "answers.jsonl"
    endpoint = ProbabilityEndpoint(base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY")
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    rows = [_fixture_row("p", ctx, [_candidate("R-x1"), _candidate("R-y1")], []) for ctx in contexts]

    written, skipped = record_to_file(rows, assessor, answers_path)

    assert written == 4
    assert skipped == 0
    assert on_disk_after_first == [2], "first row's answers must be on disk before the second is assessed"
    assert len(answers_path.read_text().splitlines()) == 4


def test_record_to_file_resumes_and_skips_recorded_keys(tmp_path):
    """A re-run skips keys already in the file, so no recorded answer is re-requested."""
    from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
    from scripts.record_prefilter_answers import record_to_file

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        requested.append(body["context"])
        scores = [{"id": item["id"], "p": 0.5} for item in body["items"]]
        return httpx.Response(200, json={"scores": scores})

    answers_path = tmp_path / "answers.jsonl"
    endpoint = ProbabilityEndpoint(base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY")
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    rows = [
        _fixture_row("p", "ctx-a", [_candidate("R-x1")], []),
        _fixture_row("p", "ctx-b", [_candidate("R-y1")], []),
    ]

    first_written, first_skipped = record_to_file(rows, assessor, answers_path)
    second_written, second_skipped = record_to_file(rows, assessor, answers_path)

    assert (first_written, first_skipped) == (2, 0)
    assert (second_written, second_skipped) == (0, 2)
    assert requested == ["ctx-a", "ctx-b"], "the re-run must not re-request any recorded context"
    assert len(answers_path.read_text().splitlines()) == 2, "the re-run must not duplicate rows"


def test_record_to_file_resumes_a_partially_recorded_run(tmp_path):
    """A crash mid-run leaves earlier rows recorded; the re-run records only the rest."""
    from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
    from scripts.record_prefilter_answers import record_to_file

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        scores = [{"id": item["id"], "p": 0.5} for item in body["items"]]
        return httpx.Response(200, json={"scores": scores})

    answers_path = tmp_path / "answers.jsonl"
    endpoint = ProbabilityEndpoint(base_url="http://localhost:9100/v1", model="prob-model", api_key_env="PROB_API_KEY")
    assessor = HttpProbabilityAssessor(endpoint, transport=httpx.MockTransport(handler))
    rows = [
        _fixture_row("p", "ctx-a", [_candidate("R-x1")], []),
        _fixture_row("p", "ctx-b", [_candidate("R-y1")], []),
    ]

    # Simulate a crash after the first row: only its answers are on disk.
    from scripts.record_prefilter_answers import record_answers

    first_row_only = record_answers([rows[0]], assessor)
    answers_path.write_text("\n".join(json.dumps(a) for a in first_row_only) + "\n")

    written, skipped = record_to_file(rows, assessor, answers_path)

    assert written == 1
    assert skipped == 1
    on_disk = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]
    assert {a["risk_id"] for a in on_disk} == {"R-x1", "R-y1"}
    assert len(on_disk) == 2, "no duplicate rows after the resume"


def test_record_answers_script_rejects_empty_fixtures(tmp_path):
    """A fixtures file with no rows exits non-zero with a message."""
    fixtures = _write_fixtures(tmp_path, [])
    answers = tmp_path / "answers.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(fixtures),
            str(answers),
            "--base-url",
            "http://localhost:9100/v1",
            "--model",
            "prob-model",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "no fixture rows" in completed.stderr
