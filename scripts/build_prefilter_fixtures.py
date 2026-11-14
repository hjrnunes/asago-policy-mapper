#!/usr/bin/env python3
"""Build prefilter fixtures from a battery run directory.

Re-chunks each policy with the chunking settings the run recorded, rebuilds
the chunk contexts, and reconstructs the (candidate, grounded) population
from the recorded grounding calls. Expansion-grounding records
(``ground-expand-*``) are excluded: the prefilter sits before grounding
proper, so its population is the candidates the grounding stage judged,
not the ones sibling expansion produced.

Each output row is one chunk: the policy name, chunk index, context and
its sha256, the candidates put to grounding for that chunk, and the ids
any grounding pass actually grounded.

Usage:
    python scripts/build_prefilter_fixtures.py <battery-run-dir> [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from asago_policy_mapper.extract.parse import chunk_documents, parse_document
from asago_policy_mapper.extract.retrieve import build_chunk_contexts

_CANDIDATE_SECTION = "## Candidate risks"
"""The heading that opens the rendered candidate list in the grounding prompt."""

_SYNTHETIC_NAME = re.compile(r"\(\d+ variants?\)$")
"""A variant-group parent entry, named by the index collapse."""


def _is_expansion(call_id: str) -> bool:
    """Tell whether a grounding record came from sibling expansion."""
    return call_id.startswith("ground-expand-")


def grounding_population(records: Iterable[dict]) -> dict[str, bool]:
    """Map each risk id to whether any non-expansion grounding pass grounded it."""
    grounded: dict[str, bool] = {}
    for record in records:
        if record.get("stage") != "grounding" or _is_expansion(str(record.get("call_id", ""))):
            continue
        for row in record.get("response", []):
            if not isinstance(row, dict):
                continue
            rid = row.get("risk_id")
            if isinstance(rid, str) and row.get("grounded"):
                grounded[rid] = True
            elif isinstance(rid, str) and rid not in grounded:
                grounded[rid] = False
    return grounded


def _parse_candidate_line(line: str, risk_id: str) -> dict | None:
    """Parse one rendered candidate line for a known risk id.

    The name runs from after ``- **<id>** (`` to the ``): `` that closes
    the name's opening parenthesis, found by tracking paren depth, because
    risk names and descriptions may themselves contain parentheses (the
    battery's "Model Attacks (membership inference, model inversion)"
    parent entry does, in both fields). A risk may carry an empty
    description (the battery's ``finos:ri-10`` does). Returns None when
    the line does not render the id.
    """
    prefix = f"- **{risk_id}** ("
    if not line.startswith(prefix):
        return None
    body = line[len(prefix) :]
    separator = _name_end(body)
    if separator < 0:
        return None
    name = body[:separator]
    description = body[separator + 3 :]
    if not name:
        return None
    return {
        "risk_id": risk_id,
        "name": name,
        "description": description,
        "synthetic": bool(_SYNTHETIC_NAME.search(name)),
    }


def _name_end(body: str) -> int:
    """Find where the rendered name ends: the ``): `` closing the name's opening paren.

    Returns the index of the ``)`` of the first ``): `` at paren depth 1
    (the depth of the name itself), or -1 when no such separator exists.
    """
    depth = 1
    for i, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and body[i : i + 3] == "): ":
                return i
    return -1


def _match_line(stripped: str, remaining: list[str]) -> dict | None:
    """Parse ``stripped`` as the rendered line of the first remaining id it matches."""
    for rid in remaining:
        parsed = _parse_candidate_line(stripped, rid)
        if parsed is not None:
            return parsed
    return None


def candidates_from_record(record: dict) -> list[dict]:
    """Parse the candidates a grounding record put to the model, in batch order.

    The record's ``risk_ids`` names the batch; the rendered user prompt is
    the source of truth for each candidate's name and description. Each
    line is matched against the ids not yet parsed, so a policy chunk that
    happens to contain a candidate-shaped line cannot inject a fictitious
    candidate, and a line that fails to parse for its id cannot derail the
    rest of the batch.
    """
    messages = record.get("messages", [])
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    section = user.split(_CANDIDATE_SECTION, 1)
    if len(section) < 2:
        return []
    risk_ids = [rid for rid in record.get("risk_ids", []) if isinstance(rid, str)]
    candidates: list[dict] = []
    remaining = list(risk_ids)
    for line in section[1].splitlines():
        parsed = _match_line(line.strip(), remaining)
        if parsed is not None:
            remaining.remove(parsed["risk_id"])
            candidates.append(parsed)
    return candidates


def _chunk_records_by_index(records: Iterable[dict], n_contexts: int) -> dict[int, list[dict]]:
    """Group non-expansion grounding records by their chunk index, in batch order."""
    by_chunk: dict[int, list[dict]] = {}
    for record in records:
        if record.get("stage") != "grounding" or _is_expansion(str(record.get("call_id", ""))):
            continue
        index = record.get("chunk_index", -1)
        if not isinstance(index, int) or index < 0 or index >= n_contexts:
            continue
        by_chunk.setdefault(index, []).append(record)
    return by_chunk


def _merge_chunk(records: list[dict]) -> tuple[list[dict], set[str]]:
    """Deduplicate one chunk's candidates in batch order and collect its grounded ids."""
    candidates: list[dict] = []
    seen: set[str] = set()
    grounded: set[str] = set()
    for record in records:
        for candidate in candidates_from_record(record):
            if candidate["risk_id"] not in seen:
                seen.add(candidate["risk_id"])
                candidates.append(candidate)
        grounded.update(_grounded_ids(record))
    return candidates, grounded


def _grounded_ids(record: dict) -> set[str]:
    """Collect the ids this record's own batch grounded.

    Membership checks against the record's ``risk_ids``, not the whole
    chunk, so a response naming an id from another batch of the same chunk
    cannot mark it grounded here.
    """
    batch_ids = {rid for rid in record.get("risk_ids", []) if isinstance(rid, str)}
    grounded: set[str] = set()
    for row in record.get("response", []):
        if isinstance(row, dict) and row.get("grounded"):
            rid = row.get("risk_id")
            if isinstance(rid, str) and rid in batch_ids:
                grounded.add(rid)
    return grounded


def build_rows(policy: str, contexts: list[str], records: Iterable[dict]) -> list[dict]:
    """Build one row per chunk, ordered by chunk index then batch order."""
    by_chunk = _chunk_records_by_index(records, len(contexts))

    rows = []
    for index in sorted(by_chunk):
        candidates, grounded = _merge_chunk(by_chunk[index])
        context = contexts[index]
        rows.append(
            {
                "policy": policy,
                "chunk_index": index,
                "context": context,
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "candidates": candidates,
                "grounded_ids": sorted(grounded),
            }
        )
    return rows


def _chunk_policy(sources: list, max_tokens: int, policy: str) -> list[str]:
    """Parse and re-chunk a run's source documents in recorded order, then build contexts."""
    if not sources:
        print(f"  Warning: {policy} has no source documents; skipping", file=sys.stderr)
        return []
    parsed = []
    for source in sources:
        path = Path(str(source))
        if not path.exists():
            print(f"  Warning: source {source} not found; skipping policy", file=sys.stderr)
            return []
        parsed.append(parse_document(path))
    chunks = chunk_documents(parsed, max_tokens=max_tokens)
    if not chunks:
        print(f"  Warning: no chunks for {policy}; skipping", file=sys.stderr)
        return []
    return build_chunk_contexts(chunks)


def _iter_run_dirs(battery_dir: Path) -> list[tuple[str, Path]]:
    """List (policy name, run dir) pairs for every run in the battery."""
    pairs = []
    for run_dir in sorted(p for p in battery_dir.iterdir() if p.is_dir()):
        result = run_dir / "risk-extraction.json"
        if result.exists():
            pairs.append((run_dir.name, run_dir))
    return pairs


def build_fixtures(battery_dir: Path) -> list[dict]:
    """Build fixture rows for every policy run in a battery directory."""
    rows: list[dict] = []
    for policy, run_dir in _iter_run_dirs(battery_dir):
        result = json.loads((run_dir / "risk-extraction.json").read_text())
        metadata = result.get("metadata", {})
        max_tokens = metadata.get("chunk_max_tokens", 512)
        contexts = _chunk_policy(result.get("source_documents", []), max_tokens, policy)
        if not contexts:
            continue
        expected_chunks = result.get("retrieval_stats", {}).get("total_chunks")
        if isinstance(expected_chunks, int) and expected_chunks != len(contexts):
            print(
                f"  Warning: {policy} re-chunked to {len(contexts)} chunks but the run recorded"
                f" {expected_chunks}; contexts may not match the run. Skipping policy",
                file=sys.stderr,
            )
            continue
        policy_rows = build_rows(policy, contexts, result.get("llm_calls", []))
        rows.extend(policy_rows)
        print(f"  {policy}: {len(policy_rows)} chunks, {sum(len(r['candidates']) for r in policy_rows)} candidates")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build prefilter fixtures from a battery run directory")
    parser.add_argument("battery_dir", type=Path, help="directory of per-policy extraction runs")
    parser.add_argument(
        "--out", type=Path, default=None, help="output JSONL path (default: <battery_dir>/prefilter-fixtures.jsonl)"
    )
    args = parser.parse_args(argv)

    if not args.battery_dir.is_dir():
        parser.error(f"{args.battery_dir} is not a directory")

    rows = build_fixtures(args.battery_dir)
    out = args.out or args.battery_dir / "prefilter-fixtures.jsonl"
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    total_candidates = sum(len(r["candidates"]) for r in rows)
    total_grounded = sum(len(r["grounded_ids"]) for r in rows)
    print(f"Wrote {len(rows)} rows ({total_candidates} candidates, {total_grounded} grounded pairs) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
