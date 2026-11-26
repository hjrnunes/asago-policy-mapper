#!/usr/bin/env python3
"""Check that not_found quotes are absent from their source documents.

For every ``not_found`` evidence span in a run directory's
``risk-extraction.json`` files, asks whether the quote appears in the
source policy after a comparison that shares no code with the
pipeline's own verification: casefold-only substring containment, plus a
``difflib.SequenceMatcher`` longest-match pass that reports the largest
contiguous run the document shares with the quote. The two numbers
together separate genuine paraphrase (absent whole, small sub-runs) from
stitched summaries (absent whole, a sub-run covering most of the quote,
typically the document sentence behind a prepended risk name).

The pipeline's verification compares a quote against its grounding
windows; this script compares against the whole parsed document, so it
can also flag quotes that fail window verification but exist elsewhere
in the document.

Usage:
    python scripts/verify_not_found.py <run_dir> <policy_dir> [--min-run-frac 0.4] [--report FILE]

``run_dir`` holds one subdirectory per policy (as the battery runner
writes them); ``policy_dir`` holds the source documents, matched to each
subdirectory by file stem.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from asago_policy_mapper.extract.parse import parse_document


def _joined(text: str) -> str:
    """Casefold and collapse whitespace — no other normalization."""
    return " ".join(text.casefold().split())


def check_quote(quote: str, document_text: str, min_run_frac: float) -> dict:
    """Compare one quote against a document with independent mechanics.

    Returns whether the quote appears whole (casefold substring only) and
    the largest contiguous run it shares with the document as a fraction
    of its length. Nothing here imports or calls the pipeline's
    normalize/locate_quote: the point is a second opinion.
    """
    quote_joined = _joined(quote)
    doc_joined = _joined(document_text)
    whole = quote_joined in doc_joined
    if whole:
        run_frac = 1.0
    else:
        sm = difflib.SequenceMatcher(None, doc_joined, quote_joined, autojunk=False)
        match = sm.find_longest_match(0, len(doc_joined), 0, len(quote_joined))
        run_frac = match.size / len(quote_joined) if quote_joined else 0.0
    return {"whole": whole, "run_frac": round(run_frac, 3), "stitched": (not whole) and run_frac >= min_run_frac}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Battery run directory (one subdirectory per policy)")
    parser.add_argument("policy_dir", type=Path, help="Directory holding the source policy documents")
    parser.add_argument(
        "--min-run-frac", type=float, default=0.4, help="Sub-run fraction counting as stitched (default 0.4)"
    )
    parser.add_argument("--report", type=Path, default=None, help="Write the JSON report here")
    args = parser.parse_args()

    results = []
    for result_file in sorted(args.run_dir.glob("*/risk-extraction.json")):
        policy = result_file.parent.name
        candidates = list(args.policy_dir.glob(f"{policy}.*")) + list(args.policy_dir.glob(f"{policy}"))
        if not candidates:
            print(f"warning: no source document for {policy}, skipped", file=sys.stderr)
            continue
        document = parse_document(candidates[0])
        data = json.loads(result_file.read_text())
        for risk in data.get("risks", []):
            for ev in risk.get("evidence", []):
                if ev.get("verification") != "not_found":
                    continue
                check = check_quote(ev["text"], document.content, args.min_run_frac)
                results.append({"policy": policy, "risk_id": risk["risk_id"], "quote": ev["text"], **check})

    whole = sum(1 for r in results if r["whole"])
    stitched = sum(1 for r in results if r["stitched"])
    summary = {
        "not_found_total": len(results),
        "present_whole": whole,
        "stitched_sub_run": stitched,
        "min_run_frac": args.min_run_frac,
    }
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.write_text(json.dumps({"summary": summary, "quotes": results}, indent=2))
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
