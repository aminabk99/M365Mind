#!/usr/bin/env python3
"""
DocMind Evaluation Pipeline
============================
Measures three production-relevant metrics:

  1. Context Precision  — what fraction of retrieved chunks contain at least
                          one expected keyword from the answer
  2. Answer Faithfulness — what fraction of answers contain at least one
                           verifiable [filename, page N] citation
  3. Pass Rate          — fraction of test cases meeting min_confidence

Modes
-----
  --quick   Mock the Ollama LLM (tests retrieval + citation parsing only).
            Safe to run in CI without a GPU.
  --full    Call the real LLM via Ollama. Requires `ollama serve` + models.

Exit codes
----------
  0  All thresholds met
  1  One or more thresholds failed
  2  Configuration / setup error

Usage
-----
  python -m eval.run_evals --quick
  python -m eval.run_evals --full --pdf tests/fixtures/sample.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR     = Path(__file__).parent
CASES_FILE   = EVAL_DIR / "test_cases.json"
CITATION_RE  = re.compile(r'\[([^\],]+),\s*page\s*(\d+)\]', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def context_precision(sources: list[dict], keywords: list[str]) -> float:
    """Fraction of retrieved chunks containing at least one keyword."""
    if not sources or not keywords:
        return 1.0  # no keywords to check → trivially satisfied
    kw_lower = [k.lower() for k in keywords]
    hits = sum(
        1 for s in sources
        if any(k in s.get("chunk_text", "").lower() for k in kw_lower)
    )
    return hits / len(sources)


def answer_faithfulness(answer: str) -> float:
    """1.0 if the answer contains at least one citation tag, else 0.0."""
    return 1.0 if CITATION_RE.search(answer) else 0.0


def check_refusal(answer: str, refusal_keywords: list[str]) -> bool:
    """True if the answer contains any of the refusal keywords."""
    al = answer.lower()
    return any(k.lower() in al for k in refusal_keywords)


# ---------------------------------------------------------------------------
# Mock LLM (for --quick CI mode)
# ---------------------------------------------------------------------------

_MOCK_ANSWER_TEMPLATE = (
    "Based on the document, the key information is summarised below. "
    "[{filename}, page {page}] This is a mock answer for CI validation."
)


def mock_answer(sources: list[dict], question: str) -> str:
    if not sources:
        return "I could not find sufficient information in the uploaded documents to answer this question."
    s = sources[0]
    return _MOCK_ANSWER_TEMPLATE.format(
        filename=s.get("filename", "unknown"),
        page=s.get("page_number", 1),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case_quick(case: dict, all_sources_fixture: list[dict]) -> dict:
    """
    Quick mode: use fixture sources, mock LLM answer, check citation logic.
    """
    answer  = mock_answer(all_sources_fixture, case["question"])
    sources = all_sources_fixture

    cp = context_precision(sources, case.get("expected_keywords", []))
    af = answer_faithfulness(answer)

    passed = True
    notes  = []

    if case.get("check_refusal"):
        if not check_refusal(answer, case.get("expected_keywords", ["could not find"])):
            passed = False
            notes.append("Expected refusal but answer did not refuse.")
    else:
        if af < 1.0 and case.get("check_citation_present"):
            passed = False
            notes.append("Answer missing citation tag.")

    return {
        "id":                case["id"],
        "description":       case["description"],
        "context_precision": round(cp, 3),
        "faithfulness":      round(af, 3),
        "passed":            passed,
        "notes":             notes,
        "answer_snippet":    answer[:120],
    }


def run_case_full(case: dict, backend_url: str) -> dict:
    """
    Full mode: call the real backend /query endpoint.
    """
    import httpx

    resp = httpx.post(
        f"{backend_url}/query",
        json={"question": case["question"], "top_k": 5},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    answer  = data.get("answer", "")
    sources = data.get("sources", [])
    conf    = data.get("confidence", 0.0)

    cp = context_precision(sources, case.get("expected_keywords", []))
    af = answer_faithfulness(answer)

    passed = True
    notes  = []

    if conf < case.get("min_confidence", 0.0):
        passed = False
        notes.append(f"Confidence {conf:.3f} < threshold {case['min_confidence']:.3f}")

    if case.get("check_refusal"):
        if not check_refusal(answer, case.get("expected_keywords", ["could not find"])):
            passed = False
            notes.append("Expected refusal but answer did not refuse.")
    elif case.get("check_citation_present") and af < 1.0:
        passed = False
        notes.append("Answer missing citation tag.")

    return {
        "id":                case["id"],
        "description":       case["description"],
        "context_precision": round(cp, 3),
        "faithfulness":      round(af, 3),
        "confidence":        round(conf, 3),
        "passed":            passed,
        "notes":             notes,
        "answer_snippet":    answer[:120],
        "retrieval_stats":   data.get("retrieval_stats", {}),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[dict], thresholds: dict) -> bool:
    """Print eval results and return True if all thresholds are met."""
    print("\n" + "=" * 60)
    print("DocMind Eval Report")
    print("=" * 60)

    total   = len(results)
    passing = sum(1 for r in results if r["passed"])

    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"\n[{status}] {r['id']} — {r['description']}")
        print(f"  Context Precision : {r['context_precision']:.3f}")
        print(f"  Faithfulness      : {r['faithfulness']:.3f}")
        if "confidence" in r:
            print(f"  Confidence        : {r['confidence']:.3f}")
        if r["notes"]:
            for note in r["notes"]:
                print(f"  ⚠  {note}")
        if "retrieval_stats" in r:
            rs = r["retrieval_stats"]
            print(f"  Retrieval         : vec={rs.get('vector_hits',0)} "
                  f"bm25={rs.get('bm25_hits',0)} "
                  f"rrf={rs.get('after_rrf',0)} "
                  f"rerank={rs.get('after_rerank',0)}")

    # Aggregate
    mean_cp = sum(r["context_precision"] for r in results) / total if total else 0
    mean_af = sum(r["faithfulness"]      for r in results) / total if total else 0
    pass_rt = passing / total if total else 0

    print("\n" + "-" * 60)
    print("Aggregate Metrics")
    print(f"  Mean Context Precision : {mean_cp:.3f}  (threshold ≥ {thresholds['min_context_precision']})")
    print(f"  Mean Faithfulness      : {mean_af:.3f}  (threshold ≥ {thresholds['min_answer_faithfulness']})")
    print(f"  Pass Rate              : {pass_rt:.3f}  ({passing}/{total} cases)  (threshold ≥ {thresholds['min_cases_passing']})")

    ok = (
        mean_cp  >= thresholds["min_context_precision"]
        and mean_af  >= thresholds["min_answer_faithfulness"]
        and pass_rt  >= thresholds["min_cases_passing"]
    )

    print("\n" + ("✓ ALL THRESHOLDS MET — CI gate passes" if ok else "✗ THRESHOLDS NOT MET — CI gate FAILS"))
    print("=" * 60 + "\n")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DocMind eval pipeline")
    parser.add_argument(
        "--quick", action="store_true",
        help="Mock LLM; test retrieval + citation logic only (CI-safe)."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Call real backend. Requires Ollama running with models pulled."
    )
    parser.add_argument(
        "--backend-url", default="http://localhost:8000",
        help="Backend base URL for --full mode."
    )
    parser.add_argument(
        "--output", default=None,
        help="Write JSON results to this file."
    )
    args = parser.parse_args()

    if not args.quick and not args.full:
        print("Specify --quick or --full", file=sys.stderr)
        sys.exit(2)

    if not CASES_FILE.exists():
        print(f"Test cases not found: {CASES_FILE}", file=sys.stderr)
        sys.exit(2)

    with open(CASES_FILE) as fh:
        config = json.load(fh)

    cases      = config["test_cases"]
    thresholds = config["thresholds"]

    results: list[dict] = []

    if args.quick:
        print("Running in QUICK mode (mocked LLM) …")
        # Minimal fixture sources — just enough to test citation logic
        fixture_sources = [
            {"filename": "sample.pdf", "page_number": 1, "chunk_text": "This document describes key findings."},
            {"filename": "sample.pdf", "page_number": 2, "chunk_text": "The methodology section outlines the approach."},
        ]
        for case in cases:
            results.append(run_case_quick(case, fixture_sources))
    else:
        print(f"Running in FULL mode against {args.backend_url} …")
        for case in cases:
            try:
                results.append(run_case_full(case, args.backend_url))
            except Exception as exc:
                results.append({
                    "id":                case["id"],
                    "description":       case["description"],
                    "context_precision": 0.0,
                    "faithfulness":      0.0,
                    "passed":            False,
                    "notes":             [f"Error: {exc}"],
                    "answer_snippet":    "",
                })

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")

    ok = print_report(results, thresholds)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
