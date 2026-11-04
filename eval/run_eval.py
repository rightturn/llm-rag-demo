"""Measure retrieval and answer quality on a small question set.

Usage:  python -m eval.run_eval
Reports: retrieval hit rate (is the right document in the top-k?) and answer match rate
(does the answer contain the expected fact?). Works offline; set OPENAI_API_KEY to evaluate
the real models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.api import DOCS_DIR, build_service

QUESTIONS = Path(__file__).with_name("questions.json")


def run() -> dict:
    service = build_service()
    service.ingest_directory(DOCS_DIR)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    hits = matches = 0
    for item in questions:
        result = service.ask(item["question"])
        sources = [c.source for c in result.citations]
        hit = item["expected_source"] in sources
        match = item["must_contain"].lower() in result.answer.lower()
        hits += hit
        matches += match
        print(f"{'OK ' if hit and match else 'MISS'} {item['question']}  ->  {result.answer[:70]}")
    n = len(questions)
    summary = {"questions": n, "retrieval_hit_rate": hits / n, "answer_match_rate": matches / n}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    out = run()
    sys.exit(0 if out["answer_match_rate"] >= 0.8 and out["retrieval_hit_rate"] >= 0.8 else 1)
