#!/usr/bin/env python3
"""Prepare original, matched-confidence and point-guided flat prediction JSONL."""
from argparse import ArgumentParser
import json

from counting_help_detection.rerank import MissingCounterQueriesError, run_reranking


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--detector", "--detectors", dest="detector", required=True)
    parser.add_argument("--counter", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--vocabulary", "--vocab", dest="vocabulary", required=True)
    parser.add_argument("--out", required=True, help="New output directory")
    args = parser.parse_args()
    try:
        result = run_reranking(args.detector, args.counter, args.images, args.vocabulary, args.out)
    except MissingCounterQueriesError as exc:
        parser.exit(2, json.dumps({"status": "MISSING_COUNTER_QUERIES",
                    "queries": [{"image_id": i, "category_id": c} for i, c in exc.queries]}) + "\n")
    print(json.dumps({"status": result["status"], "images": result["images"],
                      "candidates": result["candidate_count"], "out": args.out}))


if __name__ == "__main__":
    main()
