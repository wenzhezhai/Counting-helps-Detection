#!/usr/bin/env python3
"""Fill independent image/text counter queries with frozen Count Anything C0."""

from argparse import ArgumentParser
import json
from counting_help_detection.counter import run_counter


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Pinned Count Anything source root")
    parser.add_argument("--checkpoint", required=True, help="Native count_anything.pt")
    parser.add_argument("--config", default="configs/counter.json")
    parser.add_argument("--images", required=True)
    parser.add_argument("--vocabulary", "--vocab", dest="vocabulary", required=True)
    parser.add_argument("--queries", required=True, help="JSON or JSONL image/category queries")
    parser.add_argument("--image-root")
    parser.add_argument("--cache", action="append", default=[], help="Read-only independent cache JSONL; may repeat")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=.25)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run_counter(args.source, args.checkpoint, args.config, args.images,
                         args.vocabulary, args.queries, args.out, args.image_root,
                         args.cache, args.device, args.memory_fraction,
                         args.cpu_threads, args.preflight_only, args.limit_queries)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
