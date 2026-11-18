#!/usr/bin/env python3
"""Opt-in complete-pool disk-indexed counting and confidence reranking."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counting_help_detection.evaluation_large import configure_cpu, cpu_slot
from counting_help_detection.rerank_large import run_reranking_large


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("detector", "counter", "images", "vocabulary", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu-lock", type=Path)
    parser.add_argument("--min-free-memory-gib", type=float, default=0.)
    args = parser.parse_args()
    configure_cpu()
    with cpu_slot(args.cpu_lock, args.min_free_memory_gib):
        seal = run_reranking_large(args.detector, args.counter, args.images, args.vocabulary, args.out)
    print(f"Sealed {seal['candidate_count']} complete candidates in each of three arms.")


if __name__ == "__main__":
    main()
