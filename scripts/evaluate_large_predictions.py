#!/usr/bin/env python3
"""Opt-in complete-pool serial stock AP and original correct top-five retention."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counting_help_detection.evaluation_large import evaluate_files_large


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("gt", "manifest", "original", "output", "work-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--control-arm")
    parser.add_argument("--gt-sha256")
    parser.add_argument("--metadata", type=Path, action="append", default=[])
    parser.add_argument("--cpu-lock", type=Path)
    parser.add_argument("--min-free-memory-gib", type=float, default=0.)
    args = parser.parse_args()
    arms = {}
    for value in args.arm:
        name, separator, filename = value.partition("=")
        if not separator or not name or not filename or name in arms or name == "original":
            parser.error("Each --arm must have a unique nonreserved name and a file, written NAME=FILE")
        arms[name] = Path(filename)
    if args.output.exists():
        parser.error("Choose a new output result file.")
    report = evaluate_files_large(args.gt, args.manifest, args.original, arms, args.control_arm,
                                  args.gt_sha256, args.metadata, work_dir=args.work_dir,
                                  cpu_lock=args.cpu_lock, min_free_memory_gib=args.min_free_memory_gib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Evaluated {report['image_count']} images with serial complete-pool stock AP.")


if __name__ == "__main__":
    main()
