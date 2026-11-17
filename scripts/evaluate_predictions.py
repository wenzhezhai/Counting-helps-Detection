#!/usr/bin/env python3
"""Evaluate fixed candidates with COCO AP and original correct top-five retention."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counting_help_detection.evaluation import evaluate_files


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Original COCO ground-truth JSON")
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen image-only JSON manifest")
    parser.add_argument("--original", type=Path, required=True, help="Native flat prediction JSONL")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=FILE",
                        help="Reranked flat prediction JSONL, repeat for each arm")
    parser.add_argument("--control-arm", help="Arm used for the AP difference against confidence")
    parser.add_argument("--gt-sha256", help="Optional ground-truth checksum fixed before scoring")
    parser.add_argument("--metadata", type=Path, action="append", default=[],
                        help="Reranking or model metadata to include by checksum")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    args = parser.parse_args()
    arms = {}
    for value in args.arm:
        name, separator, filename = value.partition("=")
        if not separator or not name or not filename or name in arms or name == "original":
            parser.error("Each --arm must have a unique nonreserved name and a file, written NAME=FILE")
        arms[name] = Path(filename)
    if args.output.exists():
        parser.error("The output file already exists. Choose a new result path.")
    report = evaluate_files(args.gt, args.manifest, args.original, arms, args.control_arm,
                            args.gt_sha256, args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Evaluated {report['image_count']} images and {report['category_count']} categories.")
    print(args.output)


if __name__ == "__main__":
    main()
