#!/usr/bin/env python3
"""Export the released Group DETR v2 COCO candidates without opening ground truth."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counting_help_detection.group_detr import GROUP_SPEC, OfficialGroupDetector
from run_detector import run


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--pool-cap", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.model = "group-detr-v2-coco"
    run(args, GROUP_SPEC, detector_class=OfficialGroupDetector)


if __name__ == "__main__":
    main()
