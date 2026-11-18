#!/usr/bin/env python3
"""Prepare image/class counter queries from a sealed detector candidate stream."""

from argparse import ArgumentParser
import json
from pathlib import Path

from counting_help_detection.rerank import (
    load_images,
    load_vocabulary,
    read_detector_predictions,
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    images = load_images(args.images)
    vocabulary = load_vocabulary(args.vocabulary)
    detections = read_detector_predictions(args.detector, images, vocabulary)
    queries = sorted({(image_id, row['category_id'])
                      for image_id, rows in detections.items() for row in rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump([dict(image_id=image, category_id=category) for image, category in queries],
                  stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(f'Prepared {len(queries)} unique image/class queries.')


if __name__ == '__main__':
    main()
