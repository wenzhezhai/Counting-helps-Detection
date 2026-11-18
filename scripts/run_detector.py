#!/usr/bin/env python3
"""Export a released detector's candidates without opening ground truth."""

from __future__ import annotations

from argparse import ArgumentParser
import fcntl
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import sys
from time import monotonic
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counting_help_detection.detectors import (
    DetectorOptions, OfficialDetector, file_sha256, load_models,
)


def write_json(path, value):
    pending = path.with_name("." + path.name + "." + uuid4().hex)
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    pending.rename(path)


def image_path(row, root):
    if root is None and row.get("image_path"):
        return Path(row["image_path"]).resolve()
    relative = PurePosixPath(row["file_name"])
    if root is None or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Supply an image root and safe relative file names, or legacy image_path fields.")
    return (root / relative).resolve()


def read_images(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "annotations" in data:
            raise ValueError("The inference manifest must contain image metadata without annotations.")
        data = data["images"]
    ids = []
    for row in data:
        image_id = row.get("image_id", row.get("id"))
        if type(image_id) is not int or image_id < 0:
            raise ValueError("Image IDs must be nonnegative integers.")
        if any(key in row for key in ("annotations", "boxes", "labels", "ground_truth")):
            raise ValueError("The inference manifest contains ground-truth fields.")
        row["image_id"] = image_id
        ids.append(image_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("The manifest is empty or contains duplicate image IDs.")
    return data


def run(args, spec, detector_class=OfficialDetector):
    from PIL import Image

    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume for an existing run.")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    images = read_images(args.images)
    images_hash = file_sha256(args.images)
    source = args.source_dir.resolve()
    config = args.config.resolve() if args.config else None
    detector = detector_class(DetectorOptions(
        model_name=args.model, source_dir=source, checkpoint=args.checkpoint,
        registry=spec, config=config, device=args.device, threads=args.threads,
        gpu_memory_fraction=args.gpu_memory_fraction,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        source_manifest=getattr(args, "source_manifest", None),
    ))
    protocol = {
        "detector": detector.protocol(),
        "image_manifest_sha256": images_hash,
        "images": len(images),
        "exported_pool_cap": args.pool_cap,
        "runner_sha256": file_sha256(Path(__file__)),
        "ground_truth_read": False,
    }
    protocol_path = args.output / "PROTOCOL_PRE_GT.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("The source, checkpoint, input or inference settings changed since the run began.")
    else:
        write_json(protocol_path, protocol)
    protocol_hash = file_sha256(protocol_path)
    chunks = args.output / "chunks"
    chunks.mkdir(exist_ok=True)
    started = monotonic()
    for index, row in enumerate(images, 1):
        chunk = chunks / f"{row['image_id']:012d}.json"
        path = image_path(row, args.image_root)
        raw = path.read_bytes()
        image_hash = sha256(raw).hexdigest()
        if row.get("image_sha256") and row["image_sha256"] != image_hash:
            raise ValueError(f"Image bytes changed for image {row['image_id']}.")
        if chunk.exists():
            old = json.loads(chunk.read_text())
            if old["protocol_sha256"] != protocol_hash or old["image_sha256"] != image_hash:
                raise ValueError(f"The cached chunk differs from the frozen input for image {row['image_id']}.")
            continue
        with Image.open(BytesIO(raw)) as opened:
            image = opened.convert("RGB")
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"Decoded image dimensions differ from the manifest for image {row['image_id']}.")
        values = detector.predict(row["image_id"], image, args.pool_cap, image_path=path)
        if file_sha256(path) != image_hash:
            raise ValueError(f"Image bytes changed during inference for image {row['image_id']}.")
        write_json(chunk, {
            "image_id": row["image_id"], "image_sha256": image_hash,
            "protocol_sha256": protocol_hash, "predictions": values,
            "excluded_native_labels": getattr(detector, "last_excluded_labels", {}),
        })
        if index % 50 == 0 or index == len(images):
            print(json.dumps({"images": index, "total": len(images), "elapsed_seconds": monotonic() - started}), flush=True)
    expected = {row["image_id"] for row in images}
    if {int(path.stem) for path in chunks.glob("*.json")} != expected:
        raise ValueError("The prediction chunks do not exactly cover the image manifest.")
    target = args.output / "predictions.jsonl"
    pending = target.with_name("." + target.name + "." + uuid4().hex)
    stream = None if target.exists() else pending.open("x", encoding="utf-8")
    digest = sha256()
    row_count = 0
    nonpositive = 0
    excluded_labels = {}
    receipts = []
    try:
        for row in images:
            chunk = chunks / f"{row['image_id']:012d}.json"
            values = json.loads(chunk.read_text())
            if values["protocol_sha256"] != protocol_hash or values["image_id"] != row["image_id"]:
                raise ValueError("A cached chunk does not belong to this run.")
            receipts.append({"image_id": row["image_id"], "sha256": file_sha256(chunk)})
            for label, count in values.get("excluded_native_labels", {}).items():
                excluded_labels[label] = excluded_labels.get(label, 0) + count
            for prediction in values["predictions"]:
                encoded = json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n"
                digest.update(encoded.encode("utf-8"))
                if stream is not None:
                    stream.write(encoded)
                box = prediction["bbox_xyxy"]
                nonpositive += int(box[2] <= box[0] or box[3] <= box[1])
                row_count += 1
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if stream is not None:
            stream.close()
    if target.exists():
        if file_sha256(target) != digest.hexdigest():
            raise ValueError("The existing flat prediction stream differs from its chunks.")
    else:
        pending.rename(target)
    if file_sha256(args.images) != images_hash:
        raise ValueError("The image manifest changed during inference.")
    receipt = {
        "status": "PREDICTIONS_SEALED_BEFORE_GT", "ground_truth_read": False,
        "images": len(images), "candidate_rows": row_count,
        "protocol_sha256": protocol_hash, "prediction_sha256": digest.hexdigest(),
        "nonpositive_native_boxes_retained": nonpositive, "chunks": receipts,
        "excluded_native_labels": excluded_labels,
    }
    complete = args.output / "GT_FREE_COMPLETE.json"
    if complete.exists():
        if json.loads(complete.read_text()) != receipt:
            raise ValueError("The prediction seal differs from the saved result.")
    else:
        write_json(complete, receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "chunks"}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--models-json", type=Path, default=Path(__file__).resolve().parents[1] / "configs/models.json")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-manifest", type=Path, help="Archive receipt containing commit/source_revision and relative file SHA-256 values.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--pool-cap", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    models = load_models(args.models_json)
    if args.list_models:
        print(json.dumps({name: value["loader_status"] for name, value in models.items()}, indent=2))
        return
    if args.model not in models:
        parser.error("Select a model listed by --list-models.")
    for field in ("source_dir", "checkpoint", "images", "output"):
        if getattr(args, field) is None:
            parser.error("Missing --" + field.replace("_", "-"))
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.model in {"gtr-x", "gtr-x-objects365"}:
        os.environ["GTR_DISABLE_CUDA_GLA"] = "1"
    run(args, models[args.model])


if __name__ == "__main__":
    main()
