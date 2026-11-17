"""COCO box AP and correct-box retention for a fixed candidate pool."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TOP5_FIELDS = (
    "baseline_correct_top5",
    "matched_gt",
    "original_correct_boxes_displaced",
    "original_matched_targets_lost",
    "new_targets_gained",
    "positive_image_class_queries",
    "positive_queries_with_candidates",
)


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, dict), f"{path.name}:{line_number} must be a JSON object")
            records.append(row)
    return records


def _validate_inputs(
    ground_truth: Mapping[str, Any],
    images: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[int], list[int], dict[int, str]]:
    image_ids = [row["image_id"] for row in images]
    _require(bool(image_ids), "The image manifest is empty")
    _require(all(type(value) is int for value in image_ids), "Manifest image IDs must be integers")
    _require(len(image_ids) == len(set(image_ids)), "Manifest image IDs are duplicated")
    selected_image_ids = set(image_ids)
    source_images = {row["id"]: row for row in ground_truth["images"]}
    _require(len(source_images) == len(ground_truth["images"]), "Ground-truth image IDs are duplicated")
    _require(selected_image_ids <= set(source_images), "The manifest includes images absent from the ground truth")
    for image in images:
        reference = source_images[image["image_id"]]
        for dimension in ("width", "height"):
            if dimension in image:
                _require(image[dimension] == reference[dimension],
                         f"Image {image['image_id']} has different {dimension} in the manifest and ground truth")
    names = {row["id"]: row["name"] for row in ground_truth["categories"]}
    _require(bool(names) and len(names) == len(ground_truth["categories"]),
             "Ground-truth category IDs must be unique and nonempty")
    _require(all(type(value) is int for value in names), "Category IDs must be integers")
    annotation_ids = set()
    for row in ground_truth["annotations"]:
        _require(type(row["id"]) is int and row["id"] not in annotation_ids,
                 "Ground-truth annotation IDs must be unique integers")
        annotation_ids.add(row["id"])
        _require(row["image_id"] in source_images and row["category_id"] in names,
                 "A ground-truth annotation refers to an unknown image or category")
        _require(len(row["bbox"]) == 4 and all(map(_finite_number, row["bbox"])),
                 f"Ground-truth annotation {row['id']} has invalid coordinates")
    for arm_name, rows in predictions.items():
        identities = set()
        for row in rows:
            image_id, category_id, index = row["image_id"], row["category_id"], row["candidate_index"]
            _require(type(image_id) is int and image_id in selected_image_ids,
                     f"{arm_name} includes an image outside the manifest")
            _require(type(category_id) is int and category_id in names,
                     f"{arm_name} includes a category outside the ground-truth taxonomy")
            _require(type(index) is int, f"{arm_name} candidate_index must be an integer")
            identity = (image_id, index)
            _require(identity not in identities, f"{arm_name} duplicates candidate {identity}")
            identities.add(identity)
            _require(_finite_number(row["score"]), f"{arm_name} includes a nonfinite score")
            box = row["bbox_xyxy"]
            _require(isinstance(box, (list, tuple)) and len(box) == 4
                     and all(map(_finite_number, box)), f"{arm_name} includes invalid coordinates")
    return image_ids, sorted(names), names


def _geometry(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], tuple[Any, ...]]:
    return {
        (row["image_id"], row["candidate_index"]):
        (row["category_id"], *row["bbox_xyxy"])
        for row in rows
    }


def _group_top_five(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["image_id"], row["category_id"])].append(row)
    return {
        key: sorted(group, key=lambda row: (-row["score"], row["candidate_index"]))[:5]
        for key, group in groups.items()
    }


def _iou_xyxy_xywh(box: Sequence[float], target: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    tx, ty, width, height = target
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return 0.0
    intersection = max(0.0, min(x2, tx + width) - max(x1, tx)) * max(
        0.0, min(y2, ty + height) - max(y1, ty)
    )
    union = (x2 - x1) * (y2 - y1) + width * height - intersection
    return intersection / union if union > 0 else 0.0


def _match_top_five(
    predictions: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]
) -> dict[int, int]:
    matches = {}
    used_targets = set()
    for row in predictions:
        options = (
            (_iou_xyxy_xywh(row["bbox_xyxy"], target["bbox"]), -target["id"])
            for target in targets if target["id"] not in used_targets
        )
        overlap, negative_id = max(options, default=(0.0, 0))
        if overlap >= 0.5:
            matches[row["candidate_index"]] = -negative_id
            used_targets.add(-negative_id)
    return matches


def _top_five_metrics(
    original: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    targets: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    excluded_queries: set[tuple[int, int]],
    category_ids: Sequence[int],
) -> dict[str, Any]:
    old_groups, new_groups = _group_top_five(original), _group_top_five(current)
    pooled = Counter({field: 0 for field in TOP5_FIELDS})
    per_class = {category_id: Counter(pooled) for category_id in category_ids}
    for key, ground_truth in targets.items():
        if key in excluded_queries:
            continue
        before_rows, after_rows = old_groups.get(key, []), new_groups.get(key, [])
        before, after = _match_top_five(before_rows, ground_truth), _match_top_five(after_rows, ground_truth)
        retained_indices = {row["candidate_index"] for row in after_rows}
        previous_targets, current_targets = set(before.values()), set(after.values())
        counts = {
            "baseline_correct_top5": len(before),
            "matched_gt": len(after),
            "original_correct_boxes_displaced": len(set(before) - retained_indices),
            "original_matched_targets_lost": len(previous_targets - current_targets),
            "new_targets_gained": len(current_targets - previous_targets),
            "positive_image_class_queries": 1,
            "positive_queries_with_candidates": int(bool(before_rows)),
        }
        pooled.update(counts)
        per_class[key[1]].update(counts)
    _require(pooled["matched_gt"] == pooled["baseline_correct_top5"]
             - pooled["original_matched_targets_lost"] + pooled["new_targets_gained"],
             "Top-five target accounting does not reconcile")
    return {"pooled": dict(pooled), "by_category": {str(key): dict(value) for key, value in per_class.items()}}


def _stock_ap(
    ground_truth: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    category_ids: Sequence[int],
) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError("Evaluation requires pycocotools. Install it before running this command.") from error
    order = {image_id: index for index, image_id in enumerate(image_ids)}
    predictions = []
    for row in sorted(rows, key=lambda item: (order[item["image_id"]], -item["score"], item["candidate_index"])):
        x1, y1, x2, y2 = row["bbox_xyxy"]
        predictions.append({
            "image_id": row["image_id"], "category_id": row["category_id"],
            "score": row["score"], "bbox": [x1, y1, x2 - x1, y2 - y1],
        })
    with redirect_stdout(StringIO()):
        reference = COCO()
        reference.dataset = dict(ground_truth)
        selected_images = set(image_ids)
        reference.dataset["annotations"] = [
            dict(row) for row in ground_truth["annotations"] if row["image_id"] in selected_images
        ]
        reference.dataset.setdefault("info", {})
        reference.createIndex()
        if predictions:
            detected = reference.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {"images": list(ground_truth["images"]),
                                "categories": list(ground_truth["categories"]), "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(reference, detected, "bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = list(category_ids)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metric_names = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
                    "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge")
    metrics = {key: float(value * 100) if value >= 0 else None
               for key, value in zip(metric_names, evaluator.stats, strict=True)}
    by_category = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = evaluator.eval["precision"][:, :, index, 0, -1]
        valid = values[values >= 0]
        by_category[str(category_id)] = float(valid.mean() * 100) if valid.size else None
    return {"metrics": metrics, "AP_by_category": by_category}


def evaluate_predictions(
    ground_truth: Mapping[str, Any],
    images: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    control_arm: str | None = None,
) -> dict[str, Any]:
    """Evaluate score-only changes while retaining native finite box coordinates."""
    _require("original" not in arms, "The arm name 'original' is reserved")
    predictions = {"original": original, **arms}
    image_ids, category_ids, names = _validate_inputs(ground_truth, images, predictions)
    _require(control_arm is None or control_arm in arms, "The selected control arm does not exist")
    geometry = _geometry(original)
    for name, rows in arms.items():
        _require(_geometry(rows) == geometry, f"{name} changed candidate identities, categories or coordinates")
    targets = defaultdict(list)
    excluded_queries = set()
    selected_images = set(image_ids)
    for row in ground_truth["annotations"]:
        if row["image_id"] not in selected_images:
            continue
        key = (row["image_id"], row["category_id"])
        if row.get("iscrowd", 0) or row.get("ignore", 0):
            excluded_queries.add(key)
        else:
            targets[key].append(row)
    results = {}
    for name, rows in predictions.items():
        results[name] = {
            **_stock_ap(ground_truth, rows, image_ids, category_ids),
            "top5_vs_original": _top_five_metrics(original, rows, targets, excluded_queries, category_ids),
            "candidates": len(rows),
        }
    for row in results.values():
        ap, baseline = row["metrics"]["AP"], results["original"]["metrics"]["AP"]
        row["AP_change_vs_original"] = ap - baseline if ap is not None and baseline is not None else None
        if control_arm is not None:
            control = results[control_arm]["metrics"]["AP"]
            row["AP_change_vs_control"] = ap - control if ap is not None and control is not None else None
    geometry_rows = [[image_id, index, *value] for (image_id, index), value in sorted(geometry.items())]
    geometry_digest = sha256(json.dumps(geometry_rows, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "schema": 1, "status": "PAIRED_EVALUATION_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(image_ids), "category_count": len(category_ids),
        "categories": [{"id": category_id, "name": names[category_id]} for category_id in category_ids],
        "geometry_sha256": geometry_digest, "control_arm": control_arm,
        "metrics_definition": {
            "AP": "Stock COCO bbox AP IoU .50:.95, maxDets100, AP points 0..100, all ground-truth categories",
            "top5": "Same-image same-class, descending score then candidate_index, greedy one-to-one IoU >=.50, lowest GT ID on ties",
            "displaced_box": "An originally matched candidate ID absent from that image/class's new top five",
            "lost_target": "An originally matched GT annotation ID absent from new matches",
            "recovered_target": "A newly matched GT annotation ID absent from original matches",
        },
        "top5_excluded_crowd_or_ignored_queries": len(excluded_queries),
        "results": results,
    }


def evaluate_files(
    ground_truth_path: Path,
    manifest_path: Path,
    original_path: Path,
    arm_paths: Mapping[str, Path],
    control_arm: str | None = None,
    expected_gt_sha256: str | None = None,
    metadata_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    files = {"ground_truth": Path(ground_truth_path), "manifest": Path(manifest_path),
             "original": Path(original_path), **{f"arm:{name}": Path(path) for name, path in arm_paths.items()}}
    files.update({f"metadata:{index}": Path(path) for index, path in enumerate(metadata_paths)})
    hashes = {name: file_sha256(path) for name, path in files.items()}
    _require(expected_gt_sha256 is None or hashes["ground_truth"] == expected_gt_sha256,
             "Ground-truth SHA-256 differs from the declared value")
    with files["ground_truth"].open(encoding="utf-8") as stream:
        ground_truth = json.load(stream)
    with files["manifest"].open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    images = manifest["images"] if isinstance(manifest, dict) else manifest
    report = evaluate_predictions(ground_truth, images, read_predictions(files["original"]),
                                  {name: read_predictions(Path(path)) for name, path in arm_paths.items()}, control_arm)
    _require(all(file_sha256(path) == hashes[name] for name, path in files.items()),
             "An input file changed during evaluation")
    report["provenance"] = {name: {"name": path.name, "sha256": hashes[name]} for name, path in files.items()}
    source = Path(__file__)
    report["evaluation_source_sha256"] = file_sha256(source) if source.is_file() else None
    return report
