"""Opt-in complete-pool evaluation: disk indexing, serial stock AP, streamed top five."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from . import evaluation as reference
from .rerank_large import CandidateIndex, require_reference as require_reranking_reference


REFERENCE_SHA256 = "c970a4f88a381e7f3848af8e01c5a9c05ccac7384616a3b16113fa270bb7099b"


def require_reference():
    require_reranking_reference()
    if reference.file_sha256(Path(reference.__file__)) != REFERENCE_SHA256:
        raise ValueError("The frozen reference evaluation implementation differs.")


@contextmanager
def cpu_slot(lock_path=None, min_free_memory_gib=0.):
    """Optional shared admission; no GPU access or change to numerical evaluation."""
    if not math.isfinite(min_free_memory_gib) or min_free_memory_gib < 0:
        raise ValueError("The available-memory admission threshold must be finite and nonnegative.")
    stream = None
    try:
        if lock_path is not None:
            path = Path(lock_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+")
            fcntl.flock(stream, fcntl.LOCK_EX)
        if min_free_memory_gib:
            meminfo = Path("/proc/meminfo")
            if not meminfo.exists():
                raise RuntimeError("Memory admission requires Linux /proc/meminfo, or threshold zero.")
            while True:
                available = next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                                 if line.startswith("MemAvailable:"))
                if available >= min_free_memory_gib * 1024**3:
                    break
                time.sleep(5)
        yield
    finally:
        if stream is not None:
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()


def peak_rss_kib():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw // 1024 if sys.platform == "darwin" else raw


def configure_cpu():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def index_evaluation_arm(store, path, arm, image_ids, category_ids):
    selected, categories = set(image_ids), set(category_ids)
    with Path(path).open(encoding="utf-8") as stream:
        sequence = 0
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            reference._require(isinstance(row, dict), f"{Path(path).name}:{line_number} must be a JSON object")
            image_id, category, index = row["image_id"], row["category_id"], row["candidate_index"]
            reference._require(type(image_id) is int and image_id in selected, f"{arm} includes an image outside the manifest")
            reference._require(type(category) is int and category in categories, f"{arm} includes a category outside the ground-truth taxonomy")
            reference._require(type(index) is int, f"{arm} candidate_index must be an integer")
            reference._require(reference._finite_number(row["score"]), f"{arm} includes a nonfinite score")
            box = row["bbox_xyxy"]
            reference._require(isinstance(box, (list, tuple)) and len(box) == 4
                               and all(map(reference._finite_number, box)), f"{arm} includes invalid coordinates")
            store.add(arm, row, sequence)
            sequence += 1
    store.connection.commit()
    return sequence


def validate_geometry_and_hash(store, names, image_ids):
    """Reproduce the reference evaluator's ONE sorted compact JSON-array digest."""
    digest = sha256(b"[")
    first = True
    for image_id in sorted(image_ids):
        original = store.rows("original", image_id)
        geometry = reference._geometry(original)
        for name in names:
            reference._require(reference._geometry(store.rows(name, image_id)) == geometry,
                               f"{name} changed candidate identities, categories or coordinates")
        for row in sorted(original, key=lambda value: value["candidate_index"]):
            if not first:
                digest.update(b",")
            value = [row["image_id"], row["candidate_index"], row["category_id"], *row["bbox_xyxy"]]
            digest.update(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
            first = False
    digest.update(b"]")
    return digest.hexdigest()


def streamed_top_five(store, names, image_ids, category_ids, ground_truth):
    selected = set(image_ids)
    targets = defaultdict(lambda: defaultdict(list))
    excluded = set()
    for row in ground_truth["annotations"]:
        if row["image_id"] not in selected:
            continue
        key = row["image_id"], row["category_id"]
        if row.get("iscrowd", 0) or row.get("ignore", 0):
            excluded.add(key)
        else:
            targets[row["image_id"]][key].append(row)
    result = {name: {"pooled": Counter({field: 0 for field in reference.TOP5_FIELDS}),
                     "by_category": {str(category): Counter({field: 0 for field in reference.TOP5_FIELDS})
                                     for category in category_ids}} for name in names}
    for image_id in image_ids:
        original = store.rows("original", image_id)
        for name in names:
            current = original if name == "original" else store.rows(name, image_id)
            values = reference._top_five_metrics(original, current, targets[image_id], excluded, category_ids)
            result[name]["pooled"].update(values["pooled"])
            for category, counts in values["by_category"].items():
                result[name]["by_category"][category].update(counts)
    for values in result.values():
        values["pooled"] = dict(values["pooled"])
        values["by_category"] = {key: dict(counts) for key, counts in values["by_category"].items()}
    return result, len(excluded)


def _stock_worker(gt_path, manifest_path, predictions_path, output_path, expected_gt_sha256):
    require_reference()
    configure_cpu()
    paths = tuple(map(Path, (gt_path, manifest_path, predictions_path)))
    before = [reference.file_sha256(path) for path in paths]
    reference._require(before[0] == expected_gt_sha256, "Ground-truth SHA-256 changed before stock evaluation")
    ground_truth = json.loads(paths[0].read_text())
    manifest = json.loads(paths[1].read_text())
    images = manifest["images"] if isinstance(manifest, dict) else manifest
    image_ids, category_ids, _ = reference._validate_inputs(ground_truth, images, {})
    rows = reference.read_predictions(paths[2])
    # This is the unchanged stock-COCO helper, with its original sort, all categories and maxDets.
    metrics = reference._stock_ap(ground_truth, rows, image_ids, category_ids)
    reference._require(before == [reference.file_sha256(path) for path in paths], "An input changed during stock evaluation")
    with Path(output_path).open("x", encoding="utf-8") as stream:
        json.dump(dict(stock_ap=metrics, candidate_count=len(rows),
                       peak_process_rss_kib=peak_rss_kib(),
                       reference_implementation_sha256=REFERENCE_SHA256,
                       input_sha256=before), stream, allow_nan=False)
        stream.write("\n")


def evaluate_files_large(ground_truth_path, manifest_path, original_path, arm_paths,
                         control_arm=None, expected_gt_sha256=None, metadata_paths=(), *,
                         work_dir, cpu_lock=None, min_free_memory_gib=0.):
    """Read complete pools on disk; run exactly one full stock-AP arm per fresh process."""
    require_reference()
    configure_cpu()
    reference._require("original" not in arm_paths, "The arm name 'original' is reserved")
    reference._require(control_arm is None or control_arm in arm_paths, "The selected control arm does not exist")
    files = {"ground_truth": Path(ground_truth_path), "manifest": Path(manifest_path),
             "original": Path(original_path), **{f"arm:{name}": Path(path) for name, path in arm_paths.items()}}
    files.update({f"metadata:{index}": Path(path) for index, path in enumerate(metadata_paths)})
    hashes = {name: reference.file_sha256(path) for name, path in files.items()}
    reference._require(expected_gt_sha256 is None or hashes["ground_truth"] == expected_gt_sha256,
                       "Ground-truth SHA-256 differs from the declared value")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=False)
    with cpu_slot(cpu_lock, min_free_memory_gib):
        ground_truth = json.loads(files["ground_truth"].read_text())
        manifest = json.loads(files["manifest"].read_text())
        images = manifest["images"] if isinstance(manifest, dict) else manifest
        image_ids, category_ids, names = reference._validate_inputs(ground_truth, images, {})
        paths = {"original": files["original"], **{name: Path(path) for name, path in arm_paths.items()}}
        store = CandidateIndex(work / "evaluation_index.sqlite3")
        try:
            counts = {name: index_evaluation_arm(store, path, name, image_ids, category_ids)
                      for name, path in paths.items()}
            geometry_sha = validate_geometry_and_hash(store, arm_paths, image_ids)
            top_five, excluded_queries = streamed_top_five(store, paths, image_ids, category_ids, ground_truth)
            results, resources = {}, {}
            environment = dict(os.environ)
            source_root = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
            for position, (name, path) in enumerate(paths.items()):
                output = work / f"stock_ap_{position:03d}.json"
                subprocess.run([sys.executable, "-B", "-m", "counting_help_detection.evaluation_large",
                                "--stock-ap-worker", "--gt", str(files["ground_truth"]),
                                "--manifest", str(files["manifest"]), "--predictions", str(path),
                                "--output", str(output), "--gt-sha256", hashes["ground_truth"]],
                               check=True, env=environment)
                record = json.loads(output.read_text())
                reference._require(record["candidate_count"] == counts[name], "Stock AP candidate coverage changed")
                results[name] = {**record["stock_ap"], "top5_vs_original": top_five[name], "candidates": counts[name]}
                resources[name] = {key: value for key, value in record.items() if key != "stock_ap"}
            for values in results.values():
                ap, baseline = values["metrics"]["AP"], results["original"]["metrics"]["AP"]
                values["AP_change_vs_original"] = ap - baseline if ap is not None and baseline is not None else None
                if control_arm is not None:
                    control = results[control_arm]["metrics"]["AP"]
                    values["AP_change_vs_control"] = ap - control if ap is not None and control is not None else None
            reference._require(all(reference.file_sha256(path) == hashes[name] for name, path in files.items()),
                               "An input file changed during evaluation")
            return dict(schema=1, status="PAIRED_EVALUATION_COMPLETE", created_utc=datetime.now(timezone.utc).isoformat(),
                image_count=len(image_ids), category_count=len(category_ids),
                categories=[{"id": category, "name": names[category]} for category in category_ids],
                geometry_sha256=geometry_sha, control_arm=control_arm,
                metrics_definition={
                    "AP": "Stock COCO bbox AP IoU .50:.95, maxDets100, AP points 0..100, all ground-truth categories",
                    "top5": "Same-image same-class, descending score then candidate_index, greedy one-to-one IoU >=.50, lowest GT ID on ties",
                    "displaced_box": "An originally matched candidate ID absent from that image/class's new top five",
                    "lost_target": "An originally matched GT annotation ID absent from new matches",
                    "recovered_target": "A newly matched GT annotation ID absent from original matches"},
                top5_excluded_crowd_or_ignored_queries=excluded_queries, results=results,
                provenance={name: {"name": path.name, "sha256": hashes[name]} for name, path in files.items()},
                evaluation_source_sha256=reference.file_sha256(Path(__file__)),
                reference_evaluation_source_sha256=REFERENCE_SHA256,
                execution={"mode": "disk_indexed_serial_whole_cohort_stock_ap_streamed_top5",
                           "stock_ap_processes": resources, "candidate_pool_trimmed": False,
                           "parent_peak_rss_kib": peak_rss_kib()})
        finally:
            store.close()


def _worker_main():
    parser = ArgumentParser(description="Internal single-arm stock evaluator; use the public large-evaluation CLI.")
    parser.add_argument("--stock-ap-worker", action="store_true", required=True)
    for name in ("gt", "manifest", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gt-sha256", required=True)
    args = parser.parse_args()
    _stock_worker(args.gt, args.manifest, args.predictions, args.output, args.gt_sha256)


if __name__ == "__main__":
    _worker_main()
