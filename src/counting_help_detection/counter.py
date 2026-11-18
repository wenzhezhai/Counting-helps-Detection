"""Independent Count Anything C0 point predictions.

The native model receives image pixels and a vocabulary text prompt only:
no detector boxes, geometric prompts, annotations, or trained BMD adapters.
Queries are JSON/JSONL records containing image_id and category_id. Images
and vocabulary use the schemas documented in the rerank module.

The canonical output is one JSONL record per requested image/category:
image_id, category_id, width, height, text, points (original pixel xy),
count, and optional image_sha256, point_scores, point_sources. Existing
canonical caches and historical outputs.on.points_original_pixels caches
are accepted. Cache inputs are never modified. A new output directory
contains merged predictions for the requested queries and a separate
new_predictions.jsonl for genuinely inferred queries. Missing predictions
are never replaced by empty point lists.

The model build, mixed precision, pixel resize and presence-on native
postprocessor follow the frozen C0 general-data runner. Selected source
hashes and checkpoint SHA256 are checked before model construction.
"""

from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .rerank import file_sha256, load_images, load_vocabulary


FORBIDDEN = {"annotations", "targets", "ground_truth", "gt_count", "boxes",
             "bbox", "bbox_xyxy", "candidate_boxes", "candidate_indices"}


def _dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_row(stream, row, durable=True):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":")) + "\n")
    if durable:
        stream.flush()
        os.fsync(stream.fileno())


def _records(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            value = json.load(stream)
            if isinstance(value, dict):
                if FORBIDDEN.intersection(value):
                    raise ValueError("Queries must not contain annotations or detector boxes")
                for key in ("queries", "missing_queries", "missing", "missing_image_class_ids"):
                    if key in value and isinstance(value[key], list):
                        value = value[key]
                        break
            if not isinstance(value, list):
                raise ValueError("Expected a query list or JSONL")
            yield from value


def load_queries(path, images, vocabulary):
    """Validate explicit image/text queries without consuming detector output."""
    queries = []
    seen = set()
    for row in _records(path):
        if isinstance(row, list) and len(row) == 2:
            row = dict(image_id=row[0], category_id=row[1])
        if not isinstance(row, dict) or FORBIDDEN.intersection(row):
            raise ValueError("Query records must not contain boxes or ground truth")
        key = row["image_id"], row["category_id"]
        if any(type(value) is not int for value in key):
            raise ValueError("Query image/category IDs must be integers")
        image_id, category_id = key
        if image_id not in images or category_id not in vocabulary or key in seen:
            raise ValueError(f"Unknown or duplicate query {key}")
        if "text" in row and row["text"].strip().casefold() != vocabulary[category_id].strip().casefold():
            raise ValueError(f"Query text differs from the frozen vocabulary at {key}")
        for field in ("width", "height", "image_sha256"):
            if field in row and field in images[image_id] and row[field] != images[image_id][field]:
                raise ValueError(f"Query image metadata differs at {key}")
        queries.append(key)
        seen.add(key)
    return queries


def canonical_prediction(row, image, text):
    """Convert native presence-on cache rows to validated portable point rows."""
    if row["width"] != image["width"] or row["height"] != image["height"]:
        raise ValueError("Counter image dimensions differ from the image manifest")
    if "image_sha256" in image and row.get("image_sha256") != image["image_sha256"]:
        raise ValueError("Counter image hash differs from the image manifest")
    if "text" in row and row["text"].strip().casefold() != text.strip().casefold():
        raise ValueError("Cached prompt differs from the frozen vocabulary")
    native = row.get("outputs", {}).get("on", row)
    points = native.get("points", native.get("points_original_pixels"))
    count = native.get("count", native.get("pred_count"))
    if not isinstance(points, list) or type(count) is not int or count != len(points):
        raise ValueError("A counter prediction needs an actual point list and matching count")
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
                or not 0 <= point[0] <= image["width"] or not 0 <= point[1] <= image["height"]):
            raise ValueError("Counter point is not finite original-image pixel xy")
    result = dict(image_id=row["image_id"], category_id=row["category_id"],
                  width=image["width"], height=image["height"], text=text,
                  points=points, count=count)
    if "image_sha256" in image:
        result["image_sha256"] = image["image_sha256"]
    for name in ("point_scores", "point_sources", "pdc_count", "rsc_count"):
        if name in native:
            if name.startswith("point_") and len(native[name]) != count:
                raise ValueError("Native point metadata length differs")
            result[name] = native[name]
    for name in ("presence_probability", "presence_logit"):
        if name in row:
            result[name] = row[name]
    return result


def read_requested_cache(paths, required, images, vocabulary):
    """Stream caches; retain only explicitly requested independent queries."""
    predictions = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row["image_id"], row["category_id"]
                if key not in required:
                    continue
                if key in predictions:
                    raise ValueError(f"Duplicated requested query across counter caches: {key}")
                predictions[key] = canonical_prediction(row, images[key[0]], vocabulary[key[1]])
    return predictions


def verify_native_assets(source, checkpoint, config):
    """Validate frozen native entrypoints and full checkpoint before loading."""
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    hashes = {}
    for relative, expected in config["source_sha256"].items():
        actual = file_sha256(source / relative)
        if actual != expected:
            raise ValueError(f"Frozen Count Anything source changed: {relative}")
        hashes[relative] = actual
    runtime_files = config.get("runtime_files", [])
    runtime_records = []
    for item in sorted(runtime_files, key=lambda item: item["path"]):
        relative = Path(item["path"])
        if (relative.is_absolute() or ".." in relative.parts
                or relative.parts[0] not in {"count_anything", "sam3"}):
            raise ValueError("Invalid native runtime manifest path")
        path = source / relative
        actual = dict(path=item["path"], bytes=path.stat().st_size, sha256=file_sha256(path))
        if actual != item:
            raise ValueError(f"Native runtime file differs: {relative}")
        runtime_records.append(actual)
    fingerprint = None
    if runtime_records:
        from hashlib import sha256
        fingerprint = sha256(json.dumps(runtime_records, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
        if fingerprint != config["runtime_sha256_fingerprint"]:
            raise ValueError("The complete native source fingerprint differs")
        expected_paths = {item["path"] for item in runtime_records}
        for package in ("count_anything", "sam3"):
            for path in (source / package).rglob("*"):
                if (path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml"}
                        and str(path.relative_to(source)) not in expected_paths):
                    raise ValueError("Unexpected native runtime source file")
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != config["checkpoint_sha256"]:
        raise ValueError("Count Anything checkpoint differs from the frozen C0 base")
    revision = "UNKNOWN"
    if (source / ".git").exists():
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(source),
                                 "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        revision = result.stdout.strip()
        accepted = {config["source_revision"], *config.get("compatible_source_revisions", [])}
        if revision not in accepted:
            raise ValueError("Count Anything source revision differs from the frozen C0 base")
    return dict(source_revision=revision, declared_source_revision=config["source_revision"],
                source_sha256=hashes, checkpoint_sha256=checkpoint_sha,
                native_runtime_files=len(runtime_records), runtime_sha256_fingerprint=fingerprint,
                source_hash_coverage=("All Count Anything/SAM3 native source and asset files; "
                                      "external environment dependencies are recorded separately"
                                      if runtime_records else
                                      "Selected native entrypoints; not a transitive environment freeze"))


class NativeCountAnything:
    """Frozen C0 native inference with no detector-conditioned inputs."""

    def __init__(self, source, checkpoint, config, device="cuda", memory_fraction=.25,
                 cpu_threads=1):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, str(Path(source).resolve()))
        import torch
        from count_anything.model_builder import build_count_anything_model
        from count_anything.eval.postprocessor import CountAnythingPostProcessor
        from sam3.model.sam3_image_processor import Sam3Processor
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode
        from typing import Any

        if cpu_threads < 1 or not 0 < memory_fraction <= 1:
            raise ValueError("Invalid CPU thread or GPU memory limit")
        torch.set_num_threads(cpu_threads)
        torch.manual_seed(config["seed"])
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        self.model = build_count_anything_model(
            device="cpu", eval_mode=False, checkpoint_path=None, load_from_HF=False,
            enable_segmentation=True, enable_pdc_branch=True, pdc_row=1, pdc_line=1,
            enable_encoder_cross_attn_lora=True, encoder_cross_attn_lora_r=8,
            encoder_cross_attn_lora_alpha=8, encoder_cross_attn_lora_dropout=0.)
        safe_types = [ListConfig, ContainerMetadata, Metadata, AnyNode, Any,
                      list, dict, int, defaultdict]
        with torch.serialization.safe_globals(safe_types):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            result = self.model.load_state_dict(state["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Native checkpoint did not load strictly")
        del state
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = Sam3Processor(self.model, resolution=1008, device=str(self.device))
        self.post = CountAnythingPostProcessor(**config["postprocessor"], use_presence=True)
        self.loaded_image = None
        self.features = None

    def _clone(self, value):
        torch = self.torch
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {key: self._clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone(item) for item in value]
        if isinstance(value, tuple):
            values = [self._clone(item) for item in value]
            return type(value)(*values) if hasattr(value, "_fields") else tuple(values)
        return copy.deepcopy(value)

    def predict(self, image, image_path, category_id, text):
        """Run native image/text grounding and return original-pixel points."""
        from PIL import Image
        from torchvision.transforms import functional as tvf
        torch = self.torch
        identity = (image["image_id"], str(image_path))
        with torch.inference_mode():
            if self.loaded_image != identity:
                self.features = None
                with Image.open(image_path) as picture:
                    if picture.size != (image["width"], image["height"]):
                        raise ValueError("Actual image dimensions differ from manifest")
                    tensor = tvf.to_tensor(tvf.resize(picture.convert("RGB"), (1008, 1008)))
                tensor = tvf.normalize(tensor, [.5] * 3, [.5] * 3).to(self.device).unsqueeze(0)
                self.features = {"img_batch_all_stages": tensor,
                                 **self.model.backbone.forward_image(tensor)}
                self.loaded_image = identity
            features = self._clone(self.features)
            features.update(self.model.backbone.forward_text([text], device=str(self.device)))
            prompt = self.model._get_dummy_prompt(num_prompts=1)
            find = copy.deepcopy(self.processor.find_stage)
            autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if self.device.type == "cuda" else nullcontext())
            with autocast:
                raw = self.model.forward_grounding(features, find, None, prompt)
            size = torch.tensor([[image["height"], image["width"]]], device=self.device)
            native = self.post(raw, size)[0]
            points = native["pred_count_points"]
            scores = native["pred_count_scores"]
            sources = native["pred_count_sources"]
            if not torch.isfinite(points).all() or not torch.isfinite(scores).all():
                raise RuntimeError("Native counter emitted nonfinite point predictions")
            row = dict(image_id=image["image_id"], category_id=category_id,
                       width=image["width"], height=image["height"], text=text,
                       points=points.detach().cpu().tolist(), count=int(native["pred_count"]),
                       point_scores=scores.detach().cpu().tolist(),
                       point_sources=sources.detach().cpu().tolist(),
                       pdc_count=int(native["pdc_count"]), rsc_count=int(native["rsc_count"]),
                       presence_probability=float(raw["presence_logit_dec"].sigmoid().detach().float().cpu().item()),
                       presence_logit=float(raw["presence_logit_dec"].detach().float().cpu().item()))
            if "image_sha256" in image:
                row["image_sha256"] = image["image_sha256"]
            return canonical_prediction(row, image, text)


def run_counter(source, checkpoint, config_path, images_path, vocabulary_path,
                queries_path, output_dir, image_root=None, cache_paths=(),
                device="cuda", memory_fraction=.25, cpu_threads=1, preflight_only=False,
                limit_queries=0):
    """Fill missing independent queries into a new directory; never alter caches."""
    config = json.loads(Path(config_path).read_text())
    if config.get("presence") != "on" or config.get("resolution") != 1008:
        raise ValueError("The portable adapter implements the frozen C0 presence-on 1008 contract")
    images = load_images(images_path)
    vocabulary = load_vocabulary(vocabulary_path)
    queries = load_queries(queries_path, images, vocabulary)
    if limit_queries < 0:
        raise ValueError("Query limit cannot be negative")
    if limit_queries:
        queries = queries[:limit_queries]
    inputs = dict(config=config_path, images=images_path, vocabulary=vocabulary_path,
                  queries=queries_path, **{f"cache_{i}": path for i, path in enumerate(cache_paths)})
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    cached = read_requested_cache(cache_paths, set(queries), images, vocabulary)
    missing = [key for key in queries if key not in cached]
    assets = verify_native_assets(source, checkpoint, config)
    summary = dict(status="CPU_PREFLIGHT_PASS", requested_queries=len(queries),
                   reused_queries=len(cached), new_queries=len(missing),
                   labels_read=False, detector_boxes_read=False, geometric_prompts=False,
                   assets=assets, inputs_sha256=input_hashes)
    if preflight_only:
        return summary
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    _dump(out / "run-record.json", dict(summary, status="IN_PROGRESS", device=device,
          memory_fraction=memory_fraction, cpu_threads=cpu_threads,
          started_utc=datetime.now(timezone.utc).isoformat(),
          preprocessing="RGB PIL resize1008x1008, ToTensor, normalize(.5,.5)",
          precision="image/text FP32; grounding BF16 on CUDA; native GPU postprocessor",
          postprocessor=config["postprocessor"], implementation_sha256=file_sha256(__file__)))
    model = None
    try:
        if missing:
            model = NativeCountAnything(source, checkpoint, config, device,
                                        memory_fraction, cpu_threads)
        with (out / "new_predictions.jsonl").open("x", encoding="utf-8") as new_stream:
            for index, (image_id, category_id) in enumerate(missing):
                image = images[image_id]
                if "image_path" in image:
                    path = Path(image["image_path"])
                elif image_root is not None:
                    path = Path(image_root) / image["file_name"]
                else:
                    raise ValueError("Provide --image-root or image_path records")
                if "image_sha256" in image and file_sha256(path) != image["image_sha256"]:
                    raise ValueError(f"Image bytes differ for image {image_id}")
                started = time.monotonic()
                result = model.predict(image, path, category_id, vocabulary[category_id])
                _write_row(new_stream, result)
                cached[image_id, category_id] = result
                print(json.dumps(dict(completed_new_queries=index + 1, total_new_queries=len(missing),
                                      image_id=image_id, category_id=category_id,
                                      seconds=time.monotonic() - started)), flush=True)
        gpu_metrics = {}
        if model is not None:
            if model.device.type == "cuda":
                torch = model.torch
                device_index = model.device
                torch.cuda.synchronize(device_index)
                gpu_metrics = dict(peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device_index),
                                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device_index))
            del model
            model = None
            gc.collect()
            if gpu_metrics:
                torch.cuda.empty_cache()
                gpu_metrics["gpu_allocated_after_model_release_bytes"] = torch.cuda.memory_allocated(device_index)
        _dump(out / "MODEL_RELEASED_BEFORE_MERGE.json", dict(
              status="INFERENCE_COMPLETE_MODEL_RELEASED_BEFORE_CPU_MERGE", **gpu_metrics))
        pending = out / ("predictions." + uuid.uuid4().hex + ".partial")
        final = out / "predictions.jsonl"
        with pending.open("x", encoding="utf-8") as stream:
            for key in queries:
                _write_row(stream, cached[key], durable=False)
            stream.flush()
            os.fsync(stream.fileno())
        if final.exists():
            raise FileExistsError("Merged output already exists; preserved")
        pending.rename(final)
        directory_fd = os.open(out, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if any(file_sha256(path) != input_hashes[name] for name, path in inputs.items()):
            raise RuntimeError("An inference input changed during prediction")
        for relative, digest in assets["source_sha256"].items():
            if file_sha256(Path(source) / relative) != digest:
                raise RuntimeError("Native source changed during prediction")
        for item in config.get("runtime_files", []):
            if file_sha256(Path(source) / item["path"]) != item["sha256"]:
                raise RuntimeError("Native runtime source or assets changed during prediction")
        result = dict(summary, status="INDEPENDENT_COUNTER_PREDICTIONS_COMPLETE",
                      predictions_sha256=file_sha256(out / "predictions.jsonl"),
                      new_predictions_sha256=file_sha256(out / "new_predictions.jsonl"),
                      record_sha256=file_sha256(out / "run-record.json"),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        result.update(gpu_metrics)
        _dump(out / "GT_FREE_COMPLETE.json", result)
        return result
    except Exception as error:
        _dump(out / "PARTIAL.json", dict(status="PARTIAL", error_type=type(error).__name__,
                                       error=str(error), preserve_all_outputs=True,
                                       completed_queries=len(cached)))
        raise
