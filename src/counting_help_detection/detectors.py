"""Adapters for released detector code, without vendoring upstream models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib import import_module
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
from typing import Any
from zipfile import is_zipfile


COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)

LABEL_SPACES = {"coco_contiguous_0_based", "dataset_native_category_id"}


def detector_label_space(spec: dict[str, Any]) -> str:
    """Keep dataset category semantics independent of tensor output format."""
    value = spec.get("label_space")
    if value is None:
        value = ("coco_contiguous_0_based" if spec["postprocessor"] == "deploy"
                 else "dataset_native_category_id")
    if value not in LABEL_SPACES:
        raise ValueError("The model registry names an unsupported label space.")
    return value


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_models(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["models"]


def canonical_predictions(
    image_id: int,
    labels: list[int],
    boxes_xyxy: list[list[float]],
    scores: list[float],
    *,
    contiguous_labels: bool,
    pool_cap: int | None = None,
    category_ids: tuple[int, ...] = COCO_CATEGORY_IDS,
    ignored_labels: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    """Preserve native order, coordinates and candidate identities.

    Native finite boxes with nonpositive extents remain in the output, matching
    the upstream COCO conversion. Geometry diagnostics belong in the result
    ledger, and do not silently change the detector's candidate pool.
    """
    if not len(labels) == len(boxes_xyxy) == len(scores):
        raise ValueError("The detector returned arrays with different lengths.")
    if pool_cap is not None and pool_cap < 1:
        raise ValueError("The exported pool cap must be positive.")
    limit = len(labels) if pool_cap is None else min(pool_cap, len(labels))
    result = []
    for index in range(limit):
        label, box, score = labels[index], boxes_xyxy[index], float(scores[index])
        if type(label) is not int:
            raise ValueError("Detector class IDs must be integers.")
        if label in ignored_labels:
            continue
        if contiguous_labels:
            if not 0 <= label < len(COCO_CATEGORY_IDS):
                raise ValueError(f"Unexpected contiguous COCO label {label}.")
            label = COCO_CATEGORY_IDS[label]
        if label not in category_ids:
            raise ValueError(f"Unexpected dataset category {label}.")
        box = list(map(float, box))
        if len(box) != 4 or not all(map(isfinite, box)):
            raise ValueError("The detector returned a nonfinite box.")
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError("The detector returned an invalid score.")
        result.append({
            "image_id": image_id,
            "category_id": label,
            "candidate_index": index,
            "bbox_xyxy": box,
            "score": score,
        })
    return result


def source_snapshot(directory: Path) -> dict[str, Any]:
    """Record executable Python and YAML contents for an upstream checkout."""
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
            if not any(part in {".git", "__pycache__"} for part in path.parts):
                files[str(path.relative_to(directory))] = file_sha256(path)
    revision = None
    if (directory / ".git").exists():
        command = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True,
        )
        if command.returncode == 0:
            revision = command.stdout.strip()
    return {"git_revision": revision, "files": files}


def verify_source_receipt(directory: Path, receipt_path: Path, expected_revision: str) -> dict[str, Any]:
    """Check an archive receipt with commit/source_revision and relative file hashes.

    ``files`` accepts either a mapping of paths to SHA-256 values or a list of
    objects with ``path`` and ``sha256`` fields. Receipts accompany an archive
    acquired from the pinned upstream revision. The receipt itself is retained
    by hash in the inference protocol.
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    revision = receipt.get("commit", receipt.get("source_revision"))
    if revision != expected_revision:
        raise ValueError("The source archive receipt names a different upstream revision.")
    entries = receipt["files"]
    pairs = entries.items() if isinstance(entries, dict) else ((row["path"], row["sha256"]) for row in entries)
    checked = set()
    for name, digest in pairs:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in checked:
            raise ValueError("The source receipt contains an unsafe or duplicate file path.")
        if file_sha256(directory / relative) != digest:
            raise ValueError(f"Source archive checksum mismatch for {name}.")
        checked.add(name)
    if not checked:
        raise ValueError("The source archive receipt has no file hashes.")
    return {"receipt_sha256": file_sha256(receipt_path), "checked_files": len(checked), "revision": revision}


@dataclass(frozen=True)
class DetectorOptions:
    model_name: str
    source_dir: Path
    checkpoint: Path
    registry: dict[str, Any]
    config: Path | None = None
    device: str = "cpu"
    threads: int = 1
    gpu_memory_fraction: float | None = None
    expected_checkpoint_sha256: str | None = None
    source_manifest: Path | None = None


class OfficialDetector:
    """Load one upstream model family per Python process."""

    def __init__(self, options: DetectorOptions):
        self.options = options
        spec = options.registry
        self.spec = spec
        self.label_space = detector_label_space(spec)
        if options.threads < 1:
            raise ValueError("CPU thread count must be positive.")
        if spec.get("requires_cuda") and not options.device.startswith("cuda"):
            raise ValueError("This adapter uses the author's CUDA GLA implementation. Select a CUDA device.")
        self.source_dir = options.source_dir.resolve()
        self.execution_dir = self.source_dir / spec.get("source_subdirectory", "")
        self.config = (options.config or self.execution_dir / spec["config"]).resolve()
        self.checkpoint = options.checkpoint.resolve()
        self.checkpoint_hash = file_sha256(self.checkpoint)
        expected = options.expected_checkpoint_sha256 or spec.get("checkpoint_sha256")
        if expected and expected != self.checkpoint_hash:
            raise ValueError("The checkpoint SHA-256 differs from the requested release.")
        self.source_evidence = source_snapshot(self.source_dir)
        fingerprint = sha256(json.dumps(
            self.source_evidence["files"], sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.source_evidence["fingerprint_sha256"] = fingerprint
        if fingerprint != spec["source_fingerprint_sha256"]:
            raise ValueError("Upstream executable files differ from this model's verified source snapshot.")
        observed = self.source_evidence["git_revision"]
        if observed is not None and observed != spec["source_revision"]:
            raise ValueError("The upstream checkout is at a different revision from this model's reference.")
        if options.source_manifest is not None:
            self.source_evidence["archive_receipt"] = verify_source_receipt(
                self.source_dir, options.source_manifest, spec["source_revision"],
            )
        prefix = spec["python_namespace"]
        existing = sys.modules.get(prefix)
        if existing is not None:
            location = getattr(existing, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(self.execution_dir):
                raise RuntimeError("Another upstream detector package is loaded. Use a separate process for this model.")
        sys.path.insert(0, str(self.execution_dir))
        import torch
        import torchvision.transforms as transforms

        self.torch = torch
        device = torch.device(options.device)
        if device.type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
            torch.cuda.set_device(device)
        self.device = str(device)
        torch.set_num_threads(options.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261117)
        torch.backends.cudnn.benchmark = False
        if options.gpu_memory_fraction is not None:
            if not options.device.startswith("cuda") or not 0 < options.gpu_memory_fraction <= 1:
                raise ValueError("GPU memory fraction requires a CUDA device and a value in (0, 1].")
            torch.cuda.set_per_process_memory_fraction(options.gpu_memory_fraction, device)
        config_type = import_module(prefix + ".core").YAMLConfig
        kwargs = {"resume": str(self.checkpoint)}
        if options.model_name.startswith("dfine-x"):
            kwargs["HGNetv2"] = {"pretrained": False}
        cfg = config_type(str(self.config), **kwargs)
        for block, values in spec.get("full_checkpoint_initialization", {}).items():
            cfg.yaml_cfg[block].update(values)
        saved = torch.load(
            self.checkpoint, map_location="cpu", weights_only=True,
            mmap=is_zipfile(self.checkpoint),
        )
        self.state_key = "ema.module" if "ema" in saved else "model"
        state = saved["ema"]["module"] if "ema" in saved else saved["model"]
        cfg.model.load_state_dict(state, strict=True)
        del state, saved
        if spec["postprocessor"] == "deploy":
            self.model = cfg.model.deploy().eval().to(self.device)
            self.post = cfg.postprocessor.deploy().eval().to(self.device)
        else:
            self.model = cfg.model.eval().to(self.device)
            self.post = cfg.postprocessor.eval().to(self.device)
        if self.post.num_classes != spec.get("head_classes", 80):
            raise ValueError("The checkpoint head differs from the recorded class count.")
        if self.post.num_classes == 80:
            upstream_map = import_module(prefix + ".data.dataset").mscoco_label2category
            if tuple(upstream_map[i] for i in range(80)) != COCO_CATEGORY_IDS:
                raise ValueError("The upstream COCO category map differs from the recorded map.")
        self.input_size = tuple(cfg.yaml_cfg["eval_spatial_size"])
        if self.input_size != tuple(spec["input_size"]):
            raise ValueError("The supplied config changes the recorded inference resolution.")
        operations = [transforms.Resize(self.input_size), transforms.ToTensor()]
        if spec["normalization"]:
            norm = spec["normalization"]
            operations.append(transforms.Normalize(norm["mean"], norm["std"]))
        self.transform = transforms.Compose(operations)
        self.native_pool = int(self.post.num_top_queries)

    def protocol(self) -> dict[str, Any]:
        return {
            "model": self.options.model_name,
            "reference": self.spec,
            "observed_source": self.source_evidence,
            "source_dir": str(self.source_dir),
            "config": str(self.config),
            "config_sha256": file_sha256(self.config),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "checkpoint_key": self.state_key,
            "strict_state_loading": True,
            "device": self.device,
            "dtype": "float32",
            "input_size": list(self.input_size),
            "normalization": self.spec["normalization"],
            "native_pool": self.native_pool,
            "label_space": self.label_space,
            "cpu_threads": self.options.threads,
            "gpu_memory_fraction": self.options.gpu_memory_fraction,
            "torch_version": self.torch.__version__,
            "adapter_sha256": file_sha256(Path(__file__)),
        }

    def predict(
        self, image_id: int, image, pool_cap: int | None = None,
        *, image_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        torch = self.torch
        batch = self.transform(image).unsqueeze(0).to(self.device)
        original = torch.tensor([image.size], device=self.device)
        with torch.inference_mode():
            result = self.post(self.model(batch), original)
        if self.spec["postprocessor"] == "deploy":
            labels, boxes, scores = (value[0].cpu().tolist() for value in result)
        else:
            result, = result
            labels = result["labels"].cpu().tolist()
            boxes = result["boxes"].cpu().tolist()
            scores = result["scores"].cpu().tolist()
        self.last_excluded_labels = {
            str(label): labels[:pool_cap].count(label) if pool_cap is not None else labels.count(label)
            for label in self.spec.get("ignored_labels", ())
        }
        return canonical_predictions(
            image_id, labels, boxes, scores,
            contiguous_labels=self.label_space == "coco_contiguous_0_based",
            pool_cap=pool_cap,
            category_ids=tuple(self.spec.get("category_ids", COCO_CATEGORY_IDS)),
            ignored_labels=tuple(self.spec.get("ignored_labels", ())),
        )
