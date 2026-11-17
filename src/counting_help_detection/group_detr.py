"""Native PaddleDetection Group DETR v2 inference for its released COCO model."""

from __future__ import annotations

from hashlib import sha256
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from typing import Any

from .detectors import (
    COCO_CATEGORY_IDS, DetectorOptions, canonical_predictions, file_sha256,
    source_snapshot,
)


GROUP_SPEC = {
    "name": "Group DETR v2 (released PaddleDetection COCO checkpoint)",
    "paper": "https://arxiv.org/abs/2211.03594",
    "source": "https://github.com/PaddlePaddle/PaddleDetection",
    "revision": "b25522a0f4bde8c80603f3ba5e3472059972e3b5",
    "executable_source_manifest_sha256": "417cd08108838910c67d280093df8f62bd87f482a32ea8207f639fc4f649f81e",
    "source_license": "Apache-2.0",
    "config": "configs/group_detr/group_dino_vit_huge_4scale_1x_coco.yml",
    "config_sha256": "714fbab5ef4c948de8105ba2a0daca32d8208e9cc7749f1b7ec72558aeca307c",
    "checkpoint_url": "https://bj.bcebos.com/v1/paddledet/models/group_dino_vit_huge_4scale_1x_coco.pdparams",
    "checkpoint_sha256": "17c5c8de0aba2f614f4d550b48999598380e06f32d96b355f8d1afcd93212aef",
    "published_coco_val_ap": 63.3,
    "native_classes": 80,
    "native_pool": 300,
}


class OfficialGroupDetector:
    """Keep the released model, reader and postprocessor in an isolated process."""

    def __init__(self, options: DetectorOptions):
        if options.threads < 1:
            raise ValueError("CPU thread count must be positive.")
        self.options = options
        self.source_dir = options.source_dir.resolve()
        self.source_evidence = source_snapshot(self.source_dir)
        revision = self.source_evidence["git_revision"]
        if revision is not None and revision != GROUP_SPEC["revision"]:
            raise ValueError("The upstream Git revision differs from the recorded release.")
        source_fingerprint = sha256(json.dumps(
            self.source_evidence["files"], sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if source_fingerprint != GROUP_SPEC["executable_source_manifest_sha256"]:
            raise ValueError("The upstream Python/YAML files differ from the pinned source archive.")
        self.source_evidence["executable_manifest_sha256"] = source_fingerprint
        self.config = (options.config or self.source_dir / GROUP_SPEC["config"]).resolve()
        self.checkpoint = options.checkpoint.resolve()
        self.checkpoint_hash = file_sha256(self.checkpoint)
        expected_hash = options.expected_checkpoint_sha256 or GROUP_SPEC["checkpoint_sha256"]
        if self.checkpoint_hash != expected_hash:
            raise ValueError("The checkpoint SHA-256 differs from the requested release.")
        if file_sha256(self.config) != GROUP_SPEC["config_sha256"]:
            raise ValueError("Use the recorded official Group DETR v2 config.")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(options.threads)
        self.device = options.device.replace("cuda", "gpu", 1)
        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif not self.device.startswith("gpu:"):
            raise ValueError("Select cpu or cuda:N for this Paddle adapter.")
        fraction = options.gpu_memory_fraction
        if fraction is not None:
            if self.device == "cpu" or not 0 < fraction <= 1:
                raise ValueError("GPU memory fraction requires a GPU and a value in (0, 1].")
            os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = str(fraction)
            os.environ["FLAGS_allocator_strategy"] = "auto_growth"
        for prefix in ("ppdet", "paddle"):
            if prefix in sys.modules:
                raise RuntimeError("Start a fresh process for Group DETR v2 so runtime flags precede imports.")
        sys.path.insert(0, str(self.source_dir))
        import numpy as np
        import paddle
        from ppdet.core.workspace import create, load_config, merge_config
        from ppdet.data.source.category import get_categories
        from ppdet.engine import Trainer
        from ppdet.modeling.transformers import group_detr_transformer as group_module

        self.np, self.paddle, self.create = np, paddle, create
        paddle.set_device(self.device)
        paddle.seed(20261117)
        cfg = load_config(str(self.config))
        cfg.use_gpu = self.device != "cpu"
        cfg.worker_num = 0
        self.overrides = {
            "GroupDINOTransformer": {"dual_queries": False},
            "DETRPostProcess": {"dual_queries": False},
            "TestDataset": {"anno_path": None, "dataset_dir": ""},
        }
        cfg = merge_config(self.overrides)
        if cfg["num_classes"] != 80:
            raise ValueError("The released Group checkpoint has a native 80-class COCO head.")
        original = textwrap.dedent(inspect.getsource(group_module.GroupDINOTransformer._get_encoder_input))
        needle = "spatial_shapes.append(paddle.concat([h, w]))"
        replacement = "spatial_shapes.append(paddle.to_tensor([h, w], dtype='int64'))"
        if original.count(needle) != 1:
            raise ValueError("The recorded eager shape compatibility adapter no longer matches upstream.")
        patched = original.replace(needle, replacement)
        scope = dict(vars(group_module))
        exec(compile(patched, "<group_detr_eager_shape_compatibility>", "exec"), scope)
        group_module.GroupDINOTransformer._get_encoder_input = scope["_get_encoder_input"]
        self.shape_adapter = {
            "function": "GroupDINOTransformer._get_encoder_input",
            "original": needle, "replacement": replacement,
            "original_function_sha256": sha256(original.encode()).hexdigest(),
            "patched_function_sha256": sha256(patched.encode()).hexdigest(),
            "source_files_modified": False,
        }
        self.trainer = Trainer(cfg, mode="test")
        self.model = self.trainer.model
        if self.model.transformer.dual_queries or self.model.post_process.dual_queries:
            raise ValueError("Training-only dual query branches were not disabled.")
        state = paddle.load(str(self.checkpoint))
        expected = self.model.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        shared = set(expected) & set(state)
        shape_bad = sorted(key for key in shared if tuple(expected[key].shape) != tuple(state[key].shape))
        dtype_bad = sorted(key for key in shared if str(expected[key].dtype).replace("paddle.", "") != str(state[key].dtype).replace("paddle.", ""))
        if len(expected) != 767 or len(state) != 767 or missing or unexpected or shape_bad or dtype_bad:
            raise ValueError({"expected": len(expected), "checkpoint": len(state), "missing": missing,
                              "unexpected": unexpected, "shape_mismatches": shape_bad, "dtype_mismatches": dtype_bad})
        self.model.set_dict(state)
        actual = self.model.state_dict()
        value_bad = sorted(key for key in expected if not np.array_equal(
            actual[key].numpy(), state[key].numpy() if hasattr(state[key], "numpy") else np.asarray(state[key]),
        ))
        if value_bad:
            raise ValueError({"loaded_value_mismatches": value_bad})
        self.coverage = {"matched_key_shape_dtype_value_tensors": 767, "missing": [], "unexpected": [],
                         "shape_mismatches": [], "dtype_mismatches": [], "value_mismatches": []}
        del expected, actual, state
        self._release_idle_cuda_cache()
        self.trainer.dataset.anno_path = None
        self.trainer.dataset.dataset_dir = ""
        if self.trainer.dataset.get_anno() is not None:
            raise ValueError("The inference reader must not open ground-truth annotations.")
        mapping, _ = get_categories("COCO", None)
        if tuple(mapping[i] for i in range(80)) != COCO_CATEGORY_IDS:
            raise ValueError("The native COCO class mapping differs from the recorded mapping.")
        self.model.eval()

    def protocol(self) -> dict[str, Any]:
        return {
            "model": self.options.model_name,
            "reference": GROUP_SPEC,
            "source_dir": str(self.source_dir),
            "observed_source": self.source_evidence,
            "config": str(self.config), "config_sha256": file_sha256(self.config),
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_hash,
            "strict_state_loading": self.coverage,
            "native_pool": 300, "native_classes": 80,
            "device": self.device, "dtype": "float32",
            "cpu_threads": self.options.threads,
            "gpu_memory_fraction": self.options.gpu_memory_fraction,
            "paddle_allocator_environment": {
                name: os.environ.get(name) for name in (
                    "FLAGS_gpu_memory_limit_mb", "FLAGS_fraction_of_gpu_memory_to_use",
                    "FLAGS_allocator_strategy",
                )
            },
            "paddle_version": self.paddle.__version__,
            "cuda_cache_policy": "gc.collect and paddle.device.cuda.empty_cache after strict load and before each image",
            "inference_overrides": self.overrides,
            "eager_shape_adapter": self.shape_adapter,
            "reader": "Official TestReader; cv2 Decode; Resize [1184, 2000] keep_ratio=True",
            "ground_truth_annotations_loaded": False,
            "adapter_sha256": file_sha256(Path(__file__)),
        }

    def _release_idle_cuda_cache(self):
        if self.device != "cpu":
            gc.collect()
            self.paddle.device.cuda.empty_cache()

    def predict(self, image_id, image, pool_cap=None, *, image_path=None):
        if image_path is None:
            raise ValueError("A file path is required to preserve the official cv2 image decoder.")
        dataset = self.trainer.dataset
        if dataset.get_anno() is not None:
            raise ValueError("Ground-truth annotations are disabled for inference.")
        self._release_idle_cuda_cache()
        dataset.set_images([str(image_path)], do_eval=False)
        loader = self.create("TestReader")(dataset, 0)
        batches = 0
        with self.paddle.no_grad():
            for data in loader:
                batches += 1
                outputs = self.model(data)
                boxes = outputs["bbox"].numpy()
                counts = outputs["bbox_num"].numpy().tolist()
        if batches != 1 or counts != [300] or boxes.shape != (300, 6):
            raise ValueError("The native single-image postprocessor must return exactly 300 candidates.")
        labels = boxes[:, 0]
        if not self.np.isfinite(labels).all() or not self.np.equal(labels, self.np.floor(labels)).all():
            raise ValueError("The detector returned a nonintegral class ID.")
        return canonical_predictions(
            image_id, labels.astype(int).tolist(), boxes[:, 2:6].tolist(), boxes[:, 1].tolist(),
            contiguous_labels=True, pool_cap=pool_cap,
        )
