# Group DETR v2

The experiment uses PaddleDetection’s released ViT-Huge Group DETR v2 implementation and COCO checkpoint. The paper reports 64.5 AP on COCO test-dev with test-time augmentation and 55.6 AP on its first 5,000 Objects365 validation images. The released PaddleDetection checkpoint reports 63.3 AP on COCO val2017 with its own evaluation settings. Keep these settings attached to each reported score. [Paper, November 2022](https://arxiv.org/abs/2211.03594), [official model zoo](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/group_detr/README.md)

## Source and checkpoint

| Item | Pinned source |
| --- | --- |
| Implementation | PaddleDetection commit `b25522a0f4bde8c80603f3ba5e3472059972e3b5` |
| Executable source manifest SHA-256 | `417cd08108838910c67d280093df8f62bd87f482a32ea8207f639fc4f649f81e` over sorted Python/YAML path-to-hash pairs |
| Source archive | [PaddleDetection archive](https://codeload.github.com/PaddlePaddle/PaddleDetection/tar.gz/b25522a0f4bde8c80603f3ba5e3472059972e3b5) |
| Model configuration | [`group_dino_vit_huge_4scale_1x_coco.yml`](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/group_detr/group_dino_vit_huge_4scale_1x_coco.yml) |
| Full detector checkpoint | [`group_dino_vit_huge_4scale_1x_coco.pdparams`](https://bj.bcebos.com/v1/paddledet/models/group_dino_vit_huge_4scale_1x_coco.pdparams) |
| Checkpoint bytes | `2621220489` |
| Checkpoint SHA-256 measured after download | `17c5c8de0aba2f614f4d550b48999598380e06f32d96b355f8d1afcd93212aef` |
| Upstream source license | [Apache 2.0](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/LICENSE) |

The full checkpoint contains the backbone and detection head. Loading this checkpoint for inference succeeded without a separate backbone download. The configuration’s `pretrain_weights` entry describes training initialization.

## Portable inference

Use a separate Python 3.12 environment with the tested inference dependencies in `configs/environments/paddle_group_inference.txt`. The adapter imports Paddle lazily. The verified CPU build is `paddlepaddle==3.3.1`. The full CUDA reproduction uses the official GPU build below. Installing this wheel with its declared dependencies provides the exact CUDA 11.8 library packages required by the wheel.

```bash
python -m pip install https://paddle-whl.cdn.bcebos.com/stable/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl
python -m pip install -r configs/environments/paddle_group_inference.txt
python -m pip install -e .
```

The pinned upstream repository also contains training and tracking dependencies. The recipe above records the tested inference environment, including the modules reached by PaddleDetection's eager import chain.

After obtaining the pinned upstream source and full checkpoint above, export the native candidates with:

```bash
FLAGS_gpu_memory_limit_mb=22528 FLAGS_allocator_strategy=auto_growth \
python scripts/run_group_detr.py \
  --source-dir /path/to/PaddleDetection \
  --checkpoint /path/to/group_dino_vit_huge_4scale_1x_coco.pdparams \
  --images /path/to/images_gt_free.json \
  --image-root /path/to/coco/val2017 \
  --output /path/to/new-group-output \
  --device cuda:0 --threads 1 --gpu-memory-fraction 0.9
```

Use a fresh process and a new output directory. The full run used the GPU alone while other GPU experiments retained their saved progress. For a small CPU check, use a single-image manifest and `--device cpu` without a GPU memory fraction. `--resume` verifies the frozen protocol and image hashes before reusing completed chunks. The runner rejects a different Git revision when present and verifies the complete executable source manifest even for an archive extraction. It verifies the released config and checkpoint, checks all 767 loaded state tensors, uses the native OpenCV image decoder and test reader, and seals the complete candidate stream before evaluation. It retains finite native boxes even when their extents are nonpositive. No ground-truth file is an inference argument.

The default exports all 300 native candidates. `--pool-cap` explicitly defines a smaller diagnostic pool and must be kept identical across all reranking arms. For a compatible CUDA environment, select `--device cuda:0`; `--gpu-memory-fraction` sets Paddle's allocator fraction before import. Record additional allocator limits in the run environment and verify actual memory use before a full run. The GPU adapter releases idle allocator cache after the verified checkpoint load and before each image, and records this policy in the frozen protocol. This changes memory reuse only.

## Inference settings

For the CPU probe, the adapter applies the following overrides through `ppdet.core.workspace.merge_config` before constructing the model. CUDA runs set `use_gpu=True` and select the requested CUDA device.

```python
overrides = {
    "use_gpu": False,
    "worker_num": 0,
    "GroupDINOTransformer": {"dual_queries": False},
    "DETRPostProcess": {"dual_queries": False},
    "TestDataset": {"anno_path": None, "dataset_dir": ""},
}
```

`dual_queries=False` selects the single inference query group. The published checkpoint contains all 767 state tensors for this configuration. The training configuration additionally constructs 140 tensors for extra query groups. Strict coverage must check the inference configuration before loading weights.

The native test reader resizes with target size `[1184, 2000]` and `keep_ratio=True`, normalizes with the published ImageNet mean and standard deviation, and returns 300 detections per image. Export box coordinates back in original-image pixels. Map contiguous model labels to the original 80 sparse COCO category IDs. Preserve the native candidate index for all score adjustments.

For an image-only run, the instantiated `ImageFolder` must have `anno_path=None`. This permits prediction without reading ground-truth annotations.

## Paddle eager compatibility

At the pinned source revision, `GroupDINOTransformer._get_encoder_input` attempts to concatenate two integer dimensions using `paddle.concat([h, w])`. With the verified Paddle 3.3.1 eager CPU runtime, the shape construction requires the following equivalent integer tensor.

```python
spatial_shapes.append(paddle.to_tensor([h, w], dtype="int64"))
```

The experiment records this single-expression compatibility change, the original function hash, and the adapted function hash. The checkpoint, learned parameters, and preprocessing retain their published values. Keep the upstream source revision and the compatibility change in the run record.

## Completed verification

On 17 November 2026, a real COCO image passed a single-thread CPU forward with the released checkpoint. All 767 model state tensors matched the checkpoint by key, shape, dtype, and loaded value before inference. The native forward returned 300 boxes for 80 COCO categories. The image-only reader used no ground-truth annotations.

The validated CPU environment used Python 3.12, PaddlePaddle 3.3.1, NumPy 1.26.4, OpenCV 4.11.0.86, imgaug 0.4.0, PyYAML 6.0.3, and pycocotools 2.0.11.

The public `run_group_detr.py` entry also completed real-image CUDA 11.8 runs in FP32 on an RTX 4090 with PaddlePaddle GPU 3.3.1. It passed the complete source fingerprint and strict 767-tensor load checks, emitted 300 native candidates, and sealed the outputs before reading ground truth. A square image resized to `1184 × 1184` used peak allocated memory of 8.00 GiB and reserved memory of 10.97 GiB under a 12 GiB allocator limit.

An image selected solely for the largest transformed pixel area in the frozen manifest resized to `1184 × 1999`. It failed a 16 GiB limit even after releasing idle loading cache: 12.84 GiB was active when another 5.02 GiB was requested. It then passed in a separate process under a 22 GiB limit, with peak allocated memory of 17.86 GiB and reserved memory of 18.66 GiB. Forward and chunk writing took about 1.17 seconds. CPU fallback and native CUDA outputs can reorder low-scoring candidates, so the production CUDA export defines the frozen candidate identities.

## Completed full COCO result

On 18 November 2026, the native CUDA export completed all 5,000 COCO val2017 images and sealed 1,500,000 candidates before opening ground-truth annotations. Stock COCO box evaluation over the original 80 classes with `maxDets=100` measured **63.299883 AP**, displayed as **63.30 AP**, matching the released checkpoint's reported 63.3 AP.

The experiment used the previously evaluated fixed COCO benchmark. Its scope is released-detector reproduction on this fixed benchmark. The original COCO ground-truth SHA-256 was checked against its prior sealed value, `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. The native prediction stream SHA-256 is `e4e34025f016785235b20ddf94f593a502549f859b955c17988d20e498d556ad`, and its frozen protocol SHA-256 is `dc6ffaf4ef7a8691357063bbed0d94ee6e0c06cb98362bf6ffccbdf1669b21e1`.

This result verifies the original detector baseline. Matched confidence and counting-point results for this detector remain pending.

## Evaluation

Score the complete original COCO taxonomy on the frozen image manifest. Use stock COCO box AP with IoU thresholds 0.50–0.95 and `maxDets=100`. Within the same detector, compare original scores, matched-budget confidence, and counting-point scores with identical candidate IDs, categories, and coordinates.

For each image and class, sort by descending score and then candidate index. Greedily match the first five candidates one-to-one to ground truth at IoU ≥ 0.50, choosing the lowest annotation ID on an IoU tie. Report original correct candidate IDs that drop out, previously matched targets lost, and newly recovered targets separately. Image/class queries containing crowd or ignored annotations are excluded from this diagnostic. The AP evaluator retains the stock COCO treatment of those annotations.

The canonical Objects365 experiment uses 5,000 fixed images and its complete 365-category taxonomy. A matching released 365-class Group DETR v2 checkpoint has not been verified. A COCO-head transfer test would need a separately declared category mapping and evaluation scope.
