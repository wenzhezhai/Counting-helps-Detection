# DINOv3 ViT-7B/16 + Plain-DETR

This entry uses the official DINOv3 backbone and COCO detection head. It exports predictions from frozen released weights before evaluation.

The selected reproduction uses the official **1536-pixel short-edge** single-scale setting. The paper's full COCO validation reference for this setting is **65.4 AP**. The **2048-pixel** setting has a published reference of **65.6 AP**. Its largest geometry exceeded available device memory. The recorded fallback was selected from memory measurements before DINO AP evaluation.

The paper specifies short-edge resizing and rounding to a patch multiple. Its detector evaluation precision and initial image interpolation are unspecified. This run uses explicit whole-model CUDA BF16 autocast, BF16 Linear/Conv parameter storage and PIL bilinear image resizing. The 1,500-output parity test compares the storage adapter against unmodified window operations under that same recorded runtime. [DINOv3, Appendix D.9](https://arxiv.org/html/2508.10104v1), [released detector entry](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/detectors.py#L23-L37)

## Source and weights

- [Official repository](https://github.com/facebookresearch/dinov3), revision `6876159a11b4df116f30f667f8c9888617df0751`.
- [Official pretrained detection heads and weight access](https://github.com/facebookresearch/dinov3#pretrained-heads---detector-trained-on-coco2017-dataset).
- [DINOv3 License Agreement](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md). Obtain the weights through the official access process.
- [Pinned native window operations](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/detection/models/windows.py) and [backbone/position joiner](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/detection/models/backbone.py).

The executable source fingerprint and both complete weight checksums are recorded in `configs/dino_vit7b16.json`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` | 26,864,407,683 | `a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7` |
| `dinov3_vit7b16_coco_detr_head-b0235ff7.pth` | 1,549,231,874 | `b0235ff7ea0a037b521f58eb7915d3a022c75319e93307dd48dfc5fd85c6f031` |

The feature-storage adapter retains the upstream copyright and license notices. The repository includes the [complete DINOv3 agreement](../third_party/notices/DINOv3-LICENSE.md) and [original source attributions](../third_party/notices/DINOv3-NOTICE.md). The original window/backbone operations also credit their Plain-DETR, Deformable DETR, and DETR lineage. The separate materializer is an independent tensor-mapping adapter that invokes the unmodified upstream architecture.

## Inference environment

The tested runtime uses Linux, Python 3.12.14, PyTorch 2.7.1 with CUDA 12.6, torchvision 0.22.1, NumPy 1.26.4, and Pillow 12.3.0. This detection path uses PyTorch, torchvision, NumPy, and Pillow. The project package supplies the common scoring dependencies, including pycocotools and SciPy.

```bash
python3.12 -m venv .venv-dino
.venv-dino/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.venv-dino/bin/python -m pip install numpy==1.26.4 Pillow==12.3.0
.venv-dino/bin/python -m pip install -e . -c configs/environments/torch_inference.txt
```

The tested profile used an otherwise idle GPU with a 22-GiB PyTorch allocation cap on a 24-GiB-class device, batch size one, and `expandable_segments:True`. The memory cap is a configurable resource budget; it does not change the model. The exporter checks that the requested budget fits the currently free memory before loading. Each run records these inference settings.

## What the adapter preserves

- Native architecture and complete checkpoint coverage, with a strict map of 932 unique parameter/buffer objects and 988 persistent state keys.
- Linear/Conv parameters are materialized in BF16 for the CUDA autocast path. Other parameters and buffers retain their native types, including FP32 normalization, LayerScale, position encoding, tokens, and embeddings. Forward inference uses CUDA BF16 autocast.
- Aspect ratio is retained before rounding each resized dimension up to a multiple of 16. Images use PIL bilinear resize, ImageNet normalization and the full aspect-ratio-dependent long edge.
- The native head has 91 sparse COCO slots and returns 1,500 candidates. All 1,500 are archived. A separate eligible stream selects the 80 native COCO category IDs **after** native top-k and records all 11 unused slots separately. Original candidate indices remain attached. Finite boxes with nonpositive extent remain in the output and audit counts.
- The native postprocessor receives both resized and original image sizes. Its original-pixel coordinates enter the canonical prediction stream directly.

The memory adapter stores completed feature destinations in BF16 before concatenation. The global backbone output, global interpolation, and position encodings retain their native FP32 storage. Validation established exact equality of all 1,500 predictions and the BF16 input received by the next projection, followed by successful runs on four predefined image geometries. The parity receipt covers the resulting native predictions and projection inputs.

## Run

The published configuration records the actual native1536 parity, four-geometry profile, normal-entry verification, and their proof hashes. The published verified-policy configuration supplies the portable admission evidence. Optional `--storage-admission` validates an independently supplied receipt and its referenced parity/profile files when those files are available.

```bash
.venv-dino/bin/python scripts/run_dino.py \
  --source /path/to/dinov3-pinned-source \
  --backbone /path/to/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
  --head /path/to/dinov3_vit7b16_coco_detr_head-b0235ff7.pth \
  --images configs/benchmarks/coco_val2017/images.json \
  --image-root /path/to/coco/val2017 \
  --short-edge 1536 --gpu-limit-gib 22 \
  --allocator-config expandable_segments:True \
  --out /path/to/new-dino-output
```

The backbone's tensor metadata is read automatically through mmap/meta headers. `--metadata` can supply an existing metadata file. `--pilot-count 500` freezes the first 500 input images before inference and seals that prefix while the full worker continues. For a previously declared prefix, use `--pilot-images` and optionally `--pilot-protocol`. `--protocol` binds a separately frozen full-scope declaration. The prefix and scope declarations are frozen before evaluation.

The official `--short-edge 2048` option records its own run setting. The published successful geometry profiles cover 1536. Memory budget and allocator settings are configurable for other hardware. Each run records its requested settings, while source, weights, architecture, preprocessing, and native candidate policy stay frozen.

`--resume` reuses only chunks whose image bytes and frozen protocol match. `--feature-storage native` bypasses the feature-storage adapter while retaining the same strict mixed parameter storage; its memory needs differ from the admitted configuration.

## Outputs and result scope

Each `chunks/<12-digit-image-id>.json` stores the complete native records and their eligible/ignored partition.

- `native_predictions.jsonl` contains all native rows, including unused slots. A complete run has 7,500,000 rows for 5,000 images.
- `predictions.jsonl` contains eligible sparse COCO IDs in the common original-pixel XYXY schema.
- `ignored_native_slots.jsonl` contains the remaining native rows with their original candidate indices.
- `GT_FREE_COMPLETE.json` records image coverage, protocol/input/prediction hashes, candidate counts and chunk checksums.

Evaluate `predictions.jsonl` only after its seal exists, using stock COCO evaluation with all 80 categories and `maxDets=100`. Prefix reports state their 500-image scope. The full validation report covers all 5,000 images and uses its matching paper reference. Counting-guidance experiments add a separately frozen point cache and matched controls.

## Complete-pool counting controls

After completing all required counter queries through the counter commands in the README, the disk-indexed entries retain every eligible candidate. They use the frozen scoring rule and stock COCO helper. Each stock-AP arm runs in its own fresh process, and correct-box retention is accumulated image by image. Choose new output and work directories for each evaluation.

```bash
python scripts/rerank_large_predictions.py \
  --detector runs/dino_coco/detector/predictions.jsonl \
  --counter runs/dino_coco/counter/predictions.jsonl \
  --images configs/benchmarks/coco_val2017/images.json \
  --vocabulary configs/benchmarks/coco_val2017/vocabulary.json \
  --out runs/dino_coco/arms

python scripts/evaluate_large_predictions.py \
  --gt data/coco/instances_val2017.json \
  --manifest configs/benchmarks/coco_val2017/images.json \
  --original runs/dino_coco/arms/original.jsonl \
  --arm confidence=runs/dino_coco/arms/confidence.jsonl \
  --arm points=runs/dino_coco/arms/points.jsonl \
  --control-arm confidence \
  --metadata runs/dino_coco/arms/PREDICTIONS_SEALED_PRE_GT.json \
  --output runs/dino_coco/results.json \
  --work-dir runs/dino_coco/evaluation_work
```

Optional `--cpu-lock PATH` serializes competing CPU jobs. On Linux, `--min-free-memory-gib 24` waits for available system memory before starting. The appropriate admission threshold depends on the complete candidate count and measured process memory.

Strict metadata, source and scope checks have passed. At 1536 pixels, all 1,500 native predictions and the BF16 projection input matched exactly on the predefined reference image, and all four geometry probes completed. The full normal-entry run uses an immutable snapshot of the same numerical materializer functions, storage helper, model, preprocessing and export logic. The portable entry only replaces the private-receipt requirement with the published verified-policy check and records any resource override. The full normal-entry run has started. Its first image matched all 1,500 predictions from the unmodified-window, same-mixed-precision baseline byte-for-byte, with SHA-256 `394e89709ea23e4491ee9fc132f6813eedadcc23e3a58c26bf333a2282d97e65`. The complete benchmark result is pending.
