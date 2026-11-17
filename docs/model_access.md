# Model sources and execution status

Status snapshot is 17 November 2026. Full scores and run evidence are in [results](results.md). Source and checkpoint hashes live in `configs/models.json`. Keep each upstream model in its own compatible environment.

| Model | Public source | Checkpoint and current status |
| --- | --- | --- |
| GTR-X | [Official repository](https://github.com/Intellindust-AI-Lab/GTR) | Released COCO checkpoint, complete local reproduction |
| D-FINE-X | [Official releases](https://github.com/Peterande/D-FINE/releases/tag/v1.0) | Separate `dfine_x_obj2coco.pth` and `dfine_x_obj365.pth` preserve their native heads |
| TinyFormer-XL-PBM | [Author model zoo](https://github.com/mmpmmpmmpjosh/TinyFormer) | `TinyFormer-XL-pbm_obj2coco_24e.pth` is the Objects365-pretrained COCO detector |
| DINOv3 + Plain-DETR | [Official access instructions](https://github.com/facebookresearch/dinov3#pretrained-heads---detector-trained-on-coco2017-dataset) | Approval supplies both the ViT-7B backbone and COCO detection head, complete local files awaiting server integration |
| Group DETR v2 | [Released Paddle model zoo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/group_detr) | Released COCO checkpoint verified, native Objects365-head availability remains unresolved |
| RF-DETR-2XL | [Official model repository](https://github.com/roboflow/rf-detr) | Execution deferred by the user pending the platform license |

## DINOv3 files

The approved download email supplies these files.

```text
dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
dinov3_vit7b16_coco_detr_head-b0235ff7.pth
```

The head in the approval email uses `coco_detr_head`. Current source builds a `coco2017_detr_head` default name. Pass the actual approved files explicitly when integrating the detector. The model-zoo 65.6 AP setting uses shortest edge 2048, single scale. A 1536-pixel setting has its own reported score. Check full file hashes and profile actual model memory before full inference.

## Verified runtimes

- GTR inference uses PyTorch 2.7.1+cu126 and flash-linear-attention / fla-core 0.5.0 in an isolated dependency layer. The full raw run uses batch one, FP32, official 640-pixel preprocessing and the released EMA weights.
- D-FINE uses its pinned source and the matching native configuration. COCO and Objects365 require separate class heads and original category mapping.
- Group DETR v2 CPU verification uses Python 3.12, PaddlePaddle 3.3.1, NumPy 1.26.4 and OpenCV 4.11.0.86. See [Group setup](group_detr_v2.md) for exact loading and eager compatibility.
- Count Anything uses the pinned native model and independent image/text path in `configs/counter.json`. Grounding inference uses the native BF16 autocast path while image/text inputs retain the frozen FP32 preprocessing.

For upstream source trees, check out the exact commit in the model registry. The adapters verify the source revision and strict checkpoint loading. Frozen source archives can be accompanied by the wrapper's source receipt. Install upstream dependencies according to that pinned revision.

Signed approval URLs are personal, temporary download credentials. Keep these in the approved download workflow. Model artifacts and dataset files belong in the local `pretrained/` and `data/` directories.

## Frozen general-benchmark inputs

- `configs/benchmarks/coco_val2017/images.json` contains all 5,000 COCO val2017 images in the frozen order. `vocabulary.json` contains the 80 native category IDs and text prompts.
- `configs/benchmarks/objects365_fixed5000/images.json` contains the fixed 5,000-image original-validation panel with paths relative to the image root. Its vocabulary preserves all 365 categories.
- Each `benchmark.json` records the original ground-truth, vocabulary and manifest checksums, plus the public sanitized image-manifest checksum. Download original images and annotations from their publishers and check the recorded hashes.

Native Objects365 inference uses its dedicated model registry entry.

```bash
python scripts/run_detector.py \
  --model dfine-x-objects365 \
  --source-dir ../detector-sources/D-FINE \
  --checkpoint pretrained/dfine_x_obj365.pth \
  --images configs/benchmarks/objects365_fixed5000/images.json \
  --image-root data/objects365/images \
  --device cuda:0 --threads 1 \
  --output runs/dfine_objects365/detector
```

The native classifier has 366 slots. Slot zero is unused and logged separately. The adapter preserves native category IDs 1–365.

## GTR source and tested dependency installation

The following commands describe the isolated PyTorch environment used in our reproduction. Group DETR uses the separate Paddle environment documented on its setup page.

```bash
git clone https://github.com/Intellindust-AI-Lab/GTR.git ../detector-sources/GTR
git -C ../detector-sources/GTR checkout 5c40b2c705807ab34387624de242ca22e533e32a
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r configs/environments/torch_inference.txt
```

The adapter sets `GTR_DISABLE_CUDA_GLA=1` and uses the verified FLA 0.5.0 path. The upstream reference environment uses a newer PyTorch/CUDA combination. The actual environment above produced the full AP in the results table. Keep the chosen environment with each run.

## Tested counter dependencies

The independent counter checks passed with the same Python 3.12.14 and PyTorch 2.7.1+cu126 runtime. Additional verified packages include OmegaConf 2.3.1, hydra-core 1.3.7, ftfy 6.1.1, regex 2026.9.10 and iopath 0.1.10. Use the pinned Count Anything source and its native SAM3 requirements in the counter environment. The lightweight evaluation package installs the scoring components.
