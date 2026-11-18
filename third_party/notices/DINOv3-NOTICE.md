# DINOv3-derived detection adapter notices

Parts of `src/counting_help_detection/dino_storage.py` derive from the native window, backbone/position joiner, and positional-encoding operations in Meta's DINOv3 repository. The adapter changes feature destination storage and preserves the source attribution below.

Pinned upstream revision: `6876159a11b4df116f30f667f8c9888617df0751`.

The complete, unmodified DINOv3 agreement from this revision is included in [DINOv3-LICENSE.md](DINOv3-LICENSE.md). [Original license](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md).

## Native window operations

Source: [dinov3/eval/detection/models/windows.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/detection/models/windows.py#L1-L4).

Original header:

```text
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
```

## Backbone and positional-encoding lineage

Sources:

- [dinov3/eval/detection/models/backbone.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/detection/models/backbone.py#L1-L17).
- [dinov3/eval/detection/models/position_encoding.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/detection/models/position_encoding.py#L1-L17).

Original headers retained from both files:

```text
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
```

The project names, copyright holders, years, and license references above are reproduced from the pinned source headers.
