# Experiment evidence

## Full GTR-X COCO reproduction

Completed 17 November 2026 at 22:42:20 UTC. Source commit `5c40b2c705807ab34387624de242ca22e533e32a`. Checkpoint SHA-256 `547cc2c2d29b32315fa9d533431e35ed0b17807e82f0803779a91f268ff878fb`.

| Metric | Local reproduction |
| --- | ---: |
| Images | 5,000 |
| Native categories | 80 |
| Exported candidates per image | 300 |
| AP | 59.37986094 |
| AP50 | 77.32193595 |
| AP75 | 64.67229937 |

The released model reports COCO AP 59.4. The local result agrees after rounding. All per-image chunks, the flat native output and the original ground-truth checksum passed independent checks. One native finite negative-height box remained in the standard evaluation stream, consistent with the official coordinate conversion.

| Artifact | SHA-256 |
| --- | --- |
| Native prediction stream | `88c80bbec96d39f6a2aeb597c2e6af2b235f82bea90ccf2a682ab05955cd060d` |
| Canonical prediction derivative | `5e00535f2880600871a661de765743c2c6f1dcba97fb1c6dc45949cead0747c4` |
| Raw AP completion record | `e24f4ab499e7b08f23cae614770cc545c2bdf209b4e63289cde854fc02950956` |

The new model produces candidates in 154,139 image/category queries. The archived independent counter cache lacks 35,126 of these queries. Counting-guided comparison awaits explicit query completion.

## Historical matched D-FINE-X controls

The following evaluation completed on 9 November 2026. Result and completion hashes were rechecked during the current follow-up.

| Dataset | Images | Original AP | Confidence AP | Points AP | Points minus confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| COCO val2017 | 5,000 | 59.3070459351 | 59.4492599711 | 59.6736813328 | +0.2244213617 |
| Original Objects365 fixed validation panel | 5,000 | 46.5709481305 | 46.8030096616 | 47.4085062123 | +0.6054965506 |

Counting points displaced 249 originally correct top-five candidates on COCO and 339 on Objects365. A separate rule selected on two development panels reduced the displacement counts to 194 and 321, respectively. Its AP was 59.6082 and 47.3300. Preserve both AP and displacement in interpreting protection.

## Previous BMD-45 panel

This is the prior 1,084-image car/bus/truck panel, with original COCO detector weights and fixed boxes. It has already been examined.

| Arm | AP | Correct top-five candidates displaced | Previously matched targets lost | Targets recovered |
| --- | ---: | ---: | ---: | ---: |
| Original | 52.8852 | 0 | 0 | 0 |
| Matched confidence | 53.3647 | 0 | 0 | 0 |
| Counting points | 53.9103 | 19 | 16 | 38 |

A paired image-level bootstrap of 500 draws estimates a point-over-confidence gain of +0.55 AP with percentile interval [0.35, 0.82]. Density-stratified analysis is retrospective. Public camera IDs are unavailable.

## Independent joint-method reference

A complete native GeCo2 run used all 1,190 FSCD147 test images, released FSCD weights and SAM2, and the task's three visual exemplars per image. Reproduced AP is 46.87, AP50 75.89, count MAE 7.64 and RMSE 39.37. The paper reports 46.81, 75.89, 7.64 and 39.39, respectively. This benchmark has a distinct few-shot visual-exemplar protocol. [GeCo2, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37780)

## Pending and deferred work

- Full TinyFormer-XL-PBM COCO inference follows checkpoint transfer and strict-load verification.
- Group DETR v2 has strict CPU and native GPU checks, including the largest transformed image. Full COCO inference is active and AP awaits completion.
- DINOv3 head is verified on the server. The complete local backbone is transferring. Architecture and memory profiling precede full COCO inference.
- RF-DETR-2XL execution remains deferred at the user's request pending the applicable license.
- BMD-45 detector training stopped safely after 35 completed epochs at a user priority change. The resumable state remains archived. Counter adaptation and fresh-holdout evaluation remain pending.
- ECDet-L has strict COCO and native Objects365 CPU checks. Its partial full-list export stopped at the priority change. Full AP remains pending.
