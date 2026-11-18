# Experiment evidence

## Current full detector reproductions

The following raw detector runs completed on 17–18 November 2026. COCO uses all 5,000 validation images and 80 original categories. Objects365 uses the same frozen first 5,000 original-validation images for both native heads. Its evaluation requests all 365 categories, with 362 non-crowd-positive categories in this panel.

| Dataset | Detector | AP | AP50 | AP75 |
| --- | --- | ---: | ---: | ---: |
| COCO val2017 | DINOv3 + Plain-DETR, 1536 BF16 runtime | 65.37532620 | 82.04021351 | 72.14318565 |
| COCO val2017 | Group DETR v2 released Paddle | 63.29988325 | 81.14385784 | 69.79891 |
| COCO val2017 | TinyFormer-XL-PBM | 62.48417511 | 79.65635792 | 68.43765864 |
| COCO val2017 | GTR-X | 59.37986094 | 77.32193595 | 64.67229937 |
| COCO val2017 | D-FINE-X | 59.30895121 | 76.82009333 | 64.62456459 |
| Objects365 fixed panel | GTR-X native Objects365 | 52.65319320 | 66.11301300 | 57.55272201 |
| Objects365 fixed panel | D-FINE-X native Objects365 | 46.57414485 | 58.53046672 | 50.82714350 |

Group, TinyFormer, GTR and D-FINE each export 300 native candidates per image. Their 1,500,000 candidates remain in each evaluation stream. DINO exports all 1,500 native rows per image. Its 7,500,000 native rows partition into 7,498,300 eligible COCO candidates and 1,700 unused-slot rows. All 6,039 finite nonpositive-extent eligible DINO boxes remain in stock evaluation. GTR COCO contains one finite negative-height box, and fresh D-FINE COCO contains eight finite bad-extent boxes. The scoring arms preserve these candidates consistently. Group DETR, TinyFormer and the fresh native Objects365 exports have valid extents throughout.

| Native prediction stream | SHA-256 |
| --- | --- |
| Group DETR v2 COCO | `e4e34025f016785235b20ddf94f593a502549f859b955c17988d20e498d556ad` |
| TinyFormer COCO | `d2885873fb1d4c87538d122ff612ae6804ae31655ef32d467f8886856c8dcdde` |
| Fresh D-FINE COCO | `4f371f67ca2713179e1f20bf998220d563b41055a95e040e7713c4549f27af35` |
| Native GTR Objects365 | `80947f20409acbfcf38dec34cc624b4b4d40d0adc835f612e6ff6d705a08d3ae` |
| Fresh native D-FINE Objects365 | `c38af5c4c45f2bf61fe89bd15108cdbbe511821c304fa0dc0c55e55d78ee1fcf` |

The released Group DETR checkpoint's matching single-scale COCO validation reference is 63.3. Its full result, strict 767-tensor load and native high-resolution CUDA setting are documented in [Group setup](group_detr_v2.md). TinyFormer uses the full released EMA checkpoint, native FP32 preprocessing and its pinned source revision. Each completed COCO score agrees with its matching released score after rounding.

## Full DINOv3 + Plain-DETR reproduction

Completed on 18 November 2026 using all 5,000 COCO val2017 images, all 80 categories and stock maxDets 100. AP is **65.37532620011595**, AP50 **82.04021350665717** and AP75 **72.14318564718812**. The matching 1536-pixel paper reference is 65.4. The actual runtime uses BF16 parameter storage/autocast and PIL bilinear initial resizing. The paper leaves inference precision and initial interpolation unspecified. [Source and complete setup](dinov3.md)

The predeclared 1536 setting followed a physical-memory failure on the widest 2048 input before DINO AP was read. The complete raw evaluation retained every eligible native candidate and measured 17,845,972,992 bytes peak process RSS. The full result supersedes the separately declared pilot for the main report.

| Full-scope artifact | SHA-256 |
| --- | --- |
| Raw AP report | `d02839a60b3f0b1403be0048386bab8f5cb9d6044c8e3871ceb66dd9e678557f` |
| GT-free producer seal | `18ff18e13a8a3f207255941c9255289e7eb14bfd44a99afb58099854a32795b1` |
| All native predictions | `7dd6f09c7d5e62f0a9bae72d386a592c20bde36dcd10e946d14eeb77d37f0b00` |
| Eligible COCO predictions | `d955947973108581cea752102b4fc02a14906b5bf9df5f6d6029b6f8327c2da3` |
| Ignored native slots | `faeb27ab2a902ac05560db395eb6fd12370c747ad27abd7914478c5733508644` |
| Complete required counter-query set | `611d35bc9d26eb147b6874a3e999835d12a4d8fe2b4bf4de0c1c7e8bb90b5f34` |

Full DINO counting controls require 257,445 image/category queries. At preflight, sealed caches supplied 181,494 and 75,951 remained. The completed TinyFormer new-query journal contributes another 18,841 reusable queries. The actual full DINO preflight verifies 200,335 cached predictions and 57,110 new queries across 4,916 images. Acquisition is running before scoring the counting arms.

## Current matched counting controls

Completed on 18 November 2026. Each complete detector-specific counter cohort is sealed before generating the three scoring arms and reading evaluation annotations. GTR and Group share the complete 181,891-query union. Fresh D-FINE uses 158,641 required queries, while TinyFormer uses its full 216,339-query cohort. Confidence and points support identical budgets on identical candidates.

| Detector | Original AP | Confidence AP | Point AP | Point minus confidence | Point minus original |
| --- | ---: | ---: | ---: | ---: | ---: |
| GTR-X | 59.37986094198712 | 59.41395727747628 | 59.644652270173424 | +0.23069499 | +0.26479133 |
| Group DETR v2 | 63.29988324954671 | 63.18468808422115 | 63.242682150569394 | +0.05799407 | −0.05720110 |
| TinyFormer-XL-PBM | 62.484175107589415 | 62.50493864708171 | 62.56842832746897 | +0.06348968 | +0.08425322 |
| Fresh D-FINE-X | 59.30895120949836 | 59.450555951244965 | 59.67300828452975 | +0.22245233 | +0.36405708 |

| Point arm | Correct candidate IDs displaced | Original matched targets lost | New targets recovered | Final top-five matched targets |
| --- | ---: | ---: | ---: | ---: |
| GTR-X | 276 | 239 | 369 | 23,761 |
| Group DETR v2 | 305 | 275 | 294 | 24,349 |
| TinyFormer-XL-PBM | 252 | 222 | 277 | 24,062 |
| Fresh D-FINE-X | 251 | 219 | 350 | 23,711 |

The original top-five matching denominators are 23,631 for GTR, 24,330 for Group, 23,580 for fresh D-FINE and 24,007 for TinyFormer. Matched confidence preserves those top-five identities. GTR improves over original AP and confidence. Group's original AP remains higher than its point arm, while the point arm exceeds confidence. The positive net Group target recovery is measured at IoU 0.50 in the per-image, per-class top five. Full AP integrates IoU thresholds 0.50–0.95 with maxDets 100.

| Completed artifact | SHA-256 |
| --- | --- |
| Independent counter predictions | `778beaf7edd4161ebcc53745704a4890aa2abf6d63f68ddcf1b6e694d35025ed` |
| Counter completion seal | `671bc683719208334faeca6b7fa517a68b847115eb518c041059df894d081e5b` |
| GTR scoring-arm seal | `66e8292da17170ebb8acd7f32209ebb0f55ed18b7ef66272e62b92efde253edb` |
| GTR paired AP and retention | `d62df199134ed1e0dd92e865eb86b1def303ef698e7311e44bfcbb68cdf9a39c` |
| Group scoring-arm seal | `4fb3f8c51cade3c56852a3a5ef11c66db8dcea3f500b08132c93cf4f40e3edd1` |
| Group paired AP and retention | `cfa14bb0181d8bb0db666d48d4c062041fddaa5df54149251b4aebf17b55b99f` |
| Completed paired suite | `5f1fd451d6fc51e5c29ada4bc997e5663a25627a474d8def740fb19d9a6554fb` |

All 45,789 new counter queries were acquired and durably saved. A CPU-only recovery reused the complete query cache with zero new predictions and sealed the final stream after an interrupted slow file merge. The preserved source journal has SHA-256 `ed01938c3af7fe7483f3b71c852e94696f924008d38635e08acd8e752cba5c24`. Recovery changed the output-writing stage while preserving acquired predictions.

## Fresh TinyFormer counting controls

All 5,000 images, 80 categories and 1,500,000 candidates per scoring arm completed. The counter sealed 216,339 required queries, including 29,423 new predictions. Point guidance gives AP 62.56842832746897, exceeding original AP 62.484175107589415 and matched-confidence AP 62.50493864708171. It displaced 252 originally correct candidate IDs, lost 222 original target matches and recovered 277. The final top-five matched-target count is 24,062 from an original 24,007.

| TinyFormer artifact | SHA-256 |
| --- | --- |
| Paired AP and retention report | `ec4af775af3db9d790affd2c2def3f736d0978cae86090677a5fff606ef4c40b` |
| Complete counter predictions | `62c86b602b199250c72fed995092de8306c61c68c497b86ad1f288dc78414b3e` |
| Newly acquired predictions | `aefe34319cd7ea88da8cb36477547a7cef3f443e8addc97305fe3ae8401fd476` |
| Counter completion seal | `344c4e297ca39fe584c3a44a9d48880ef59f81449cbed1ae69dc6225e7c48829` |
| Scoring-arm seal | `e6168f83b07fbb97b5d64975892b0710d5872508568e4e83170f19765b1b2080` |

The same frozen scoring and evaluation functions preserve candidate geometry and match support budgets. The completed new-query journal is independently checked before its applicable predictions enter the full DINO counter cache.

## Fresh native D-FINE Objects365 counting controls

The complete original 365-category protocol uses the frozen 5,000-image panel, with non-crowd targets in 362 categories. All 244,302 required image/category counter queries were completed, including 481 additional queries. Native predictions and all score arms retain the same 1,500,000 candidates.

| Original AP | Confidence AP | Point AP | Point minus confidence | Point minus original |
| ---: | ---: | ---: | ---: | ---: |
| 46.57414484785549 | 46.80211338148044 | 47.40967143382833 | +0.60755805 | +0.83552659 |

The point arm displaced 338 originally correct top-five candidate IDs, lost 261 originally matched targets and recovered 570. The original matched-target count was 43,609, and the final count was 43,918. The complete paired report SHA-256 is `772bf0f52172ca3abc318d059386ee55471c05f1e4cf83b06453be92b75c0886`. This is the fresh released-weight run, with its own predictions and controls.

The fresh D-FINE COCO counter completed 187 new queries after reusing the sealed GTR/Group union and historical raw caches. Complete coverage is 158,641 required queries. Its counter prediction SHA-256 is `05faf9095b3f753a4fb819916ee38b9d024db510caa9051c11c260d64d34f4b6`, the scoring-arm seal is `48290f6613aad18b8296b92172a957bb99442b7f159ba8c9638fcd14ea86077d`, and the paired report is `9e5d73ae3fc86832118f7c189c67f8ce3b451ab073fe80c7cb92f41022a2c47a`.

## Scope of counting comparisons

The current general-benchmark image panels have already been examined in earlier experiments. Input seals record zero image-ID and image-SHA overlap relative to a separate development panel. Training-image membership for the released Count Anything/CLOC checkpoint remains unverified because a weight-linked training image manifest is unavailable in the audited source. The counter runtime fingerprint establishes source and checkpoint identity. Fixed-cohort counting comparisons are reported with this provenance boundary.

Raw completion records for TinyFormer and native GTR Objects365 have SHA-256 `8b40b9da9da3f795ebe9d551bc0e1707ace41226bc0f5bb5dcf67e43fbc61d61` and `490bd52378cef258409bcb68ec5e0ae5fdfff9e3d4d0f02cae61dbd7d6938f09`. Their completions were at 00:40:35 and 00:49:27 UTC on 18 November 2026.

The fresh candidate streams require explicit counter-query completion and new scoring seals. The historical matched results below retain their original run identity.

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

The model produces candidates in 154,139 image/category queries. The archived cache initially lacked 35,126 of these queries. The frozen combined GTR/Group acquisition and CPU reseal completed the required coverage. The full paired result appears above.

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

A complete native GeCo2 run used all 1,190 FSCD147 test images, released FSCD weights and SAM2, and the task's three visual exemplars per image. Its native evaluation uses maxDets 10,000. Reproduced AP is 46.87, AP50 75.89, count MAE 7.64 and RMSE 39.37. The paper reports 46.81, 75.89, 7.64 and 39.39, respectively. This benchmark has a distinct few-shot visual-exemplar protocol. [GeCo2, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37780)

## Pending and deferred work

- GTR, Group, TinyFormer and fresh D-FINE COCO matched controls and native D-FINE Objects365 controls are complete. Full DINO counter acquisition is running. Native GTR Objects365 follows its separately frozen cohort.
- Full DINOv3 raw COCO reproduction is complete. Full counting acquisition is running with verified TinyFormer cache reuse, preserving the full 257,445-query scope and native candidate pool. Native GTR Objects365 acquisition follows DINO.
- RF-DETR-2XL execution remains deferred at the user's request pending the applicable license.
- Group DETR's released COCO checkpoint is fully reproduced. Availability of its exact native Objects365 head remains unresolved.
- BMD-45 detector training stopped safely after 35 completed epochs at a user priority change. The resumable state remains archived. Counter adaptation and fresh-holdout evaluation remain pending.
- ECDet-L has strict COCO and native Objects365 CPU checks. Its partial full-list export stopped at the priority change. Full AP remains pending.
