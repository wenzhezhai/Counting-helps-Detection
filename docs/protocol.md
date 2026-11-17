# Fixed-candidate evaluation protocol

## Data and model scope

COCO uses all 5,000 val2017 images and its 80 original sparse category IDs. Objects365 uses a fixed ordered panel of 5,000 original validation images and all 365 native categories. A checkpoint with a COCO classification head requires its own declared category mapping for a cross-dataset transfer test. Native 365-class AP uses the released 365-class checkpoint.

BMD-45 historical traffic analysis uses cars, buses and trucks on a previously frozen 1,084-image panel. A separate fresh adaptation split remains sealed for future scoring after training and development-only selection. Public BMD-45 files supply no reliable per-image camera identifiers, so that holdout is image-level.

## Files

Image metadata, independent of annotations, is a JSON list.

```json
[{"image_id": 1, "file_name": "000000000001.jpg", "width": 640, "height": 480}]
```

The vocabulary preserves the dataset's category IDs.

```json
[{"category_id": 3, "text": "car"}]
```

Detector predictions are flat JSONL, one record per candidate. Candidate indices are unique within an image. Coordinates use original-image pixels.

```json
{"image_id":1,"category_id":3,"candidate_index":0,"bbox_xyxy":[12,20,88,95],"score":0.72}
```

Counter predictions are flat JSONL, one record per image/category query, including zero-count queries.

```json
{"image_id":1,"category_id":3,"width":640,"height":480,"points":[[50,55]],"count":1}
```

Counter query manifests may be a JSON list of records or a JSON list of `[image_id, category_id]` pairs. The counter adapter also accepts the coverage audit's `missing_image_class_ids` list. Legacy independent cache rows with `outputs.on.points_original_pixels` are supported.

## Frozen matching and scores

For each image and class, match independent points one-to-one with positive-area candidate boxes. The Hungarian cost is normalized point-to-box-center distance minus the original detector score. A point is eligible within 1.15 times the box's width and height, centered on the box. Dummy assignments cost 2.0.

Let K be the number of supported candidates. The confidence control supports the K candidates with highest original confidence, using candidate index to break score ties. Apply the same transformation to both arms.

```text
supported score   = 0.75*s + 0.25*(2*s - s*s)
unsupported score = 0.75*s + 0.25*s*s
```

Original coordinates, categories, candidate identities and candidate counts remain identical across arms. Finite native boxes with nonpositive extents remain in the prediction stream and standard AP. Such boxes have zero overlap and receive zero point support in the diagnostics. Missing counter queries must be explicitly inferred before the complete matched comparison.

Seal detector and counter predictions and then all score arms before opening evaluation annotations. Preserve failed attempts and record recoveries as separate revisions. Parameter choice uses source/development data. These historical benchmarks have already been inspected, and subsequent checks on them are reproduction and transfer analyses. Fresh-heldout claims require the separate frozen evaluation panel.

## AP

Use stock pycocotools bbox AP with IoU thresholds 0.50 to 0.95 in steps of 0.05, all native evaluation categories and `maxDets=[1,10,100]`. Report AP on a 0–100 scale. Candidate export may retain 300 native boxes while the evaluator applies its standard maximum per image and category.

## Top-five retention

Within every image and class, rank candidates by descending score and then candidate index. Greedily match the first five candidates one-to-one to labeled boxes at IoU ≥ 0.50. On equal IoU, prefer the lowest annotation ID. Exclude queries containing crowd or ignored annotations from this diagnostic and preserve standard COCO crowd handling in AP.

Report separately

- Originally correct top-five candidate IDs that leave the top five.
- Previously matched targets that become unmatched.
- Previously unmatched targets that become matched.

A displaced box can have another candidate still matching its target, so candidate displacement and target loss differ. The evaluator checks identical candidate identities, categories and geometry across every arm before scoring.
