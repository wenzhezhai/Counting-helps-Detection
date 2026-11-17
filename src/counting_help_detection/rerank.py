"""Frozen point-guided reranking with a matched confidence control.

Detector JSONL uses one row per candidate with image_id, category_id,
candidate_index, bbox_xyxy=[x1,y1,x2,y2] in original-image pixels, and score
in [0,1]. Candidate indices are unique within each image. Legacy per-image
rows with boxes in xywh, category_ids, scores and optional candidate_indices
are also accepted.

Independent counter JSONL uses image_id, category_id, width, height and
points in original-image pixel xy coordinates. count is optional and must agree
with len(points). Legacy outputs.on.points_original_pixels / pred_count
records are accepted. The image manifest is a list of image_id, width and
height records (or an object containing that list as images). Vocabulary
records contain category_id and text. Counter queries are required for
every image/category that contains detector candidates. Missing queries
raise MissingCounterQueriesError.

The positive-box matching contract is adapted from this project's frozen
C2 Hungarian reference, source SHA256
f8d11174263318142cee7c514d89aac0c9f5ff7959e868d70b6a326a76613e6b.
The exact mixture used by the archived fixed-point rule is preserved:
0.75*s + 0.25*(2*s-s*s) for supported boxes, otherwise
0.75*s + 0.25*s*s. SciPy supplies the Hungarian assignment.

Finite boxes with nonpositive extents remain in each arm and receive no
point matches. Box coordinates and candidate identities never change.
The caller must supply independent counter predictions. This module never
opens an annotation file or selects parameters from evaluation results.
"""

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path


ARMS = ("original", "confidence", "points")
SCHEMA_VERSION = 1


def file_sha256(path):
    """Hash a file without loading it in memory."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value):
    return type(value) is int


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at JSONL line {line_number}")
                yield row


def _compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def load_images(path):
    """Read image metadata without opening image files or annotations."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    forbidden = {"annotations", "targets", "ground_truth"}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError("The image manifest contains ground-truth fields")
        value = value["images"]
    if not isinstance(value, list) or not value:
        raise ValueError("An image manifest must contain a nonempty image list")
    images = {}
    for row in value:
        if forbidden.intersection(row):
            raise ValueError("An image record contains ground-truth fields")
        image_id = row["image_id"] if "image_id" in row else row["id"]
        if not _integer(image_id) or image_id in images:
            raise ValueError("Image IDs must be unique integers")
        if not all(_integer(row[k]) and row[k] > 0 for k in ("width", "height")):
            raise ValueError("Image dimensions must be positive integers")
        images[image_id] = dict(row, image_id=image_id)
    return images


def load_vocabulary(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("Vocabulary must be a nonempty list")
    vocabulary = {}
    for row in value:
        category = row["category_id"]
        text = row["text"]
        if not _integer(category) or category in vocabulary:
            raise ValueError("Vocabulary category IDs must be unique integers")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each vocabulary category needs a text prompt")
        vocabulary[category] = text
    return vocabulary


def _identity_metadata(row, image):
    for key in ("width", "height", "image_sha256"):
        if key in row and key in image and row[key] != image[key]:
            raise ValueError(f"Image metadata differs for image {image['image_id']}")


def read_detector_predictions(path, images, vocabulary):
    """Return validated per-image candidate groups, retaining native geometry."""
    groups = defaultdict(list)
    identities = set()
    for row in _jsonl(path):
        image_id = row["image_id"]
        if not _integer(image_id) or image_id not in images:
            raise ValueError(f"Unknown detector image ID {image_id}")
        _identity_metadata(row, images[image_id])
        if "bbox_xyxy" in row:
            candidates = [row]
        else:
            boxes, scores, categories = row["boxes"], row["scores"], row["category_ids"]
            indices = row.get("candidate_indices", list(range(len(boxes))))
            if len({len(boxes), len(scores), len(categories), len(indices)}) != 1:
                raise ValueError("Legacy detector arrays have different lengths")
            candidates = []
            for index, category, box, score in zip(indices, categories, boxes, scores):
                if len(box) != 4 or not all(_number(x) for x in box):
                    raise ValueError("Invalid legacy detector box")
                x, y, width, height = box
                candidates.append(dict(image_id=image_id, category_id=category,
                    candidate_index=index, bbox_xyxy=[x, y, x + width, y + height], score=score))
        for candidate in candidates:
            index = candidate["candidate_index"]
            category = candidate["category_id"]
            box = candidate["bbox_xyxy"]
            score = candidate["score"]
            if not _integer(index) or not _integer(category) or category not in vocabulary:
                raise ValueError("Candidate index/category differs from the fixed vocabulary")
            identity = image_id, index
            if identity in identities:
                raise ValueError(f"Duplicate candidate index {identity}")
            if len(box) != 4 or not all(_number(x) for x in box):
                raise ValueError("Box coordinates must be finite xyxy values")
            if not _number(score) or not 0 <= score <= 1:
                raise ValueError("Original scores must be finite values in [0,1]")
            identities.add(identity)
            groups[image_id].append(dict(candidate))
    return dict(groups)


def read_counter_predictions(path, images, vocabulary):
    """Read independent points. Missing candidate-bearing queries are checked later."""
    result = {}
    for row in _jsonl(path):
        image_id, category = row["image_id"], row["category_id"]
        key = image_id, category
        if (not _integer(image_id) or not _integer(category) or image_id not in images
                or category not in vocabulary or key in result):
            raise ValueError(f"Unknown or duplicated counter query {key}")
        image = images[image_id]
        _identity_metadata(row, image)
        if "text" in row and (not isinstance(row["text"], str)
                or row["text"].strip().casefold() != vocabulary[category].strip().casefold()):
            raise ValueError(f"Counter prompt differs from vocabulary at {key}")
        prediction = row.get("outputs", {}).get("on", row)
        points = prediction.get("points", prediction.get("points_original_pixels"))
        if not isinstance(points, list):
            raise ValueError(f"Missing counter point list for {key}")
        count = prediction.get("count", prediction.get("pred_count", len(points)))
        if not _integer(count) or count < 0 or count != len(points):
            raise ValueError(f"Counter count and point list differ at {key}")
        for point in points:
            if len(point) != 2 or not all(_number(x) for x in point):
                raise ValueError(f"Counter points must be finite pixel xy pairs at {key}")
            if not 0 <= point[0] <= image["width"] or not 0 <= point[1] <= image["height"]:
                raise ValueError(f"Counter point lies outside the original image at {key}")
        result[key] = dict(image_id=image_id, category_id=category, points=points, count=count)
    return result


class MissingCounterQueriesError(ValueError):
    """An explicit coverage failure. queries contains missing image/category pairs."""

    def __init__(self, queries):
        self.queries = sorted(queries)
        super().__init__(f"Independent counter predictions are missing for {len(self.queries)} candidate-bearing queries")


def match_points(boxes_xyxy, scores, points):
    """Return point/box pairs using the frozen score-aware Hungarian contract."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    score = np.asarray(scores, dtype=np.float64)
    locations = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    eligible = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    if len(eligible) == 0 or len(locations) == 0:
        return []
    chosen = boxes[eligible]
    wh = np.maximum(chosen[:, 2:] - chosen[:, :2], 1e-8)
    centre = chosen[:, :2] + wh / 2
    delta = np.abs(locations[:, None, :] - centre[None, :, :]) / wh[None, :, :]
    valid = np.all(delta <= 1.15 / 2, axis=-1)
    cost = np.linalg.norm(delta, axis=-1) - score[eligible][None, :]
    cost[~valid] = 1e6
    cost = np.concatenate((cost, np.full((len(locations), len(locations)), 2.0)), axis=1)
    rows, columns = linear_sum_assignment(cost)
    return [(int(point), int(eligible[box])) for point, box in zip(rows, columns)
            if box < len(eligible) and valid[point, box]]


def adjusted_score(score, supported):
    """Preserve the fixed 0.25 mixture coefficient used in the archived method."""
    return .75 * score + .25 * (2 * score - score * score if supported else score * score)


def ranked_indices(rows):
    return [row["candidate_index"] for row in sorted(rows, key=lambda row: (-row["score"], row["candidate_index"]))]


def geometry_digest(rows):
    """Hash identities, categories and pixel geometry independently of scores."""
    digest = sha256()
    ordered = sorted(rows, key=lambda row: (row["image_id"], row["candidate_index"]))
    for row in ordered:
        digest.update((_compact([row["image_id"], row["category_id"], row["candidate_index"], row["bbox_xyxy"]]) + "\n").encode())
    return digest.hexdigest()


def rerank_image(detections, counter_queries):
    """Score one image. counter_queries maps category ID to independent point lists.

    Return (arms, evidence), where arms maps original/confidence/points to
    flat candidate records and evidence records each category's common K.
    """
    grouped = defaultdict(list)
    for row in detections:
        grouped[row["category_id"]].append(row)
    missing = {(row["image_id"], category) for category, rows in grouped.items()
               for row in rows[:1] if category not in counter_queries}
    if missing:
        raise MissingCounterQueriesError(missing)
    values = {name: [] for name in ARMS}
    evidence = []
    for category, rows in sorted(grouped.items()):
        point_pairs = match_points([row["bbox_xyxy"] for row in rows],
                                   [row["score"] for row in rows], counter_queries[category])
        supported = {rows[index]["candidate_index"] for _, index in point_pairs}
        ordered = ranked_indices(rows)
        confidence = set(ordered[:len(supported)])
        original, control, points = [], [], []
        for row in rows:
            index = row["candidate_index"]
            original.append(dict(row))
            control.append(dict(row, score=adjusted_score(row["score"], index in confidence)))
            points.append(dict(row, score=adjusted_score(row["score"], index in supported)))
        if ranked_indices(original) != ranked_indices(control):
            raise RuntimeError("Matched confidence changed within-class candidate order")
        for arm, group in zip(ARMS, (original, control, points)):
            values[arm].extend(group)
        evidence.append(dict(image_id=rows[0]["image_id"], category_id=category,
            supported_box_count=len(supported), point_supported_candidate_indices=sorted(supported),
            confidence_supported_candidate_indices=sorted(confidence),
            point_box_matches=[dict(point_index=p, candidate_index=rows[b]["candidate_index"]) for p, b in point_pairs]))
    expected = geometry_digest(detections)
    if any(geometry_digest(values[name]) != expected for name in ARMS):
        raise RuntimeError("A score adjustment changed candidate identities or geometry")
    return values, evidence


def run_reranking(detector_path, counter_path, image_path, vocabulary_path, output_dir):
    """Write three fixed-candidate arms and a pre-GT prediction seal to a new folder."""
    input_paths = {"detector": Path(detector_path), "counter": Path(counter_path),
                   "images": Path(image_path), "vocabulary": Path(vocabulary_path)}
    before = {name: file_sha256(path) for name, path in input_paths.items()}
    images = load_images(image_path)
    vocabulary = load_vocabulary(vocabulary_path)
    detections = read_detector_predictions(detector_path, images, vocabulary)
    queries = read_counter_predictions(counter_path, images, vocabulary)
    required = {(image_id, row["category_id"]) for image_id, rows in detections.items() for row in rows}
    missing = required - queries.keys()
    if missing:
        raise MissingCounterQueriesError(missing)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    counts = {name: 0 for name in ARMS}
    geometry = {name: sha256() for name in ARMS}
    filenames = {name: name + ".jsonl" for name in ARMS}
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = {name: stack.enter_context((out / filename).open("x", encoding="utf-8"))
                   for name, filename in filenames.items()}
        proof = stack.enter_context((out / "support_budget.jsonl").open("x", encoding="utf-8"))
        for image_id in images:
            rows = detections.get(image_id, [])
            point_queries = {category: queries[(image_id, category)]["points"]
                             for category in {row["category_id"] for row in rows}}
            values, evidence = rerank_image(rows, point_queries)
            for name in ARMS:
                ordered = sorted(values[name], key=lambda row: row["candidate_index"])
                for row in ordered:
                    streams[name].write(_compact(row) + "\n")
                    geometry[name].update((_compact([row["image_id"], row["category_id"], row["candidate_index"], row["bbox_xyxy"]]) + "\n").encode())
                    counts[name] += 1
            for row in evidence:
                proof.write(_compact(row) + "\n")
    if any(file_sha256(path) != before[name] for name, path in input_paths.items()):
        raise RuntimeError("An input changed while preparing the score arms")
    geometry = {name: digest.hexdigest() for name, digest in geometry.items()}
    if len(set(geometry.values())) != 1 or len(set(counts.values())) != 1:
        raise RuntimeError("Candidate counts or geometry differ between arms")
    seal = dict(schema_version=SCHEMA_VERSION, status="PREDICTIONS_SEALED_BEFORE_GT",
        created_utc=datetime.now(timezone.utc).isoformat(), ground_truth_read=False,
        images=len(images), categories=len(vocabulary), candidate_count=counts["original"],
        inputs={name: dict(path=str(path.resolve()), sha256=before[name]) for name, path in input_paths.items()},
        geometry_sha256=geometry["original"],
        arms={name: dict(file=filenames[name], sha256=file_sha256(out / filenames[name])) for name in ARMS},
        support_budget=dict(file="support_budget.jsonl", sha256=file_sha256(out / "support_budget.jsonl")),
        matching=dict(method="Hungarian", margin=1.15, dummy_cost=2.0, uses_original_score=True),
        score_adjustment=dict(point_weight=.25, identical_transform_for_confidence_control=True),
        implementation_sha256=file_sha256(Path(__file__)))
    with (out / "PREDICTIONS_SEALED_PRE_GT.json").open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return seal
