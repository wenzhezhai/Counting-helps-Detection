"""Opt-in disk-indexed execution of the unchanged fixed-point reranker."""

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from . import rerank as reference


REFERENCE_SHA256 = "9fe88f7a24f6d49a6936f133280c4da878edbd95436260de4e3db92cdc5bc12b"


def require_reference():
    if reference.file_sha256(Path(reference.__file__)) != REFERENCE_SHA256:
        raise ValueError("The frozen reference reranking implementation differs.")


class CandidateIndex:
    """Keep arbitrary input ordering on disk, with globally unique image/index IDs."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("""CREATE TABLE candidates (
            arm TEXT NOT NULL, image_id TEXT NOT NULL, candidate_index TEXT NOT NULL,
            category_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(arm,image_id,candidate_index)) WITHOUT ROWID""")
        self.connection.execute("CREATE INDEX image_sequence ON candidates(arm,image_id,sequence)")
        self.connection.execute("""CREATE TABLE queries (
            image_id TEXT NOT NULL, category_id TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(image_id,category_id)) WITHOUT ROWID""")

    def add(self, arm, row, sequence):
        try:
            self.connection.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?)", (
                arm, str(row["image_id"]), str(row["candidate_index"]), str(row["category_id"]),
                sequence, reference._compact(row)))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Duplicate candidate {(row['image_id'], row['candidate_index'])}") from error
        if sequence % 10000 == 0:
            self.connection.commit()

    def rows(self, arm, image_id, *, index_order=False):
        cursor = self.connection.execute(
            "SELECT payload FROM candidates WHERE arm=? AND image_id=? ORDER BY sequence",
            (arm, str(image_id)))
        rows = [json.loads(row[0]) for row in cursor]
        return sorted(rows, key=lambda row: row["candidate_index"]) if index_order else rows

    def count(self, arm):
        return self.connection.execute("SELECT COUNT(*) FROM candidates WHERE arm=?", (arm,)).fetchone()[0]

    def close(self):
        self.connection.close()


def index_detector(store, path, images, vocabulary, arm="original"):
    """Use the frozen reader's validation and retain every field and within-image order."""
    sequence = 0
    for row in reference._jsonl(path):
        image_id = row["image_id"]
        if not reference._integer(image_id) or image_id not in images:
            raise ValueError(f"Unknown detector image ID {image_id}")
        reference._identity_metadata(row, images[image_id])
        if "bbox_xyxy" in row:
            candidates = [row]
        else:
            boxes, scores, categories = row["boxes"], row["scores"], row["category_ids"]
            indices = row.get("candidate_indices", list(range(len(boxes))))
            if len({len(boxes), len(scores), len(categories), len(indices)}) != 1:
                raise ValueError("Legacy detector arrays have different lengths")
            candidates = []
            for index, category, box, score in zip(indices, categories, boxes, scores):
                if len(box) != 4 or not all(reference._number(x) for x in box):
                    raise ValueError("Invalid legacy detector box")
                x, y, width, height = box
                candidates.append(dict(image_id=image_id, category_id=category,
                    candidate_index=index, bbox_xyxy=[x, y, x + width, y + height], score=score))
        for candidate in candidates:
            index, category = candidate["candidate_index"], candidate["category_id"]
            box, score = candidate["bbox_xyxy"], candidate["score"]
            if not reference._integer(index) or not reference._integer(category) or category not in vocabulary:
                raise ValueError("Candidate index/category differs from the fixed vocabulary")
            if len(box) != 4 or not all(reference._number(x) for x in box):
                raise ValueError("Box coordinates must be finite xyxy values")
            if not reference._number(score) or not 0 <= score <= 1:
                raise ValueError("Original scores must be finite values in [0,1]")
            store.add(arm, candidate, sequence)
            sequence += 1
    store.connection.commit()
    return sequence


def index_counter(store, path, images, vocabulary):
    count_queries = 0
    for row in reference._jsonl(path):
        image_id, category = row["image_id"], row["category_id"]
        key = image_id, category
        if (not reference._integer(image_id) or not reference._integer(category)
                or image_id not in images or category not in vocabulary):
            raise ValueError(f"Unknown or duplicated counter query {key}")
        image = images[image_id]
        reference._identity_metadata(row, image)
        if "text" in row and (not isinstance(row["text"], str)
                or row["text"].strip().casefold() != vocabulary[category].strip().casefold()):
            raise ValueError(f"Counter prompt differs from vocabulary at {key}")
        prediction = row.get("outputs", {}).get("on", row)
        points = prediction.get("points", prediction.get("points_original_pixels"))
        if not isinstance(points, list):
            raise ValueError(f"Missing counter point list for {key}")
        count = prediction.get("count", prediction.get("pred_count", len(points)))
        if not reference._integer(count) or count < 0 or count != len(points):
            raise ValueError(f"Counter count and point list differ at {key}")
        for point in points:
            if len(point) != 2 or not all(reference._number(x) for x in point):
                raise ValueError(f"Counter points must be finite pixel xy pairs at {key}")
            if not 0 <= point[0] <= image["width"] or not 0 <= point[1] <= image["height"]:
                raise ValueError(f"Counter point lies outside the original image at {key}")
        normalized = dict(image_id=image_id, category_id=category, points=points, count=count)
        try:
            store.connection.execute("INSERT INTO queries VALUES (?,?,?)", (
                str(image_id), str(category), reference._compact(normalized)))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Unknown or duplicated counter query {key}") from error
        count_queries += 1
        if count_queries % 10000 == 0:
            store.connection.commit()
    store.connection.commit()
    missing = store.connection.execute("""SELECT DISTINCT c.image_id,c.category_id
        FROM candidates c LEFT JOIN queries q
        ON c.image_id=q.image_id AND c.category_id=q.category_id
        WHERE c.arm='original' AND q.image_id IS NULL""").fetchall()
    if missing:
        raise reference.MissingCounterQueriesError({(int(image), int(category)) for image, category in missing})
    return count_queries


def run_reranking_large(detector_path, counter_path, image_path, vocabulary_path, output_dir):
    """Run identical per-image mathematics while keeping complete input pools on disk."""
    require_reference()
    paths = {"detector": Path(detector_path), "counter": Path(counter_path),
             "images": Path(image_path), "vocabulary": Path(vocabulary_path)}
    before = {name: reference.file_sha256(path) for name, path in paths.items()}
    images = reference.load_images(image_path)
    vocabulary = reference.load_vocabulary(vocabulary_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    store = CandidateIndex(out / "candidate_query_index.sqlite3")
    try:
        index_detector(store, detector_path, images, vocabulary)
        query_count = index_counter(store, counter_path, images, vocabulary)
        counts = Counter({name: 0 for name in reference.ARMS})
        geometry = {name: sha256() for name in reference.ARMS}
        filenames = {name: name + ".jsonl" for name in reference.ARMS}
        with ExitStack() as stack:
            streams = {name: stack.enter_context((out / filename).open("x", encoding="utf-8"))
                       for name, filename in filenames.items()}
            proof = stack.enter_context((out / "support_budget.jsonl").open("x", encoding="utf-8"))
            for image_id in images:
                rows = store.rows("original", image_id)
                queries = {int(category): json.loads(payload)["points"] for category, payload in
                           store.connection.execute("SELECT category_id,payload FROM queries WHERE image_id=?", (str(image_id),))}
                values, evidence = reference.rerank_image(rows, queries)
                for name in reference.ARMS:
                    for row in sorted(values[name], key=lambda value: value["candidate_index"]):
                        streams[name].write(reference._compact(row) + "\n")
                        geometry[name].update((reference._compact([
                            row["image_id"], row["category_id"], row["candidate_index"], row["bbox_xyxy"]]) + "\n").encode())
                        counts[name] += 1
                for row in evidence:
                    proof.write(reference._compact(row) + "\n")
        if any(reference.file_sha256(path) != before[name] for name, path in paths.items()):
            raise RuntimeError("An input changed while preparing the score arms")
        geometry = {name: digest.hexdigest() for name, digest in geometry.items()}
        if len(set(geometry.values())) != 1 or len(set(counts.values())) != 1:
            raise RuntimeError("Candidate counts or geometry differ between arms")
        seal = dict(schema_version=reference.SCHEMA_VERSION, status="PREDICTIONS_SEALED_BEFORE_GT",
            created_utc=datetime.now(timezone.utc).isoformat(), ground_truth_read=False,
            images=len(images), categories=len(vocabulary), candidate_count=counts["original"],
            inputs={name: dict(path=str(path.resolve()), sha256=before[name]) for name, path in paths.items()},
            geometry_sha256=geometry["original"],
            arms={name: dict(file=filenames[name], sha256=reference.file_sha256(out / filenames[name]))
                  for name in reference.ARMS},
            support_budget=dict(file="support_budget.jsonl", sha256=reference.file_sha256(out / "support_budget.jsonl")),
            matching=dict(method="Hungarian", margin=1.15, dummy_cost=2.0, uses_original_score=True),
            score_adjustment=dict(point_weight=.25, identical_transform_for_confidence_control=True),
            implementation_sha256=reference.file_sha256(Path(__file__)),
            reference_implementation_sha256=REFERENCE_SHA256,
            execution=dict(mode="disk_indexed_per_image", independent_counter_queries=query_count,
                           preserves_original_input_order_for_matching=True, candidate_pool_trimmed=False))
        with (out / "PREDICTIONS_SEALED_PRE_GT.json").open("x", encoding="utf-8") as stream:
            json.dump(seal, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return seal
    finally:
        store.close()
