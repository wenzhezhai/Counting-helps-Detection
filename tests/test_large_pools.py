"""Contract parity for optional disk-indexed execution; artifacts use an explicit workspace."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from uuid import uuid4

from counting_help_detection import evaluation, rerank
from counting_help_detection.evaluation_large import evaluate_files_large
from counting_help_detection.rerank_large import run_reranking_large


class CompletePoolParity(unittest.TestCase):
    def setUp(self):
        parent = os.environ.get("COUNTING_LARGE_TEST_ROOT")
        if not parent:
            self.skipTest("Set COUNTING_LARGE_TEST_ROOT to an explicit artifact workspace.")
        self.root = Path(parent) / uuid4().hex
        self.root.mkdir(parents=True)
        self.images = [dict(image_id=2, width=100, height=100), dict(image_id=1, width=100, height=100)]
        self.vocabulary = [dict(category_id=i, text=name) for i, name in [(1, "car"), (2, "bus"), (3, "other")]]
        self.rows = [
            dict(image_id=2, category_id=1, candidate_index=9, bbox_xyxy=[0, 0, 20, 20], score=.8, extra="保持"),
            dict(image_id=1, category_id=2, candidate_index=6, bbox_xyxy=[10, 10, 30, 30], score=.7),
            dict(image_id=2, category_id=1, candidate_index=2, bbox_xyxy=[0, 0, 20, 20], score=.8),
            dict(image_id=2, category_id=1, candidate_index=4, bbox_xyxy=[-0.0, 0.0, -1.0, 12], score=.1),
            dict(image_id=2, category_id=1, candidate_index=8, bbox_xyxy=[30, 30, 50, 50], score=.4),
            dict(image_id=1, category_id=1, candidate_index=5, bbox_xyxy=[40, 40, 60, 60], score=.2),
            dict(image_id=2, category_id=1, candidate_index=3, bbox_xyxy=[40, 40, 60, 60], score=.2),
            dict(image_id=2, category_id=1, candidate_index=7, bbox_xyxy=[60, 60, 80, 80], score=.3),
            dict(image_id=2, category_id=3, candidate_index=99, bbox_xyxy=[80, 80, 90, 90], score=.3),
        ]
        self.queries = [
            dict(image_id=2, category_id=1, points=[[70, 70], [10, 10]]),
            dict(image_id=1, category_id=2, points=[[20, 20]]),
            dict(image_id=1, category_id=1, points=[]),
            dict(image_id=1, category_id=3, points=[]),
            dict(image_id=2, category_id=3, points=[]),
        ]
        self.gt = dict(images=[dict(id=i, width=100, height=100) for i in (1, 2)],
            categories=[dict(id=row["category_id"], name=row["text"]) for row in self.vocabulary],
            annotations=[
                dict(id=1, image_id=2, category_id=1, bbox=[0, 0, 20, 20], area=400, iscrowd=0),
                dict(id=2, image_id=2, category_id=1, bbox=[60, 60, 20, 20], area=400, iscrowd=0),
                dict(id=3, image_id=2, category_id=1, bbox=[0, 0, 10, 10], area=100, iscrowd=1),
                dict(id=10, image_id=1, category_id=2, bbox=[10, 10, 20, 20], area=400, iscrowd=0),
                dict(id=11, image_id=1, category_id=2, bbox=[10, 10, 20, 20], area=400, iscrowd=0),
                dict(id=12, image_id=1, category_id=1, bbox=[40, 40, 20, 20], area=400, iscrowd=0, ignore=1),
            ])
        self.write_json("images.json", self.images)
        self.write_json("vocabulary.json", self.vocabulary)
        self.write_json("gt.json", self.gt)
        self.write_jsonl("detector.jsonl", self.rows)
        self.write_jsonl("counter.jsonl", self.queries)

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def write_jsonl(self, name, rows):
        (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run_pair(self):
        args = [self.root / name for name in ("detector.jsonl", "counter.jsonl", "images.json", "vocabulary.json")]
        small = rerank.run_reranking(*args, self.root / "small")
        large = run_reranking_large(*args, self.root / "large")
        return small, large

    def test_shuffled_input_ties_geometry_extra_fields_and_complete_metrics(self):
        small, large = self.run_pair()
        for filename in ("original.jsonl", "confidence.jsonl", "points.jsonl", "support_budget.jsonl"):
            self.assertEqual((self.root / "small" / filename).read_bytes(), (self.root / "large" / filename).read_bytes())
        self.assertEqual(small["geometry_sha256"], large["geometry_sha256"])
        self.assertEqual(large["candidate_count"], len(self.rows))
        evidence = [json.loads(line) for line in (self.root / "large/support_budget.jsonl").read_text().splitlines()]
        tie = next(row for row in evidence if (row["image_id"], row["category_id"]) == (2, 1))
        self.assertIn(9, tie["point_supported_candidate_indices"])
        self.assertNotIn(2, tie["point_supported_candidate_indices"])
        self.assertEqual(tie["point_box_matches"], [dict(point_index=0, candidate_index=7),
                                                   dict(point_index=1, candidate_index=9)])
        paths = {name: self.root / "large" / (name + ".jsonl") for name in ("confidence", "points")}
        original = self.root / "large/original.jsonl"
        expected = evaluation.evaluate_files(self.root / "gt.json", self.root / "images.json", original, paths, "confidence")
        actual = evaluate_files_large(self.root / "gt.json", self.root / "images.json", original, paths,
                                      "confidence", work_dir=self.root / "large_evaluation")
        for key in ("results", "geometry_sha256", "categories", "metrics_definition", "image_count",
                    "category_count", "control_arm", "top5_excluded_crowd_or_ignored_queries"):
            self.assertEqual(expected[key], actual[key], key)
        self.assertEqual(actual["top5_excluded_crowd_or_ignored_queries"], 2)

    def test_raw_only(self):
        self.run_pair()
        original = self.root / "large/original.jsonl"
        expected = evaluation.evaluate_files(self.root / "gt.json", self.root / "images.json", original, {})
        actual = evaluate_files_large(self.root / "gt.json", self.root / "images.json", original, {},
                                      work_dir=self.root / "raw_evaluation")
        self.assertEqual(expected["results"], actual["results"])
        self.assertEqual(expected["geometry_sha256"], actual["geometry_sha256"])

    def test_arbitrary_python_integer_identifiers(self):
        image, category, index = 2**80, 2**85, 2**90
        for row in self.images + self.rows + self.queries:
            if row["image_id"] == 1:
                row["image_id"] = image
        for row in self.rows + self.queries:
            if row["category_id"] == 3:
                row["category_id"] = category
        self.vocabulary[2]["category_id"] = category
        self.rows[0]["candidate_index"] = index
        self.rows[3]["candidate_index"] = -index
        self.write_json("images.json", self.images)
        self.write_json("vocabulary.json", self.vocabulary)
        self.write_jsonl("detector.jsonl", self.rows)
        self.write_jsonl("counter.jsonl", self.queries)
        self.run_pair()
        for filename in ("original.jsonl", "confidence.jsonl", "points.jsonl", "support_budget.jsonl"):
            self.assertEqual((self.root / "small" / filename).read_bytes(), (self.root / "large" / filename).read_bytes())

    def test_legacy_arrays_and_counter_schema(self):
        legacy = []
        for image in self.images:
            rows = [row for row in self.rows if row["image_id"] == image["image_id"]]
            legacy.append(dict(image_id=image["image_id"], candidate_indices=[row["candidate_index"] for row in rows],
                category_ids=[row["category_id"] for row in rows], scores=[row["score"] for row in rows],
                boxes=[[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in [row["bbox_xyxy"] for row in rows]]))
        queries = [dict(image_id=row["image_id"], category_id=row["category_id"],
                        outputs=dict(on=dict(points_original_pixels=row["points"], pred_count=len(row["points"]))))
                   for row in self.queries]
        self.write_jsonl("detector.jsonl", legacy)
        self.write_jsonl("counter.jsonl", queries)
        self.run_pair()
        for filename in ("original.jsonl", "confidence.jsonl", "points.jsonl", "support_budget.jsonl"):
            self.assertEqual((self.root / "small" / filename).read_bytes(), (self.root / "large" / filename).read_bytes())

    def test_platform_rss_units(self):
        from counting_help_detection import evaluation_large
        with patch.object(evaluation_large.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=4096)):
            with patch.object(evaluation_large.sys, "platform", "darwin"):
                self.assertEqual(evaluation_large.peak_rss_kib(), 4)
            with patch.object(evaluation_large.sys, "platform", "linux"):
                self.assertEqual(evaluation_large.peak_rss_kib(), 4096)

    def test_cross_category_duplicate_candidate_is_rejected(self):
        self.write_jsonl("detector.jsonl", self.rows + [dict(self.rows[0], category_id=2)])
        self.assert_both_reject(ValueError)

    def test_unused_duplicate_counter_query_is_rejected(self):
        self.write_jsonl("counter.jsonl", self.queries + [self.queries[3]])
        self.assert_both_reject(ValueError)

    def test_missing_query_is_not_zero_filled(self):
        self.write_jsonl("counter.jsonl", [row for row in self.queries if (row["image_id"], row["category_id"]) != (1, 2)])
        self.assert_both_reject(rerank.MissingCounterQueriesError)

    def assert_both_reject(self, error):
        args = [self.root / name for name in ("detector.jsonl", "counter.jsonl", "images.json", "vocabulary.json")]
        for function, name in [(rerank.run_reranking, "small"), (run_reranking_large, "large")]:
            with self.assertRaises(error):
                function(*args, self.root / name)


if __name__ == "__main__":
    unittest.main()
