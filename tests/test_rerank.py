"""Regression for fixed candidate geometry and a genuinely matched support budget."""
import unittest

from counting_help_detection.rerank import geometry_digest, ranked_indices, rerank_image


class MatchedRerankingRegression(unittest.TestCase):
    def test_same_candidates_budget_and_monotone_confidence_order(self):
        rows = [
            dict(image_id=1, category_id=3, candidate_index=0, bbox_xyxy=[50., 50., 60., 60.], score=.80),
            dict(image_id=1, category_id=3, candidate_index=1, bbox_xyxy=[0., 0., 10., 10.], score=.60),
            dict(image_id=1, category_id=3, candidate_index=2, bbox_xyxy=[70., 70., 65., 75.], score=.20),
            dict(image_id=1, category_id=6, candidate_index=3, bbox_xyxy=[80., 80., 90., 90.], score=.40),
        ]
        arms, evidence = rerank_image(rows, {3: [[5., 5.]], 6: []})
        expected_geometry = geometry_digest(rows)
        self.assertTrue(all(geometry_digest(values) == expected_geometry for values in arms.values()))
        for category in (3, 6):
            original = [r for r in arms["original"] if r["category_id"] == category]
            confidence = [r for r in arms["confidence"] if r["category_id"] == category]
            self.assertEqual(ranked_indices(original), ranked_indices(confidence))
        car, bus = evidence
        self.assertEqual(car["point_supported_candidate_indices"], [1])
        self.assertEqual(car["confidence_supported_candidate_indices"], [0])
        self.assertEqual(car["supported_box_count"], 1)
        self.assertEqual(bus["supported_box_count"], 0)
        self.assertEqual(car["point_box_matches"], [dict(point_index=0, candidate_index=1)])
        point_scores = {r["candidate_index"]: r["score"] for r in arms["points"]}
        self.assertAlmostEqual(point_scores[1], .75*.6 + .25*(2*.6-.6*.6))
        self.assertAlmostEqual(point_scores[0], .75*.8 + .25*.8*.8)
        self.assertNotIn(2, car["point_supported_candidate_indices"])
        self.assertEqual(rows[1]["score"], .6)


if __name__ == "__main__":
    unittest.main()
