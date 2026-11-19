"""Tests for ML dataset generation, training, and hazard prediction."""

from __future__ import annotations

import unittest

from geoagent_ai.ml import HazardPredictor, train_and_save_models


class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = train_and_save_models(samples_per_class=220)
        cls.predictor = HazardPredictor()

    def test_training_metrics_quality(self) -> None:
        self.assertGreaterEqual(self.metrics["classification_accuracy"], 0.92)
        self.assertGreaterEqual(self.metrics["classification_weighted_f1"], 0.92)
        self.assertGreaterEqual(self.metrics["severity_r2"], 0.85)
        self.assertEqual(len(self.metrics["classes"]), 6)

    def test_classes_cover_ppt_disasters(self) -> None:
        expected = {"No Risk", "Flood", "Fire", "Cyclone", "Landslide", "Earthquake"}
        self.assertEqual(set(self.metrics["classes"]), expected)


if __name__ == "__main__":
    unittest.main()
