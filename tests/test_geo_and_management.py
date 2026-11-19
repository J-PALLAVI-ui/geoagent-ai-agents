"""Tests for geospatial computations, custom coordinates, and feedback loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from geoagent_ai.agents import GeoAgentOrchestrator
from geoagent_ai.management import AgentManagerMonitor
from geoagent_ai.schemas import FeedbackSubmission, LocationQuery


class TestGeoAndManagement(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.monitor = AgentManagerMonitor(db_path=Path(self.tmp_dir.name) / "test_mgmt.db")
        self.orchestrator = GeoAgentOrchestrator(monitor=self.monitor)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_custom_map_pin_generates_valid_geojson_and_shelters(self) -> None:
        res = self.orchestrator.run(
            LocationQuery(
                location_name="Custom Coastal Pin",
                latitude=16.5062,
                longitude=80.6480,
                mode="simulation",
                scenario="flood_severe",
            )
        )
        poly = res.geo_analysis.affected_polygon_geojson
        self.assertEqual(poly["geometry"]["type"], "Polygon")
        coords = poly["geometry"]["coordinates"][0]
        self.assertGreaterEqual(len(coords), 20)
        self.assertEqual(coords[0], coords[-1])

        shelters = res.response_plan.nearest_shelters
        self.assertEqual(len(shelters), 3)
        distances = [s.distance_km for s in shelters]
        self.assertEqual(distances, sorted(distances))

    def test_feedback_loop_calibrates_future_severity(self) -> None:
        query = LocationQuery(
            location_name="Chennai Coast, Tamil Nadu",
            mode="simulation",
            scenario="cyclone_medium",
        )
        first = self.orchestrator.run(query)
        self.monitor.submit_feedback(
            FeedbackSubmission(
                run_id=first.run_id,
                location_name=first.collected_data.resolved_location,
                detected_disaster=first.classification.detected_disaster,
                predicted_risk=first.risk_assessment.risk_level,
                verdict="underestimated",
                notes="Field team noted higher wind surge",
            )
        )
        second = self.orchestrator.run(query)
        self.assertGreater(second.risk_assessment.calibration_applied, 0.0)
        self.assertGreater(second.risk_assessment.severity_score, first.risk_assessment.severity_score)

        summary = self.monitor.get_monitoring_summary()
        self.assertEqual(summary["total_pipeline_runs"], 2)
        self.assertEqual(summary["total_feedback_entries"], 1)


if __name__ == "__main__":
    unittest.main()
