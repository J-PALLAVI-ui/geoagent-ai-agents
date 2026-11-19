"""Integration tests for the 6-agent LangGraph StateGraph across all disaster types."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from geoagent_ai.agents import GeoAgentOrchestrator
from geoagent_ai.management import AgentManagerMonitor
from geoagent_ai.schemas import LocationQuery


class TestAgentsGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.monitor = AgentManagerMonitor(db_path=Path(self.tmp_dir.name) / "test_audit.db")
        self.orchestrator = GeoAgentOrchestrator(monitor=self.monitor)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_all_hazard_scenarios_autonomous_detection(self) -> None:
        cases = [
            ("Ernakulam, Kerala", "Flood", True),
            ("Bandipur Forest, Karnataka", "Fire", True),
            ("Puri Coast, Odisha", "Cyclone", True),
            ("Wayanad, Kerala", "Landslide", True),
            ("Bhuj, Gujarat", "Earthquake", True),
            ("Bengaluru, Karnataka", "No Risk", False),
        ]
        for location, expected_disaster, expect_alert in cases:
            with self.subTest(location=location):
                res = self.orchestrator.run(
                    LocationQuery(location_name=location, mode="simulation")
                )
                self.assertEqual(res.classification.detected_disaster, expected_disaster)
                self.assertEqual(res.alert.alert_triggered, expect_alert)
                self.assertEqual(len(res.agent_logs), 6)
                self.assertGreaterEqual(len(res.response_plan.nearest_shelters), 3)
                self.assertGreaterEqual(len(res.response_plan.evacuation_routes), 3)

    def test_live_mode_graceful_fallback_without_network(self) -> None:
        with patch("requests.get", side_effect=requests.ConnectionError("Simulated offline sandbox")):
            res = self.orchestrator.run(
                LocationQuery(location_name="Ernakulam, Kerala", mode="live")
            )
        self.assertEqual(len(res.agent_logs), 6)
        self.assertGreaterEqual(len(res.collected_data.fallbacks_triggered), 1)
        self.assertEqual(res.agent_logs[0].status, "fallback")
        self.assertIn(
            res.classification.detected_disaster,
            {"No Risk", "Flood", "Fire", "Cyclone", "Landslide", "Earthquake"},
        )


if __name__ == "__main__":
    unittest.main()
