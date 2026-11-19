"""API endpoint verification tests using FastAPI TestClient."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import app


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_and_regions(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        regions = self.client.get("/api/regions")
        self.assertEqual(regions.status_code, 200)
        self.assertGreaterEqual(len(regions.json()["regions"]), 8)

    def test_analyze_and_monitoring_endpoints(self) -> None:
        res = self.client.post(
            "/api/analyze",
            json={"location_name": "Ernakulam, Kerala", "mode": "simulation"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["classification"]["detected_disaster"], "Flood")
        self.assertTrue(data["alert"]["alert_triggered"])

        mon = self.client.get("/api/monitoring/summary")
        self.assertEqual(mon.status_code, 200)
        self.assertGreaterEqual(mon.json()["summary"]["total_pipeline_runs"], 1)


if __name__ == "__main__":
    unittest.main()
