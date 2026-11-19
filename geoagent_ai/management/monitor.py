"""SQLite-backed Agent Registry, Performance Monitor, Audit Trail, and Feedback Calibration."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from geoagent_ai.config import DB_PATH
from geoagent_ai.schemas import FeedbackSubmission, PipelineExecutionResult

AGENT_REGISTRY: list[dict[str, str]] = [
    {
        "agent_id": "agent-1-data-collection",
        "name": "Data Collection Agent",
        "role": "Multi-source Weather, NASA FIRMS, Sentinel-2, SRTM DEM, USGS Seismic & Historical ingestion",
        "version": "1.0.0",
        "status": "active",
    },
    {
        "agent_id": "agent-2-hazard-classification",
        "name": "Disaster Classification Agent",
        "role": "Ensemble ML + Cross-Source Validation classifying Flood/Fire/Cyclone/Landslide/Earthquake/No Risk",
        "version": "1.0.0",
        "status": "active",
    },
    {
        "agent_id": "agent-3-geo-location",
        "name": "Geo-Location Agent",
        "role": "Spatial impact footprint polygon, area calculation & population exposure mapping",
        "version": "1.0.0",
        "status": "active",
    },
    {
        "agent_id": "agent-4-risk-prediction",
        "name": "Risk Prediction Agent",
        "role": "Severity scoring (0-100), risk level (Low/Medium/High) & feedback-calibrated probability",
        "version": "1.0.0",
        "status": "active",
    },
    {
        "agent_id": "agent-5-alert-communication",
        "name": "Alert & Communication Agent",
        "role": "Automated SMS/Email/Push alert generation when Risk is Medium or High",
        "version": "1.0.0",
        "status": "active",
    },
    {
        "agent_id": "agent-6-response-planning",
        "name": "Response Planning Agent",
        "role": "Autonomous evacuation routes, nearest shelters, safety protocols & resource dispatch",
        "version": "1.0.0",
        "status": "active",
    },
]


class AgentManagerMonitor:
    """Manages agent registry, SQLite audit logs, performance metrics, watchlist, and feedback."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    location_name TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    mode TEXT NOT NULL,
                    detected_disaster TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    severity_score REAL NOT NULL,
                    alert_triggered INTEGER NOT NULL,
                    total_duration_ms REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    summary TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alert_history (
                    alert_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    location_name TEXT NOT NULL,
                    detected_disaster TEXT NOT NULL,
                    severity_level TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    dispatch_status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS operator_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    location_name TEXT NOT NULL,
                    detected_disaster TEXT NOT NULL,
                    predicted_risk TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    calibration_delta REAL NOT NULL,
                    notes TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monitoring_watchlist (
                    location_name TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    last_disaster TEXT,
                    last_risk TEXT,
                    last_checked_utc TEXT
                );
                """
            )

    def record_run(self, result: PipelineExecutionResult) -> None:
        """Persist full pipeline execution, per-agent latency logs, and alerts."""
        with self._session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs
                (run_id, created_at_utc, location_name, latitude, longitude, mode,
                 detected_disaster, risk_level, severity_score, alert_triggered, total_duration_ms, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.created_at_utc,
                    result.collected_data.resolved_location,
                    result.collected_data.latitude,
                    result.collected_data.longitude,
                    result.query.mode,
                    result.classification.detected_disaster,
                    result.risk_assessment.risk_level,
                    result.risk_assessment.severity_score,
                    1 if result.alert.alert_triggered else 0,
                    result.total_duration_ms,
                    result.model_dump_json(),
                ),
            )

            for log in result.agent_logs:
                conn.execute(
                    """
                    INSERT INTO agent_metrics
                    (run_id, agent_id, agent_name, status, duration_ms, summary, created_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.run_id,
                        log.agent_id,
                        log.agent_name,
                        log.status,
                        log.duration_ms,
                        log.summary,
                        result.created_at_utc,
                    ),
                )

            if result.alert.alert_triggered:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alert_history
                    (alert_id, run_id, location_name, detected_disaster, severity_level, headline, dispatch_status, created_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.alert.alert_id,
                        result.run_id,
                        result.collected_data.resolved_location,
                        result.classification.detected_disaster,
                        result.alert.severity_level,
                        result.alert.headline,
                        result.alert.dispatch_status,
                        result.alert.timestamp_utc,
                    ),
                )

            conn.execute(
                """
                INSERT INTO monitoring_watchlist (location_name, latitude, longitude, last_disaster, last_risk, last_checked_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(location_name) DO UPDATE SET
                    last_disaster = excluded.last_disaster,
                    last_risk = excluded.last_risk,
                    last_checked_utc = excluded.last_checked_utc
                """,
                (
                    result.collected_data.resolved_location,
                    result.collected_data.latitude,
                    result.collected_data.longitude,
                    result.classification.detected_disaster,
                    result.risk_assessment.risk_level,
                    result.created_at_utc,
                ),
            )

    def get_calibration_offset(self, location_name: str, detected_disaster: str) -> float:
        """Compute bounded severity calibration offset from operator feedback loop."""
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(calibration_delta), 0.0) AS total_delta
                FROM operator_feedback
                WHERE lower(location_name) = lower(?) AND lower(detected_disaster) = lower(?)
                """,
                (location_name, detected_disaster),
            ).fetchone()
        offset = float(row["total_delta"]) if row else 0.0
        return max(-15.0, min(15.0, offset))

    def submit_feedback(self, feedback: FeedbackSubmission) -> dict[str, Any]:
        """Record authority/user feedback and update adaptive calibration delta."""
        delta_map = {
            "confirmed": 1.5,
            "false_alarm": -5.0,
            "underestimated": 6.0,
            "overestimated": -4.0,
        }
        delta = delta_map[feedback.verdict]
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO operator_feedback
                (run_id, location_name, detected_disaster, predicted_risk, verdict, calibration_delta, notes, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.run_id,
                    feedback.location_name,
                    feedback.detected_disaster,
                    feedback.predicted_risk,
                    feedback.verdict,
                    delta,
                    feedback.notes,
                    now,
                ),
            )
        new_offset = self.get_calibration_offset(feedback.location_name, feedback.detected_disaster)
        return {
            "status": "recorded",
            "applied_delta": delta,
            "updated_calibration_offset": new_offset,
            "timestamp_utc": now,
        }

    def get_recent_runs(self, limit: int = 12) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT run_id, created_at_utc, location_name, latitude, longitude, mode,
                       detected_disaster, risk_level, severity_score, alert_triggered, total_duration_ms
                FROM pipeline_runs
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_alerts(self, limit: int = 15) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT alert_id, run_id, location_name, detected_disaster, severity_level,
                       headline, dispatch_status, created_at_utc
                FROM alert_history
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_watchlist(self) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM monitoring_watchlist ORDER BY last_checked_utc DESC LIMIT 20"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Return agent registry paired with live execution & performance statistics."""
        with self._session() as conn:
            total_runs = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
            total_alerts = conn.execute("SELECT COUNT(*) FROM alert_history").fetchone()[0]
            total_feedback = conn.execute("SELECT COUNT(*) FROM operator_feedback").fetchone()[0]
            perf_rows = conn.execute(
                """
                SELECT agent_id,
                       COUNT(*) AS invocations,
                       ROUND(AVG(duration_ms), 2) AS avg_duration_ms,
                       SUM(CASE WHEN status = 'fallback' THEN 1 ELSE 0 END) AS fallback_count
                FROM agent_metrics
                GROUP BY agent_id
                """
            ).fetchall()
        perf_map = {r["agent_id"]: dict(r) for r in perf_rows}

        registry_enriched = []
        for agent in AGENT_REGISTRY:
            stats = perf_map.get(
                agent["agent_id"],
                {"invocations": 0, "avg_duration_ms": 0.0, "fallback_count": 0},
            )
            registry_enriched.append({**agent, **stats})

        return {
            "total_pipeline_runs": total_runs,
            "total_alerts_issued": total_alerts,
            "total_feedback_entries": total_feedback,
            "agents": registry_enriched,
        }
