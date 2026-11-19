"""FastAPI backend serving GeoAgent AI endpoints and interactive web dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from geoagent_ai.agents import GeoAgentOrchestrator
from geoagent_ai.data_sources import get_region_catalog
from geoagent_ai.data_sources.weather_client import SCENARIO_WEATHER
from geoagent_ai.management import AgentManagerMonitor
from geoagent_ai.ml import train_and_save_models
from geoagent_ai.schemas import FeedbackSubmission, LocationQuery

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(
    title="GeoAgent AI: Autonomous Disaster Monitoring and Response System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

monitor = AgentManagerMonitor()
orchestrator = GeoAgentOrchestrator(monitor=monitor)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def serve_dashboard():
    """Serve interactive Geospatial Command Center frontend."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "system": "GeoAgent AI",
        "model_metrics": orchestrator.predictor.metrics,
    }


@app.get("/api/regions")
def list_regions() -> dict[str, Any]:
    return {
        "regions": get_region_catalog(),
        "scenarios": list(SCENARIO_WEATHER.keys()),
    }


@app.post("/api/analyze")
def analyze_location(query: LocationQuery):
    """Run autonomous 6-agent LangGraph workflow for a given location."""
    return orchestrator.run(query).model_dump()


@app.get("/api/monitoring/summary")
def monitoring_summary() -> dict[str, Any]:
    """Return agent registry, performance metrics, recent runs, alerts, and watchlist."""
    return {
        "summary": monitor.get_monitoring_summary(),
        "recent_runs": monitor.get_recent_runs(limit=12),
        "recent_alerts": monitor.get_recent_alerts(limit=15),
        "watchlist": monitor.get_watchlist(),
        "model_metrics": orchestrator.predictor.metrics,
    }


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackSubmission) -> dict[str, Any]:
    """Record operator feedback for continuous calibration loop."""
    return monitor.submit_feedback(payload)


@app.post("/api/models/retrain")
def retrain_models() -> dict[str, Any]:
    """Retrain ML hazard classifier and severity regressor and reload predictor."""
    metrics = train_and_save_models()
    orchestrator.predictor = type(orchestrator.predictor)()
    return {"status": "retrained", "metrics": metrics}
