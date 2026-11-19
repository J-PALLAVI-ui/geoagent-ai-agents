"""USGS live seismic feed and scenario seismic telemetry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from geoagent_ai.config import get_settings
from geoagent_ai.schemas import SeismicTelemetry


def fetch_seismic_telemetry(
    latitude: float,
    longitude: float,
    mode: str = "live",
    scenario: str | None = None,
) -> tuple[SeismicTelemetry, str | None]:
    """Fetch recent earthquakes near location from USGS API or simulation profile."""
    if mode == "simulation":
        if scenario == "earthquake_high":
            return (
                SeismicTelemetry(
                    seismic_magnitude=6.7,
                    depth_km=11.5,
                    recent_events_count=4,
                    source="Seismic Sensor Network Simulation (earthquake_high)",
                ),
                None,
            )
        return (
            SeismicTelemetry(
                seismic_magnitude=1.3,
                depth_km=18.0,
                recent_events_count=0,
                source="Seismic Simulation Baseline",
            ),
            None,
        )

    settings = get_settings()
    start_time = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        resp = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start_time,
                "latitude": latitude,
                "longitude": longitude,
                "maxradiuskm": 220,
                "minmagnitude": 2.5,
            },
            timeout=settings.request_timeout_sec,
        )
        resp.raise_for_status()
        features = resp.json().get("features") or []
        if not features:
            return (
                SeismicTelemetry(
                    seismic_magnitude=1.1,
                    depth_km=15.0,
                    recent_events_count=0,
                    source="USGS Earthquake Live API (Quiet)",
                ),
                None,
            )
        strongest = max(features, key=lambda f: float(f["properties"].get("mag") or 0.0))
        mag = float(strongest["properties"].get("mag") or 1.5)
        depth = float(strongest["geometry"]["coordinates"][2] or 15.0)
        return (
            SeismicTelemetry(
                seismic_magnitude=round(mag, 2),
                depth_km=round(depth, 1),
                recent_events_count=len(features),
                source="USGS Earthquake Live API",
            ),
            None,
        )
    except Exception as exc:
        return (
            SeismicTelemetry(
                seismic_magnitude=1.2,
                depth_km=16.0,
                recent_events_count=0,
                source="Seismic Baseline (Offline Fallback)",
            ),
            f"USGS seismic API fallback ({type(exc).__name__}); used quiet seismic baseline",
        )
