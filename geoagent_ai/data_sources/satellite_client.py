"""Satellite imagery indices (Sentinel-2 / Copernicus) and NASA FIRMS fire hotspots."""

from __future__ import annotations

import csv
import io
from typing import Any

import requests

from geoagent_ai.config import get_settings
from geoagent_ai.schemas import SatelliteTelemetry, WeatherParameters

SCENARIO_SATELLITE: dict[str, dict[str, Any]] = {
    "no_risk": {
        "ndvi": 0.56,
        "ndwi": 0.08,
        "nbr": 0.34,
        "fire_hotspots_count": 0,
        "frp_mw": 0.0,
        "cloud_cover_pct": 18.0,
    },
    "flood_medium": {
        "ndvi": 0.32,
        "ndwi": 0.44,
        "nbr": 0.15,
        "fire_hotspots_count": 0,
        "frp_mw": 0.0,
        "cloud_cover_pct": 78.0,
    },
    "flood_severe": {
        "ndvi": 0.21,
        "ndwi": 0.68,
        "nbr": 0.08,
        "fire_hotspots_count": 0,
        "frp_mw": 0.0,
        "cloud_cover_pct": 92.0,
    },
    "flood_high": {
        "ndvi": 0.24,
        "ndwi": 0.61,
        "nbr": 0.10,
        "fire_hotspots_count": 0,
        "frp_mw": 0.0,
        "cloud_cover_pct": 88.0,
    },
    "wildfire_high": {
        "ndvi": 0.14,
        "ndwi": -0.32,
        "nbr": -0.45,
        "fire_hotspots_count": 37,
        "frp_mw": 148.5,
        "cloud_cover_pct": 8.0,
    },
    "cyclone_medium": {
        "ndvi": 0.38,
        "ndwi": 0.35,
        "nbr": 0.20,
        "fire_hotspots_count": 0,
        "frp_mw": 0.0,
        "cloud_cover_pct": 94.0,
    },
    "cyclone_high": {
        "ndvi": 0.29,
        "ndwi": 0.52,
        "nbr": 0.12,
        "fire_hotspots_count": 0,
        "frp_mw": 0.0,
        "cloud_cover_pct": 99.0,
    },
    "landslide_high": {
        "ndvi": 0.42,
        "ndwi": 0.39,
        "nbr": 0.18,
        "fire_hotspots_count": 0,
        "frp_mw": 0.0,
        "cloud_cover_pct": 85.0,
    },
    "earthquake_high": {
        "ndvi": 0.35,
        "ndwi": -0.05,
        "nbr": 0.22,
        "fire_hotspots_count": 1,
        "frp_mw": 4.2,
        "cloud_cover_pct": 12.0,
    },
}


def fetch_satellite_telemetry(
    latitude: float,
    longitude: float,
    weather: WeatherParameters,
    mode: str = "live",
    scenario: str | None = None,
) -> tuple[SatelliteTelemetry, str | None]:
    """Fetch NASA FIRMS active thermal hotspots (when key configured) and spectral indices."""
    if mode == "simulation":
        preset = SCENARIO_SATELLITE.get(scenario or "no_risk", SCENARIO_SATELLITE["no_risk"])
        return (
            SatelliteTelemetry(
                **preset,
                source=f"Sentinel-2 + NASA FIRMS Simulation ({scenario or 'no_risk'})",
            ),
            None,
        )

    settings = get_settings()
    fire_count = 0
    total_frp = 0.0
    fallback_note: str | None = None
    source_label = "Sentinel-2 Spectral Model + FIRMS Baseline"

    if settings.nasa_firms_map_key:
        bbox = f"{longitude - 0.6:.3f},{latitude - 0.6:.3f},{longitude + 0.6:.3f},{latitude + 0.6:.3f}"
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{settings.nasa_firms_map_key}/VIIRS_SNPP_NRT/{bbox}/2"
        )
        try:
            resp = requests.get(url, timeout=settings.request_timeout_sec)
            resp.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(resp.text)))
            fire_count = len(rows)
            total_frp = round(sum(float(r.get("frp", 0.0) or 0.0) for r in rows), 2)
            source_label = "NASA FIRMS VIIRS Live + Sentinel-2 Model"
        except Exception as exc:
            fallback_note = f"NASA FIRMS API fallback ({type(exc).__name__}); derived thermal baseline"

    # Derive physically consistent NDVI / NDWI / NBR from live meteorological state
    ndwi = max(-0.4, min(0.75, (weather.rainfall_mm_24h / 300.0) * 0.65 + (weather.humidity_pct - 50.0) / 220.0))
    ndvi = max(0.1, min(0.8, 0.52 - max(0.0, weather.temperature_c - 35.0) * 0.025))
    nbr = max(-0.5, min(0.5, 0.32 - (fire_count * 0.02)))
    cloud = max(5.0, min(99.0, weather.humidity_pct * 0.85 if weather.rainfall_mm_24h > 10 else weather.humidity_pct * 0.4))

    return (
        SatelliteTelemetry(
            ndvi=round(ndvi, 3),
            ndwi=round(ndwi, 3),
            nbr=round(nbr, 3),
            fire_hotspots_count=fire_count,
            frp_mw=total_frp,
            cloud_cover_pct=round(cloud, 1),
            source=source_label,
        ),
        fallback_note,
    )
