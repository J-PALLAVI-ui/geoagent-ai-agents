"""SRTM / DEM terrain profile and soil saturation estimator."""

from __future__ import annotations

from typing import Any

import requests

from geoagent_ai.config import get_settings
from geoagent_ai.schemas import TerrainProfile, WeatherParameters


def fetch_terrain_profile(
    latitude: float,
    longitude: float,
    weather: WeatherParameters,
    catalog_entry: dict[str, Any] | None = None,
    mode: str = "live",
    scenario: str | None = None,
) -> tuple[TerrainProfile, str | None]:
    """Fetch SRTM/DEM elevation and derive slope, drainage, and soil moisture."""
    base_elev = float(catalog_entry["elevation_m"]) if catalog_entry else 120.0
    base_slope = float(catalog_entry["slope_deg"]) if catalog_entry else 5.0
    land_cover = catalog_entry["land_cover"] if catalog_entry else "Mixed Terrain"
    source = "SRTM DEM Regional Catalog"
    fallback_note: str | None = None

    if mode == "live":
        settings = get_settings()
        try:
            resp = requests.get(
                "https://api.open-elevation.com/api/v1/lookup",
                params={"locations": f"{latitude:.4f},{longitude:.4f}"},
                timeout=settings.request_timeout_sec,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if results:
                base_elev = float(results[0]["elevation"])
                source = "Open-Elevation SRTM DEM Live"
        except Exception as exc:
            fallback_note = f"DEM elevation API fallback ({type(exc).__name__}); using local SRTM baseline"

    # Override soil moisture and slope emphasis for specific simulation demos
    soil_moisture = min(0.98, max(0.08, 0.24 + (weather.rainfall_mm_24h / 320.0) * 0.65 + (weather.humidity_pct / 400.0)))
    if mode == "simulation" and scenario == "wildfire_high":
        soil_moisture = 0.09
    elif mode == "simulation" and scenario == "landslide_high":
        base_slope = max(base_slope, 29.0)
        soil_moisture = 0.91

    drainage_density = round(1.0 + (max(0.0, 40.0 - base_slope) / 40.0) * 1.4, 2)

    return (
        TerrainProfile(
            elevation_m=round(base_elev, 1),
            slope_deg=round(base_slope, 1),
            soil_moisture=round(soil_moisture, 3),
            drainage_density=drainage_density,
            land_cover=land_cover,
            source=source,
        ),
        fallback_note,
    )
