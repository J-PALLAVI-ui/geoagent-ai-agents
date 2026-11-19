"""Scenario simulation baselines and live weather API integration."""

from __future__ import annotations

from typing import Any

import requests

from geoagent_ai.config import get_settings
from geoagent_ai.schemas import WeatherParameters

SCENARIO_WEATHER: dict[str, dict[str, Any]] = {
    "no_risk": {
        "rainfall_mm_24h": 4.2,
        "rainfall_mm_3h": 0.5,
        "wind_speed_kmh": 14.0,
        "pressure_hpa": 1014.0,
        "temperature_c": 26.5,
        "humidity_pct": 58.0,
        "condition": "Partly Cloudy / Stable",
    },
    "flood_medium": {
        "rainfall_mm_24h": 152.0,
        "rainfall_mm_3h": 54.0,
        "wind_speed_kmh": 34.0,
        "pressure_hpa": 1003.0,
        "temperature_c": 25.0,
        "humidity_pct": 91.0,
        "condition": "Heavy Monsoon Rain",
    },
    "flood_severe": {
        "rainfall_mm_24h": 278.0,
        "rainfall_mm_3h": 98.0,
        "wind_speed_kmh": 46.0,
        "pressure_hpa": 996.0,
        "temperature_c": 24.0,
        "humidity_pct": 96.0,
        "condition": "Extreme Cloudburst & Deluge",
    },
    "flood_high": {
        "rainfall_mm_24h": 255.0,
        "rainfall_mm_3h": 88.0,
        "wind_speed_kmh": 41.0,
        "pressure_hpa": 997.5,
        "temperature_c": 24.8,
        "humidity_pct": 95.0,
        "condition": "Severe Basin Rainfall",
    },
    "wildfire_high": {
        "rainfall_mm_24h": 0.0,
        "rainfall_mm_3h": 0.0,
        "wind_speed_kmh": 42.0,
        "pressure_hpa": 1008.0,
        "temperature_c": 41.5,
        "humidity_pct": 16.0,
        "condition": "Hot Dry Gale & Thermal Anomaly",
    },
    "cyclone_medium": {
        "rainfall_mm_24h": 128.0,
        "rainfall_mm_3h": 48.0,
        "wind_speed_kmh": 88.0,
        "pressure_hpa": 984.0,
        "temperature_c": 27.0,
        "humidity_pct": 92.0,
        "condition": "Cyclonic Storm Squalls",
    },
    "cyclone_high": {
        "rainfall_mm_24h": 215.0,
        "rainfall_mm_3h": 84.0,
        "wind_speed_kmh": 158.0,
        "pressure_hpa": 952.0,
        "temperature_c": 26.2,
        "humidity_pct": 96.0,
        "condition": "Severe Cyclonic Storm Eye-Wall",
    },
    "landslide_high": {
        "rainfall_mm_24h": 245.0,
        "rainfall_mm_3h": 82.0,
        "wind_speed_kmh": 36.0,
        "pressure_hpa": 999.0,
        "temperature_c": 19.5,
        "humidity_pct": 97.0,
        "condition": "Persistent Orographic Downpour",
    },
    "earthquake_high": {
        "rainfall_mm_24h": 0.0,
        "rainfall_mm_3h": 0.0,
        "wind_speed_kmh": 16.0,
        "pressure_hpa": 1013.0,
        "temperature_c": 31.0,
        "humidity_pct": 42.0,
        "condition": "Clear Sky (Seismic Alert)",
    },
}


def _describe_wmo_code(code: int | None) -> str:
    if code is None:
        return "Live Observed"
    if code >= 95:
        return "Thunderstorm / Squall"
    if code >= 61:
        return "Moderate to Heavy Rain"
    if code >= 51:
        return "Drizzle / Light Rain"
    if code >= 1:
        return "Partly Cloudy"
    return "Clear Sky"


def fetch_weather_parameters(
    latitude: float,
    longitude: float,
    mode: str = "live",
    scenario: str | None = None,
) -> tuple[WeatherParameters, str | None]:
    """Fetch live weather from OpenWeatherMap / Open-Meteo, or return scenario telemetry."""
    if mode == "simulation":
        preset = SCENARIO_WEATHER.get(scenario or "no_risk", SCENARIO_WEATHER["no_risk"])
        return (
            WeatherParameters(
                **preset,
                source=f"Simulation Preset ({scenario or 'no_risk'})",
            ),
            None,
        )

    settings = get_settings()
    if settings.openweather_api_key:
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "appid": settings.openweather_api_key,
                    "units": "metric",
                },
                timeout=settings.request_timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            rain_1h = float(payload.get("rain", {}).get("1h", 0.0))
            rain_3h = float(payload.get("rain", {}).get("3h", rain_1h * 2.5))
            return (
                WeatherParameters(
                    rainfall_mm_24h=round(rain_3h * 5.0, 2),
                    rainfall_mm_3h=round(rain_3h, 2),
                    wind_speed_kmh=round(float(payload["wind"]["speed"]) * 3.6, 2),
                    pressure_hpa=float(payload["main"].get("pressure", 1013.0)),
                    temperature_c=float(payload["main"].get("temp", 26.0)),
                    humidity_pct=float(payload["main"].get("humidity", 60.0)),
                    condition=str(payload["weather"][0]["description"]).title(),
                    source="OpenWeatherMap Live API",
                ),
                None,
            )
        except Exception as exc:
            fallback_note = f"OpenWeatherMap failed ({type(exc).__name__}); tried Open-Meteo"
    else:
        fallback_note = None

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,weather_code",
                "hourly": "precipitation",
                "past_days": 1,
                "forecast_days": 1,
                "timezone": "UTC",
            },
            timeout=settings.request_timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        cur = data.get("current", {})
        hourly_rain = [float(v or 0.0) for v in data.get("hourly", {}).get("precipitation", [])]
        rain_24h = sum(hourly_rain[-24:]) if hourly_rain else float(cur.get("precipitation", 0.0)) * 6.0
        rain_3h = sum(hourly_rain[-3:]) if hourly_rain else float(cur.get("precipitation", 0.0)) * 2.0
        return (
            WeatherParameters(
                rainfall_mm_24h=round(rain_24h, 2),
                rainfall_mm_3h=round(rain_3h, 2),
                wind_speed_kmh=round(float(cur.get("wind_speed_10m", 12.0)), 2),
                pressure_hpa=round(float(cur.get("surface_pressure", 1012.0)), 2),
                temperature_c=round(float(cur.get("temperature_2m", 26.0)), 2),
                humidity_pct=round(float(cur.get("relative_humidity_2m", 60.0)), 2),
                condition=_describe_wmo_code(cur.get("weather_code")),
                source="Open-Meteo Live API (IMD-Aligned)",
            ),
            fallback_note,
        )
    except Exception as exc:
        # Fault-tolerant deterministic baseline derived from latitude/climate band
        base = dict(SCENARIO_WEATHER["no_risk"])
        if abs(latitude) < 15.0:
            base["temperature_c"] = 28.5
            base["humidity_pct"] = 70.0
        return (
            WeatherParameters(
                **base,
                source="Regional Meteorological Baseline (Offline Fallback)",
            ),
            f"Weather API unreachable ({type(exc).__name__}); activated regional weather baseline",
        )
