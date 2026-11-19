"""Configuration and domain constants for GeoAgent AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DB_PATH = BASE_DIR / "geoagent_audit.db"

DISASTER_TYPES = [
    "No Risk",
    "Flood",
    "Fire",
    "Cyclone",
    "Landslide",
    "Earthquake",
]

RISK_LEVELS = ["Low", "Medium", "High"]

FEATURE_COLUMNS = [
    "rainfall_mm_24h",
    "rainfall_mm_3h",
    "wind_speed_kmh",
    "pressure_hpa",
    "temperature_c",
    "humidity_pct",
    "soil_moisture",
    "ndvi",
    "ndwi",
    "fire_hotspots_count",
    "frp_mw",
    "elevation_m",
    "slope_deg",
    "seismic_magnitude",
]


@dataclass(frozen=True)
class Settings:
    """Application runtime configuration resolved from environment variables."""

    openweather_api_key: str = field(
        default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", "").strip()
    )
    nasa_firms_map_key: str = field(
        default_factory=lambda: os.getenv("NASA_FIRMS_MAP_KEY", "").strip()
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip()
    )
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "").strip())
    smtp_port: int = field(
        default_factory=lambda: int(os.getenv("SMTP_PORT", "587") or 587)
    )
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", "").strip())
    smtp_password: str = field(
        default_factory=lambda: os.getenv("SMTP_PASSWORD", "").strip()
    )
    alert_webhook_url: str = field(
        default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL", "").strip()
    )
    request_timeout_sec: float = 4.5
    db_path: Path = DB_PATH


def get_settings() -> Settings:
    """Return fresh runtime settings."""
    return Settings()
