"""Region catalog lookup, geocoding, and historical disaster matching."""

from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from typing import Any

import requests

from geoagent_ai.config import DATA_DIR, get_settings
from geoagent_ai.schemas import HistoricalMatch


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two coordinates."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return round(2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


@lru_cache(maxsize=1)
def get_region_catalog() -> list[dict[str, Any]]:
    """Load preconfigured regions, shelters, and emergency contacts."""
    catalog_path = DATA_DIR / "regions_catalog.json"
    with catalog_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("regions", [])


@lru_cache(maxsize=1)
def _load_historical_rows() -> list[dict[str, Any]]:
    csv_path = DATA_DIR / "historical_disasters.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def match_catalog_region(
    location_name: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any] | None:
    """Find best matching catalog entry by name/alias or nearest coordinate (<60 km)."""
    query = location_name.strip().lower()
    catalog = get_region_catalog()

    for region in catalog:
        if query == region["id"].lower() or query == region["name"].lower():
            return region
        if query in region["name"].lower() or any(
            query in alias.lower() or alias.lower() in query
            for alias in region.get("aliases", [])
        ):
            return region

    if latitude is not None and longitude is not None:
        nearest: dict[str, Any] | None = None
        best_dist = 60.0
        for region in catalog:
            dist = haversine_km(latitude, longitude, region["latitude"], region["longitude"])
            if dist < best_dist:
                best_dist = dist
                nearest = region
        return nearest

    return None


def resolve_location(
    location_name: str,
    latitude: float | None = None,
    longitude: float | None = None,
    allow_network: bool = True,
) -> tuple[str, float, float, str, dict[str, Any] | None, str | None]:
    """Resolve location string/coords into (name, lat, lon, admin, catalog_entry, fallback_note)."""
    catalog_entry = match_catalog_region(location_name, latitude, longitude)

    if latitude is not None and longitude is not None:
        resolved_name = (
            catalog_entry["name"]
            if catalog_entry and location_name.strip().lower() in {"custom", "map pin", ""}
            else location_name.strip()
        )
        admin = catalog_entry["state_or_country"] if catalog_entry else "Custom Coordinates"
        return resolved_name, float(latitude), float(longitude), admin, catalog_entry, None

    if catalog_entry is not None:
        return (
            catalog_entry["name"],
            float(catalog_entry["latitude"]),
            float(catalog_entry["longitude"]),
            catalog_entry["state_or_country"],
            catalog_entry,
            None,
        )

    # Try Open-Meteo free geocoding API if online
    if allow_network:
        settings = get_settings()
        try:
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location_name, "count": 1, "language": "en", "format": "json"},
                timeout=settings.request_timeout_sec,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if results:
                top = results[0]
                admin_parts = [p for p in [top.get("admin1"), top.get("country")] if p]
                return (
                    top.get("name", location_name),
                    float(top["latitude"]),
                    float(top["longitude"]),
                    ", ".join(admin_parts) or "Global",
                    None,
                    None,
                )
        except Exception as exc:
            return (
                location_name.strip(),
                12.9716,
                77.5946,
                "Fallback Coordinate Baseline",
                get_region_catalog()[0],
                f"Geocoding API unavailable ({type(exc).__name__}); defaulted to baseline coordinate",
            )

    default_region = get_region_catalog()[0]
    return (
        location_name.strip() or default_region["name"],
        float(default_region["latitude"]),
        float(default_region["longitude"]),
        default_region["state_or_country"],
        default_region,
        "Unknown region offline; mapped to primary monitoring baseline",
    )


def find_historical_matches(
    latitude: float,
    longitude: float,
    limit: int = 4,
    max_radius_km: float = 450.0,
) -> list[HistoricalMatch]:
    """Return nearest historical disaster records from the historical dataset."""
    matches: list[HistoricalMatch] = []
    for row in _load_historical_rows():
        dist = haversine_km(
            latitude,
            longitude,
            float(row["latitude"]),
            float(row["longitude"]),
        )
        if dist <= max_radius_km:
            matches.append(
                HistoricalMatch(
                    event_id=row["event_id"],
                    region=row["region"],
                    disaster_type=row["disaster_type"],
                    year=int(row["year"]),
                    severity=row["severity"],
                    casualties_or_impact=row["casualties_or_impact"],
                    distance_km=dist,
                )
            )
    matches.sort(key=lambda m: m.distance_km)
    return matches[:limit]
