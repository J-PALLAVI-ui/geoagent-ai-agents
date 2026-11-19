"""Multi-source environmental, satellite, terrain, seismic, and historical data clients."""

from geoagent_ai.data_sources.historical_store import (
    find_historical_matches,
    get_region_catalog,
    resolve_location,
)
from geoagent_ai.data_sources.satellite_client import fetch_satellite_telemetry
from geoagent_ai.data_sources.seismic_client import fetch_seismic_telemetry
from geoagent_ai.data_sources.terrain_client import fetch_terrain_profile
from geoagent_ai.data_sources.weather_client import fetch_weather_parameters

__all__ = [
    "fetch_weather_parameters",
    "fetch_satellite_telemetry",
    "fetch_terrain_profile",
    "fetch_seismic_telemetry",
    "find_historical_matches",
    "get_region_catalog",
    "resolve_location",
]
