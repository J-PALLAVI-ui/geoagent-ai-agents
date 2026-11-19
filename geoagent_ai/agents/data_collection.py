"""Agent 1: Data Collection Agent — multi-source ingestion with fault-tolerant fallbacks."""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from geoagent_ai.agents.state import GeoAgentState, run_timed_step
from geoagent_ai.data_sources import (
    fetch_satellite_telemetry,
    fetch_seismic_telemetry,
    fetch_terrain_profile,
    fetch_weather_parameters,
    find_historical_matches,
    resolve_location,
)
from geoagent_ai.schemas import MultiSourceBundle


def data_collection_node(state: GeoAgentState) -> dict:
    """Collect weather, satellite/FIRMS, SRTM terrain, USGS seismic & historical data."""

    def _execute():
        query = state["query"]
        allow_net = query.mode == "live"
        (
            resolved_name,
            lat,
            lon,
            admin,
            catalog_entry,
            geo_fallback,
        ) = resolve_location(
            query.location_name,
            query.latitude,
            query.longitude,
            allow_network=allow_net,
        )

        scenario = query.scenario
        if query.mode == "simulation" and not scenario:
            scenario = (catalog_entry or {}).get("default_scenario", "no_risk")

        fallbacks: list[str] = [geo_fallback] if geo_fallback else []

        weather, w_fb = fetch_weather_parameters(lat, lon, mode=query.mode, scenario=scenario)
        if w_fb:
            fallbacks.append(w_fb)

        satellite, sat_fb = fetch_satellite_telemetry(
            lat, lon, weather, mode=query.mode, scenario=scenario
        )
        if sat_fb:
            fallbacks.append(sat_fb)

        terrain, ter_fb = fetch_terrain_profile(
            lat, lon, weather, catalog_entry=catalog_entry, mode=query.mode, scenario=scenario
        )
        if ter_fb:
            fallbacks.append(ter_fb)

        seismic, sei_fb = fetch_seismic_telemetry(lat, lon, mode=query.mode, scenario=scenario)
        if sei_fb:
            fallbacks.append(sei_fb)

        historical = find_historical_matches(lat, lon)
        sources_used = [
            weather.source,
            satellite.source,
            terrain.source,
            seismic.source,
            f"Historical Disaster Store ({len(historical)} nearby events)",
        ]

        bundle = MultiSourceBundle(
            resolved_location=resolved_name,
            latitude=lat,
            longitude=lon,
            state_or_country=admin,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            weather=weather,
            satellite=satellite,
            terrain=terrain,
            seismic=seismic,
            historical_records=historical,
            data_sources_used=sources_used,
            fallbacks_triggered=fallbacks,
        )

        status = "fallback" if fallbacks else "completed"
        summary = (
            f"Ingested 5 data streams for {resolved_name} ({lat:.3f}, {lon:.3f}); "
            f"{len(fallbacks)} fallback(s)."
        )
        msg = AIMessage(
            content=(
                f"[DataCollectionAgent] Collected weather (rain={weather.rainfall_mm_24h}mm, "
                f"wind={weather.wind_speed_kmh}km/h), FIRMS hotspots={satellite.fire_hotspots_count}, "
                f"slope={terrain.slope_deg}°, seismic=M{seismic.seismic_magnitude}."
            )
        )
        return (
            {
                "collected_data": bundle,
                "catalog_entry": catalog_entry,
                "messages": [msg],
            },
            status,
            summary,
            {
                "resolved_location": resolved_name,
                "fallbacks_count": len(fallbacks),
                "historical_matches": len(historical),
            },
        )

    return run_timed_step("agent-1-data-collection", "Data Collection Agent", _execute)
