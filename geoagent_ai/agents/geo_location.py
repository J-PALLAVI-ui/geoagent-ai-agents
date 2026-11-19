"""Agent 3: Geo-Location Agent — spatial footprint, area, and population exposure analysis."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent_ai.agents.state import GeoAgentState, run_timed_step
from geoagent_ai.geo import analyze_geospatial_impact


def geo_location_node(state: GeoAgentState) -> dict:
    """Compute affected GeoJSON polygon, radius, area, and exposed population."""

    def _execute():
        bundle = state["collected_data"]
        classification = state["classification"]
        assert bundle is not None and classification is not None

        geo = analyze_geospatial_impact(
            bundle=bundle,
            detected_disaster=classification.detected_disaster,
            catalog_entry=state.get("catalog_entry"),
        )
        msg = AIMessage(
            content=(
                f"[GeoLocationAgent] Mapped {geo.affected_area_km2} km² impact zone "
                f"({geo.zone_classification}); ~{geo.estimated_exposed_population:,} estimated exposed."
            )
        )
        return (
            {"geo_analysis": geo, "messages": [msg]},
            "completed",
            f"Mapped {geo.impact_radius_km} km radius ({geo.affected_area_km2} km²)",
            {
                "impact_radius_km": geo.impact_radius_km,
                "affected_area_km2": geo.affected_area_km2,
                "estimated_exposed_population": geo.estimated_exposed_population,
            },
        )

    return run_timed_step("agent-3-geo-location", "Geo-Location Agent", _execute)
