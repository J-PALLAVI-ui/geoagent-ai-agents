"""Spatial impact polygon, population exposure, shelter matching, and evacuation routing."""

from __future__ import annotations

import math
from typing import Any

from geoagent_ai.data_sources.historical_store import haversine_km
from geoagent_ai.schemas import (
    EvacuationRoute,
    GeoAnalysisResult,
    MultiSourceBundle,
    ShelterInfo,
)

DISASTER_BASE_RADIUS_KM = {
    "No Risk": 3.5,
    "Landslide": 5.5,
    "Fire": 9.0,
    "Flood": 14.0,
    "Earthquake": 26.0,
    "Cyclone": 40.0,
}


def _offset_lat_lon(lat: float, lon: float, dx_km: float, dy_km: float) -> tuple[float, float]:
    """Offset lat/lon by dx (east) and dy (north) kilometers."""
    d_lat = dy_km / 111.32
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    d_lon = dx_km / (111.32 * cos_lat)
    return round(lat + d_lat, 5), round(lon + d_lon, 5)


def _circle_polygon_geojson(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    disaster_type: str,
    steps: int = 32,
) -> dict[str, Any]:
    """Generate an organic hazard footprint GeoJSON Feature."""
    coords: list[list[float]] = []
    # Elliptical stretch for cyclones/floods/landslides to represent directional corridor
    stretch_x = 1.25 if disaster_type in {"Cyclone", "Flood"} else 1.0
    stretch_y = 0.85 if disaster_type == "Landslide" else 1.0

    for i in range(steps + 1):
        angle = (2.0 * math.pi * i) / steps
        # Slight deterministic harmonic perturbation for realistic hazard perimeter
        wobble = 1.0 + (0.08 * math.sin(3 * angle) if disaster_type != "No Risk" else 0.0)
        dx = math.cos(angle) * radius_km * stretch_x * wobble
        dy = math.sin(angle) * radius_km * stretch_y * wobble
        p_lat, p_lon = _offset_lat_lon(center_lat, center_lon, dx, dy)
        coords.append([p_lon, p_lat])

    return {
        "type": "Feature",
        "properties": {"disaster_type": disaster_type, "radius_km": radius_km},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def _generate_hotspot_points(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    disaster_type: str,
    bundle: MultiSourceBundle,
) -> list[dict[str, Any]]:
    """Create geospatial sub-feature points (e.g. NASA FIRMS fires, river gauges, epicenter)."""
    if disaster_type == "No Risk":
        return []

    offsets = [
        (0.0, 0.0, "Primary Hazard Core"),
        (0.32, 0.25, "Secondary Sensor Alert"),
        (-0.28, 0.30, "Perimeter Telemetry Node"),
        (0.22, -0.34, "Monitoring Station"),
    ]
    if disaster_type == "Fire" and bundle.satellite.fire_hotspots_count > 10:
        offsets.append((-0.42, -0.22, "NASA FIRMS Thermal Cluster"))

    points: list[dict[str, Any]] = []
    for idx, (fx, fy, label) in enumerate(offsets):
        p_lat, p_lon = _offset_lat_lon(center_lat, center_lon, fx * radius_km, fy * radius_km)
        points.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"NODE-{idx + 1}",
                    "label": label,
                    "disaster_type": disaster_type,
                },
                "geometry": {"type": "Point", "coordinates": [p_lon, p_lat]},
            }
        )
    return points


def analyze_geospatial_impact(
    bundle: MultiSourceBundle,
    detected_disaster: str,
    catalog_entry: dict[str, Any] | None = None,
) -> GeoAnalysisResult:
    """Compute affected polygon, exposed area, population exposure, and zone class."""
    base_r = DISASTER_BASE_RADIUS_KM.get(detected_disaster, 8.0)

    # Scale footprint dynamically based on observed telemetry intensity
    scale = 1.0
    if detected_disaster == "Flood":
        scale = 0.8 + min(0.8, bundle.weather.rainfall_mm_24h / 300.0)
    elif detected_disaster == "Cyclone":
        scale = 0.8 + min(0.75, bundle.weather.wind_speed_kmh / 180.0)
    elif detected_disaster == "Fire":
        scale = 0.8 + min(0.7, bundle.satellite.fire_hotspots_count / 50.0)
    elif detected_disaster == "Earthquake":
        scale = 0.7 + min(1.0, max(0.0, bundle.seismic.seismic_magnitude - 4.0) / 3.5)

    radius_km = round(base_r * scale, 2)
    area_km2 = round(math.pi * (radius_km ** 2), 2)
    pop_density = float(catalog_entry.get("population_density_per_km2", 650)) if catalog_entry else 650.0

    exposure_factor = 0.05 if detected_disaster == "No Risk" else 0.42
    exposed_pop = int(area_km2 * pop_density * exposure_factor)

    zone_class = (
        catalog_entry["zone_type"]
        if catalog_entry
        else ("Highland Terrain" if bundle.terrain.elevation_m > 500 else "Lowland Basin")
    )

    polygon = _circle_polygon_geojson(
        bundle.latitude, bundle.longitude, radius_km, detected_disaster
    )
    hotspots = _generate_hotspot_points(
        bundle.latitude, bundle.longitude, radius_km, detected_disaster, bundle
    )

    return GeoAnalysisResult(
        center_lat=bundle.latitude,
        center_lon=bundle.longitude,
        impact_radius_km=radius_km,
        affected_area_km2=area_km2,
        estimated_exposed_population=exposed_pop,
        zone_classification=zone_class,
        affected_polygon_geojson=polygon,
        hotspot_points_geojson=hotspots,
    )


def compute_shelters_and_routes(
    center_lat: float,
    center_lon: float,
    impact_radius_km: float,
    detected_disaster: str,
    catalog_entry: dict[str, Any] | None = None,
) -> tuple[list[ShelterInfo], list[EvacuationRoute]]:
    """Rank nearest emergency shelters and generate multi-waypoint evacuation corridors."""
    raw_shelters = list(catalog_entry.get("shelters", [])) if catalog_entry else []

    if not raw_shelters:
        safe_dist = max(6.0, impact_radius_km * 1.15)
        bearings = [(0.8, 0.7, "North-East Safe Relief Camp", "District Stadium", 1800),
                    (-0.9, 0.5, "North-West High-Ground Shelter", "Community Hub", 1200),
                    (0.4, -0.95, "South Regional Evacuation Center", "Civil Complex", 2200)]
        for bx, by, name, stype, cap in bearings:
            s_lat, s_lon = _offset_lat_lon(center_lat, center_lon, bx * safe_dist, by * safe_dist)
            raw_shelters.append(
                {
                    "name": name,
                    "latitude": s_lat,
                    "longitude": s_lon,
                    "capacity": cap,
                    "type": stype,
                }
            )

    shelters: list[ShelterInfo] = []
    for item in raw_shelters:
        dist = haversine_km(center_lat, center_lon, item["latitude"], item["longitude"])
        shelters.append(
            ShelterInfo(
                name=item["name"],
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                distance_km=dist,
                capacity=int(item["capacity"]),
                type=item["type"],
            )
        )
    shelters.sort(key=lambda s: s.distance_km)

    routes: list[EvacuationRoute] = []
    avg_speed_kmh = 26.0 if detected_disaster in {"Flood", "Landslide", "Cyclone"} else 35.0
    for idx, shelter in enumerate(shelters[:3]):
        # Midpoint waypoint slightly offset perpendicular to path to avoid direct core hazard crossing
        mid_lat = (center_lat + shelter.latitude) / 2.0
        mid_lon = (center_lon + shelter.longitude) / 2.0
        perp_lat = -(shelter.longitude - center_lon) * 0.14
        perp_lon = (shelter.latitude - center_lat) * 0.14
        wp_mid = [round(mid_lat + perp_lat, 5), round(mid_lon + perp_lon, 5)]

        road_dist = round(shelter.distance_km * 1.18, 2)
        eta_min = max(6, int(round((road_dist / avg_speed_kmh) * 60.0)))
        safety_status = (
            "Primary Open Corridor"
            if idx == 0
            else ("Alternate Safe Corridor" if idx == 1 else "Heavy Vehicle Staging Route")
        )
        routes.append(
            EvacuationRoute(
                route_id=f"EVAC-{idx + 1}",
                target_shelter=shelter.name,
                distance_km=road_dist,
                estimated_time_min=eta_min,
                safety_status=safety_status,
                waypoints=[
                    [round(center_lat, 5), round(center_lon, 5)],
                    wp_mid,
                    [round(shelter.latitude, 5), round(shelter.longitude, 5)],
                ],
            )
        )

    return shelters, routes


def build_folium_map(
    geo: GeoAnalysisResult,
    risk_level: str,
    detected_disaster: str,
    shelters: list[ShelterInfo],
    routes: list[EvacuationRoute],
) -> Any:
    """Build an interactive Folium map (used by Streamlit dashboard when folium is installed)."""
    import folium  # Imported lazily so core API remains dependency-light

    color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
    color = color_map.get(risk_level, "#3b82f6")

    fmap = folium.Map(location=[geo.center_lat, geo.center_lon], zoom_start=11, tiles="CartoDB positron")

    folium.GeoJson(
        geo.affected_polygon_geojson,
        name="Affected Hazard Zone",
        style_function=lambda _: {
            "fillColor": color,
            "color": color,
            "weight": 2.5,
            "fillOpacity": 0.25,
        },
        tooltip=f"{detected_disaster} ({risk_level} Risk) — {geo.affected_area_km2} km²",
    ).add_to(fmap)

    folium.Marker(
        [geo.center_lat, geo.center_lon],
        popup=f"<b>Monitored Location</b><br>Hazard: {detected_disaster}<br>Risk: {risk_level}",
        icon=folium.Icon(color="red" if risk_level == "High" else ("orange" if risk_level == "Medium" else "green"), icon="info-sign"),
    ).add_to(fmap)

    for shelter in shelters:
        folium.Marker(
            [shelter.latitude, shelter.longitude],
            popup=f"<b>{shelter.name}</b><br>Type: {shelter.type}<br>Capacity: {shelter.capacity}<br>Distance: {shelter.distance_km} km",
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(fmap)

    for route in routes:
        folium.PolyLine(
            locations=route.waypoints,
            color="#2563eb",
            weight=3.5,
            dash_array="6, 6",
            tooltip=f"{route.route_id}: {route.target_shelter} ({route.distance_km} km, ~{route.estimated_time_min} min)",
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap
