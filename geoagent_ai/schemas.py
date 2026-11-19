"""Typed Pydantic schemas for multi-source data, agent outputs, and API responses."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


DisasterType = Literal["No Risk", "Flood", "Fire", "Cyclone", "Landslide", "Earthquake"]
RiskLevel = Literal["Low", "Medium", "High"]


class LocationQuery(BaseModel):
    """Minimal user input: only location/region is required."""

    location_name: str = Field(..., min_length=1, description="City, region, or place name")
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    mode: Literal["live", "simulation"] = Field(
        default="live",
        description="Live fetches external APIs with safe fallbacks; simulation injects scenario presets",
    )
    scenario: str | None = Field(
        default=None,
        description="Optional scenario preset name when mode='simulation'",
    )


class WeatherParameters(BaseModel):
    rainfall_mm_24h: float = 0.0
    rainfall_mm_3h: float = 0.0
    wind_speed_kmh: float = 0.0
    pressure_hpa: float = 1013.0
    temperature_c: float = 26.0
    humidity_pct: float = 55.0
    condition: str = "Clear"
    source: str = "Open-Meteo / Baseline"


class SatelliteTelemetry(BaseModel):
    ndvi: float = 0.45
    ndwi: float = 0.05
    nbr: float = 0.25
    fire_hotspots_count: int = 0
    frp_mw: float = 0.0
    cloud_cover_pct: float = 20.0
    source: str = "Sentinel-2 / NASA FIRMS Model"


class TerrainProfile(BaseModel):
    elevation_m: float = 100.0
    slope_deg: float = 4.0
    soil_moisture: float = 0.32
    drainage_density: float = 1.2
    land_cover: str = "Mixed Urban/Vegetation"
    source: str = "SRTM DEM"


class SeismicTelemetry(BaseModel):
    seismic_magnitude: float = 1.2
    depth_km: float = 15.0
    recent_events_count: int = 0
    source: str = "USGS Earthquake Feed"


class HistoricalMatch(BaseModel):
    event_id: str
    region: str
    disaster_type: str
    year: int
    severity: str
    casualties_or_impact: str
    distance_km: float


class MultiSourceBundle(BaseModel):
    resolved_location: str
    latitude: float
    longitude: float
    state_or_country: str = ""
    timestamp_utc: str
    weather: WeatherParameters
    satellite: SatelliteTelemetry
    terrain: TerrainProfile
    seismic: SeismicTelemetry
    historical_records: list[HistoricalMatch] = Field(default_factory=list)
    data_sources_used: list[str] = Field(default_factory=list)
    fallbacks_triggered: list[str] = Field(default_factory=list)


class DisasterClassification(BaseModel):
    detected_disaster: DisasterType
    confidence: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
    key_indicators: list[str]
    cross_source_agreement: float = Field(ge=0.0, le=1.0)
    explanation: str


class GeoAnalysisResult(BaseModel):
    center_lat: float
    center_lon: float
    impact_radius_km: float
    affected_area_km2: float
    estimated_exposed_population: int
    zone_classification: str
    affected_polygon_geojson: dict[str, Any]
    hotspot_points_geojson: list[dict[str, Any]] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    severity_score: float = Field(ge=0.0, le=100.0)
    risk_probability_pct: float = Field(ge=0.0, le=100.0)
    model_used: str
    contributing_factors: list[str]
    calibration_applied: float = 0.0


class AlertPayload(BaseModel):
    alert_triggered: bool
    alert_id: str
    severity_level: RiskLevel
    headline: str
    alert_message: str
    affected_area: str
    channels: list[str]
    sms_preview: str
    email_subject: str
    email_body: str
    push_notification: str
    dispatch_status: str
    timestamp_utc: str


class ShelterInfo(BaseModel):
    name: str
    latitude: float
    longitude: float
    distance_km: float
    capacity: int
    type: str


class EvacuationRoute(BaseModel):
    route_id: str
    target_shelter: str
    distance_km: float
    estimated_time_min: int
    safety_status: str
    waypoints: list[list[float]]  # [[lat, lon], ...]


class EmergencyContact(BaseModel):
    agency: str
    number: str
    role: str


class ResponsePlan(BaseModel):
    priority_level: str
    recommended_actions: list[str]
    safety_measures: list[str]
    resource_allocation: list[str]
    nearest_shelters: list[ShelterInfo]
    evacuation_routes: list[EvacuationRoute]
    emergency_contacts: list[EmergencyContact]


class AgentExecutionLog(BaseModel):
    agent_id: str
    agent_name: str
    status: Literal["completed", "fallback", "skipped", "error"]
    duration_ms: float
    summary: str
    outputs: dict[str, Any] = Field(default_factory=dict)


class PipelineExecutionResult(BaseModel):
    run_id: str
    created_at_utc: str
    query: LocationQuery
    collected_data: MultiSourceBundle
    classification: DisasterClassification
    geo_analysis: GeoAnalysisResult
    risk_assessment: RiskAssessment
    alert: AlertPayload
    response_plan: ResponsePlan
    agent_logs: list[AgentExecutionLog]
    total_duration_ms: float


class FeedbackSubmission(BaseModel):
    run_id: str
    location_name: str
    detected_disaster: str
    predicted_risk: RiskLevel
    verdict: Literal["confirmed", "false_alarm", "underestimated", "overestimated"]
    notes: str = ""
