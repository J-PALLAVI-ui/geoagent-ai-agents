"""Streamlit + Folium dashboard matching Slide 11 tools of GeoAgent AI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import streamlit.components.v1 as components

from geoagent_ai.agents import GeoAgentOrchestrator
from geoagent_ai.data_sources import get_region_catalog
from geoagent_ai.data_sources.weather_client import SCENARIO_WEATHER
from geoagent_ai.geo import build_folium_map
from geoagent_ai.management import AgentManagerMonitor
from geoagent_ai.schemas import FeedbackSubmission, LocationQuery

st.set_page_config(
    page_title="GeoAgent AI — Autonomous Disaster Monitoring",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_resource
def load_system():
    monitor = AgentManagerMonitor()
    orchestrator = GeoAgentOrchestrator(monitor=monitor)
    return orchestrator, monitor


orchestrator, monitor = load_system()

st.title("🛰️ GeoAgent AI: Autonomous Disaster Monitoring and Response System")
st.caption(
    "Multi-Agent LangGraph System · Scikit-Learn/XGBoost Ensemble · Folium Geospatial Mapping"
)

with st.sidebar:
    st.header("Location Surveillance Input")
    catalog = get_region_catalog()
    region_names = [r["name"] for r in catalog]
    selected_preset = st.selectbox("Quick Region Preset", region_names, index=1)
    custom_location = st.text_input("Or Enter Any Location / City", value=selected_preset)
    mode = st.radio("Data Ingestion Mode", ["simulation", "live"], index=0)
    scenario_opt = st.selectbox(
        "Scenario Override (Optional)",
        ["Auto-Detect from Telemetry", *list(SCENARIO_WEATHER.keys())],
    )
    run_clicked = st.button("Run Autonomous Agents", type="primary", use_container_width=True)

if "last_result" not in st.session_state or run_clicked:
    query = LocationQuery(
        location_name=custom_location.strip() or selected_preset,
        mode=mode,  # type: ignore[arg-type]
        scenario=None if scenario_opt == "Auto-Detect from Telemetry" else scenario_opt,
    )
    with st.spinner("Coordinating 6 autonomous GeoAgent nodes..."):
        st.session_state["last_result"] = orchestrator.run(query)

res = st.session_state["last_result"]

# Agent Pipeline Strip
st.subheader("Autonomous Multi-Agent Pipeline Trace")
cols = st.columns(len(res.agent_logs))
for col, log in zip(cols, res.agent_logs):
    with col:
        st.metric(label=log.agent_name.replace(" Agent", ""), value=f"{log.duration_ms} ms", delta=log.status.upper())
        st.caption(log.summary)

# KPIs
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Detected Disaster", res.classification.detected_disaster, f"{res.classification.confidence * 100:.1f}% conf")
k2.metric("Risk Level", res.risk_assessment.risk_level, f"{res.risk_assessment.risk_probability_pct}% prob")
k3.metric("Severity Score", f"{res.risk_assessment.severity_score:.1f} / 100")
k4.metric("Exposed Area / Pop", f"{res.geo_analysis.affected_area_km2} km²", f"~{res.geo_analysis.estimated_exposed_population:,} people")

if res.alert.alert_triggered:
    if res.alert.severity_level == "High":
        st.error(f"🚨 **{res.alert.headline}** — {res.alert.alert_message}")
    else:
        st.warning(f"⚠️ **{res.alert.headline}** — {res.alert.alert_message}")
else:
    st.success(f"✅ **{res.alert.headline}** — {res.alert.alert_message}")

left_col, right_col = st.columns([1.2, 1.0])

with left_col:
    st.subheader("Folium Geospatial Risk & Evacuation Map")
    try:
        fmap = build_folium_map(
            geo=res.geo_analysis,
            risk_level=res.risk_assessment.risk_level,
            detected_disaster=res.classification.detected_disaster,
            shelters=res.response_plan.nearest_shelters,
            routes=res.response_plan.evacuation_routes,
        )
        components.html(fmap._repr_html_(), height=460)
    except ImportError:
        st.info("Install `folium` (`pip install folium`) to display the interactive Folium map.")

    st.subheader("Multi-Source Telemetry")
    t1, t2 = st.columns(2)
    t1.json(res.collected_data.weather.model_dump())
    t2.json(res.collected_data.satellite.model_dump())

with right_col:
    tab1, tab2, tab3 = st.tabs(["Response & Shelters", "Alerts & Indicators", "Agent Ops & Feedback"])
    with tab1:
        st.markdown(f"**Priority:** `{res.response_plan.priority_level}`")
        st.markdown("**Recommended Actions**")
        for act in res.response_plan.recommended_actions:
            st.write(f"- {act}")
        st.markdown("**Nearest Shelters & Routes**")
        for s in res.response_plan.nearest_shelters:
            st.write(f"- **{s.name}** ({s.type}) — `{s.distance_km} km`, cap `{s.capacity}`")
        st.markdown("**Emergency Contacts**")
        for c in res.response_plan.emergency_contacts:
            st.write(f"- {c.agency}: `{c.number}` ({c.role})")

    with tab2:
        st.markdown("**Explainable AI Reasoning**")
        st.write(res.classification.explanation)
        for ind in res.classification.key_indicators:
            st.write(f"- {ind}")
        st.markdown("**SMS Preview**")
        st.code(res.alert.sms_preview)
        st.markdown("**Email Advisory**")
        st.code(res.alert.email_body)

    with tab3:
        st.markdown("**Authority Feedback Loop**")
        fb_col1, fb_col2 = st.columns(2)
        if fb_col1.button("Confirm Hazard"):
            out = monitor.submit_feedback(
                FeedbackSubmission(
                    run_id=res.run_id,
                    location_name=res.collected_data.resolved_location,
                    detected_disaster=res.classification.detected_disaster,
                    predicted_risk=res.risk_assessment.risk_level,
                    verdict="confirmed",
                )
            )
            st.success(f"Calibration updated: {out['updated_calibration_offset']}")
        if fb_col2.button("Mark False Alarm"):
            out = monitor.submit_feedback(
                FeedbackSubmission(
                    run_id=res.run_id,
                    location_name=res.collected_data.resolved_location,
                    detected_disaster=res.classification.detected_disaster,
                    predicted_risk=res.risk_assessment.risk_level,
                    verdict="false_alarm",
                )
            )
            st.warning(f"Calibration updated: {out['updated_calibration_offset']}")

        st.markdown("**Agent Monitoring Summary**")
        st.json(monitor.get_monitoring_summary())
