"""Agent 5: Alert & Communication Agent — issues warnings if risk is Medium or High."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent_ai.agents.state import GeoAgentState, run_timed_step
from geoagent_ai.notifications import dispatch_alert_notifications


def alert_communication_node(state: GeoAgentState) -> dict:
    """Trigger automated multi-channel alerts when Risk Level is Medium or High."""

    def _execute():
        bundle = state["collected_data"]
        classification = state["classification"]
        geo = state["geo_analysis"]
        risk = state["risk_assessment"]
        assert bundle and classification and geo and risk

        alert = dispatch_alert_notifications(
            location_name=bundle.resolved_location,
            detected_disaster=classification.detected_disaster,
            risk_level=risk.risk_level,
            severity_score=risk.severity_score,
            affected_area_km2=geo.affected_area_km2,
            key_indicators=classification.key_indicators,
        )
        msg = AIMessage(
            content=(
                f"[AlertAgent] {'TRIGGERED' if alert.alert_triggered else 'SUPPRESSED'}: "
                f"{alert.headline} ({alert.dispatch_status})."
            )
        )
        return (
            {"alert": alert, "messages": [msg]},
            "completed" if alert.alert_triggered else "skipped",
            alert.headline,
            {
                "alert_triggered": alert.alert_triggered,
                "channels": alert.channels,
                "dispatch_status": alert.dispatch_status,
            },
        )

    return run_timed_step(
        "agent-5-alert-communication",
        "Alert & Communication Agent",
        _execute,
    )
