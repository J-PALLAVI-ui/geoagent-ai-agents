"""Agent 4: Risk & Severity Prediction Agent — ML regressor + adaptive feedback calibration."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent_ai.agents.state import GeoAgentState, run_timed_step
from geoagent_ai.management import AgentManagerMonitor
from geoagent_ai.ml import HazardPredictor
from geoagent_ai.schemas import RiskAssessment


def build_risk_prediction_node(predictor: HazardPredictor, monitor: AgentManagerMonitor):
    """Create LangGraph node predicting disaster risk level & severity score."""

    def risk_prediction_node(state: GeoAgentState) -> dict:
        def _execute():
            bundle = state["collected_data"]
            classification = state["classification"]
            assert bundle is not None and classification is not None

            offset = monitor.get_calibration_offset(
                bundle.resolved_location, classification.detected_disaster
            )
            risk_dict = predictor.predict_risk(
                bundle=bundle,
                detected_disaster=classification.detected_disaster,
                calibration_offset=offset,
            )
            assessment = RiskAssessment(**risk_dict)
            msg = AIMessage(
                content=(
                    f"[RiskPredictionAgent] Risk={assessment.risk_level}, "
                    f"Severity={assessment.severity_score}/100, Probability={assessment.risk_probability_pct}%."
                )
            )
            return (
                {
                    "risk_assessment": assessment,
                    "calibration_offset": offset,
                    "messages": [msg],
                },
                "completed",
                f"{assessment.risk_level} Risk ({assessment.severity_score}/100, {assessment.risk_probability_pct}%)",
                {
                    "risk_level": assessment.risk_level,
                    "severity_score": assessment.severity_score,
                    "risk_probability_pct": assessment.risk_probability_pct,
                },
            )

        return run_timed_step("agent-4-risk-prediction", "Risk Prediction Agent", _execute)

    return risk_prediction_node
