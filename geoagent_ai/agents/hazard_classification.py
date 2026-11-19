"""Agent 2: Disaster Classification Agent — automatic hazard detection via ML + cross-validation."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent_ai.agents.state import GeoAgentState, run_timed_step
from geoagent_ai.ml import HazardPredictor
from geoagent_ai.schemas import DisasterClassification


def build_classification_node(predictor: HazardPredictor):
    """Create LangGraph node bound to a loaded HazardPredictor."""

    def hazard_classification_node(state: GeoAgentState) -> dict:
        def _execute():
            bundle = state["collected_data"]
            assert bundle is not None, "collected_data missing before classification"
            result = predictor.classify(bundle)
            classification = DisasterClassification(**result)
            msg = AIMessage(
                content=(
                    f"[DisasterClassificationAgent] Detected '{classification.detected_disaster}' "
                    f"(confidence={classification.confidence:.2f}, agreement={classification.cross_source_agreement:.2f})."
                )
            )
            return (
                {"classification": classification, "messages": [msg]},
                "completed",
                f"Classified as {classification.detected_disaster} ({classification.confidence * 100:.1f}% confidence)",
                {
                    "detected_disaster": classification.detected_disaster,
                    "confidence": classification.confidence,
                    "cross_source_agreement": classification.cross_source_agreement,
                },
            )

        return run_timed_step(
            "agent-2-hazard-classification",
            "Disaster Classification Agent",
            _execute,
        )

    return hazard_classification_node
