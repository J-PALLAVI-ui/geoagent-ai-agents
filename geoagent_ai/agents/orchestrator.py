"""LangGraph multi-agent orchestrator coordinating all 6 autonomous agents."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from geoagent_ai.agents.alert_agent import alert_communication_node
from geoagent_ai.agents.data_collection import data_collection_node
from geoagent_ai.agents.geo_location import geo_location_node
from geoagent_ai.agents.hazard_classification import build_classification_node
from geoagent_ai.agents.response_agent import response_planning_node
from geoagent_ai.agents.risk_prediction import build_risk_prediction_node
from geoagent_ai.agents.state import GeoAgentState
from geoagent_ai.management import AgentManagerMonitor
from geoagent_ai.ml import HazardPredictor
from geoagent_ai.schemas import LocationQuery, PipelineExecutionResult


class GeoAgentOrchestrator:
    """Compiled LangGraph multi-agent system with integrated management & audit monitor."""

    def __init__(self, monitor: AgentManagerMonitor | None = None) -> None:
        self.monitor = monitor or AgentManagerMonitor()
        self.predictor = HazardPredictor()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GeoAgentState)
        builder.add_node("data_collection", data_collection_node)
        builder.add_node("hazard_classification", build_classification_node(self.predictor))
        builder.add_node("geo_location", geo_location_node)
        builder.add_node(
            "risk_prediction", build_risk_prediction_node(self.predictor, self.monitor)
        )
        builder.add_node("alert_communication", alert_communication_node)
        builder.add_node("response_planning", response_planning_node)

        builder.add_edge(START, "data_collection")
        builder.add_edge("data_collection", "hazard_classification")
        builder.add_edge("hazard_classification", "geo_location")
        builder.add_edge("geo_location", "risk_prediction")
        builder.add_edge("risk_prediction", "alert_communication")
        builder.add_edge("alert_communication", "response_planning")
        builder.add_edge("response_planning", END)

        return builder.compile()

    def run(self, query: LocationQuery) -> PipelineExecutionResult:
        """Execute full autonomous multi-agent workflow for a location."""
        t0 = time.perf_counter()
        run_id = f"RUN-{uuid.uuid4().hex[:10].upper()}"
        initial_state: GeoAgentState = {
            "run_id": run_id,
            "query": query,
            "catalog_entry": None,
            "calibration_offset": 0.0,
            "collected_data": None,
            "classification": None,
            "geo_analysis": None,
            "risk_assessment": None,
            "alert": None,
            "response_plan": None,
            "agent_logs": [],
            "messages": [],
        }

        final_state = self.graph.invoke(initial_state)
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        result = PipelineExecutionResult(
            run_id=run_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            query=query,
            collected_data=final_state["collected_data"],
            classification=final_state["classification"],
            geo_analysis=final_state["geo_analysis"],
            risk_assessment=final_state["risk_assessment"],
            alert=final_state["alert"],
            response_plan=final_state["response_plan"],
            agent_logs=final_state["agent_logs"],
            total_duration_ms=total_ms,
        )
        self.monitor.record_run(result)
        return result
