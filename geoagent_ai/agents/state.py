"""Shared LangGraph AgentState and execution timing utilities."""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import BaseMessage

from geoagent_ai.schemas import (
    AgentExecutionLog,
    AlertPayload,
    DisasterClassification,
    GeoAnalysisResult,
    LocationQuery,
    MultiSourceBundle,
    ResponsePlan,
    RiskAssessment,
)


class GeoAgentState(TypedDict):
    """Shared state passed across all nodes in the LangGraph StateGraph."""

    run_id: str
    query: LocationQuery
    catalog_entry: dict[str, Any] | None
    calibration_offset: float
    collected_data: MultiSourceBundle | None
    classification: DisasterClassification | None
    geo_analysis: GeoAnalysisResult | None
    risk_assessment: RiskAssessment | None
    alert: AlertPayload | None
    response_plan: ResponsePlan | None
    agent_logs: Annotated[list[AgentExecutionLog], operator.add]
    messages: Annotated[list[BaseMessage], operator.add]


def run_timed_step(
    agent_id: str,
    agent_name: str,
    fn: Callable[[], tuple[dict[str, Any], str, str, dict[str, Any]]],
) -> dict[str, Any]:
    """Execute an agent node step with consistent latency tracking and fault logging."""
    start = time.perf_counter()
    updates, status, summary, outputs = fn()
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    log_entry = AgentExecutionLog(
        agent_id=agent_id,
        agent_name=agent_name,
        status=status,  # type: ignore[arg-type]
        duration_ms=elapsed_ms,
        summary=summary,
        outputs=outputs,
    )
    updates["agent_logs"] = [log_entry]
    return updates
