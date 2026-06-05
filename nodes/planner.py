"""
nodes/planner.py
─────────────────────────────────────────────────────────────────────────────
Planner Node — Entry point of the LangGraph execution.

Responsibilities:
  1. Validate and parse the incoming raw anomaly alert JSON.
  2. Assign a globally unique trace_id for Langfuse observability.
  3. Initialise all list fields that parallel nodes will append to.
  4. Log the execution metadata for dashboards.

Returns a partial EcoGuardState delta (only the fields it sets).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from state import (
    NodeExecutionMeta,
    NodeStatus,
    RawAlert,
    EcoGuardState,
    OperationalMode,
)

logger = logging.getLogger(__name__)


async def planner_node(state: EcoGuardState) -> dict[str, Any]:
    """
    LangGraph node: Planner (Evidence Collector / Agent 0).

    Args:
        state: Current EcoGuardState (only `raw_alert` is expected at this stage).

    Returns:
        Partial state delta initialising core fields for downstream nodes.

    Raises:
        ValueError:          If raw_alert is missing from the state.
        pydantic.ValidationError: If the alert payload violates the RawAlert schema.
    """
    start_ms = time.monotonic() * 1000
    node_name = "planner_evidence_collector"
    logger.info("[%s] Starting — received state keys: %s", node_name, list(state.keys()))

    # ── 1. Validate raw_alert presence ────────────────────────────────────────
    raw_alert_dict: dict[str, Any] | None = state.get("raw_alert")
    if not raw_alert_dict:
        raise ValueError("EcoGuardState.raw_alert is missing — cannot proceed.")

    # ── 2. Parse and validate against RawAlert Pydantic schema ───────────────
    try:
        alert = RawAlert(**raw_alert_dict)
        logger.info(
            "[%s] Alert validated: id=%s severity=%s source=%s",
            node_name, alert.alert_id, alert.severity, alert.raw_payload.get('source_system', 'Unknown'),
        )
    except ValidationError as exc:
        logger.error("[%s] RawAlert schema validation failed:\n%s", node_name, exc)
        raise

    # ── 3. Determine Operational Mode ─────────────────────────────────────────
    # Heuristic based on incident type
    incident_type = alert.raw_payload.get("incident_type", "Unknown")
    
    if incident_type in ["Water Contamination", "Resource Waste"]:
        mode = OperationalMode.CIRCULAR_ECONOMY_INTELLIGENCE
    elif incident_type in ["Environmental Hazard", "Sustainability Violation", "Illegal Waste Dumping"]:
        mode = OperationalMode.ENVIRONMENTAL_INTELLIGENCE
    else:
        mode = OperationalMode.COMMUNITY_INTELLIGENCE
        
    logger.info("[%s] Selected Operational Mode: %s", node_name, mode.value)

    # ── 4. Assign immutable trace_id ──────────────────────────────────────────
    trace_id = str(uuid.uuid4())

    # ── 5. Record execution metadata ──────────────────────────────────────────
    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=node_name,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info("[%s] Completed in %.1f ms — trace_id=%s", node_name, duration_ms, trace_id)

    return {
        # Persist the validated, normalised alert (with auto-assigned alert_id)
        "raw_alert": alert.model_dump(),

        # Immutable trace identifier for this incident
        "trace_id": trace_id,
        
        # Operational Mode
        "operational_mode": mode.value,

        # Initialise append-safe lists for parallel node outputs
        "agent_scores": [],
        "node_execution_log": [exec_meta.model_dump()],

        # Reset consensus flags for this run (NOT hitl_decision — that is
        # owned by the HITL gate and must not be clobbered here, since
        # callers may pre-seed it for automated/test execution)
        "hitl_required": False,
    }
