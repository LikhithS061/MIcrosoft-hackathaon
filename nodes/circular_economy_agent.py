"""
nodes/circular_economy_agent.py
─────────────────────────────────────────────────────────────────────────────
Circular Economy Agent Node — Parallel Branch (Agent 7).

Domain: Symbiosis, waste-to-value, circular economy opportunities.

Outputs:
  - circular_opportunity_report: Structured report on opportunities.
  - symbiosis_index: Opportunity score.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from state import (
    NodeExecutionMeta,
    NodeStatus,
    EcoGuardState,
    CircularOpportunityReport,
    OperationalMode,
)

logger = logging.getLogger(__name__)

_AGENT_ID = "circular_economy_agent"


async def circular_economy_node(state: EcoGuardState) -> dict[str, Any]:
    """LangGraph node: Circular Economy Agent."""
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    mode = state.get("operational_mode", OperationalMode.COMMUNITY_INTELLIGENCE.value)
    logger.info("[%s] Starting — trace_id=%s mode=%s", _AGENT_ID, trace_id, mode)

    raw_alert: dict[str, Any] = state["raw_alert"]
    incident_type = raw_alert.get("raw_payload", {}).get("incident_type", "Unknown")
    payload = raw_alert.get("raw_payload", {})
    
    # Calculate Opportunity via heuristic (Mock mode logic)
    symbiosis_index = 0.0
    recipients = []
    benefits = "None identified"
    env_savings = "N/A"
    economic_value = "N/A"
    opp_type = "None"
    
    # Simple heuristics based on incident type to simulate opportunity discovery
    if incident_type == "Resource Waste" or incident_type == "Water Contamination":
        if payload.get("water_loss_liters_per_day", 0) > 0 or payload.get("ph_level", 0) > 0:
            symbiosis_index = 85.0
            recipients = ["Community Landscaping", "Local Industrial Cooling"]
            benefits = "Reuse treated wastewater for non-potable needs."
            env_savings = "Saves 10,000+ Liters of fresh water daily."
            economic_value = "₹15,000/month reduction in municipal water cost."
            opp_type = "Wastewater Reuse"
            
    elif incident_type == "Fire Hazard" or payload.get("temperature_rise_c", 0) > 0:
        symbiosis_index = 72.0
        recipients = ["Community Heating", "Drying Facilities"]
        benefits = "Waste heat recovery for localized heating needs."
        env_savings = "Offsets 500 kWh of fossil-fuel heating."
        economic_value = "₹8,000/month energy savings."
        opp_type = "Waste Heat Recovery"
        
    elif incident_type == "Illegal Waste Dumping" or payload.get("waste_overflow_pct", 0) > 50:
        symbiosis_index = 78.0
        recipients = ["Composting Plant", "Recycling Partners"]
        benefits = "Convert organic waste to compost."
        env_savings = "Reduces landfill usage by 4 tons."
        economic_value = "Sale of compost at ₹1,200/ton."
        opp_type = "Organic Waste Composting"

    # If we are explicitly in CIRCULAR mode, boost the score
    if mode == OperationalMode.CIRCULAR_ECONOMY_INTELLIGENCE.value and symbiosis_index > 0:
        symbiosis_index = min(100.0, symbiosis_index + 10.0)

    report = CircularOpportunityReport(
        symbiosis_index=symbiosis_index,
        potential_recipients=recipients,
        estimated_benefits=benefits,
        environmental_savings=env_savings,
        economic_value=economic_value,
        opportunity_type=opp_type
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_AGENT_ID,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info("[%s] Done — SymbiosisIndex=%.1f  %.0fms",
                _AGENT_ID, symbiosis_index, duration_ms)

    return {
        "circular_opportunity_report": report.model_dump(),
        "symbiosis_index": symbiosis_index,
        "node_execution_log": [exec_meta.model_dump()],
    }
