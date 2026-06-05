"""
nodes/sustainability_agent.py
─────────────────────────────────────────────────────────────────────────────
Sustainability Agent Node — Parallel Branch C (Agent 3).

Domain: Environmental impact, resource loss, ecological sustainability.
Domain Weight (w_i): 0.35

Live mode (USE_MOCK_LLM=false):
  → ChatGroq with SustainabilityExtraction structured output.

Mock mode (USE_MOCK_LLM=true):
  → Deterministic heuristic based on incident type and severity.

Outputs:
  - hazard_score:       Environmental hazard score C_i ∈ [0, 100]
  - sustainability_score: Dedicated sustainability metric (stored separately)
  - confidence:         α_i ∈ [0, 1]
  - extended_report:    environmental_damage_level, resource_loss_estimate
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from nodes.llm_client import (
    USE_MOCK_LLM,
    SustainabilityExtraction,
    build_structured_llm,
)
from state import (
    AgentScore,
    CitationLink,
    DOMAIN_WEIGHTS,
    NodeExecutionMeta,
    NodeStatus,
    EcoGuardState,
)

logger = logging.getLogger(__name__)

_AGENT_ID = "sustainability_agent"
_AGENT_TYPE = "Environmental Sustainability Analyst"
_DOMAIN_WEIGHT = DOMAIN_WEIGHTS[_AGENT_ID]

_structured_llm = build_structured_llm(SustainabilityExtraction)

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Sustainability Agent for ECO GUARD, a multi-agent safety platform.

Your domain weight is {domain_weight}. You assess ENVIRONMENTAL and SUSTAINABILITY impacts.

SUSTAINABILITY SCORE (0–100):
  0–20   → Negligible environmental impact
  21–40  → Minor, localized resource impact
  41–60  → Moderate damage, resource recovery needed
  61–80  → Significant environmental degradation
  81–100 → Severe/irreversible ecological damage

Assess:
  • Water resource loss or contamination volume
  • Air quality degradation (PM2.5, NOx, gas emissions)
  • Solid/hazardous waste environmental impact
  • Long-term soil/groundwater contamination risk
  • Sustainability policy compliance

CONFIDENCE (0.0–1.0): Base 0.80. Reduce if environmental data is sparse.\
"""

_HUMAN_PROMPT = """\
Alert Payload:
{alert_data}

Produce your structured environmental sustainability assessment now.\
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_PROMPT),
])

# Mock scoring tables
_SUSTAINABILITY_BY_TYPE: dict[str, float] = {
    "Water Contamination":    75.0,
    "Illegal Waste Dumping":  82.0,
    "Sustainability Violation": 85.0,
    "Environmental Hazard":   78.0,
    "Fire Hazard":            60.0,
    "Gas Leak":               55.0,
    "Flooding":               50.0,
    "Electrical Fault":       35.0,
    "Infrastructure Failure": 25.0,
    "Resource Waste":         68.0,
    "Public Safety":          15.0,
}

_SEVERITY_MODIFIER: dict[str, float] = {
    "LOW": -10, "MEDIUM": 0, "HIGH": 5, "CRITICAL": 12, "CATASTROPHIC": 18,
}


async def sustainability_node(state: EcoGuardState) -> dict[str, Any]:
    """LangGraph node: Sustainability Agent (parallel branch C)."""
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting — trace_id=%s", _AGENT_ID, trace_id)

    raw_alert: dict[str, Any] = state["raw_alert"]
    incident_type = raw_alert.get("raw_payload", {}).get("incident_type", "Unknown")

    if not USE_MOCK_LLM:
        result = await _score_via_groq(raw_alert)
        hazard_score = result.sustainability_score
        confidence = result.confidence
        rationale = result.rationale
        resource_loss = result.resource_loss_estimate
        damage_level = result.environmental_damage_level
    else:
        hazard_score, confidence, rationale, resource_loss, damage_level = _score_mock(
            raw_alert, incident_type
        )

    citations = _build_citations(incident_type)

    score = AgentScore(
        agent_id=_AGENT_ID,
        agent_type=_AGENT_TYPE,
        hazard_score=hazard_score,
        confidence=confidence,
        domain_weight=_DOMAIN_WEIGHT,
        rationale=rationale,
        citations=citations,
        status=NodeStatus.COMPLETED,
        extended_report={
            "sustainability_score": hazard_score,
            "resource_loss_estimate": resource_loss,
            "environmental_damage_level": damage_level,
        },
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_AGENT_ID,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info("[%s] Done — C_i=%.1f α_i=%.3f  %.0fms",
                _AGENT_ID, hazard_score, confidence, duration_ms)

    return {
        "agent_scores": [score.model_dump()],
        "sustainability_score": round(hazard_score, 2),
        "environmental_impact_report": rationale,
        "node_execution_log": [exec_meta.model_dump()],
    }


async def _score_via_groq(raw_alert: dict[str, Any]) -> SustainabilityExtraction:
    try:
        chain = _PROMPT | _structured_llm
        return await chain.ainvoke({
            "domain_weight": _DOMAIN_WEIGHT,
            "alert_data": json.dumps(raw_alert, indent=2, default=str),
        })
    except Exception as exc:
        logger.error("[%s] Groq failure: %s — using fallback.", _AGENT_ID, exc)
        return SustainabilityExtraction(
            sustainability_score=50.0,
            confidence=0.10,
            rationale=f"[DEGRADED] Groq API failure: {exc}. Manual review required.",
            resource_loss_estimate="Unknown",
            environmental_damage_level="MODERATE",
        )


def _score_mock(
    raw_alert: dict[str, Any],
    incident_type: str,
) -> tuple[float, float, str, str, str]:
    severity = raw_alert.get("severity", "MEDIUM")
    payload = raw_alert.get("raw_payload", {})

    base = _SUSTAINABILITY_BY_TYPE.get(incident_type, 40.0)
    modifier = _SEVERITY_MODIFIER.get(severity.upper(), 0)
    score = max(0.0, min(100.0, base + modifier))

    # Special indicators
    resource_loss = "Unknown"
    damage_level = "MODERATE"

    if payload.get("ph_level", 0) > 10:
        score = min(100.0, score + 8)
        resource_loss = f"pH {payload.get('ph_level')} — alkaline contamination of water supply"
        damage_level = "HIGH"
    elif payload.get("water_loss_liters_per_day"):
        wl = payload["water_loss_liters_per_day"]
        resource_loss = f"{wl:,} liters/day water waste"
        damage_level = "MODERATE"
    elif payload.get("pm25_level", 0) > 150:
        score = min(100.0, score + 5)
        resource_loss = f"PM2.5: {payload.get('pm25_level')} μg/m³ (Hazardous threshold: 150)"
        damage_level = "HIGH"
    elif payload.get("waste_volume_tons", 0) > 0:
        resource_loss = f"{payload['waste_volume_tons']} tons hazardous waste dumped"
        damage_level = "HIGH"

    confidence = 0.78
    rationale = (
        f"[MOCK] Sustainability analysis for {incident_type} (severity={severity}). "
        f"Base score: {base:.1f}, adjusted to {score:.1f}. "
        f"Resource loss: {resource_loss}. Environmental damage: {damage_level}. "
        f"C_i={score:.1f} α_i={confidence:.2f}."
    )
    return round(score, 2), confidence, rationale, resource_loss, damage_level


def _build_citations(incident_type: str) -> list[CitationLink]:
    """Build regulatory citation links for sustainability findings."""
    citation_map: dict[str, list[dict]] = {
        "Water Contamination": [
            {"id": "CPCB-WATER-001", "label": "CPCB IS 10500 Drinking Water Standard"},
        ],
        "Air Quality Degradation": [
            {"id": "CPCB-AIR-001", "label": "CPCB NAAQS PM2.5 Standards"},
        ],
        "Environmental Hazard": [
            {"id": "CPCB-AIR-001", "label": "CPCB NAAQS Air Quality Standards"},
        ],
        "Illegal Waste Dumping": [
            {"id": "CPCB-WASTE-001", "label": "CPCB Solid Waste Management Rules 2016"},
            {"id": "CPCB-HAZWASTE-001", "label": "Hazardous Waste Rules 2016"},
        ],
        "Sustainability Violation": [
            {"id": "CPCB-WASTE-001", "label": "CPCB Solid Waste Management Rules 2016"},
        ],
        "Resource Waste": [
            {"id": "WATER-CONS-001", "label": "Municipal Water Conservation Guidelines"},
        ],
    }
    refs = citation_map.get(incident_type, [{"id": "CPCB-WATER-001", "label": "CPCB Environmental Guidelines"}])
    return [
        CitationLink(
            source_type="REGULATORY_KB",
            identifier=r["id"],
            label=r["label"],
            confidence_score=0.92,
            url=f"https://cpcb.nic.in/regulations/{r['id']}",
        )
        for r in refs
    ]
