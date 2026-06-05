"""
nodes/safety_auditor.py
─────────────────────────────────────────────────────────────────────────────
Safety Auditor Node — Parallel Branch D (Agent 4).

Domain: Human safety, emergency procedures, life-critical risk.
Domain Weight (w_i): 0.50

Live mode (USE_MOCK_LLM=false):
  → ChatGroq with SafetyRiskExtraction structured output.

Mock mode (USE_MOCK_LLM=true):
  → Deterministic heuristic based on severity and incident type.

Outputs:
  - hazard_score:       Safety risk score C_i ∈ [0, 100]
  - safety_risk_score:  Dedicated safety metric (stored separately)
  - confidence:         α_i ∈ [0, 1]
  - extended_report:    immediate_danger, evacuation_recommended, emergency_procedures
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from nodes.llm_client import (
    USE_MOCK_LLM,
    SafetyRiskExtraction,
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

_AGENT_ID = "safety_auditor"
_AGENT_TYPE = "Safety & Emergency Procedures Auditor"
_DOMAIN_WEIGHT = DOMAIN_WEIGHTS[_AGENT_ID]

_structured_llm = build_structured_llm(SafetyRiskExtraction)

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Safety Auditor for ECO GUARD.

Your domain weight is {domain_weight}. You assess IMMEDIATE DANGER to life, \
injury risk, and draft emergency response procedures.

SAFETY RISK SCORE (0–100):
  0–20   → Routine issue, no safety risk
  21–50  → Minor safety risk, caution required
  51–75  → Significant risk to health/safety, active mitigation needed
  76–90  → Immediate danger to life/limb, evacuation likely
  91–100 → Mass casualty risk, catastrophic life safety failure

Draft 3-5 specific, actionable EMERGENCY PROCEDURES.
ONLY provide safety guidelines and measures established by the Government of India. Do NOT provide measures from other countries.
If the issue needs to be solved, explicitly state that they should meet or contact the National Disaster Management Authority (NDMA) or the specific relevant Indian government department.
Set immediate_danger=True if score > 75 or if the incident involves gas leaks, \
electrical arcs, structural failure, or hazardous contamination.

CONFIDENCE (0.0–1.0): Base 0.85.\
"""

_HUMAN_PROMPT = """\
Alert Payload:
{alert_data}

Produce your structured safety and emergency assessment now.\
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_PROMPT),
])

# Mock scoring tables
_SEVERITY_SCORE_MAP: dict[str, float] = {
    "LOW": 32.5,         # 20-45 range
    "MEDIUM": 57.5,      # 45-70 range
    "HIGH": 82.5,        # 70-95 range
    "CRITICAL": 92.5,    # 85-100 range
    "CATASTROPHIC": 98.0,
}


async def safety_auditor_node(state: EcoGuardState) -> dict[str, Any]:
    """LangGraph node: Safety Auditor (parallel branch D)."""
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting — trace_id=%s", _AGENT_ID, trace_id)

    raw_alert: dict[str, Any] = state["raw_alert"]
    incident_type = raw_alert.get("raw_payload", {}).get("incident_type", "Unknown")

    if not USE_MOCK_LLM:
        result = await _score_via_groq(raw_alert)
        hazard_score = result.safety_risk_score
        confidence = result.confidence
        rationale = result.rationale
        immediate_danger = result.immediate_danger
        evacuation = result.evacuation_recommended
        procedures = result.emergency_procedures
    else:
        hazard_score, confidence, rationale, immediate_danger, evacuation, procedures = _score_mock(
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
            "safety_risk_score": hazard_score,
            "immediate_danger": immediate_danger,
            "evacuation_recommended": evacuation,
            "emergency_procedures": procedures,
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
        "safety_risk_score": round(hazard_score, 2),
        "emergency_response_plan": procedures,
        "node_execution_log": [exec_meta.model_dump()],
    }


async def _score_via_groq(raw_alert: dict[str, Any]) -> SafetyRiskExtraction:
    try:
        chain = _PROMPT | _structured_llm
        return await chain.ainvoke({
            "domain_weight": _DOMAIN_WEIGHT,
            "alert_data": json.dumps(raw_alert, indent=2, default=str),
        })
    except Exception as exc:
        logger.error("[%s] Groq failure: %s — using fallback.", _AGENT_ID, exc)
        return SafetyRiskExtraction(
            safety_risk_score=50.0,
            confidence=0.10,
            rationale=f"[DEGRADED] Groq API failure: {exc}. Manual review required.",
            immediate_danger=False,
            evacuation_recommended=False,
            emergency_procedures=["[DEGRADED] Unable to generate specific procedures. Escalate to human command."],
        )


def _score_mock(
    raw_alert: dict[str, Any],
    incident_type: str,
) -> tuple[float, float, str, bool, bool, list[str]]:
    severity = raw_alert.get("severity", "MEDIUM")
    payload = raw_alert.get("raw_payload", {})

    score = _SEVERITY_SCORE_MAP.get(severity.upper(), 57.5)

    immediate_danger = False
    evacuation = False
    procedures = []

    if incident_type == "Gas Leak":
        immediate_danger = True
        evacuation = True
        procedures = [
            "Initiate immediate building evacuation",
            "Isolate main gas supply valve",
            "Deploy emergency ventilation fans",
            "Establish 100m exclusion zone",
        ]
    elif incident_type == "Fire Hazard":
        if payload.get("smoke_observed"):
            immediate_danger = True
        procedures = [
            "Dispatch fire response team",
            "Clear surrounding area of flammable material",
            "Prepare for localized evacuation if fire spreads",
        ]
    elif incident_type == "Electrical Fault":
        immediate_danger = True
        procedures = [
            "Isolate affected electrical panel immediately",
            "Ensure no personnel enter the electrical room",
            "Call DISCOM emergency response",
        ]
    elif incident_type == "Flooding":
        procedures = [
            "Shut down basement electrical panels",
            "Deploy submersible pumps",
            "Relocate vehicles from basement parking",
        ]
    else:
        procedures = [
            f"Assess site safety before deploying personnel for {incident_type}",
            "Maintain communication with incident command",
            "Monitor severity indicators for escalation",
        ]
        if score > 75.0:
            immediate_danger = True

    confidence = 0.82
    rationale = (
        f"[MOCK] Safety analysis for {incident_type} (severity={severity}). "
        f"Base risk based on severity, score: {score:.1f}. "
        f"Danger: {'High' if immediate_danger else 'Low'}. "
        f"C_i={score:.1f} α_i={confidence:.2f}."
    )
    return round(score, 2), confidence, rationale, immediate_danger, evacuation, procedures


def _build_citations(incident_type: str) -> list[CitationLink]:
    """Build regulatory/safety citation links for safety findings."""
    citation_map: dict[str, list[dict]] = {
        "Gas Leak": [
            {"id": "NDMA-GAS-001", "label": "NDMA Gas Leak Response Protocol"},
            {"id": "NBC-GAS-001", "label": "NBC Gas Installations Safety"},
        ],
        "Fire Hazard": [
            {"id": "NBC-FIRE-001", "label": "NBC Fire and Life Safety"},
            {"id": "NDMA-FIRE-001", "label": "NDMA Fire Management Guidelines"},
        ],
        "Electrical Fault": [
            {"id": "CEA-ELEC-001", "label": "CEA Electrical Safety Regulations"},
        ],
        "Flooding": [
            {"id": "NDMA-FLOOD-001", "label": "NDMA Urban Flooding SOP"},
        ],
    }
    refs = citation_map.get(incident_type, [{"id": "NDMA-COMMUNITY-001", "label": "NDMA Community Safety"}])
    return [
        CitationLink(
            source_type="REGULATORY_KB",
            identifier=r["id"],
            label=r["label"],
            confidence_score=0.95,
            url=f"https://ndma.gov.in/guidelines/{r['id']}",
        )
        for r in refs
    ]
