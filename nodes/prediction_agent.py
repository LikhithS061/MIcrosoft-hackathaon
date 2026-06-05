"""
nodes/prediction_agent.py
─────────────────────────────────────────────────────────────────────────────
Prediction Agent Node — Sequential Branch (Agent 6).

Domain: Forecasting future incidents, pattern analysis, cascading risk.
Domain Weight (w_i): 0.30

This agent runs AFTER the 5 parallel domain agents have completed.
It queries the SQLite Learning Engine for historical similar cases to predict
what will happen next if the current incident is not mitigated.

Live mode (USE_MOCK_LLM=false):
  → ChatGroq with PredictionExtraction structured output.

Mock mode (USE_MOCK_LLM=true):
  → Deterministic prediction based on historical outcomes and incident type.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from learning_engine import get_learning_engine
from nodes.llm_client import (
    USE_MOCK_LLM,
    PredictionExtraction,
    build_structured_llm,
)
from state import (
    AgentScore,
    CitationLink,
    DOMAIN_WEIGHTS,
    HistoricalCase,
    NodeExecutionMeta,
    NodeStatus,
    PredictedRisk,
    PredictionReport,
    EcoGuardState,
)

logger = logging.getLogger(__name__)

_AGENT_ID = "prediction_agent"
_AGENT_TYPE = "Future Risk Forecasting & Pattern Learning Engine"
_DOMAIN_WEIGHT = DOMAIN_WEIGHTS[_AGENT_ID]

_structured_llm = build_structured_llm(PredictionExtraction)

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Prediction Agent for ECO GUARD.

Your domain weight is {domain_weight}. Your task is to FORECAST FUTURE CASCADING RISKS \
based on the current incident and historical similar cases.

HAZARD SCORE (0–100):
  Score the severity of the PREDICTED future state if no mitigation occurs.
  (If the current incident will inevitably cascade into a worse disaster, score high).

PREDICTION:
  Identify the top 2 likely follow-on incidents.
  Provide probability scores (0.0 - 1.0) and time horizons.

HISTORICAL CONTEXT:
{historical_context}

Use the historical outcomes to inform your prediction probabilities.\
"""

_HUMAN_PROMPT = """\
Current Alert:
{alert_data}

Produce your structured prediction analysis now.\
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_PROMPT),
])

# Mock cascade chains
_PREDICTION_CHAINS: dict[str, list[dict]] = {
    "Fire Hazard": [
        {"type": "Structural Compromise", "prob": 0.85, "horizon": "4 hours", "action": "Evacuation and fire suppression"},
        {"type": "Air Quality Degradation", "prob": 0.70, "horizon": "24 hours", "action": "Distribute masks to downwind communities"},
    ],
    "Gas Leak": [
        {"type": "Explosion / Fire Outbreak", "prob": 0.92, "horizon": "Immediate", "action": "Shut off main electrical breakers, full evacuation"},
        {"type": "Mass Asphyxiation", "prob": 0.88, "horizon": "1 hour", "action": "Deploy ventilation systems"},
    ],
    "Flooding": [
        {"type": "Electrical Short Circuit", "prob": 0.89, "horizon": "2 hours", "action": "De-energise basement level"},
        {"type": "Waterborne Disease Outbreak", "prob": 0.65, "horizon": "5 days", "action": "Hyper-chlorinate drinking water"},
    ],
    "Water Contamination": [
        {"type": "Public Health Epidemic (Gastro)", "prob": 0.84, "horizon": "48 hours", "action": "Provide alternative drinking water sources immediately"},
        {"type": "Corrosion of Piping Infrastructure", "prob": 0.40, "horizon": "3 months", "action": "Flush system to restore neutral pH"},
    ],
    "Electrical Fault": [
        {"type": "Panel Fire Outbreak", "prob": 0.80, "horizon": "1 hour", "action": "Isolate panel, deploy CO2 extinguishers"},
        {"type": "Extended Power Outage", "prob": 0.95, "horizon": "24 hours", "action": "Ensure backup generator fuel supply"},
    ],
}


async def prediction_node(state: EcoGuardState) -> dict[str, Any]:
    """LangGraph node: Prediction Agent (sequential, runs after parallel agents)."""
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting — trace_id=%s", _AGENT_ID, trace_id)

    raw_alert: dict[str, Any] = state["raw_alert"]
    incident_type = raw_alert.get("raw_payload", {}).get("incident_type", "Unknown")

    # 1. Query Learning Engine for historical similar cases
    engine = get_learning_engine()
    raw_historical = await engine.find_similar(raw_alert, top_k=3)

    historical_cases: list[HistoricalCase] = []
    for h in raw_historical:
        historical_cases.append(HistoricalCase(
            incident_id=h["id"],
            incident_type=h["incident_type"],
            location=h.get("location", ""),
            severity=h.get("severity", ""),
            hazard_score=h.get("hazard_score", 0.0),
            outcome=h.get("outcome", ""),
            lessons_learned=h.get("lessons_learned", ""),
            similarity_score=h.get("similarity_score", 0.0),
            timestamp=h.get("timestamp", ""),
        ))

    hist_context_str = "\n".join(
        f"- [ID: {h.incident_id} | Sim: {h.similarity_score}] {h.incident_type} ({h.severity}): "
        f"Outcome: {h.outcome}. Lesson: {h.lessons_learned}"
        for h in historical_cases
    ) if historical_cases else "No similar historical cases found in learning DB."

    # 2. Score via LLM or Mock
    if not USE_MOCK_LLM:
        result = await _score_via_groq(raw_alert, hist_context_str)
        hazard_score = min(100.0, result.top_risk_probability * 100 + 15)  # proxy score
        confidence = result.prediction_confidence
        rationale = result.rationale
        risks = [
            PredictedRisk(
                incident_type=result.top_predicted_risk_type,
                probability=result.top_risk_probability,
                confidence=confidence,
                time_horizon="Short-term",
                reasoning=rationale,
                preventive_action=result.preventive_actions[0] if result.preventive_actions else "Monitor",
            ),
            PredictedRisk(
                incident_type=result.secondary_risk_type,
                probability=result.secondary_risk_probability,
                confidence=confidence * 0.8,
                time_horizon="Medium-term",
                reasoning="Secondary cascade effect.",
                preventive_action=result.preventive_actions[1] if len(result.preventive_actions) > 1 else "Monitor",
            )
        ]
    else:
        hazard_score, confidence, rationale, risks = _score_mock(raw_alert, incident_type, historical_cases)

    # 3. Build Citations (pointing to Learning DB)
    citations: list[CitationLink] = [
        CitationLink(
            source_type="LEARNING_DB",
            identifier=h.incident_id,
            label=f"Historical Case: {h.incident_type} ({h.severity})",
            confidence_score=h.similarity_score,
        )
        for h in historical_cases
    ]

    report = PredictionReport(
        predicted_risks=risks,
        historical_pattern_summary=rationale,
        learning_db_incidents_used=len(historical_cases),
    )

    score = AgentScore(
        agent_id=_AGENT_ID,
        agent_type=_AGENT_TYPE,
        hazard_score=hazard_score,
        confidence=confidence,
        domain_weight=_DOMAIN_WEIGHT,
        rationale=rationale,
        citations=citations,
        status=NodeStatus.COMPLETED,
        extended_report=report.model_dump(),
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_AGENT_ID,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info("[%s] Done — top_pred='%s' prob=%.2f  %.0fms",
                _AGENT_ID, risks[0].incident_type if risks else "None",
                risks[0].probability if risks else 0.0, duration_ms)

    return {
        "agent_scores": [score.model_dump()],
        "historical_similar_cases": [h.model_dump() for h in historical_cases],
        "prediction_report": report.model_dump(),
        "node_execution_log": [exec_meta.model_dump()],
    }


async def _score_via_groq(raw_alert: dict[str, Any], hist_context: str) -> PredictionExtraction:
    try:
        chain = _PROMPT | _structured_llm
        return await chain.ainvoke({
            "domain_weight": _DOMAIN_WEIGHT,
            "historical_context": hist_context,
            "alert_data": json.dumps(raw_alert, indent=2, default=str),
        })
    except Exception as exc:
        logger.error("[%s] Groq failure: %s — using fallback.", _AGENT_ID, exc)
        return PredictionExtraction(
            top_predicted_risk_type="Unknown System Cascade",
            top_risk_probability=0.5,
            secondary_risk_type="None",
            secondary_risk_probability=0.0,
            prediction_confidence=0.1,
            rationale=f"[DEGRADED] Groq API failure: {exc}.",
            preventive_actions=["Manual review required"],
        )


def _score_mock(
    raw_alert: dict[str, Any],
    incident_type: str,
    historical_cases: list[HistoricalCase],
) -> tuple[float, float, str, list[PredictedRisk]]:
    severity = raw_alert.get("severity", "MEDIUM").upper()
    chains = _PREDICTION_CHAINS.get(incident_type, [
        {"type": "General Situation Deterioration", "prob": 0.60, "horizon": "12 hours", "action": "Implement mitigation procedures"},
        {"type": "Secondary Infrastructure Strain", "prob": 0.40, "horizon": "48 hours", "action": "Monitor system load"},
    ])

    # Adjust probabilities and score based on severity
    severity_factors = {"LOW": 0.35, "MEDIUM": 0.70, "HIGH": 0.90, "CRITICAL": 1.0, "CATASTROPHIC": 1.1}
    factor = severity_factors.get(severity, 0.70)

    adjusted_chains = []
    for c in chains:
        adj_p = max(0.05, min(1.0, c["prob"] * factor))
        adjusted_chains.append({
            "type": c["type"],
            "prob": adj_p,
            "horizon": c["horizon"],
            "action": c["action"]
        })

    base_score = (50.0 + (adjusted_chains[0]["prob"] * 40.0)) * factor
    base_score = max(5.0, min(100.0, base_score))
    confidence = 0.85 if historical_cases else 0.60

    risks = [
        PredictedRisk(
            incident_type=c["type"],
            probability=c["prob"],
            confidence=confidence,
            time_horizon=c["horizon"],
            reasoning=f"Based on cascade patterns for {incident_type} (severity={severity}).",
            preventive_action=c["action"],
        )
        for c in adjusted_chains
    ]

    hist_note = f" (Validated against {len(historical_cases)} past cases from Learning DB)" if historical_cases else " (No direct historical precedent)"
    rationale = f"[MOCK] Prediction analysis indicates probability of {adjusted_chains[0]['type']} within {adjusted_chains[0]['horizon']} is {adjusted_chains[0]['prob']:.2f}.{hist_note}"

    return round(base_score, 2), confidence, rationale, risks
