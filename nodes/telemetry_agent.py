"""
nodes/telemetry_agent.py
─────────────────────────────────────────────────────────────────────────────
Telemetry Agent Node — Parallel Branch A.

Domain: Physical sensor telemetry (seismic, meteorological, IoT mesh).
Domain Weight (w_i): 0.55  — as defined in state.DOMAIN_WEIGHTS.

Live mode  (USE_MOCK_LLM=false):
  → ChatGroq (llama-3.3-70b-versatile) with structured output.
    The LLM fills TelemetryScoreExtraction (hazard_score, confidence, rationale).
    Citations are still assembled from Qdrant + Neo4j tool calls (not LLM-generated).

Mock mode  (USE_MOCK_LLM=true):
  → Deterministic heuristic. Tests always run in this mode.
  → USE_MOCK_LLM is read from .env at import time via llm_client.

Failure strategy:
  → On any Groq API error the node returns a fallback AgentScore with
    confidence=0.10 (near-zero) so the consensus math still runs but H
    will be pulled low enough to force HITL — the system degrades safely.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from nodes.llm_client import (
    USE_MOCK_LLM,
    TelemetryScoreExtraction,
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
from tools import neo4j_mock, qdrant_mock

logger = logging.getLogger(__name__)

_AGENT_ID = "telemetry_agent"
_AGENT_TYPE = "Physical Sensor & Telemetry Expert"
_DOMAIN_WEIGHT = DOMAIN_WEIGHTS[_AGENT_ID]

# ── Build the structured LLM once at module import (singleton) ────────────────
_structured_llm = build_structured_llm(TelemetryScoreExtraction)

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Telemetry Agent for ECO GUARD, an automated \
disaster mitigation pipeline operating under NIMS/ICS protocols.

Your domain weight is {domain_weight} (physical sensor data carries the \
highest evidentiary authority in this pipeline).

Your task: analyse the raw anomaly alert below and compute two metrics.

HAZARD SCORE (0–100):
  Assess the raw physical severity of the event. Use this scale:
    0–20   → Negligible / background noise
    21–40  → Minor, monitoring recommended
    41–60  → Moderate, precautionary alerts warranted
    61–75  → Significant, activate response protocols
    76–90  → Severe, mandatory evacuation likely
    91–100 → Catastrophic, maximum response

CONFIDENCE (0.0–1.0):
  Reflect how complete and reliable the telemetry data is.
    • Start at 0.90 for fully instrumented events (magnitude + coordinates + sensor ID)
    • Drop by 0.15 for each missing critical field
    • Minimum 0.10 — never report 0.0 (that means no data at all)

RATIONALE:
  Write 2–3 sentences. Reference specific fields from the alert payload \
(magnitude, PGA, depth, sensor codes, etc.). Be precise and technical.

Do NOT fabricate sensor readings not present in the payload. \
If data is sparse, reflect that in a lower confidence score.\
"""

_HUMAN_PROMPT = """\
Raw Alert Payload:
{alert_data}

Produce your structured assessment now.\
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_PROMPT),
])

# ── Mock severity ladder (offline / test path) ────────────────────────────────
# ── Mock severity ladder (offline / test path) ────────────────────────────────
_SEVERITY_SCORE_MAP: dict[str, float] = {
    "LOW": 30.0,         # 20-40 range
    "MEDIUM": 52.5,      # 40-65 range
    "HIGH": 80.0,        # 70-90 range
    "CRITICAL": 92.5,    # 85-100 range
    "CATASTROPHIC": 98.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Node entry-point
# ──────────────────────────────────────────────────────────────────────────────

async def telemetry_node(state: EcoGuardState) -> dict[str, Any]:
    """
    LangGraph node: Telemetry Agent (parallel branch A).

    Returns:
        Partial state delta with one AgentScore appended to agent_scores.
    """
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting — trace_id=%s  live_llm=%s",
                _AGENT_ID, trace_id, not USE_MOCK_LLM)

    raw_alert: dict[str, Any] = state["raw_alert"]

    # ── 1. GraphRAG tool calls (always run — provide citations regardless) ────
    citations = await _gather_citations(raw_alert)

    # ── 2. Score computation ──────────────────────────────────────────────────
    if not USE_MOCK_LLM:
        hazard_score, confidence, rationale = await _score_via_groq(raw_alert)
    else:
        hazard_score, confidence, rationale = _score_mock(raw_alert)

    # ── 3. Assemble validated AgentScore ─────────────────────────────────────
    score = AgentScore(
        agent_id=_AGENT_ID,
        agent_type=_AGENT_TYPE,
        hazard_score=hazard_score,
        confidence=confidence,
        domain_weight=_DOMAIN_WEIGHT,
        rationale=rationale,
        citations=citations,
        status=NodeStatus.COMPLETED,
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_AGENT_ID,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info("[%s] Done — C_i=%.1f α_i=%.3f w_i=%.2f  %.0fms",
                _AGENT_ID, hazard_score, confidence, _DOMAIN_WEIGHT, duration_ms)

    return {
        "agent_scores": [score.model_dump()],
        "node_execution_log": [exec_meta.model_dump()],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private: live Groq path
# ──────────────────────────────────────────────────────────────────────────────

async def _score_via_groq(
    raw_alert: dict[str, Any],
) -> tuple[float, float, str]:
    """
    Invoke Groq (llama-3.3-70b-versatile) with structured output.

    Returns (hazard_score, confidence, rationale).
    Falls back to a safe degraded score on any API/parsing failure.
    """
    try:
        chain = _PROMPT | _structured_llm
        result: TelemetryScoreExtraction = await chain.ainvoke({
            "domain_weight": _DOMAIN_WEIGHT,
            "alert_data": _format_alert(raw_alert),
        })
        logger.info("[%s] Groq extraction OK — C_i=%.1f α_i=%.3f",
                    _AGENT_ID, result.hazard_score, result.confidence)
        return result.hazard_score, result.confidence, result.rationale

    except Exception as exc:
        logger.error("[%s] Groq API failure: %s — using fallback score.", _AGENT_ID, exc)
        return (
            50.0,   # mid-range — uncertain, not zero
            0.10,   # near-zero confidence → forces HITL via math consensus
            f"[DEGRADED — Groq API failure] {type(exc).__name__}: {exc}. "
            f"Fallback score applied. Manual review mandatory.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Private: mock path (tests / offline)
# ──────────────────────────────────────────────────────────────────────────────

def _score_mock(raw_alert: dict[str, Any]) -> tuple[float, float, str]:
    """Deterministic heuristic — mirrors the original mock implementation."""
    severity = raw_alert.get("severity", "MEDIUM")
    source   = raw_alert.get("source_system", "UNKNOWN")
    payload  = raw_alert.get("raw_payload", {})
    magnitude: float | None = payload.get("magnitude")

    base = _SEVERITY_SCORE_MAP.get(severity.upper(), 52.5)
    if magnitude is not None:
        base = min(100.0, base + max(0.0, (magnitude - 5.0) * 5.0))

    confidence = 0.75
    if source.upper() not in {"USGS_SEISMIC", "NOAA_WX", "NOAA_TSUNAMI", "PNSN"}:
        confidence -= 0.15
    confidence = max(0.10, min(1.0, confidence))

    rationale = (
        f"[MOCK] Telemetry analysis for [{source}] severity={severity}. "
        f"Base score: {_SEVERITY_SCORE_MAP.get(severity.upper(), 40.0):.1f}. "
        f"Magnitude: {magnitude or 'N/A'}. C_i={base:.1f} α_i={confidence:.2f}."
    )
    return round(base, 2), round(confidence, 3), rationale


# ──────────────────────────────────────────────────────────────────────────────
# Private: citation assembly (tool layer — always runs)
# ──────────────────────────────────────────────────────────────────────────────

async def _gather_citations(raw_alert: dict[str, Any]) -> list[CitationLink]:
    """Query Qdrant + Neo4j and build CitationLink objects for the AgentScore."""
    citations: list[CitationLink] = []
    source = raw_alert.get("source_system", "USGS_SEISMIC")
    severity = raw_alert.get("severity", "MEDIUM")
    description = raw_alert.get("description", "")

    # Qdrant semantic search
    hits = await qdrant_mock.semantic_search(
        query=f"sensor telemetry {source} {severity} hazard detection {description[:60]}",
        top_k=3,
    )
    for hit in hits:
        citations.append(CitationLink(
            source_type="QDRANT_CHUNK",
            identifier=hit.chunk_id,
            label=f"{hit.source_document} (p.{hit.page})",
            confidence_score=round(hit.score, 4),
            url=f"http://localhost:6333/collections/eco_guard_kb/points/{hit.chunk_id}",
        ))

    # Neo4j entity graph
    sensor_entity = _resolve_sensor_entity(source)
    neo4j_result = await neo4j_mock.query_entity_relationships(
        entity_id=sensor_entity, max_hops=2
    )
    for node in neo4j_result.nodes[:3]:
        citations.append(CitationLink(
            source_type="NEO4J_NODE",
            identifier=node.id,
            label=f"[{', '.join(node.labels)}] {node.properties.get('name', node.id)}",
            confidence_score=0.85,
            url=f"http://localhost:7474/browser/?id={node.id}",
        ))

    return citations


def _resolve_sensor_entity(source_system: str) -> str:
    mapping = {
        "USGS_SEISMIC": "sensor:usgs_pnsn_001",
        "NOAA_WX":      "sensor:noaa_dart_56001",
        "NOAA_TSUNAMI": "sensor:noaa_dart_56001",
        "PNSN":         "sensor:usgs_pnsn_001",
    }
    return mapping.get(source_system.upper(), "sensor:usgs_pnsn_001")


def _format_alert(alert: dict[str, Any]) -> str:
    """Pretty-format the alert dict for the LLM prompt."""
    import json
    return json.dumps(alert, indent=2, default=str)
