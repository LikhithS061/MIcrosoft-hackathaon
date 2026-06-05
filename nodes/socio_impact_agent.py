"""
nodes/socio_impact_agent.py
─────────────────────────────────────────────────────────────────────────────
Socio-Economic Impact Agent Node — Parallel Branch B.

Domain: Population exposure, displacement, infrastructure interdependency,
        secondary health/economic cascades.
Domain Weight (w_i): 0.45  — as defined in state.DOMAIN_WEIGHTS.

Live mode  (USE_MOCK_LLM=false):
  → ChatGroq (llama-3.3-70b-versatile) with structured output.
    The LLM fills SocioScoreExtraction (hazard_score, confidence, rationale).
    Neo4j relationship data is injected into the prompt context to ground the
    LLM's reasoning in deterministic entity data — preventing hallucination.

Mock mode  (USE_MOCK_LLM=true):
  → Deterministic heuristic (population-weighted scoring).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from nodes.llm_client import (
    USE_MOCK_LLM,
    SocioScoreExtraction,
    build_structured_llm,
)
from state import (
    AgentScore,
    CitationLink,
    DOMAIN_WEIGHTS,
    NodeExecutionMeta,
    NodeStatus,
    EcoGuardState,
    VARIANCE_CRITICAL,
)
from tools import neo4j_mock, qdrant_mock

logger = logging.getLogger(__name__)

_AGENT_ID = "socio_impact_agent"
_AGENT_TYPE = "Socio-Economic & Population Impact Analyst"
_DOMAIN_WEIGHT = DOMAIN_WEIGHTS[_AGENT_ID]

# ── Build the structured LLM once at module import (singleton) ────────────────
_structured_llm = build_structured_llm(SocioScoreExtraction)

# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Socio-Economic Impact Agent for ECO GUARD, a \
disaster mitigation pipeline. Your domain weight is {domain_weight}.

Your task: assess the SECONDARY human, social, and economic consequences \
of the physical event described in the alert. You are NOT assessing the \
raw physical severity — the Telemetry Agent handles that. You assess:

  • Population directly exposed and at risk of displacement
  • Critical infrastructure cascade failures (power, transport, comms)
  • Secondary health system strain (hospitals, water supply, shelter)
  • Economic disruption (GDP impact, supply chain severance)
  • Governance/response gap (is there a protocol deficit?)

HAZARD SCORE (0–100):
  0–20   → Minimal social disruption
  21–40  → Localised community impact, manageable with local resources
  41–60  → Regional impact, multi-agency coordination required
  61–75  → Large-scale displacement, sustained multi-week response
  76–90  → Mass casualty / displacement scenario, federal/international aid
  91–100 → Civilisation-level infrastructure collapse, generational recovery

CONFIDENCE (0.0–1.0):
  Base 0.85. Reduce by 0.10 for each major data gap (population unknown,
  infrastructure map unavailable, no protocol coverage identified).

GRAPH CONTEXT (verified Neo4j entity relationships):
{graph_context}

Use the graph context as ground truth. Do not contradict it.\
"""

_HUMAN_PROMPT = """\
Raw Alert:
{alert_data}

Produce your structured socio-economic impact assessment now.\
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_PROMPT),
])

# ── Mock severity ladder (offline / test path) ────────────────────────────────
_SEVERITY_SCORE_MAP: dict[str, float] = {
    "LOW": 25.0,         # 15-35 range
    "MEDIUM": 47.5,      # 35-60 range
    "HIGH": 77.5,        # 65-90 range
    "CRITICAL": 90.0,    # 80-100 range
    "CATASTROPHIC": 98.0,
}
_POPULATION_THRESHOLDS: list[tuple[int, float]] = [
    (10_000_000, 30.0), (5_000_000, 20.0),
    (1_000_000, 12.0),  (100_000, 5.0), (0, 0.0),
]


# ──────────────────────────────────────────────────────────────────────────────
# Node entry-point
# ──────────────────────────────────────────────────────────────────────────────

async def socio_impact_node(state: EcoGuardState) -> dict[str, Any]:
    """
    LangGraph node: Socio-Economic Impact Agent (parallel branch B).

    Returns:
        Partial state delta with one AgentScore appended to agent_scores.
    """
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting — trace_id=%s  live_llm=%s",
                _AGENT_ID, trace_id, not USE_MOCK_LLM)

    raw_alert: dict[str, Any] = state["raw_alert"]
    geo: list = raw_alert.get("geo_coordinates", [0.0, 0.0])

    # ── 1. GraphRAG: Neo4j graph for LLM context + citations ─────────────────
    region_entity = _geo_to_region_entity(geo)
    neo4j_result = await neo4j_mock.query_entity_relationships(
        entity_id=region_entity, max_hops=2,
        relationship_types=["THREATENS", "IMPACTS", "GENERATES"],
    )

    citations: list[CitationLink] = []
    exposed_population = 0
    graph_facts: list[str] = []

    for node in neo4j_result.nodes:
        if "Region" in node.labels:
            pop = node.properties.get("population", 0)
            exposed_population = max(exposed_population, pop)
            graph_facts.append(
                f"Region '{node.properties.get('name', node.id)}': "
                f"population={pop:,}, GDP=${node.properties.get('gdp_bn_usd', 'N/A')}B"
            )
        elif "Infrastructure" in node.labels:
            graph_facts.append(
                f"Infrastructure '{node.properties.get('name', node.id)}' "
                f"[{', '.join(node.labels)}]: "
                f"criticality={node.properties.get('criticality', 'UNKNOWN')}"
            )
        elif "Hazard" in node.labels:
            graph_facts.append(
                f"Hazard '{node.id}': {json.dumps(node.properties, default=str)}"
            )

        citations.append(CitationLink(
            source_type="NEO4J_NODE",
            identifier=node.id,
            label=f"[{', '.join(node.labels)}] {node.properties.get('name', node.id)}",
            confidence_score=0.88,
            url=f"http://localhost:7474/browser/?id={node.id}",
        ))

    # ── 2. Qdrant: humanitarian literature citations ───────────────────────────
    severity = raw_alert.get("severity", "MEDIUM")
    description = raw_alert.get("description", "")
    qdrant_hits = await qdrant_mock.semantic_search(
        query=(
            f"population displacement humanitarian response shelter "
            f"{severity} disaster {description[:60]}"
        ),
        top_k=3,
    )
    for hit in qdrant_hits:
        citations.append(CitationLink(
            source_type="QDRANT_CHUNK",
            identifier=hit.chunk_id,
            label=f"{hit.source_document} (p.{hit.page})",
            confidence_score=round(hit.score, 4),
            url=f"http://localhost:6333/collections/eco_guard_kb/points/{hit.chunk_id}",
        ))

    # ── 3. Score computation ──────────────────────────────────────────────────
    graph_context_str = (
        "\n".join(f"  • {f}" for f in graph_facts)
        if graph_facts
        else "  • No graph entities found for this geographic region."
    )

    if not USE_MOCK_LLM:
        hazard_score, confidence, rationale = await _score_via_groq(
            raw_alert=raw_alert,
            graph_context=graph_context_str,
        )
    else:
        hazard_score, confidence, rationale = _score_mock(
            severity=severity,
            exposed_population=exposed_population,
            neo4j_node_count=len(neo4j_result.nodes),
            qdrant_hits=qdrant_hits,
        )

    # ── 4. Assemble validated AgentScore ─────────────────────────────────────
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

    logger.info("[%s] Done — C_i=%.1f α_i=%.3f w_i=%.2f  exposed_pop=%d  %.0fms",
                _AGENT_ID, hazard_score, confidence, _DOMAIN_WEIGHT,
                exposed_population, duration_ms)

    return {
        "agent_scores": [score.model_dump()],
        "node_execution_log": [exec_meta.model_dump()],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private: live Groq path
# ──────────────────────────────────────────────────────────────────────────────

async def _score_via_groq(
    raw_alert: dict[str, Any],
    graph_context: str,
) -> tuple[float, float, str]:
    """Invoke Groq with Neo4j graph context injected into the system prompt."""
    try:
        chain = _PROMPT | _structured_llm
        result: SocioScoreExtraction = await chain.ainvoke({
            "domain_weight": _DOMAIN_WEIGHT,
            "graph_context": graph_context,
            "alert_data": json.dumps(raw_alert, indent=2, default=str),
        })
        logger.info("[%s] Groq extraction OK — C_i=%.1f α_i=%.3f",
                    _AGENT_ID, result.hazard_score, result.confidence)
        return result.hazard_score, result.confidence, result.rationale

    except Exception as exc:
        logger.error("[%s] Groq API failure: %s — using fallback.", _AGENT_ID, exc)
        return (
            50.0, 0.10,
            f"[DEGRADED — Groq API failure] {type(exc).__name__}: {exc}. "
            f"Fallback score applied. Manual review mandatory.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Private: mock path
# ──────────────────────────────────────────────────────────────────────────────

def _score_mock(
    severity: str,
    exposed_population: int,
    neo4j_node_count: int,
    qdrant_hits: list,
) -> tuple[float, float, str]:
    base = _SEVERITY_SCORE_MAP.get(severity.upper(), 47.5)
    pop_bonus = next(
        (bonus for threshold, bonus in _POPULATION_THRESHOLDS
         if exposed_population >= threshold), 0.0
    )
    base = min(100.0, base + pop_bonus)
    if neo4j_node_count > 5:
        base = min(100.0, base * 1.08)

    confidence = min(0.90, 0.55 + len(qdrant_hits) * 0.10)
    confidence = max(0.10, confidence)

    return (
        round(base, 2), round(confidence, 3),
        f"[MOCK] Socio-economic impact: severity={severity}, "
        f"exposed_pop={exposed_population:,}, "
        f"graph_nodes={neo4j_node_count}. C_i={base:.1f} α_i={confidence:.2f}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _geo_to_region_entity(geo: list) -> str:
    lat, lon = float(geo[0]), float(geo[1])
    if 42.0 <= lat <= 50.0 and -125.0 <= lon <= -116.0:
        return "region:pacific_northwest"
    if 33.0 <= lat <= 34.5 and -119.0 <= lon <= -117.0:
        return "region:los_angeles"
    return "region:pacific_northwest"
