"""
nodes/regulatory_agent.py
─────────────────────────────────────────────────────────────────────────────
Regulatory Intelligence Agent Node — Parallel Branch E (Agent 5).

Domain: Indian government regulations, compliance, legal considerations.
Domain Weight (w_i): 0.40

Unlike other agents, this agent heavily relies on the RAG tool
(tools.regulatory_rag) to retrieve actual regulations (zero hallucination).
The LLM is only used to summarize the retrieved regulations in the context of
the specific alert, producing the RegulatoryReport and `hazard_score` (based on
compliance violation severity).

Outputs:
  - hazard_score:       Regulatory risk score C_i ∈ [0, 100]
  - confidence:         α_i ∈ [0, 1]
  - regulatory_report:  RegulatoryReport dict (applicable laws, compliance)
  - stakeholders_to_notify: list of authorities
"""

from __future__ import annotations

import logging
import time
from typing import Any

from nodes.llm_client import USE_MOCK_LLM
from state import (
    AgentScore,
    CitationLink,
    DOMAIN_WEIGHTS,
    NodeExecutionMeta,
    NodeStatus,
    RegulatoryFinding,
    RegulatoryReport,
    EcoGuardState,
)
from tools import regulatory_rag

logger = logging.getLogger(__name__)

_AGENT_ID = "regulatory_agent"
_AGENT_TYPE = "Government Regulatory Compliance Expert"
_DOMAIN_WEIGHT = DOMAIN_WEIGHTS[_AGENT_ID]


async def regulatory_node(state: EcoGuardState) -> dict[str, Any]:
    """LangGraph node: Regulatory Intelligence Agent (parallel branch E)."""
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting — trace_id=%s", _AGENT_ID, trace_id)

    raw_alert: dict[str, Any] = state["raw_alert"]
    incident_type = raw_alert.get("raw_payload", {}).get("incident_type", "Unknown")
    severity = raw_alert.get("severity", "MEDIUM")
    description = raw_alert.get("description", "")

    # 1. Retrieve applicable regulations via ChromaDB RAG (or curated fallback)
    regs = await regulatory_rag.retrieve_regulations(
        incident_type=incident_type,
        description=description,
        severity=severity,
        top_k=3,
    )

    # 2. Score compliance risk based on retrieved regulations
    hazard_score, confidence, rationale, findings, stakeholders = _evaluate_compliance(
        regs, severity, incident_type
    )

    # 3. Build Citations (straight from the retrieved regulations)
    citations: list[CitationLink] = []
    for r in findings:
        source_type = "CHROMA_RAG" if r.retrieval_method == "ChromaDB RAG" else "REGULATORY_KB"
        citations.append(
            CitationLink(
                source_type=source_type,
                identifier=r.regulation_id,
                label=f"{r.source}: {r.title}",
                confidence_score=0.98,
                url=f"gov://{r.source.replace(' ', '_').lower()}/{r.regulation_id}",
            )
        )

    # 4. Assemble RegulatoryReport
    violating_ids = [r.regulation_id for r in findings if r.compliance_status == "VIOLATION"]
    report = RegulatoryReport(
        applicable_regulations=findings,
        primary_violations=violating_ids,
        compliance_summary=rationale,
        legal_action_required=len(violating_ids) > 0,
        stakeholders_to_notify=stakeholders,
    )

    status = NodeStatus.COMPLETED if hazard_score != -1.0 else "INSUFFICIENT REGULATORY DATA"
    score = AgentScore(
        agent_id=_AGENT_ID,
        agent_type=_AGENT_TYPE,
        hazard_score=hazard_score if hazard_score != -1.0 else 0.0,
        confidence=confidence,
        domain_weight=_DOMAIN_WEIGHT,
        rationale=rationale,
        citations=citations,
        status=status,
        extended_report=report.model_dump(),
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_AGENT_ID,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info("[%s] Done — C_i=%.1f α_i=%.3f regs=%d  %.0fms",
                _AGENT_ID, hazard_score, confidence, len(regs), duration_ms)

    return {
        "agent_scores": [score.model_dump()],
        "regulatory_report": report.model_dump(),
        "stakeholders_to_notify": stakeholders,
        "node_execution_log": [exec_meta.model_dump()],
    }


def _evaluate_compliance(
    retrieved_regs: list[dict[str, Any]],
    severity: str,
    incident_type: str,
) -> tuple[float, float, str, list[RegulatoryFinding], list[str]]:
    """
    Evaluate compliance risk. In live mode, we could pass the retrieved regs to Groq
    to determine exactly which ones are violated based on the alert payload.
    For this node, since we need deterministic adherence to the laws, we use a hybrid
    heuristic that sets VIOLATION if the alert severity matches or exceeds the law's threshold.
    """
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4, "CATASTROPHIC": 5}
    alert_rank = severity_rank.get(severity.upper(), 2)

    findings: list[RegulatoryFinding] = []
    max_risk_score = 0.0
    stakeholders: set[str] = set()

    for r in retrieved_regs:
        reg_rank = severity_rank.get(r.get("severity_threshold", "MEDIUM"), 2)

        if alert_rank >= reg_rank:
            status = "VIOLATION"
            implications = "Statutory non-compliance. Immediate remediation required."
            risk_contribution = 85.0 + (alert_rank - reg_rank) * 5
        else:
            status = "AT_RISK"
            implications = "Pre-emptive compliance monitoring required."
            risk_contribution = 40.0

        max_risk_score = max(max_risk_score, risk_contribution)
        stakeholders.add(r.get("source", "Local Authority"))

        findings.append(RegulatoryFinding(
            regulation_id=r["regulation_id"],
            title=r["title"],
            source=r["source"],
            section=r["section"],
            summary=r["summary"],
            compliance_status=status,
            legal_implications=implications,
            retrieval_method=r.get("retrieval_method", "Curated Fallback"),
        ))

    if not findings:
        return -1.0, 0.30, f"No specific regulations found for {incident_type}.", [], ["Local Municipal Corporation"]

    max_risk_score = min(100.0, max_risk_score)
    confidence = 0.90  # High confidence because we are using actual retrieved laws

    viol_count = sum(1 for f in findings if f.compliance_status == "VIOLATION")
    rationale = (
        f"Regulatory assessment identified {len(findings)} applicable guidelines. "
        f"{viol_count} active statutory violations detected based on severity thresholds. "
        f"C_i={max_risk_score:.1f} (Compliance Risk)."
    )

    # Standardize stakeholders based on incident type
    stakeholder_list = list(stakeholders)
    if "Fire Hazard" in incident_type:
        stakeholder_list.append("State Fire Department")
    if "Security" in incident_type or "Public Safety" in incident_type:
        stakeholder_list.append("Local Police Station")

    return max_risk_score, confidence, rationale, findings, stakeholder_list
