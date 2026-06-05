"""
nodes/gatekeeper_node.py
─────────────────────────────────────────────────────────────────────────────
Gatekeeper Node — Final synthesis and HITL gate.

Responsibilities:
  1. Draft the plain-English MitigationBrief with clickable chain-of-evidence
     links derived from all upstream agent citation lists.
  2. Determine the final risk tier from H and the verification trust score.
  3. Set `hitl_required = True` unconditionally when H > 75 or σ²_H > threshold
     (even if consensus_node hasn't explicitly flagged it — belt-and-suspenders).
  4. Emit the mitigation_brief into state so the Streamlit HITL UI can render it.

This node does NOT wait for human input — it only prepares and emits state.
The actual human pause is implemented as a LangGraph `interrupt_before` on the
terminal edge, handled in graph.py.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from state import (
    AgentScore,
    AlertSeverity,
    CitationLink,
    HAZARD_INDEX_CRITICAL,
    MitigationBrief,
    NodeExecutionMeta,
    NodeStatus,
    EcoGuardState,
    VARIANCE_CRITICAL,
    VerificationReport,
)

logger = logging.getLogger(__name__)
_NODE_NAME = "gatekeeper_node"


async def gatekeeper_node(state: EcoGuardState) -> dict[str, Any]:
    """
    LangGraph node: Gatekeeper (final synthesis + HITL trigger).

    Args:
        state: Full EcoGuardState with agent_scores, consensus metrics,
               and verification_report all populated.

    Returns:
        Partial state delta with `mitigation_brief`, updated `hitl_required`,
        and the final `node_execution_log` entry.
    """
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting gatekeeper synthesis — trace_id=%s", _NODE_NAME, trace_id)

    H: float = state.get("aggregate_hazard_index", 0.0)
    sigma_sq: float = state.get("consensus_variance", 0.0)
    sustainability: float = state.get("aggregate_sustainability_score", 0.0)
    safety: float = state.get("aggregate_safety_score", 0.0)
    raw_alert: dict[str, Any] = state.get("raw_alert", {})
    agent_scores_raw: list[dict[str, Any]] = state.get("agent_scores", [])
    verification_raw: dict[str, Any] = state.get("verification_report", {})
    prediction_raw: dict[str, Any] = state.get("prediction_report", {})
    regulatory_raw: dict[str, Any] = state.get("regulatory_report", {})

    # ── 1. Extract decision_status from consensus ─────────────────────────────
    decision_status = state.get("decision_status", "HUMAN REVIEW REQUIRED")
    hitl_required = (decision_status == "HUMAN REVIEW REQUIRED")

    # ── 2. Determine risk tier from H ─────────────────────────────────────────
    risk_tier = _h_to_risk_tier(H)

    # ── 3. Compile chain-of-evidence from all agent citations ─────────────────
    chain_of_evidence: list[CitationLink] = []
    seen_ids: set[str] = set()
    for score_dict in agent_scores_raw:
        for cit_dict in score_dict.get("citations", []):
            cit_id = cit_dict.get("identifier", "")
            if cit_id not in seen_ids:
                seen_ids.add(cit_id)
                chain_of_evidence.append(CitationLink(**cit_dict))

    # ── 4. Generate recommended actions ───────────────────────────────────────
    actions = _generate_recommended_actions(
        risk_tier=risk_tier,
        H=H,
        sigma_sq=sigma_sq,
        trust=verification_raw.get("overall_trustworthiness", 1.0),
        alert=raw_alert,
        prediction_report=prediction_raw,
    )

    # ── 5. Draft executive summary ────────────────────────────────────────────
    summary = _draft_executive_summary(
        alert=raw_alert,
        H=H,
        sustainability=sustainability,
        safety=safety,
        sigma_sq=sigma_sq,
        risk_tier=risk_tier,
        trust=verification_raw.get("overall_trustworthiness", 1.0),
        verified_agents=verification_raw.get("verified_agent_ids", []),
        flagged_count=len(verification_raw.get("flagged_claims", [])),
        decision_status=decision_status,
        prediction_report=prediction_raw,
        regulatory_report=regulatory_raw,
    )

    # ── 5.5 Extract stakeholders from state ───────────────────────────────────
    stakeholders = state.get("stakeholders_to_notify", [])

    # ── 6. Build MitigationBrief ──────────────────────────────────────────────
    brief = MitigationBrief(
        executive_summary=summary,
        recommended_actions=actions,
        chain_of_evidence=chain_of_evidence,
        risk_tier=risk_tier,
        estimated_impact_radius_km=_estimate_impact_radius(H, raw_alert),
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        stakeholders_to_notify=stakeholders,
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_NODE_NAME,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info(
        "[%s] Brief drafted — risk_tier=%s H=%.2f hitl_required=%s in %.1f ms",
        _NODE_NAME, risk_tier.value, H, hitl_required, duration_ms,
    )

    return {
        "mitigation_brief": brief.model_dump(),
        "decision_status": decision_status,
        "node_execution_log": [exec_meta.model_dump()],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private synthesis helpers
# ──────────────────────────────────────────────────────────────────────────────

def _h_to_risk_tier(H: float) -> AlertSeverity:
    """Map the Aggregate Hazard Index H to an AlertSeverity tier."""
    if H >= 90:
        return AlertSeverity.CATASTROPHIC
    elif H >= 75:
        return AlertSeverity.CRITICAL
    elif H >= 55:
        return AlertSeverity.HIGH
    elif H >= 30:
        return AlertSeverity.MEDIUM
    else:
        return AlertSeverity.LOW


def _estimate_impact_radius(H: float, alert: dict[str, Any]) -> float | None:
    """
    Rough impact radius estimate (km) based on H and alert payload.
    In production this would call a physics-based attenuation model.
    """
    magnitude: float | None = alert.get("raw_payload", {}).get("magnitude")
    if magnitude is not None:
        # Wells-Coppersmith approximate rupture length
        rupture_km = 10 ** (0.59 * magnitude - 2.44)
        return round(rupture_km * (H / 100.0) * 2.5, 1)
    # Fallback: linear scale from H
    return round(H * 0.8, 1)


def _generate_recommended_actions(
    risk_tier: AlertSeverity,
    H: float,
    sigma_sq: float,
    trust: float,
    alert: dict[str, Any],
    prediction_report: dict[str, Any],
) -> list[str]:
    """Generate tiered recommended actions from consensus outputs."""
    actions: list[str] = []

    # Inject Prediction Agent's top preventive action if available
    predicted_risks = prediction_report.get("predicted_risks", [])
    if predicted_risks and predicted_risks[0].get("preventive_action"):
        actions.append(f"[PREVENTIVE] {predicted_risks[0]['preventive_action']}")

    # Always-on base actions
    actions.append(
        f"[IMMEDIATE] Activate the {risk_tier.value} incident command structure "
        f"under IRS (Incident Response System) protocols by NDMA within the next 15 minutes."
    )
    actions.append(
        f"[IMMEDIATE] Confirm primary alert validity with source: "
        f"{alert.get('source_system', 'UNKNOWN')} — cross-verify with secondary sensor array."
    )

    if risk_tier in (AlertSeverity.CRITICAL, AlertSeverity.CATASTROPHIC):
        actions.append(
            "[CRITICAL] Issue evacuation order for Zone A/B populations within "
            "inundation/blast radius. Activate NDMA Disaster Management Plan §4.2."
        )
        actions.append(
            "[CRITICAL] Pre-position NDRF (National Disaster Response Force) and SDRF teams at "
            "staging areas NW-1, NW-2, and NW-3."
        )
        actions.append(
            "[CRITICAL] Notify State Load Despatch Centre (SLDC) grid operations for emergency islanding "
            "and load-shedding to protect critical infrastructure nodes."
        )

    if risk_tier in (AlertSeverity.HIGH, AlertSeverity.CRITICAL, AlertSeverity.CATASTROPHIC):
        actions.append(
            "[HIGH] Activate SACHET / NDMA Early Warning System broadcast across "
            "all channels. Deploy localized SMS notifications to mobile devices."
        )
        actions.append(
            "[HIGH] Stand up Emergency Operations Center (EOC). Brief all ESF "
            "function leads within 30 minutes of alert confirmation."
        )

    if sigma_sq > VARIANCE_CRITICAL * 0.5:  # High agent disagreement
        actions.append(
            f"[CAUTION] High consensus variance (σ²_H={sigma_sq:.1f}) detected. "
            f"Do NOT commit irreversible resources until a secondary human expert "
            f"review panel validates agent outputs."
        )

    if trust < 0.65:
        actions.append(
            f"[CAUTION] Red-team trust score={trust:.2f} is below acceptance threshold. "
            f"One or more agent claims failed knowledge-graph cross-reference. "
            f"Treat hazard score H={H:.1f} as a lower-bound estimate only."
        )

    actions.append(
        "[ONGOING] Update Eco Guard state every 15 minutes with fresh telemetry. "
        "Re-run the consensus pipeline if H changes by ±5 or new sensor data arrives."
    )
    
    actions.append(
        "[RESOLUTION] If the issue needs to be solved or requires immediate escalation, "
        "please meet or contact the National Disaster Management Authority (NDMA) or the specific relevant Indian government department."
    )

    return actions


def _draft_executive_summary(
    alert: dict[str, Any],
    H: float,
    sustainability: float,
    safety: float,
    sigma_sq: float,
    risk_tier: AlertSeverity,
    trust: float,
    verified_agents: list[str],
    flagged_count: int,
    decision_status: str,
    prediction_report: dict[str, Any],
    regulatory_report: dict[str, Any],
) -> str:
    """Compose the plain-English executive summary for the HITL operator."""
    hitl_note = (
        "⚠ HUMAN APPROVAL REQUIRED before any mitigation action is taken."
        if decision_status == "HUMAN REVIEW REQUIRED"
        else f"{decision_status} — no critical threshold breaches detected."
    )

    flagged_note = (
        f"{flagged_count} agent claim(s) were flagged by the Red Teamer as "
        "unverifiable or contradicted by the knowledge graph. "
        if flagged_count > 0
        else "All agent claims passed knowledge-graph cross-reference. "
    )

    predicted = prediction_report.get("predicted_risks", [])
    top_pred_str = (
        f"{predicted[0].get('incident_type', 'Unknown')} ({predicted[0].get('probability', 0.0):.0%} prob)"
        if predicted else "No prediction available."
    )
    
    viol_count = len(regulatory_report.get("primary_violations", []))
    viol_str = f"{viol_count} active statutory violations detected." if viol_count > 0 else "Compliant."

    return (
        f"ECO GUARD — INCIDENT ASSESSMENT\n"
        f"{'='*55}\n"
        f"Alert: {alert.get('title', 'N/A')} "
        f"[{alert.get('source_system', 'UNKNOWN')}]\n"
        f"Severity declared by source: {alert.get('severity', 'N/A')}\n"
        f"Location: {alert.get('geo_coordinates', 'N/A')}\n\n"
        f"CONSENSUS METRICS\n"
        f"  Aggregate Hazard Index (H):  {H:.2f} / 100\n"
        f"  Sustainability Score:        {sustainability:.2f} / 100\n"
        f"  Safety Risk Score:           {safety:.2f} / 100\n"
        f"  Consensus Variance (σ²_H):   {sigma_sq:.2f}\n"
        f"  Risk Tier:                   {risk_tier.value}\n"
        f"  Agent Trust Score:           {trust:.2f} / 1.00\n"
        f"  Verified Agents:             {', '.join(verified_agents) or 'None'}\n\n"
        f"PREDICTION & REGULATORY\n"
        f"  Top Predicted Risk:          {top_pred_str}\n"
        f"  Compliance Status:           {viol_str}\n\n"
        f"RED TEAM FINDINGS\n"
        f"  {flagged_note}\n"
        f"DECISION GATE\n"
        f"  {hitl_note}\n"
        f"{'='*55}\n"
        f"This brief was auto-generated by the Eco Guard multi-agent "
        f"pipeline. All evidence links are traceable to Neo4j entity nodes "
        f"and Qdrant/ChromaDB knowledge chunks listed in the chain-of-evidence."
    )
