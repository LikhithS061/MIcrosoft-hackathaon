"""
nodes/consensus_node.py
─────────────────────────────────────────────────────────────────────────────
Mathematical Consensus Node — Evaluation layer.

Responsibilities:
  1. Deserialise all AgentScore dicts from EcoGuardState.agent_scores.
  2. Call the pure math module to compute H and σ²_H.
  3. Evaluate both values against configured critical thresholds.
  4. Set hitl_required = True if either threshold is breached (forced HITL).
  5. Write ConsensusMetrics into the state for the Gatekeeper node.

This node does NO I/O and NO LLM calls — it is a pure deterministic
computation layer, which is why it has its own isolated module.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import ValidationError

from math_consensus import (
    AgentScoreInput,
    ConsensusComputationError,
    ConsensusResult,
    agent_scores_from_state,
    compute_full_consensus,
    evaluate_hitl_threshold,
)
from state import (
    ConsensusMetrics,
    HAZARD_INDEX_CRITICAL,
    NodeExecutionMeta,
    NodeStatus,
    EcoGuardState,
    VARIANCE_CRITICAL,
)

logger = logging.getLogger(__name__)
_NODE_NAME = "consensus_node"


async def consensus_node(state: EcoGuardState) -> dict[str, Any]:
    """
    LangGraph node: Mathematical Consensus Evaluator.

    Args:
        state: EcoGuardState with `agent_scores` and `verification_report`.

    Returns:
        Partial state delta with consensus metrics and `hitl_required` flag.

    Raises:
        ConsensusComputationError: On degenerate inputs (all-zero confidence).
        ValidationError:           On schema violations in agent_scores.
    """
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting consensus computation — trace_id=%s", _NODE_NAME, trace_id)

    raw_scores: list[dict[str, Any]] = state.get("agent_scores", [])
    if not raw_scores:
        raise ConsensusComputationError(
            "consensus_node: agent_scores is empty — cannot compute H or σ²_H."
        )

    # ── 1. Deserialise and validate ───────────────────────────────────────────
    try:
        scores: list[AgentScoreInput] = agent_scores_from_state(raw_scores)
    except ValidationError as exc:
        logger.error("[%s] AgentScoreInput validation failed:\n%s", _NODE_NAME, exc)
        raise

    # ── 1.5 Handle missing regulatory data ────────────────────────────────────
    regulatory_agent_idx = next((i for i, s in enumerate(scores) if s.agent_id == "regulatory_agent"), None)
    if regulatory_agent_idx is not None:
        raw_reg_score = raw_scores[regulatory_agent_idx]
        if raw_reg_score.get("hazard_score") == -1.0 or raw_reg_score.get("status") == "INSUFFICIENT REGULATORY DATA":
            other_scores = [s.hazard_score for i, s in enumerate(scores) if i != regulatory_agent_idx]
            avg_score = sum(other_scores) / len(other_scores) if other_scores else 0.0
            
            # Override Regulatory Agent score with average of others
            scores[regulatory_agent_idx] = scores[regulatory_agent_idx].model_copy(
                update={"hazard_score": avg_score, "confidence": 0.3}
            )
            logger.info("[%s] Replaced missing Regulatory score with average: %.1f", _NODE_NAME, avg_score)

    logger.info("[%s] Processing %d agent score(s).", _NODE_NAME, len(scores))
    for s in scores:
        logger.debug(
            "[%s]   agent=%s C_i=%.1f α_i=%.3f w_i=%.2f",
            _NODE_NAME, s.agent_id, s.hazard_score, s.confidence, s.domain_weight,
        )

    # ── 2. Compute H and σ²_H ────────────────────────────────────────────────
    try:
        result: ConsensusResult = compute_full_consensus(scores)
    except ConsensusComputationError:
        logger.exception("[%s] Consensus computation failed.", _NODE_NAME)
        raise

    logger.info(
        "[%s] H=%.3f σ²_H=%.3f Σ(w·α)=%.4f",
        _NODE_NAME,
        result.aggregate_hazard_index,
        result.consensus_variance,
        result.effective_weight_sum,
    )

    # Also extract the dedicated scores from Agents 3 and 4
    sustainability_score = 0.0
    safety_score = 0.0
    for s in scores:
        if s.agent_id == "sustainability_agent":
            sustainability_score = s.hazard_score
        elif s.agent_id == "safety_auditor":
            safety_score = s.hazard_score

    # Log per-agent breakdown
    for agent_id, breakdown in result.per_agent_contributions.items():
        logger.info(
            "[%s]   [%s] normalized_weight=%.3f squared_deviation=%.3f",
            _NODE_NAME, agent_id,
            breakdown["normalized_weight"],
            breakdown["squared_deviation"],
        )

    symbiosis_idx = state.get("symbiosis_index")
    verification_report = state.get("verification_report", {})
    verification_status = verification_report.get("verification_status", "VERIFIED")
    
    # ── 3. Evaluate HITL thresholds ───────────────────────────────────────────
    decision_status, trigger_reason = evaluate_hitl_threshold(
        result=result,
        safety_risk=safety_score,
        verification_status=verification_status,
        h_critical=HAZARD_INDEX_CRITICAL,
        sigma_critical=VARIANCE_CRITICAL,
        symbiosis_index=symbiosis_idx,
    )

    if decision_status == "HUMAN REVIEW REQUIRED":
        logger.warning(
            "[%s] ⚠ HITL FORCED: %s", _NODE_NAME, trigger_reason
        )
    else:
        logger.info("[%s] Decision: %s. Reason: %s", _NODE_NAME, decision_status, trigger_reason)

    # ── 4. Build ConsensusMetrics ─────────────────────────────────────────────
    metrics = ConsensusMetrics(
        aggregate_hazard_index=round(result.aggregate_hazard_index, 4),
        aggregate_sustainability_score=round(sustainability_score, 4),
        aggregate_safety_score=round(safety_score, 4),
        consensus_variance=round(result.consensus_variance, 4),
        hitl_triggered_reason=trigger_reason,
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_NODE_NAME,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    return {
        "aggregate_hazard_index": metrics.aggregate_hazard_index,
        "aggregate_sustainability_score": metrics.aggregate_sustainability_score,
        "aggregate_safety_score": metrics.aggregate_safety_score,
        "consensus_variance": metrics.consensus_variance,
        "decision_status": decision_status,
        "hitl_triggered_reason": trigger_reason,
        "node_execution_log": [exec_meta.model_dump()],
    }
