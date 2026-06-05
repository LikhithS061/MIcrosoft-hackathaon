"""
math_consensus.py
─────────────────────────────────────────────────────────────────────────────
Pure, side-effect-free mathematical consensus module for The Eco Guard.

Implements the two consensus metrics as specified:

  Aggregate Hazard Index H:
    H = Σ(w_i · α_i · C_i) / Σ(w_i · α_i)

  Consensus Variance σ²_H:
    σ²_H = Σ(w_i · α_i · (C_i − H)²) / Σ(w_i · α_i)

All functions accept Pydantic-validated AgentScore objects and return typed
Python primitives.  No I/O, no logging, no external dependencies.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Input / Output contracts
# ──────────────────────────────────────────────────────────────────────────────

class AgentScoreInput(BaseModel):
    """
    Minimal validated input required by the math module.

    Deliberately decoupled from the larger AgentScore model so this module
    can be tested independently without importing the full graph state.
    """

    agent_id: str
    hazard_score: float = Field(..., ge=0.0, le=100.0, description="C_i")
    confidence: float = Field(..., ge=0.0, le=1.0, description="α_i")
    domain_weight: float = Field(..., gt=0.0, description="w_i")


class ConsensusResult(NamedTuple):
    """Immutable, typed result bundle from the consensus computation."""

    aggregate_hazard_index: float   # H  ∈ [0, 100]
    consensus_variance: float       # σ²_H ≥ 0
    effective_weight_sum: float     # Σ(w_i · α_i) — useful for diagnostics
    per_agent_contributions: dict[str, dict[str, float]]
    """Breakdown of each agent's weighted contribution for audit trails."""


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

class ConsensusComputationError(Exception):
    """Raised when the consensus calculation cannot proceed safely."""


def compute_aggregate_hazard(scores: list[AgentScoreInput]) -> float:
    """
    Compute the Aggregate Hazard Index H.

        H = Σ(w_i · α_i · C_i) / Σ(w_i · α_i)

    Args:
        scores: Non-empty list of validated agent scores.

    Returns:
        H clamped to [0.0, 100.0].

    Raises:
        ConsensusComputationError: If the effective weight sum is zero
            (all agents reported zero confidence), making H undefined.
    """
    if not scores:
        raise ConsensusComputationError(
            "Cannot compute aggregate hazard index: no agent scores provided."
        )

    numerator: float = sum(s.domain_weight * s.confidence * s.hazard_score for s in scores)
    denominator: float = sum(s.domain_weight * s.confidence for s in scores)

    if math.isclose(denominator, 0.0, abs_tol=1e-9):
        raise ConsensusComputationError(
            "Effective weight sum Σ(w_i · α_i) is zero.  "
            "All agents reported zero confidence — cannot compute H."
        )

    H = numerator / denominator
    return max(0.0, min(100.0, H))   # defensive clamp; should never trigger


def compute_consensus_variance(scores: list[AgentScoreInput], H: float) -> float:
    """
    Compute the Consensus Variance σ²_H.

        σ²_H = Σ(w_i · α_i · (C_i − H)²) / Σ(w_i · α_i)

    Args:
        scores: Non-empty list of validated agent scores.
        H:      Pre-computed Aggregate Hazard Index from compute_aggregate_hazard().

    Returns:
        σ²_H ≥ 0.

    Raises:
        ConsensusComputationError: If the effective weight sum is zero.
    """
    if not scores:
        raise ConsensusComputationError(
            "Cannot compute consensus variance: no agent scores provided."
        )

    denominator: float = sum(s.domain_weight * s.confidence for s in scores)
    if math.isclose(denominator, 0.0, abs_tol=1e-9):
        raise ConsensusComputationError(
            "Effective weight sum Σ(w_i · α_i) is zero — cannot compute σ²_H."
        )

    numerator: float = sum(
        s.domain_weight * s.confidence * (s.hazard_score - H) ** 2
        for s in scores
    )
    return numerator / denominator


def compute_full_consensus(scores: list[AgentScoreInput]) -> ConsensusResult:
    """
    Unified entry-point: compute both H and σ²_H in a single pass.

    Prefer this over calling the two functions separately to avoid
    redundant weight-sum calculations and ensure H is consistent.

    Args:
        scores: Non-empty list of validated agent scores.

    Returns:
        ConsensusResult named-tuple with all metrics and a per-agent breakdown.

    Raises:
        ConsensusComputationError: On degenerate inputs (see sub-functions).
    """
    if not scores:
        raise ConsensusComputationError("scores list must not be empty.")

    # Precompute effective weights once
    effective_weights: list[float] = [s.domain_weight * s.confidence for s in scores]
    weight_sum: float = sum(effective_weights)

    if math.isclose(weight_sum, 0.0, abs_tol=1e-9):
        raise ConsensusComputationError(
            "All agents reported zero confidence — consensus is undefined."
        )

    # ── Normalize Scores ──────────────────────────────────────────────────────
    normalized_scores = []
    for s in scores:
        clamped_score = max(0.0, min(100.0, s.hazard_score))
        normalized_scores.append(s.model_copy(update={"hazard_score": clamped_score}))
        
    # ── H ─────────────────────────────────────────────────────────────────────
    H: float = sum(ew * s.hazard_score for ew, s in zip(effective_weights, normalized_scores)) / weight_sum
    H = max(0.0, min(100.0, H))

    # ── σ²_H ──────────────────────────────────────────────────────────────────
    sigma_sq: float = (
        sum(ew * (s.hazard_score - H) ** 2 for ew, s in zip(effective_weights, normalized_scores))
        / weight_sum
    )

    # ── Per-agent audit breakdown ──────────────────────────────────────────────
    per_agent: dict[str, dict[str, float]] = {
        s.agent_id: {
            "C_i": s.hazard_score,
            "alpha_i": s.confidence,
            "w_i": s.domain_weight,
            "effective_weight": ew,
            "normalized_weight": ew / weight_sum,
            "squared_deviation": (s.hazard_score - H) ** 2,
        }
        for s, ew in zip(normalized_scores, effective_weights)
    }

    return ConsensusResult(
        aggregate_hazard_index=H,
        consensus_variance=sigma_sq,
        effective_weight_sum=weight_sum,
        per_agent_contributions=per_agent,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Circular Opportunity Engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_symbiosis_index(
    utility: float, 
    env_benefit: float, 
    econ_benefit: float, 
    compliance: float
) -> float:
    """
    Compute the Symbiosis Index (Opportunity Score) for Circular Economy mode.
    
    Weights are distributed as:
    Utility: 30%, Environmental Benefit: 30%, Economic Benefit: 20%, Compliance: 20%
    All inputs should be [0, 100]. Returns score [0, 100].
    """
    weights = [0.3, 0.3, 0.2, 0.2]
    values = [utility, env_benefit, econ_benefit, compliance]
    
    score = sum(w * v for w, v in zip(weights, values))
    return max(0.0, min(100.0, score))


# ──────────────────────────────────────────────────────────────────────────────
# Threshold evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_hitl_threshold(
    result: ConsensusResult,
    safety_risk: float = 0.0,
    verification_status: str = "VERIFIED",
    h_critical: float = 90.0,
    sigma_critical: float = 150.0,
    symbiosis_index: float | None = None,
    symbiosis_critical: float = 70.0,
) -> tuple[str, str | None]:
    """
    Determine the tier of the decision based on the defined Decision Matrix.

    Returns:
        (decision_status: str, reason: str | None)
        Status is one of: AUTO APPROVED, MONITORING, REVIEW RECOMMENDED, HUMAN REVIEW REQUIRED.
    """
    H = result.aggregate_hazard_index
    Var = result.consensus_variance
    
    reasons = []
    
    # 1. Mandatory Human Review checks
    if H >= h_critical:
        reasons.append(f"Aggregate Hazard Index (H={H:.2f}) >= {h_critical}")
    if Var >= sigma_critical:
        reasons.append(f"Consensus Variance (σ²_H={Var:.2f}) >= {sigma_critical}")
    if safety_risk >= 90.0:
        reasons.append(f"Safety Risk ({safety_risk:.2f}) >= 90")
    if verification_status != "VERIFIED":
        reasons.append(f"Verification Status is '{verification_status}'")
    if symbiosis_index is not None and symbiosis_index >= symbiosis_critical:
        reasons.append(f"Symbiosis Index={symbiosis_index:.2f} indicates a high-value circular opportunity")
        
    if reasons:
        return "HUMAN REVIEW REQUIRED", "Triggered because: " + " | ".join(reasons)
        
    # 2. Recommended Review
    if 75 <= H < 90 and Var < 150:
        return "REVIEW RECOMMENDED", f"H is {H:.2f} (High Risk) and Variance is acceptable."
        
    # 3. Monitoring
    if 50 <= H < 75 and Var < 100:
        return "MONITORING", f"H is {H:.2f} (Moderate Risk) with stable consensus."
        
    # 4. Auto Approved
    if H < 50 and Var < 50:
        return "AUTO APPROVED", f"H is {H:.2f} (Low Risk) with strong consensus."
        
    # Fallback to REVIEW RECOMMENDED if it hits some odd edge case
    return "REVIEW RECOMMENDED", "Metrics did not match strict auto-approval thresholds."


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build AgentScoreInput list from EcoGuardState's agent_scores
# ──────────────────────────────────────────────────────────────────────────────

def agent_scores_from_state(raw_scores: list[dict]) -> list[AgentScoreInput]:
    """
    Deserialize the raw dict list stored in EcoGuardState.agent_scores
    into validated AgentScoreInput objects.

    Raises:
        pydantic.ValidationError: If any score dict violates the schema.
    """
    return [AgentScoreInput(**s) for s in raw_scores]
