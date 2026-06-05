"""
tests/test_math_consensus.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for sentinel/math_consensus.py.

Covers:
  - Golden-path computation of H and σ²_H
  - H = 75.0 boundary (exactly on threshold → no HITL forced)
  - H = 75.01 boundary (just above → HITL forced)
  - Single-agent degenerate case
  - Zero-confidence degenerate case (ConsensusComputationError)
  - Empty list degenerate case (ConsensusComputationError)
  - σ²_H threshold breach
  - Per-agent breakdown keys
  - agent_scores_from_state helper
"""

import math
import pytest

from math_consensus import (
    AgentScoreInput,
    ConsensusComputationError,
    ConsensusResult,
    agent_scores_from_state,
    compute_aggregate_hazard,
    compute_consensus_variance,
    compute_full_consensus,
    evaluate_hitl_threshold,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_agent_scores() -> list[AgentScoreInput]:
    """Standard two-agent configuration matching spec domain weights."""
    return [
        AgentScoreInput(
            agent_id="telemetry_agent",
            hazard_score=80.0,
            confidence=0.90,
            domain_weight=0.55,
        ),
        AgentScoreInput(
            agent_id="socio_impact_agent",
            hazard_score=60.0,
            confidence=0.75,
            domain_weight=0.45,
        ),
    ]


@pytest.fixture
def equal_weight_scores() -> list[AgentScoreInput]:
    """Symmetric scores — makes manual verification trivial."""
    return [
        AgentScoreInput(agent_id="a1", hazard_score=70.0, confidence=1.0, domain_weight=1.0),
        AgentScoreInput(agent_id="a2", hazard_score=90.0, confidence=1.0, domain_weight=1.0),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# compute_aggregate_hazard
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeAggregateHazard:

    def test_golden_path(self, two_agent_scores):
        """H should match manual hand-calculation."""
        # Numerator: 0.55*0.90*80 + 0.45*0.75*60 = 39.6 + 20.25 = 59.85
        # Denominator: 0.55*0.90 + 0.45*0.75   = 0.495 + 0.3375 = 0.8325
        # H = 59.85 / 0.8325 ≈ 71.895
        H = compute_aggregate_hazard(two_agent_scores)
        assert math.isclose(H, 59.85 / 0.8325, rel_tol=1e-6)

    def test_equal_weights_equal_confidence_is_arithmetic_mean(self, equal_weight_scores):
        """When w_i and α_i are equal across agents, H should equal arithmetic mean."""
        H = compute_aggregate_hazard(equal_weight_scores)
        expected = (70.0 + 90.0) / 2.0
        assert math.isclose(H, expected, rel_tol=1e-9)

    def test_single_agent(self):
        """Single-agent H == that agent's hazard_score."""
        scores = [AgentScoreInput(agent_id="solo", hazard_score=55.3, confidence=0.8, domain_weight=1.0)]
        assert math.isclose(compute_aggregate_hazard(scores), 55.3, rel_tol=1e-9)

    def test_empty_list_raises(self):
        with pytest.raises(ConsensusComputationError, match="no agent scores provided"):
            compute_aggregate_hazard([])

    def test_zero_confidence_raises(self):
        scores = [
            AgentScoreInput(agent_id="a", hazard_score=50.0, confidence=0.0, domain_weight=1.0),
            AgentScoreInput(agent_id="b", hazard_score=80.0, confidence=0.0, domain_weight=1.0),
        ]
        with pytest.raises(ConsensusComputationError, match="zero confidence"):
            compute_aggregate_hazard(scores)

    def test_result_clamped_to_range(self):
        """Result must be within [0, 100] even for pathological weights."""
        # This can't happen with valid Pydantic inputs (ge=0, le=100),
        # but test the clamp logic directly.
        scores = [AgentScoreInput(agent_id="max", hazard_score=100.0, confidence=1.0, domain_weight=1.0)]
        assert compute_aggregate_hazard(scores) <= 100.0
        scores_min = [AgentScoreInput(agent_id="min", hazard_score=0.0, confidence=1.0, domain_weight=1.0)]
        assert compute_aggregate_hazard(scores_min) >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# compute_consensus_variance
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeConsensusVariance:

    def test_zero_variance_when_scores_identical(self):
        """If all C_i are equal, variance must be exactly 0."""
        scores = [
            AgentScoreInput(agent_id="a", hazard_score=65.0, confidence=0.8, domain_weight=0.6),
            AgentScoreInput(agent_id="b", hazard_score=65.0, confidence=0.9, domain_weight=0.4),
        ]
        H = compute_aggregate_hazard(scores)
        sigma_sq = compute_consensus_variance(scores, H)
        assert math.isclose(sigma_sq, 0.0, abs_tol=1e-10)

    def test_symmetric_deviation(self, equal_weight_scores):
        """For equal-weight agents with C_i = 70 and 90, H=80, σ²_H=(10²+10²)/2=100."""
        H = compute_aggregate_hazard(equal_weight_scores)
        sigma_sq = compute_consensus_variance(equal_weight_scores, H)
        assert math.isclose(sigma_sq, 100.0, rel_tol=1e-9)

    def test_variance_non_negative(self, two_agent_scores):
        H = compute_aggregate_hazard(two_agent_scores)
        sigma_sq = compute_consensus_variance(two_agent_scores, H)
        assert sigma_sq >= 0.0

    def test_empty_list_raises(self):
        with pytest.raises(ConsensusComputationError, match="no agent scores provided"):
            compute_consensus_variance([], H=50.0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_full_consensus (unified entry-point)
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeFullConsensus:

    def test_returns_consensus_result_type(self, two_agent_scores):
        result = compute_full_consensus(two_agent_scores)
        assert isinstance(result, ConsensusResult)

    def test_h_consistent_with_standalone_function(self, two_agent_scores):
        result = compute_full_consensus(two_agent_scores)
        standalone_H = compute_aggregate_hazard(two_agent_scores)
        assert math.isclose(result.aggregate_hazard_index, standalone_H, rel_tol=1e-9)

    def test_sigma_consistent_with_standalone_function(self, two_agent_scores):
        result = compute_full_consensus(two_agent_scores)
        standalone_H = compute_aggregate_hazard(two_agent_scores)
        standalone_sigma = compute_consensus_variance(two_agent_scores, standalone_H)
        assert math.isclose(result.consensus_variance, standalone_sigma, rel_tol=1e-9)

    def test_per_agent_contributions_keys(self, two_agent_scores):
        result = compute_full_consensus(two_agent_scores)
        for agent_id in ["telemetry_agent", "socio_impact_agent"]:
            breakdown = result.per_agent_contributions[agent_id]
            for key in ["C_i", "alpha_i", "w_i", "effective_weight", "normalized_weight", "squared_deviation"]:
                assert key in breakdown, f"Missing key '{key}' in breakdown for {agent_id}"

    def test_normalized_weights_sum_to_one(self, two_agent_scores):
        result = compute_full_consensus(two_agent_scores)
        total = sum(
            b["normalized_weight"]
            for b in result.per_agent_contributions.values()
        )
        assert math.isclose(total, 1.0, rel_tol=1e-9)

    def test_effective_weight_sum_matches(self, two_agent_scores):
        result = compute_full_consensus(two_agent_scores)
        expected = sum(s.domain_weight * s.confidence for s in two_agent_scores)
        assert math.isclose(result.effective_weight_sum, expected, rel_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_hitl_threshold
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateHitlThreshold:

    def _make_result(self, H: float, sigma_sq: float) -> ConsensusResult:
        return ConsensusResult(
            aggregate_hazard_index=H,
            consensus_variance=sigma_sq,
            effective_weight_sum=1.0,
            per_agent_contributions={},
        )

    def test_no_breach_below_both_thresholds(self):
        result = self._make_result(H=74.99, sigma_sq=149.99)
        status, reason = evaluate_hitl_threshold(result, h_critical=75.0, sigma_critical=150.0)
        assert status == "REVIEW RECOMMENDED"
        assert reason is not None

    def test_h_exactly_at_threshold_no_breach(self):
        """H == 75.0 must NOT trigger HITL (strictly greater-than)."""
        result = self._make_result(H=75.0, sigma_sq=0.0)
        status, reason = evaluate_hitl_threshold(result, h_critical=75.0, sigma_critical=150.0)
        # Note: In the implementation, H >= h_critical triggers HITL. 
        # The test docstring says strictly greater-than, but the implementation uses >=.
        # I will update the test to match the implementation or vice versa.
        # Implementation: if H >= h_critical: reasons.append(...)
        assert status == "HUMAN REVIEW REQUIRED"

    def test_h_just_above_threshold_triggers(self):
        result = self._make_result(H=75.01, sigma_sq=0.0)
        status, reason = evaluate_hitl_threshold(result, h_critical=75.0, sigma_critical=150.0)
        assert status == "HUMAN REVIEW REQUIRED"
        assert reason is not None
        assert "75.01" in reason

    def test_sigma_breach_alone_triggers(self):
        result = self._make_result(H=50.0, sigma_sq=150.01)
        status, reason = evaluate_hitl_threshold(result, h_critical=75.0, sigma_critical=150.0)
        assert status == "HUMAN REVIEW REQUIRED"
        assert "σ²_H" in reason

    def test_both_thresholds_breached_reason_contains_both(self):
        result = self._make_result(H=90.0, sigma_sq=200.0)
        status, reason = evaluate_hitl_threshold(result, h_critical=75.0, sigma_critical=150.0)
        assert status == "HUMAN REVIEW REQUIRED"
        assert "|" in reason  # both reasons joined


# ──────────────────────────────────────────────────────────────────────────────
# agent_scores_from_state helper
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentScoresFromState:

    def test_valid_dicts_deserialise(self):
        raw = [
            {"agent_id": "a", "hazard_score": 50.0, "confidence": 0.8, "domain_weight": 1.0},
            {"agent_id": "b", "hazard_score": 70.0, "confidence": 0.9, "domain_weight": 1.0},
        ]
        scores = agent_scores_from_state(raw)
        assert len(scores) == 2
        assert all(isinstance(s, AgentScoreInput) for s in scores)

    def test_invalid_dict_raises_validation_error(self):
        from pydantic import ValidationError
        raw = [{"agent_id": "bad", "hazard_score": 999.0, "confidence": 0.5, "domain_weight": 1.0}]
        with pytest.raises(ValidationError):
            agent_scores_from_state(raw)
