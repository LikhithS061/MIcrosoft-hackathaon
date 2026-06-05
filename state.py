"""
state.py
─────────────────────────────────────────────────────────────────────────────
Central state contract for ECO GUARD.

Extends v2.0 with 5 additional agent domains:
  Agent 3 — Sustainability Agent
  Agent 4 — Safety Auditor
  Agent 5 — Regulatory Intelligence Agent
  Agent 6 — Prediction Agent
  Agent 7 — Verification Agent (unchanged from v2.0)

Every node receives the full EcoGuardState and returns a *partial dict*
containing only the fields it modifies.  LangGraph merges those deltas via
its built-in reducer (last-write-wins for scalar fields, append for lists
annotated with operator.add).

Strict Pydantic sub-models are used for every nested payload so that
schema violations surface immediately at node boundaries, not silently.
"""

from __future__ import annotations

import operator
import uuid
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    """ISO-aligned severity ladder."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    CATASTROPHIC = "CATASTROPHIC"


class OperationalMode(str, Enum):
    """The three operating modes of Eco Guard X."""
    COMMUNITY_INTELLIGENCE = "COMMUNITY_INTELLIGENCE"
    ENVIRONMENTAL_INTELLIGENCE = "ENVIRONMENTAL_INTELLIGENCE"
    CIRCULAR_ECONOMY_INTELLIGENCE = "CIRCULAR_ECONOMY_INTELLIGENCE"


class HITLDecision(str, Enum):
    """Possible human decisions at the approval gate."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INVESTIGATE = "INVESTIGATE"


class NodeStatus(str, Enum):
    """Execution status for each graph node — used for observability."""
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    INSUFFICIENT_DATA = "INSUFFICIENT REGULATORY DATA"


class IncidentType(str, Enum):
    """Canonical incident type labels used across all 10 demo incidents."""
    WATER_CONTAMINATION = "Water Contamination"
    FIRE_HAZARD = "Fire Hazard"
    GAS_LEAK = "Gas Leak"
    INFRASTRUCTURE_FAILURE = "Infrastructure Failure"
    FLOODING = "Flooding"
    ELECTRICAL_FAULT = "Electrical Fault"
    RESOURCE_WASTE = "Resource Waste"
    SUSTAINABILITY_VIOLATION = "Sustainability Violation"
    ENVIRONMENTAL_HAZARD = "Environmental Hazard"
    PUBLIC_SAFETY = "Public Safety"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models (Pydantic v2)
# ──────────────────────────────────────────────────────────────────────────────

class RawAlert(BaseModel):
    """The validated, deserialized anomaly alert ingested by the Planner."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=3, max_length=256)
    description: str = Field(..., min_length=10)
    severity: AlertSeverity
    location: str = Field(
        ...,
        description="e.g. 'Apartment Complex', 'Industrial Park', 'Water Treatment Facility'",
    )
    timestamp_utc: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of the anomaly detection",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
    )
    raw_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Pass-through of the original upstream JSON for auditability",
    )


class AgentScore(BaseModel):
    """
    Scored output produced by a single domain agent (Agents 1–6).

    Each agent produces:
      • hazard_score  C_i ∈ [0, 100]
      • confidence    α_i ∈ [0, 1]
      • domain_weight w_i  (predefined per agent type)
      • citations     – list of evidence links used to derive the score
    """

    agent_id: str = Field(..., description="Stable identifier, e.g. 'telemetry_agent'")
    agent_type: str = Field(..., description="Human-readable domain label")
    hazard_score: float = Field(..., ge=0.0, le=100.0, description="C_i")
    confidence: float = Field(..., ge=0.0, le=1.0, description="α_i")
    domain_weight: float = Field(..., gt=0.0, description="w_i — predefined, never learned")
    rationale: str = Field(..., min_length=10, description="Plain-English justification")
    citations: list["CitationLink"] = Field(default_factory=list)
    status: NodeStatus = Field(default=NodeStatus.COMPLETED)
    # Agent-specific extended fields (optional)
    extended_report: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific structured report (sustainability/safety/regulatory/prediction)",
    )

    @field_validator("domain_weight")
    @classmethod
    def weight_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("domain_weight must be strictly positive")
        return v


class CitationLink(BaseModel):
    """A verifiable evidence pointer that can be rendered in the HITL UI."""

    source_type: str = Field(
        ...,
        description="'NEO4J_NODE', 'QDRANT_CHUNK', 'SENSOR_RAW', 'REGULATORY_KB', 'CHROMA_RAG', 'LEARNING_DB'",
    )
    identifier: str = Field(..., description="Node ID, chunk UUID, regulation ID, etc.")
    label: str = Field(..., description="Human-readable label for the link")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    url: str | None = Field(
        default=None,
        description="Optional deep-link for HITL UI",
    )


class VerificationReport(BaseModel):
    """Output produced by the Verification Agent (Agent 7)."""

    verified_agent_ids: list[str] = Field(
        default_factory=list,
        description="Agent IDs whose claims passed cross-reference checks",
    )
    flagged_claims: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured list of hallucinated or unverifiable claims",
    )
    overall_trustworthiness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Aggregate trust score across all agent outputs",
    )
    neo4j_cross_refs: list[str] = Field(default_factory=list)
    qdrant_cross_refs: list[str] = Field(default_factory=list)
    summary: str = Field(..., min_length=10)
    verification_status: str = Field(
        default="VERIFIED",
        description="'VERIFIED' or 'VERIFICATION FAILED'",
    )


class ConsensusMetrics(BaseModel):
    """Output of the mathematical consensus node."""

    aggregate_hazard_index: float = Field(..., ge=0.0, le=100.0, description="H")
    aggregate_sustainability_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Weighted sustainability score from Agent 3",
    )
    aggregate_safety_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Safety risk score from Agent 4",
    )
    consensus_variance: float = Field(..., ge=0.0, description="σ²_H")
    agent_disagreement_level: str = Field(
        default="LOW",
        description="'LOW', 'MEDIUM', 'HIGH' — derived from consensus_variance",
    )
    hitl_triggered_reason: str | None = Field(
        default=None,
        description="Non-None when either threshold breach forces HITL",
    )


class RegulatoryFinding(BaseModel):
    """A single applicable government regulation with compliance assessment."""

    regulation_id: str
    title: str
    source: str
    section: str
    summary: str
    compliance_status: str = Field(
        ...,
        description="'VIOLATION', 'AT_RISK', 'COMPLIANT', 'UNKNOWN'",
    )
    legal_implications: str = Field(default="")
    retrieval_method: str = Field(default="Curated Fallback")


class RegulatoryReport(BaseModel):
    """Output of the Regulatory Intelligence Agent (Agent 5)."""

    applicable_regulations: list[RegulatoryFinding] = Field(default_factory=list)
    primary_violations: list[str] = Field(
        default_factory=list,
        description="List of regulation IDs in active violation",
    )
    compliance_summary: str = Field(default="")
    legal_action_required: bool = Field(default=False)
    stakeholders_to_notify: list[str] = Field(default_factory=list)


class PredictedRisk(BaseModel):
    """A single predicted future incident with probability."""

    incident_type: str
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    time_horizon: str = Field(description="e.g. '7 days', '30 days', '3 months'")
    reasoning: str
    preventive_action: str


class PredictionReport(BaseModel):
    """Output of the Prediction Agent (Agent 6)."""

    predicted_risks: list[PredictedRisk] = Field(default_factory=list)
    historical_pattern_summary: str = Field(default="")
    learning_db_incidents_used: int = Field(default=0)


class HistoricalCase(BaseModel):
    """A similar historical incident retrieved from the Learning Engine."""

    incident_id: str
    incident_type: str
    location: str
    severity: str
    hazard_score: float
    outcome: str
    lessons_learned: str
    similarity_score: float
    timestamp: str


class MitigationBrief(BaseModel):
    """The final plain-English output drafted by the Gatekeeper Node."""

    executive_summary: str
    recommended_actions: list[str]
    chain_of_evidence: list[CitationLink]
    risk_tier: AlertSeverity
    estimated_impact_radius_km: float | None = None
    generated_at_utc: str
    stakeholders_to_notify: list[str] = Field(default_factory=list)
    explainability_report: str = Field(default="")


class NodeExecutionMeta(BaseModel):
    """Tracks per-node status for observability and Langfuse tracing."""

    node_name: str
    status: NodeStatus
    duration_ms: float | None = None
    error_message: str | None = None


class CircularOpportunityReport(BaseModel):
    """Output produced by the Circular Economy Agent (Agent 7)."""

    symbiosis_index: float = Field(default=0.0, ge=0.0, le=100.0, description="Opportunity Score")
    potential_recipients: list[str] = Field(default_factory=list)
    estimated_benefits: str = Field(default="")
    environmental_savings: str = Field(default="")
    economic_value: str = Field(default="")
    opportunity_type: str = Field(default="Resource Reuse")


# ──────────────────────────────────────────────────────────────────────────────
# Top-level Graph State
# ──────────────────────────────────────────────────────────────────────────────

class EcoGuardState(TypedDict, total=False):
    """
    The single source of truth flowing through the LangGraph StateGraph.

    Design principles:
      1. All fields are optional (total=False) so nodes return minimal deltas.
      2. List fields use `Annotated[list[T], operator.add]` to enable safe
         parallel append semantics — LangGraph will *merge* lists from
         concurrent branches rather than overwriting them.
      3. Scalar fields use last-write-wins (LangGraph default).
      4. Pydantic sub-models are stored as their `.model_dump()` dicts so the
         state remains JSON-serialisable for checkpointing.
    """

    # ── Ingestion ──────────────────────────────────────────────────────────────
    trace_id: str                            # UUID injected by Planner, immutable
    raw_alert: dict[str, Any]               # Validated RawAlert.model_dump()
    operational_mode: str                   # Selected OperationalMode enum value

    # ── Historical Context (from Learning Engine) ──────────────────────────────
    historical_similar_cases: list[dict[str, Any]]  # Top-k similar past incidents

    # ── Agent Outputs (parallel-safe append) ──────────────────────────────────
    agent_scores: Annotated[list[dict[str, Any]], operator.add]
    """Each parallel agent appends one AgentScore.model_dump() here."""

    # ── Sustainability Agent output ────────────────────────────────────────────
    sustainability_score: float             # 0-100 from Agent 3
    environmental_impact_report: str

    # ── Safety Auditor output ──────────────────────────────────────────────────
    safety_risk_score: float                # 0-100 from Agent 4
    emergency_response_plan: list[str]

    # ── Regulatory Intelligence output ────────────────────────────────────────
    regulatory_report: dict[str, Any]      # RegulatoryReport.model_dump()
    stakeholders_to_notify: list[str]

    # ── Prediction Agent output ───────────────────────────────────────────────
    prediction_report: dict[str, Any]      # PredictionReport.model_dump()

    # ── Verification ──────────────────────────────────────────────────────────
    verification_report: dict[str, Any]     # VerificationReport.model_dump()

    # ── Consensus Mathematics ─────────────────────────────────────────────────
    aggregate_hazard_index: float           # H
    aggregate_sustainability_score: float   # Weighted sustainability
    aggregate_safety_score: float           # Weighted safety risk
    consensus_variance: float               # σ²_H
    agent_disagreement_level: str           # LOW / MEDIUM / HIGH
    
    # ── Circular Opportunity Engine ───────────────────────────────────────────
    circular_opportunity_report: dict[str, Any] # CircularOpportunityReport.model_dump()
    symbiosis_index: float                  # Opportunity Score for Mode 3

    decision_status: str                    # AUTO APPROVED, MONITORING, REVIEW RECOMMENDED, HUMAN REVIEW REQUIRED
    hitl_triggered_reason: str | None       # Human-readable threshold breach message

    # ── Gatekeeper Output ──────────────────────────────────────────────────────
    mitigation_brief: dict[str, Any]        # MitigationBrief.model_dump()

    # ── Human-in-the-Loop ─────────────────────────────────────────────────────
    hitl_decision: str                      # HITLDecision enum value
    human_feedback_notes: str | None        # Optional free-text from operator

    # ── Observability ─────────────────────────────────────────────────────────
    node_execution_log: Annotated[list[dict[str, Any]], operator.add]
    """Append-only execution metadata from every node."""

    langfuse_trace_url: str | None          # Set by Planner after trace creation


# ──────────────────────────────────────────────────────────────────────────────
# EcoGuard-wide configuration constants (loaded once at import time)
# ──────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()

HAZARD_INDEX_CRITICAL: float = float(os.getenv("HAZARD_INDEX_CRITICAL", "75.0"))
VARIANCE_CRITICAL: float = float(os.getenv("VARIANCE_CRITICAL", "150.0"))
SAFETY_SCORE_CRITICAL: float = float(os.getenv("SAFETY_SCORE_CRITICAL", "80.0"))

# Domain weights for all 6 scoring agents (Agent 7 is verification — no score weight)
DOMAIN_WEIGHTS: dict[str, float] = {
    "telemetry_agent":      0.55,  # Physical sensor data — highest evidentiary weight
    "socio_impact_agent":   0.45,  # Population-level secondary data
    "sustainability_agent": 0.35,  # Environmental dimension
    "safety_auditor":       0.50,  # Safety risk + emergency response
    "regulatory_agent":     0.40,  # Regulatory compliance dimension
    "prediction_agent":     0.30,  # Forecasting (lower weight — future uncertainty)
    "circular_economy_agent": 0.50, # Agent 7, strong weight in Mode 3
}
