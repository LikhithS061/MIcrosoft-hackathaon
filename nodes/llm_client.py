"""
nodes/llm_client.py
─────────────────────────────────────────────────────────────────────────────
Centralized LLM factory for ECO GUARD.

Single source of truth for model initialization. All agent nodes import
their LLM from here — no duplicate ChatGroq() calls scattered across files.

Architecture:
  - Primary:  Groq (llama-3.3-70b-versatile)  — USE_MOCK_LLM=false
  - Fallback: deterministic mock heuristic      — USE_MOCK_LLM=true
  - LLM is initialized once at module import (thread-safe singleton pattern)

Slim extraction schemas defined here:
  - TelemetryScoreExtraction  (Agent 1)
  - SocioScoreExtraction      (Agent 2)
  - SustainabilityExtraction  (Agent 3)
  - SafetyRiskExtraction      (Agent 4)
  - PredictionExtraction      (Agent 6)

Agent 5 (Regulatory) uses curated RAG + structured summarisation.
Agent 7 (Verification) is rule-based, no LLM schema.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")


# ──────────────────────────────────────────────────────────────────────────────
# Slim extraction schemas  (LLM-facing only — no nested Pydantic complexity)
# ──────────────────────────────────────────────────────────────────────────────

class TelemetryScoreExtraction(BaseModel):
    """
    Minimal structured output contract for the Telemetry Agent (Agent 1).
    The LLM fills ONLY these three fields.
    """
    hazard_score: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Hazard severity score 0–100 based on the physical/technical data. "
            "0 = no hazard, 100 = maximum catastrophic hazard."
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Confidence in the hazard score 0.0–1.0. "
            "Drop below 0.80 if critical fields are missing."
        ),
    )
    rationale: str = Field(
        ..., min_length=30,
        description=(
            "Concise plain-English justification of the score. "
            "Reference specific data fields from the alert payload."
        ),
    )


class SocioScoreExtraction(BaseModel):
    """
    Minimal structured output contract for the Socio-Economic Impact Agent (Agent 2).
    """
    hazard_score: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Socio-economic hazard impact score 0–100. "
            "Consider: population exposure, displacement risk, infrastructure "
            "cascades, health system strain, economic disruption."
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Confidence in the impact assessment 0.0–1.0. "
            "Lower confidence if population/infrastructure data is sparse."
        ),
    )
    rationale: str = Field(
        ..., min_length=30,
        description=(
            "Concise justification of the socio-economic impact score. "
            "Reference population figures, infrastructure nodes, or "
            "secondary cascade effects where applicable."
        ),
    )


class SustainabilityExtraction(BaseModel):
    """
    Structured output for the Sustainability Agent (Agent 3).
    """
    sustainability_score: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Environmental sustainability impact score 0–100. "
            "0 = no environmental impact, 100 = catastrophic irreversible damage. "
            "Consider: resource loss, environmental damage, long-term ecological effects."
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the sustainability assessment 0.0–1.0.",
    )
    rationale: str = Field(
        ..., min_length=30,
        description=(
            "Environmental impact justification. Reference specific indicators: "
            "water loss, air quality, waste volume, soil contamination."
        ),
    )
    resource_loss_estimate: str = Field(
        default="Unknown",
        description="Quantified estimate of resource loss e.g. '10,000 L/day water waste'.",
    )
    environmental_damage_level: str = Field(
        default="MODERATE",
        description="One of: NEGLIGIBLE, LOW, MODERATE, HIGH, SEVERE.",
    )


class SafetyRiskExtraction(BaseModel):
    """
    Structured output for the Safety Auditor (Agent 4).
    """
    safety_risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Safety risk score 0–100. "
            "0 = no safety risk, 100 = immediate life-threatening danger. "
            "Consider: immediate danger to life, injury risk, fire/explosion/structural risk."
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the safety assessment 0.0–1.0.",
    )
    rationale: str = Field(
        ..., min_length=30,
        description="Safety risk justification citing specific hazard indicators.",
    )
    immediate_danger: bool = Field(
        default=False,
        description="True if the incident poses immediate risk to human life.",
    )
    evacuation_recommended: bool = Field(
        default=False,
        description="True if evacuation of residents is recommended.",
    )
    emergency_procedures: list[str] = Field(
        default_factory=list,
        description="List of 3-5 specific immediate emergency actions to take.",
    )


class PredictionExtraction(BaseModel):
    """
    Structured output for the Prediction Agent (Agent 6).
    """
    top_predicted_risk_type: str = Field(
        ...,
        description="The most likely follow-on incident type e.g. 'Fire Outbreak', 'Structural Failure'.",
    )
    top_risk_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability (0.0–1.0) of the top predicted incident occurring.",
    )
    secondary_risk_type: str = Field(
        default="Unknown",
        description="Second most likely follow-on incident type.",
    )
    secondary_risk_probability: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of the secondary predicted incident.",
    )
    prediction_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall confidence in the prediction analysis 0.0–1.0.",
    )
    rationale: str = Field(
        ..., min_length=30,
        description="Reasoning behind the prediction based on current incident data and historical patterns.",
    )
    preventive_actions: list[str] = Field(
        default_factory=list,
        description="List of 2-4 preventive actions to reduce predicted risk probabilities.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# LLM factory — returns structured-output-bound chain or None (mock mode)
# ──────────────────────────────────────────────────────────────────────────────

def build_structured_llm(schema: type[BaseModel]) -> Any | None:
    """
    Build a Groq LLM bound to a specific Pydantic extraction schema.

    Args:
        schema: A Pydantic BaseModel class to bind via .with_structured_output().

    Returns:
        A LangChain Runnable (structured_llm) ready for .ainvoke(),
        or None when USE_MOCK_LLM=true (tests / offline mode).

    Raises:
        EnvironmentError: If USE_MOCK_LLM=false but GROQ_API_KEY is not set.
    """
    if USE_MOCK_LLM:
        logger.info("llm_client: Mock LLM mode active — Groq will NOT be called.")
        return None

    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set in .env but USE_MOCK_LLM=false. "
            "Either add the key or set USE_MOCK_LLM=true for offline mode."
        )

    try:
        from langchain_groq import ChatGroq  # imported lazily — not needed in mock mode
    except ImportError:
        raise ImportError(
            "langchain-groq is not installed. Run: pip install langchain-groq"
        )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,       # Low temp → deterministic scoring, less hallucination
        max_retries=2,
        api_key=GROQ_API_KEY,
    )

    structured_llm = llm.with_structured_output(schema)
    logger.info(
        "llm_client: Groq LLM ready — model=%s schema=%s",
        GROQ_MODEL, schema.__name__,
    )
    return structured_llm
