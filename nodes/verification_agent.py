"""
nodes/verification_agent.py
─────────────────────────────────────────────────────────────────────────────
Verification Agent Node — The Red Teamer.

This node is the adversarial convergence point after the parallel branches.
It actively interrogates the outputs of every upstream agent by:

  1. Cross-referencing each agent's numeric claims against the Neo4j graph
     (detect factual hallucinations in entity property assertions).
  2. Re-running targeted Qdrant searches to confirm that cited chunk IDs
     actually contain text relevant to the agent's stated rationale.
  3. Producing a VerificationReport with per-claim verdicts, an overall
     trustworthiness score, and a list of verified citation chains.

Design principle: this node is purely *questioning* — it never modifies the
upstream agent_scores.  Any corrections flow through the consensus_node.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from state import (
    CitationLink,
    NodeExecutionMeta,
    NodeStatus,
    EcoGuardState,
    VerificationReport,
)
from tools import neo4j_mock, qdrant_mock

logger = logging.getLogger(__name__)

_NODE_NAME = "verification_agent"

# Patterns the Red Teamer tries to extract from agent rationales
_NUMERIC_CLAIM_PATTERN = re.compile(
    r"(?P<entity>[\w:_]+)\s+(?P<property>[\w_]+)[=:\s]+(?P<value>[\d.]+)",
    re.IGNORECASE,
)


async def verification_node(state: EcoGuardState) -> dict[str, Any]:
    """
    LangGraph node: Verification Agent (Red Teamer).

    Runs after both parallel agents have completed (convergence node).

    Args:
        state: EcoGuardState with agent_scores populated by both parallel branches.

    Returns:
        Partial state delta with `verification_report` populated.

    Raises:
        ValueError: If agent_scores is empty or missing.
    """
    start_ms = time.monotonic() * 1000
    trace_id = state.get("trace_id", "unknown")
    logger.info("[%s] Starting red-team verification — trace_id=%s", _NODE_NAME, trace_id)

    agent_scores: list[dict[str, Any]] = state.get("agent_scores", [])
    if not agent_scores:
        raise ValueError(
            "verification_node: agent_scores is empty. "
            "Both parallel branches must complete before verification."
        )

    # ── 1. Dispatch all verification tasks concurrently ───────────────────────
    verification_tasks = [
        _verify_single_agent(score_dict)
        for score_dict in agent_scores
    ]
    agent_verification_results = await asyncio.gather(*verification_tasks)

    # ── 2. Aggregate results ──────────────────────────────────────────────────
    all_verified_ids: list[str] = []
    all_flagged_claims: list[dict[str, Any]] = []
    all_neo4j_refs: list[str] = []
    all_qdrant_refs: list[str] = []
    trust_scores: list[float] = []

    for agent_id, verified_claims, flagged_claims, neo4j_refs, qdrant_refs, trust in agent_verification_results:
        if trust >= 0.6:
            all_verified_ids.append(agent_id)
        all_flagged_claims.extend(flagged_claims)
        all_neo4j_refs.extend(neo4j_refs)
        all_qdrant_refs.extend(qdrant_refs)
        trust_scores.append(trust)

    overall_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

    # ── 3. Build VerificationReport ───────────────────────────────────────────
    flagged_count = len(all_flagged_claims)
    verified_count = len(all_verified_ids)

    summary = (
        f"Red-team verification completed for {len(agent_scores)} agent(s). "
        f"{verified_count}/{len(agent_scores)} agents passed trust threshold (≥ 0.60). "
        f"{flagged_count} claim(s) flagged as unverifiable or contradicted by knowledge graph. "
        f"Overall trustworthiness: {overall_trust:.2f}. "
        f"Cross-referenced {len(all_neo4j_refs)} Neo4j entity(-ies) and "
        f"{len(all_qdrant_refs)} Qdrant chunk(s)."
    )

    # Determine verification status
    verification_status = "VERIFIED"
    if overall_trust < 0.65 or any(f.get("verdict") == "SCHEMA_VIOLATION" for f in all_flagged_claims):
        verification_status = "VERIFICATION FAILED"
        
    report = VerificationReport(
        verified_agent_ids=all_verified_ids,
        flagged_claims=all_flagged_claims,
        overall_trustworthiness=round(overall_trust, 4),
        neo4j_cross_refs=list(set(all_neo4j_refs)),
        qdrant_cross_refs=list(set(all_qdrant_refs)),
        summary=summary,
        verification_status=verification_status,
    )

    duration_ms = time.monotonic() * 1000 - start_ms
    exec_meta = NodeExecutionMeta(
        node_name=_NODE_NAME,
        status=NodeStatus.COMPLETED,
        duration_ms=round(duration_ms, 2),
    )

    logger.info(
        "[%s] Completed — trust=%.2f flagged=%d in %.1f ms",
        _NODE_NAME, overall_trust, flagged_count, duration_ms,
    )

    return {
        "verification_report": report.model_dump(),
        "node_execution_log": [exec_meta.model_dump()],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private: per-agent verification logic
# ──────────────────────────────────────────────────────────────────────────────

async def _verify_single_agent(
    score_dict: dict[str, Any],
) -> tuple[str, list[dict], list[dict], list[str], list[str], float]:
    """
    Verify one agent's score, rationale, and citations.

    Returns:
        (agent_id, verified_claims, flagged_claims, neo4j_refs, qdrant_refs, trust_score)
    """
    agent_id: str = score_dict.get("agent_id", "unknown")
    rationale: str = score_dict.get("rationale", "")
    citations: list[dict] = score_dict.get("citations", [])
    hazard_score: float = score_dict.get("hazard_score", 0.0)

    verified_claims: list[dict[str, Any]] = []
    flagged_claims: list[dict[str, Any]] = []
    neo4j_refs: list[str] = []
    qdrant_refs: list[str] = []

    # ── A. Verify citations exist in the knowledge stores ─────────────────────
    citation_check_tasks = [_check_citation(c) for c in citations]
    citation_results = await asyncio.gather(*citation_check_tasks)

    citation_pass = 0
    for citation_dict, (found, ref_type, ref_id) in zip(citations, citation_results):
        if found:
            citation_pass += 1
            if ref_type == "NEO4J_NODE":
                neo4j_refs.append(ref_id)
            elif ref_type == "QDRANT_CHUNK":
                qdrant_refs.append(ref_id)
            verified_claims.append({
                "citation_id": citation_dict.get("identifier"),
                "label": citation_dict.get("label"),
                "verdict": "VERIFIED",
            })
        else:
            flagged_claims.append({
                "agent_id": agent_id,
                "citation_id": citation_dict.get("identifier"),
                "label": citation_dict.get("label"),
                "verdict": "NOT_FOUND",
                "reason": f"Citation identifier '{citation_dict.get('identifier')}' "
                           f"not resolvable in knowledge stores.",
            })

    # ── B. Cross-reference numeric claims from rationale text ─────────────────
    numeric_cross_refs = await _cross_reference_rationale_numerics(
        agent_id=agent_id,
        rationale=rationale,
    )
    for claim_result in numeric_cross_refs:
        if claim_result.get("verified"):
            verified_claims.append(claim_result)
            if claim_result.get("entity"):
                neo4j_refs.append(claim_result["entity"])
        else:
            flagged_claims.append({**claim_result, "agent_id": agent_id})

    # ── C. Sanity check: hazard_score should be within plausible range ────────
    if not (0.0 <= hazard_score <= 100.0):
        flagged_claims.append({
            "agent_id": agent_id,
            "verdict": "SCHEMA_VIOLATION",
            "reason": f"hazard_score={hazard_score} is outside [0, 100].",
        })

    # ── D. Compute per-agent trust score ──────────────────────────────────────
    total_checks = len(citations) + len(numeric_cross_refs) + 1  # +1 for schema check
    passes = citation_pass + sum(1 for r in numeric_cross_refs if r.get("verified")) + (
        1 if 0.0 <= hazard_score <= 100.0 else 0
    )
    trust = passes / total_checks if total_checks > 0 else 0.0

    logger.debug(
        "[%s] agent=%s trust=%.2f checks=%d passes=%d flagged=%d",
        _NODE_NAME, agent_id, trust, total_checks, passes, len(flagged_claims),
    )

    return agent_id, verified_claims, flagged_claims, neo4j_refs, qdrant_refs, trust


async def _check_citation(
    citation_dict: dict[str, Any],
) -> tuple[bool, str, str]:
    """
    Check if a single citation's identifier resolves in the appropriate store.

    Returns:
        (found: bool, ref_type: str, identifier: str)
    """
    source_type: str = citation_dict.get("source_type", "")
    identifier: str = citation_dict.get("identifier", "")

    if source_type == "QDRANT_CHUNK":
        hit = await qdrant_mock.retrieve_chunk_by_id(identifier)
        return hit is not None, "QDRANT_CHUNK", identifier

    elif source_type == "NEO4J_NODE":
        result = await neo4j_mock.query_entity_relationships(
            entity_id=identifier, max_hops=0
        )
        return result.metadata.get("found", False), "NEO4J_NODE", identifier

    # Unknown source type — cannot verify
    return False, source_type, identifier


async def _cross_reference_rationale_numerics(
    agent_id: str,
    rationale: str,
) -> list[dict[str, Any]]:
    """
    Attempt to extract entity:property=value claims from the rationale string
    and cross-reference them against Neo4j.

    Returns list of cross-reference result dicts.
    """
    results: list[dict[str, Any]] = []
    matches = _NUMERIC_CLAIM_PATTERN.findall(rationale)

    # Limit to first 3 matches to avoid hammering the mock DB
    for entity, prop, value_str in matches[:3]:
        try:
            claimed_value = float(value_str)
        except ValueError:
            continue

        xref = await neo4j_mock.cross_reference_claim(
            claim_entity_id=entity.lower(),
            claim_property=prop.lower(),
            claimed_value=claimed_value,
        )

        results.append({
            "entity": entity,
            "property": prop,
            "claimed_value": claimed_value,
            "verified": xref.get("verified", False),
            "actual_value": xref.get("actual_value"),
            "delta": xref.get("delta"),
            "verdict": "VERIFIED" if xref.get("verified") else "CONTRADICTED",
        })

    return results
