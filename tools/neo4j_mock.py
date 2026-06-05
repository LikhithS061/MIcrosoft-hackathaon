"""
tools/neo4j_mock.py
─────────────────────────────────────────────────────────────────────────────
Deterministic in-memory mock of Neo4j graph queries for The Eco Guard.

Replaces live Bolt/HTTP calls during development and unit testing.
All functions are async so call-sites remain identical between mock and live.

Switch between mock and live via USE_MOCK_TOOLS=true/false in .env.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# In-memory knowledge graph
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """Represents a Neo4j node in the mock graph."""
    id: str
    labels: list[str]
    properties: dict[str, Any]


@dataclass
class GraphEdge:
    """Represents a Neo4j relationship in the mock graph."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphQueryResult:
    """Structured result of a mock graph query."""
    query: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "nodes": [
                {"id": n.id, "labels": n.labels, "properties": n.properties}
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.relationship_type,
                    "properties": e.properties,
                }
                for e in self.edges
            ],
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Static knowledge base (simulates a seismology + emergency mgmt graph)
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_NODES: list[GraphNode] = [
    GraphNode("zone:cascadia_subduction", ["SeismicZone"],
              {"name": "Cascadia Subduction Zone", "avg_mw": 8.9, "recurrence_years": 243}),
    GraphNode("zone:san_andreas_south", ["SeismicZone"],
              {"name": "Southern San Andreas Fault", "avg_mw": 7.8, "recurrence_years": 150}),
    GraphNode("region:pacific_northwest", ["Region"],
              {"name": "Pacific Northwest", "population": 8_200_000, "gdp_bn_usd": 620}),
    GraphNode("region:los_angeles", ["Region"],
              {"name": "Los Angeles Metro", "population": 13_200_000, "gdp_bn_usd": 1_100}),
    GraphNode("hazard:tsunami_cascadia", ["Hazard", "TsunamiHazard"],
              {"max_wave_height_m": 25, "inundation_km": 5, "warning_time_min": 15}),
    GraphNode("hazard:liquefaction_sf", ["Hazard", "LiquefactionHazard"],
              {"risk_level": "HIGH", "affected_area_km2": 450}),
    GraphNode("infra:i5_corridor", ["Infrastructure", "Transportation"],
              {"name": "I-5 Corridor", "criticality": "NATIONAL", "annual_freight_bn": 800}),
    GraphNode("infra:bchydro_grid", ["Infrastructure", "PowerGrid"],
              {"name": "BC Hydro Grid", "capacity_mw": 12_600, "redundancy": "LOW"}),
    GraphNode("protocol:nims", ["Protocol", "EmergencyPlan"],
              {"name": "NIMS", "version": "2017", "jurisdiction": "FEDERAL"}),
    GraphNode("protocol:cascadia_playbook", ["Protocol", "EmergencyPlan"],
              {"name": "Cascadia Rising Playbook", "last_exercised": "2023-06-15", "gap_score": 0.42}),
    GraphNode("sensor:usgs_pnsn_001", ["Sensor", "SeismicSensor"],
              {"station_code": "PNSN-001", "latitude": 47.65, "longitude": -122.30, "status": "ONLINE"}),
    GraphNode("sensor:noaa_dart_56001", ["Sensor", "TsunamiSensor"],
              {"buoy_id": "56001", "depth_m": 2800, "last_ping_utc": "2024-06-04T11:55:00Z"}),
]

_MOCK_EDGES: list[GraphEdge] = [
    GraphEdge("zone:cascadia_subduction", "region:pacific_northwest", "THREATENS",
              {"probability_50yr": 0.37, "intensity": "X+"}),
    GraphEdge("zone:cascadia_subduction", "hazard:tsunami_cascadia", "GENERATES",
              {"coupling_coefficient": 0.72}),
    GraphEdge("zone:san_andreas_south", "region:los_angeles", "THREATENS",
              {"probability_30yr": 0.60, "intensity": "VIII"}),
    GraphEdge("zone:san_andreas_south", "hazard:liquefaction_sf", "TRIGGERS",
              {"soil_amplification": 2.1}),
    GraphEdge("hazard:tsunami_cascadia", "infra:i5_corridor", "IMPACTS",
              {"disruption_days": 180}),
    GraphEdge("hazard:tsunami_cascadia", "infra:bchydro_grid", "IMPACTS",
              {"failure_probability": 0.88}),
    GraphEdge("protocol:cascadia_playbook", "zone:cascadia_subduction", "GOVERNS"),
    GraphEdge("protocol:nims", "protocol:cascadia_playbook", "SUPERSEDES"),
    GraphEdge("sensor:usgs_pnsn_001", "zone:cascadia_subduction", "MONITORS"),
    GraphEdge("sensor:noaa_dart_56001", "hazard:tsunami_cascadia", "MONITORS"),
]

# Build adjacency index for O(1) lookups
_NODE_INDEX: dict[str, GraphNode] = {n.id: n for n in _MOCK_NODES}
_ADJACENCY: dict[str, list[GraphEdge]] = {}
for edge in _MOCK_EDGES:
    _ADJACENCY.setdefault(edge.source_id, []).append(edge)
    _ADJACENCY.setdefault(edge.target_id, []).append(edge)  # bidirectional lookup


# ──────────────────────────────────────────────────────────────────────────────
# Public async API  (mirrors what a live Neo4jClient would expose)
# ──────────────────────────────────────────────────────────────────────────────

async def query_entity_relationships(
    entity_id: str,
    max_hops: int = 2,
    relationship_types: list[str] | None = None,
) -> GraphQueryResult:
    """
    Retrieve the ego-graph of `entity_id` up to `max_hops` away.

    Args:
        entity_id:          Neo4j node ID to start traversal from.
        max_hops:           BFS depth (default 2 to avoid explosion).
        relationship_types: Optional allowlist of REL_TYPE strings to filter.

    Returns:
        GraphQueryResult with all discovered nodes and edges.

    Raises:
        KeyError: If entity_id is not found in the mock graph.
    """
    await asyncio.sleep(0.01)  # simulate network round-trip

    if entity_id not in _NODE_INDEX:
        logger.warning("neo4j_mock: entity '%s' not found, returning empty graph.", entity_id)
        return GraphQueryResult(
            query=f"MATCH (n {{id:'{entity_id}'}})-[*0..{max_hops}]-(m) RETURN n,m",
            nodes=[],
            edges=[],
            metadata={"entity_id": entity_id, "found": False, "source": "mock"},
        )

    visited_nodes: set[str] = set()
    visited_edges: list[GraphEdge] = []
    frontier: set[str] = {entity_id}

    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for nid in frontier:
            for edge in _ADJACENCY.get(nid, []):
                neighbour = edge.target_id if edge.source_id == nid else edge.source_id
                if neighbour not in visited_nodes:
                    next_frontier.add(neighbour)
                    if edge not in visited_edges:
                        if relationship_types is None or edge.relationship_type in relationship_types:
                            visited_edges.append(edge)
        visited_nodes.update(frontier)
        frontier = next_frontier

    visited_nodes.update(frontier)
    found_nodes = [_NODE_INDEX[nid] for nid in visited_nodes if nid in _NODE_INDEX]

    logger.debug(
        "neo4j_mock: entity='%s' hops=%d → %d nodes, %d edges",
        entity_id, max_hops, len(found_nodes), len(visited_edges),
    )

    return GraphQueryResult(
        query=f"MATCH (n {{id:'{entity_id}'}})-[*0..{max_hops}]-(m) RETURN n,m",
        nodes=found_nodes,
        edges=visited_edges,
        metadata={"entity_id": entity_id, "found": True, "source": "mock"},
    )


async def query_hazard_protocols(hazard_type: str) -> list[dict[str, Any]]:
    """
    Return all emergency protocols that govern a given hazard type.

    Args:
        hazard_type: Label substring to match, e.g. 'Tsunami', 'Liquefaction'.

    Returns:
        List of protocol property dicts, sorted by gap_score descending (worst first).
    """
    await asyncio.sleep(0.005)

    matching_protocols: list[dict[str, Any]] = []
    for node in _MOCK_NODES:
        if "Protocol" in node.labels:
            # Check if any GOVERNS edge leads to a hazard matching the type
            for edge in _ADJACENCY.get(node.id, []):
                target = _NODE_INDEX.get(edge.target_id)
                if target and hazard_type.lower() in json.dumps(target.labels).lower():
                    proto_data = dict(node.properties)
                    proto_data["protocol_id"] = node.id
                    proto_data["governs"] = edge.target_id
                    matching_protocols.append(proto_data)
                    break

    matching_protocols.sort(key=lambda p: p.get("gap_score", 0.0), reverse=True)
    return matching_protocols


async def cross_reference_claim(
    claim_entity_id: str,
    claim_property: str,
    claimed_value: Any,
    tolerance: float = 0.10,
) -> dict[str, Any]:
    """
    Verify a factual claim against the mock knowledge graph.

    Used by the Verification Agent (Red Teamer) to detect hallucinations.

    Args:
        claim_entity_id:  The node whose property we want to check.
        claim_property:   Property key, e.g. 'avg_mw', 'population'.
        claimed_value:    The value asserted by an agent.
        tolerance:        Relative tolerance for numeric comparisons (default 10%).

    Returns:
        Dict with keys: entity_found, property_found, verified, actual_value, delta.
    """
    await asyncio.sleep(0.005)

    node = _NODE_INDEX.get(claim_entity_id)
    if node is None:
        return {
            "entity_found": False,
            "property_found": False,
            "verified": False,
            "actual_value": None,
            "delta": None,
            "source": "mock",
        }

    actual = node.properties.get(claim_property)
    if actual is None:
        return {
            "entity_found": True,
            "property_found": False,
            "verified": False,
            "actual_value": None,
            "delta": None,
            "source": "mock",
        }

    # Numeric verification with tolerance
    if isinstance(actual, (int, float)) and isinstance(claimed_value, (int, float)):
        delta = abs(actual - claimed_value) / (abs(actual) + 1e-9)
        verified = delta <= tolerance
    else:
        delta = 0.0
        verified = str(actual).lower() == str(claimed_value).lower()

    return {
        "entity_found": True,
        "property_found": True,
        "verified": verified,
        "actual_value": actual,
        "claimed_value": claimed_value,
        "delta": delta,
        "source": "mock",
    }
