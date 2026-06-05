"""
tools/qdrant_mock.py
─────────────────────────────────────────────────────────────────────────────
Deterministic in-memory mock of Qdrant vector search for The Eco Guard.

Simulates embedding-based semantic retrieval using pre-computed keyword
overlap scoring (a simple TF-style approximation) rather than real vectors.

All functions are async and their signatures mirror the live Qdrant client
so production swap-in is a single import change.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Mock knowledge corpus (simulates chunked disaster-management documents)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeChunk:
    """Represents a Qdrant vector point in the mock collection."""
    chunk_id: str
    source_document: str
    page: int
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    # Pre-tokenised terms for scoring (populated at import time)
    _terms: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._terms = set(re.findall(r"\b\w+\b", self.text.lower()))


@dataclass
class SearchHit:
    """Single result from a semantic search query."""
    chunk_id: str
    score: float           # Approximate cosine similarity ∈ [0, 1]
    text: str
    source_document: str
    page: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text": self.text,
            "source_document": self.source_document,
            "page": self.page,
            "payload": self.payload,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Static knowledge corpus — disaster management / seismology domain
# ──────────────────────────────────────────────────────────────────────────────

_CORPUS: list[KnowledgeChunk] = [
    KnowledgeChunk(
        chunk_id="chunk:fema_p646_p1",
        source_document="FEMA P-646 Guidelines for Design of Structures for Vertical Evacuation from Tsunamis",
        page=1,
        text=(
            "Tsunami vertical evacuation refuges must be designed to withstand wave heights "
            "exceeding the maximum considered tsunami (MCT) by a factor of 1.3. "
            "Structural freeboard of at least 3 meters above the MCT run-up elevation is mandatory."
        ),
        payload={"hazard_type": "tsunami", "authority": "FEMA", "year": 2019},
    ),
    KnowledgeChunk(
        chunk_id="chunk:usgs_psha_cascadia_p22",
        source_document="USGS Open-File Report 2021-1075: PSHA for the Cascadia Subduction Zone",
        page=22,
        text=(
            "Probabilistic seismic hazard analysis for a full-margin Cascadia rupture yields "
            "peak ground acceleration (PGA) values of 0.5–1.2 g across coastal Oregon and Washington "
            "with a 2% probability of exceedance in 50 years."
        ),
        payload={"hazard_type": "seismic", "authority": "USGS", "year": 2021},
    ),
    KnowledgeChunk(
        chunk_id="chunk:who_displacement_p5",
        source_document="WHO Displacement and Health in Complex Emergencies (2022)",
        page=5,
        text=(
            "Acute displacement events exceeding 50,000 individuals within 72 hours correlate with "
            "a 4.7x increase in communicable disease incidence. Pre-positioned medical supply chains "
            "reduce excess mortality by 31% when activated within 6 hours of primary event."
        ),
        payload={"hazard_type": "displacement", "authority": "WHO", "year": 2022},
    ),
    KnowledgeChunk(
        chunk_id="chunk:asce7_liquefaction_p210",
        source_document="ASCE 7-22 Minimum Design Loads Chapter 20: Soil Liquefaction",
        page=210,
        text=(
            "Sites with SPT N-values below 15 and groundwater depth less than 10 m in seismic "
            "design category D or higher shall be evaluated for liquefaction potential. "
            "Lateral spreading of 0.5–2.0 m is expected for ML > 6.5 events on liquefiable soils."
        ),
        payload={"hazard_type": "liquefaction", "authority": "ASCE", "year": 2022},
    ),
    KnowledgeChunk(
        chunk_id="chunk:noaa_dart_operations_p3",
        source_document="NOAA DART Buoy Operations Manual v4.1",
        page=3,
        text=(
            "DART buoys detect pressure anomalies indicative of tsunami generation within 60 seconds "
            "of seafloor displacement. Data is transmitted via Iridium satellite with latency under 3 minutes. "
            "False alarm rate historically below 0.4% for amplitude threshold of 3 cm."
        ),
        payload={"hazard_type": "tsunami", "authority": "NOAA", "year": 2023},
    ),
    KnowledgeChunk(
        chunk_id="chunk:pnsn_realtime_p1",
        source_document="Pacific Northwest Seismic Network: Real-Time Monitoring Overview",
        page=1,
        text=(
            "PNSN operates over 450 broadband and strong-motion sensors across WA, OR, and ID. "
            "Magnitude ≥ 2.5 events are auto-located within 15 seconds. "
            "ShakeAlert EEW system provides 2–30 second warning before S-wave arrival at most urban sites."
        ),
        payload={"hazard_type": "seismic", "authority": "PNSN", "year": 2024},
    ),
    KnowledgeChunk(
        chunk_id="chunk:redcross_shelter_p8",
        source_document="American Red Cross: Shelter and Mass Care Guidelines (2023)",
        page=8,
        text=(
            "A minimum of 2.8 m² per person is required in emergency shelter operations. "
            "Psychosocial support must be deployed within 48 hours for populations exceeding 1,000. "
            "Water provision of 15 L/person/day is the WHO minimum during acute disaster response."
        ),
        payload={"hazard_type": "displacement", "authority": "RedCross", "year": 2023},
    ),
    KnowledgeChunk(
        chunk_id="chunk:epri_grid_seismic_p44",
        source_document="EPRI Technical Report 3002000704: Seismic Fragility of Electrical Grid Components",
        page=44,
        text=(
            "High-voltage transformers (500 kV class) have a median seismic capacity of 0.35 g PGA "
            "with lognormal standard deviation of 0.40. Grid restoration following transformer failure "
            "averages 12–18 months due to global manufacturing constraints."
        ),
        payload={"hazard_type": "infrastructure", "authority": "EPRI", "year": 2020},
    ),
]


def _tokenise(text: str) -> set[str]:
    """Lower-case word tokenisation (no stemming — fast and deterministic)."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard_score(query_terms: set[str], chunk_terms: set[str]) -> float:
    """
    Jaccard similarity as a cheap cosine proxy.
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    if not query_terms or not chunk_terms:
        return 0.0
    intersection = len(query_terms & chunk_terms)
    union = len(query_terms | chunk_terms)
    return intersection / union


# ──────────────────────────────────────────────────────────────────────────────
# Public async API  (mirrors qdrant_client.AsyncQdrantClient)
# ──────────────────────────────────────────────────────────────────────────────

async def semantic_search(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    filter_payload: dict[str, Any] | None = None,
) -> list[SearchHit]:
    """
    Perform an approximate semantic search over the mock knowledge corpus.

    Args:
        query:            Free-text query string.
        top_k:            Maximum number of results to return.
        score_threshold:  Minimum Jaccard score; chunks below this are dropped.
        filter_payload:   Optional exact-match filter on payload fields,
                          e.g. {"hazard_type": "tsunami"}.

    Returns:
        List of SearchHit objects, sorted by descending relevance score.
    """
    await asyncio.sleep(0.02)  # simulate embedding + ANN latency

    query_terms = _tokenise(query)
    hits: list[SearchHit] = []

    for chunk in _CORPUS:
        # Apply payload filter first (cheap)
        if filter_payload:
            if not all(chunk.payload.get(k) == v for k, v in filter_payload.items()):
                continue

        score = _jaccard_score(query_terms, chunk._terms)
        if score < score_threshold:
            continue

        hits.append(SearchHit(
            chunk_id=chunk.chunk_id,
            score=score,
            text=chunk.text,
            source_document=chunk.source_document,
            page=chunk.page,
            payload=chunk.payload,
        ))

    hits.sort(key=lambda h: h.score, reverse=True)
    result = hits[:top_k]

    logger.debug(
        "qdrant_mock: query='%s...' → %d/%d hits above threshold",
        query[:60], len(result), len(hits),
    )
    return result


async def retrieve_chunk_by_id(chunk_id: str) -> SearchHit | None:
    """
    Direct lookup of a specific knowledge chunk by its UUID.

    Used by the Verification Agent to pull the exact source passage
    when cross-referencing a citation.

    Args:
        chunk_id: The chunk identifier previously returned by semantic_search.

    Returns:
        SearchHit with score=1.0, or None if not found.
    """
    await asyncio.sleep(0.005)

    for chunk in _CORPUS:
        if chunk.chunk_id == chunk_id:
            return SearchHit(
                chunk_id=chunk.chunk_id,
                score=1.0,
                text=chunk.text,
                source_document=chunk.source_document,
                page=chunk.page,
                payload=chunk.payload,
            )
    logger.warning("qdrant_mock: chunk_id '%s' not found in corpus.", chunk_id)
    return None


async def upsert_chunk(
    chunk_id: str,
    text: str,
    source_document: str,
    page: int = 0,
    payload: dict[str, Any] | None = None,
) -> bool:
    """
    Insert or update a knowledge chunk in the mock collection.

    In production this would call qdrant_client.upsert() with a real embedding.
    Returns True on success (always in mock mode).
    """
    await asyncio.sleep(0.01)

    # Remove existing chunk with the same ID (idempotent upsert)
    global _CORPUS
    _CORPUS = [c for c in _CORPUS if c.chunk_id != chunk_id]

    new_chunk = KnowledgeChunk(
        chunk_id=chunk_id,
        source_document=source_document,
        page=page,
        text=text,
        payload=payload or {},
    )
    _CORPUS.append(new_chunk)
    logger.info("qdrant_mock: upserted chunk '%s'.", chunk_id)
    return True
