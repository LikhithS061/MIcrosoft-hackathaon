"""
tools/regulatory_rag.py
─────────────────────────────────────────────────────────────────────────────
Regulatory Intelligence RAG Tool for Eco Guard 3.0.

Architecture:
  Primary:  ChromaDB vector store + embedded regulatory content
            (NDMA, CPCB, BIS, NBC, Solid Waste Rules, CEA, etc.)
  Fallback: Curated lookup table by incident_type — zero hallucination

Flow:
  incident_type → query ChromaDB → retrieve relevant chunks → cite source
  if ChromaDB unavailable → use curated fallback lookup

The regulatory content is embedded in-process using sentence-transformers
(all-MiniLM-L6-v2) — no external services required.

Design Principle:
  Never return a regulation that is not in the curated knowledge base.
  Every regulation must have a source reference.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Curated Indian Regulatory Knowledge Base
# Each entry: {regulation_id, title, source, section, summary, applicable_to}
# ──────────────────────────────────────────────────────────────────────────────

REGULATORY_KB: list[dict[str, Any]] = [
    # ─── NDMA ─────────────────────────────────────────────────────────────────
    {
        "regulation_id": "NDMA-GAS-001",
        "title": "NDMA Guidelines on Chemical and Industrial Disasters",
        "source": "National Disaster Management Authority (NDMA)",
        "section": "Chapter 4 — Gas Leak Response Protocol",
        "summary": (
            "Immediate evacuation mandatory when gas concentration exceeds 25% LEL. "
            "NDRF deployment within 30 minutes for residential gas leaks. "
            "Incident Command must be established by District Collector."
        ),
        "applicable_to": ["Gas Leak", "Chemical Hazard", "Industrial Accident"],
        "severity_threshold": "CRITICAL",
    },
    {
        "regulation_id": "NDMA-FLOOD-001",
        "title": "NDMA National Flood Risk Mitigation Programme",
        "source": "National Disaster Management Authority (NDMA)",
        "section": "Section 3.2 — Urban Flooding SOP",
        "summary": (
            "States must maintain pre-positioned pumping equipment. "
            "Electrical systems must be shut down before flood response. "
            "Evacuation of basement areas mandatory when water level exceeds 15 cm."
        ),
        "applicable_to": ["Flooding", "Urban Flood", "Basement Flooding"],
        "severity_threshold": "HIGH",
    },
    {
        "regulation_id": "NDMA-FIRE-001",
        "title": "NDMA Guidelines for Management of Fire in High-Rise Buildings",
        "source": "National Disaster Management Authority (NDMA)",
        "section": "Chapter 2 — Prevention and Preparedness",
        "summary": (
            "High-rise buildings must conduct quarterly fire drills. "
            "Waste storage areas require automatic fire suppression. "
            "Fire NOC mandatory from State Fire Department; renewal every 3 years."
        ),
        "applicable_to": ["Fire Hazard", "Building Fire", "Waste Dump Fire Hazard"],
        "severity_threshold": "HIGH",
    },
    {
        "regulation_id": "NDMA-COMMUNITY-001",
        "title": "NDMA Community-Based Disaster Risk Reduction Guidelines",
        "source": "National Disaster Management Authority (NDMA)",
        "section": "Section 6 — Public Safety and Security",
        "summary": (
            "Housing societies must have a community disaster management committee. "
            "Emergency response plans must be displayed at entry points. "
            "Security incidents must be reported to local police within 1 hour."
        ),
        "applicable_to": ["Security Threat", "Public Safety", "Community Safety"],
        "severity_threshold": "MEDIUM",
    },

    # ─── CPCB ──────────────────────────────────────────────────────────────────
    {
        "regulation_id": "CPCB-WATER-001",
        "title": "CPCB Drinking Water Quality Standards",
        "source": "Central Pollution Control Board (CPCB)",
        "section": "IS 10500:2012 — Drinking Water Specifications",
        "summary": (
            "Permissible pH range: 6.5–8.5. Turbidity: max 1 NTU (acceptable 5 NTU). "
            "Residual chlorine: 0.2 mg/L minimum. "
            "pH > 9.5 requires immediate supply suspension and remediation."
        ),
        "applicable_to": ["Water Contamination", "Water Quality", "Drinking Water"],
        "severity_threshold": "HIGH",
    },
    {
        "regulation_id": "CPCB-AIR-001",
        "title": "CPCB National Ambient Air Quality Standards (NAAQS)",
        "source": "Central Pollution Control Board (CPCB)",
        "section": "Environment (Protection) Rules 1986 — Schedule VII",
        "summary": (
            "PM2.5 24-hour average standard: 60 μg/m³ (national standard). "
            "PM2.5 > 150: Hazardous — stop outdoor activities, use N95 masks. "
            "Diesel generator emissions must comply with CPCB DG set norms."
        ),
        "applicable_to": ["Air Quality Degradation", "Environmental Hazard", "Generator Emissions"],
        "severity_threshold": "HIGH",
    },
    {
        "regulation_id": "CPCB-WASTE-001",
        "title": "CPCB Solid Waste Management Rules 2016",
        "source": "Central Pollution Control Board (CPCB)",
        "section": "Rule 15 — Duties of Bulk Generators",
        "summary": (
            "Bulk generators (>100 kg/day) must segregate and process waste at source. "
            "Illegal dumping attracts fine up to ₹25,000 per incident. "
            "Hazardous waste must be handed to authorised recyclers only."
        ),
        "applicable_to": ["Illegal Waste Dumping", "Sustainability Violation", "Waste Management"],
        "severity_threshold": "HIGH",
    },
    {
        "regulation_id": "CPCB-HAZWASTE-001",
        "title": "Hazardous and Other Wastes (Management) Rules 2016",
        "source": "Central Pollution Control Board (CPCB) / MoEF",
        "section": "Rule 9 — Storage and Handling",
        "summary": (
            "Hazardous waste cannot be stored in open areas. "
            "Disposal only through CPCB-authorised Treatment Storage Disposal Facilities (TSDFs). "
            "Proximity to schools/hospitals: mandatory 500m exclusion zone."
        ),
        "applicable_to": ["Illegal Waste Dumping", "Hazardous Waste", "Sustainability Violation"],
        "severity_threshold": "HIGH",
    },

    # ─── BIS ───────────────────────────────────────────────────────────────────
    {
        "regulation_id": "BIS-WATER-001",
        "title": "BIS IS 10500:2012 — Drinking Water Standard",
        "source": "Bureau of Indian Standards (BIS)",
        "section": "Table 1 — Requirement for Drinking Water",
        "summary": (
            "pH acceptable range: 6.5–8.5; permissible up to 9.5. "
            "pH 11.2 is outside all permissible limits — mandatory cessation of supply. "
            "Turbidity: acceptable 1 NTU, permissible 5 NTU."
        ),
        "applicable_to": ["Water Contamination", "Water Quality"],
        "severity_threshold": "HIGH",
    },
    {
        "regulation_id": "BIS-LIFT-001",
        "title": "BIS IS 14665 — Electric Traction Lifts",
        "source": "Bureau of Indian Standards (BIS)",
        "section": "Part 4: Safety Rules for Construction and Installation",
        "summary": (
            "Lifts must undergo annual inspection by a competent person. "
            "Excessive motor temperature and repeated sudden stops are safety-critical defects. "
            "Lift must be taken out of service until cleared by a Lift Inspector."
        ),
        "applicable_to": ["Elevator Failure", "Infrastructure Failure", "Lift Safety"],
        "severity_threshold": "MEDIUM",
    },
    {
        "regulation_id": "BIS-ELEC-001",
        "title": "BIS IS 732 — Electrical Wiring Installations",
        "source": "Bureau of Indian Standards (BIS)",
        "section": "Section 6 — Distribution Boards and Panels",
        "summary": (
            "Distribution panels must be inspected annually by a licensed electrician. "
            "Voltage fluctuation > ±10% of nominal is a fault condition requiring isolation. "
            "Burning smell at electrical panel is a fire emergency — immediate isolation required."
        ),
        "applicable_to": ["Electrical Fault", "Electrical Hazard", "Distribution Panel"],
        "severity_threshold": "CRITICAL",
    },

    # ─── National Building Code (NBC) ─────────────────────────────────────────
    {
        "regulation_id": "NBC-FIRE-001",
        "title": "National Building Code of India — Fire and Life Safety",
        "source": "Bureau of Indian Standards / Ministry of Housing (NBC 2016)",
        "section": "Part 4 — Fire and Life Safety",
        "summary": (
            "All residential buildings >15m must have automatic fire detection. "
            "Waste storage areas must be isolated with 2-hour fire-rated walls. "
            "Emergency exit access must be kept clear at all times."
        ),
        "applicable_to": ["Fire Hazard", "Waste Dump Fire Hazard", "Building Safety"],
        "severity_threshold": "HIGH",
    },
    {
        "regulation_id": "NBC-STRUCT-001",
        "title": "National Building Code — Structural and Infrastructure Safety",
        "source": "Bureau of Indian Standards / Ministry of Housing (NBC 2016)",
        "section": "Part 6 — Structural Design",
        "summary": (
            "Basement waterproofing must comply with IS 3370. "
            "Drainage systems must be maintained and cleared every 6 months. "
            "Electrical installations in basements must meet IP65 waterproofing rating."
        ),
        "applicable_to": ["Flooding", "Infrastructure Failure", "Water Leakage"],
        "severity_threshold": "MEDIUM",
    },
    {
        "regulation_id": "NBC-GAS-001",
        "title": "National Building Code — Gas Installations",
        "source": "Bureau of Indian Standards / Ministry of Housing (NBC 2016)",
        "section": "Part 9 — Plumbing Services — Gas Supply",
        "summary": (
            "Gas pipelines must be inspected every 6 months. "
            "All gas utility rooms require mechanical ventilation achieving 10 air changes/hour. "
            "Gas detectors must be installed within 30cm of ceiling in utility rooms."
        ),
        "applicable_to": ["Gas Leak", "Gas Safety", "Utility Room"],
        "severity_threshold": "CRITICAL",
    },

    # ─── CEA ──────────────────────────────────────────────────────────────────
    {
        "regulation_id": "CEA-ELEC-001",
        "title": "Central Electricity Authority — Measures Relating to Safety",
        "source": "Central Electricity Authority (CEA) — Ministry of Power",
        "section": "CEA (Measures relating to Safety and Electric Supply) Regulations 2010",
        "summary": (
            "All electrical installations must be tested annually. "
            "Distribution panels with 18+ months since inspection are in statutory non-compliance. "
            "Defective installations must be isolated within 24 hours of fault detection."
        ),
        "applicable_to": ["Electrical Fault", "Electrical Hazard", "Power Distribution"],
        "severity_threshold": "CRITICAL",
    },

    # ─── Water Conservation ────────────────────────────────────────────────────
    {
        "regulation_id": "WATER-CONS-001",
        "title": "Municipal Water Conservation and Leakage Policy",
        "source": "Ministry of Jal Shakti / Municipal Corporation",
        "section": "Urban Water Conservation Guidelines 2024",
        "summary": (
            "Water loss > 10% of supply is a reportable event. "
            "Repair of leaks must be completed within 48 hours of report. "
            "Persistent leaks > 7 days attract penalty under the Municipal Water Act."
        ),
        "applicable_to": ["Water Leakage", "Resource Waste", "Pipeline Failure"],
        "severity_threshold": "MEDIUM",
    },

    # ─── IPC / Security ────────────────────────────────────────────────────────
    {
        "regulation_id": "IPC-SECURITY-001",
        "title": "IPC Section 441 — Criminal Trespass",
        "source": "Indian Penal Code (IPC) / Bhartiya Nyaya Sanhita 2023",
        "section": "Section 441 — Criminal Trespass",
        "summary": (
            "Unauthorized entry into a private property is criminal trespass (up to 3 months imprisonment). "
            "Housing societies must report repeated trespass attempts to local police. "
            "CCTV footage must be preserved as evidence."
        ),
        "applicable_to": ["Security Threat", "Public Safety", "Unauthorized Entry"],
        "severity_threshold": "MEDIUM",
    },
    {
        "regulation_id": "PSARA-001",
        "title": "Private Security Agencies (Regulation) Act 2005",
        "source": "Ministry of Home Affairs (PSARA)",
        "section": "Section 7 — Duties of Licensee",
        "summary": (
            "Housing societies employing security guards must use PSARA-licensed agencies. "
            "Security personnel must be trained in first response and incident reporting. "
            "Incident registers must be maintained and available for inspection."
        ),
        "applicable_to": ["Security Threat", "Public Safety"],
        "severity_threshold": "MEDIUM",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Index by incident type for fast fallback lookup
# ──────────────────────────────────────────────────────────────────────────────

def _build_index() -> dict[str, list[dict[str, Any]]]:
    idx: dict[str, list[dict[str, Any]]] = {}
    for reg in REGULATORY_KB:
        for itype in reg["applicable_to"]:
            idx.setdefault(itype, []).append(reg)
    return idx


_REGULATION_INDEX: dict[str, list[dict[str, Any]]] = _build_index()


# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB RAG (primary) + curated fallback
# ──────────────────────────────────────────────────────────────────────────────

_chroma_client = None
_chroma_collection = None
_chroma_available = False


def _try_init_chroma() -> bool:
    """Attempt to initialize ChromaDB with embedded regulatory content."""
    global _chroma_client, _chroma_collection, _chroma_available

    if _chroma_available:
        return True

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        persist_dir = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_regulatory_db")
        persist_dir = os.path.abspath(persist_dir)

        _chroma_client = chromadb.PersistentClient(path=persist_dir)

        ef = embedding_functions.DefaultEmbeddingFunction()

        _chroma_collection = _chroma_client.get_or_create_collection(
            name="indian_regulations",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Populate if empty
        existing = _chroma_collection.count()
        if existing == 0:
            logger.info("regulatory_rag: Indexing %d regulations into ChromaDB...", len(REGULATORY_KB))
            docs = [r["summary"] for r in REGULATORY_KB]
            ids = [r["regulation_id"] for r in REGULATORY_KB]
            metadatas = [
                {
                    "title": r["title"],
                    "source": r["source"],
                    "section": r["section"],
                    "applicable_to": ", ".join(r["applicable_to"]),
                    "severity_threshold": r["severity_threshold"],
                }
                for r in REGULATORY_KB
            ]
            _chroma_collection.add(documents=docs, ids=ids, metadatas=metadatas)
            logger.info("regulatory_rag: ChromaDB indexed successfully.")
        else:
            logger.info("regulatory_rag: ChromaDB already contains %d entries.", existing)

        _chroma_available = True
        return True

    except Exception as exc:
        logger.warning("regulatory_rag: ChromaDB init failed (%s) — using curated fallback.", exc)
        _chroma_available = False
        return False


async def retrieve_regulations(
    incident_type: str,
    description: str,
    severity: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """
    Retrieve applicable Indian government regulations.

    Primary:  ChromaDB semantic search over embedded regulation summaries.
    Fallback: Curated index lookup by incident_type.

    Args:
        incident_type:  e.g. "Water Contamination", "Gas Leak"
        description:    Free-text incident description for semantic search
        severity:       Alert severity string
        top_k:          Number of regulations to return

    Returns:
        List of regulation dicts with:
          regulation_id, title, source, section, summary, applicable_to
    """
    # Try ChromaDB first
    if _try_init_chroma() and _chroma_collection is not None:
        try:
            query = f"{incident_type} {description[:200]} {severity}"
            results = _chroma_collection.query(
                query_texts=[query],
                n_results=min(top_k, _chroma_collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            regulations: list[dict[str, Any]] = []
            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for reg_id, meta, doc, dist in zip(ids, metadatas, documents, distances):
                similarity = round(1.0 - dist, 4)  # cosine distance → similarity
                regulations.append({
                    "regulation_id": reg_id,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "section": meta.get("section", ""),
                    "summary": doc,
                    "applicable_to": meta.get("applicable_to", "").split(", "),
                    "severity_threshold": meta.get("severity_threshold", ""),
                    "retrieval_similarity": similarity,
                    "retrieval_method": "ChromaDB RAG",
                })

            if regulations:
                logger.info(
                    "regulatory_rag: ChromaDB returned %d regulations for '%s'",
                    len(regulations), incident_type,
                )
                return regulations

        except Exception as exc:
            logger.warning("regulatory_rag: ChromaDB query failed (%s) — falling back.", exc)

    # Curated fallback
    return _curated_lookup(incident_type, severity, top_k)


def _curated_lookup(
    incident_type: str,
    severity: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return regulations from the curated index (zero hallucination)."""
    # Direct match
    matches: list[dict[str, Any]] = _REGULATION_INDEX.get(incident_type, [])

    # Keyword fallback for partial matches
    if not matches:
        it_lower = incident_type.lower()
        for key, regs in _REGULATION_INDEX.items():
            if any(word in it_lower for word in key.lower().split()):
                matches.extend(regs)

    # Deduplicate
    seen_ids: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in matches:
        if r["regulation_id"] not in seen_ids:
            seen_ids.add(r["regulation_id"])
            unique.append({**r, "retrieval_method": "Curated Fallback"})

    # Sort by severity threshold relevance
    priority = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    unique.sort(key=lambda r: priority.get(r.get("severity_threshold", "LOW"), 3))

    result = unique[:top_k]
    logger.info(
        "regulatory_rag: Curated fallback returned %d regulations for '%s'",
        len(result), incident_type,
    )
    return result


def get_all_regulation_ids() -> list[str]:
    """Return all known regulation IDs (for verification agent cross-reference)."""
    return [r["regulation_id"] for r in REGULATORY_KB]
