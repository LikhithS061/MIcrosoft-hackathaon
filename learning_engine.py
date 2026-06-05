"""
learning_engine.py
─────────────────────────────────────────────────────────────────────────────
Learning Engine for Eco Guard 3.0.

Architecture:
  - SQLite database (aiosqlite) — primary persistent store across runs
  - Pre-seeded with the 10 demo incidents from demo_incidents.py on first init
  - Similarity search: keyword + severity matching (lightweight, no embeddings)
  - Stores full incident record after every pipeline run for future learning

Schema:
  incidents table:
    id TEXT PRIMARY KEY
    incident_type TEXT
    location TEXT
    severity TEXT
    hazard_score REAL
    sustainability_score REAL
    safety_score REAL
    regulatory_issues TEXT  (JSON list)
    recommended_actions TEXT  (JSON list)
    hitl_decision TEXT
    outcome TEXT
    lessons_learned TEXT
    timestamp TEXT

Usage:
    engine = LearningEngine()
    await engine.initialize()                    # creates DB + seeds demo data
    similar = await engine.find_similar(alert)  # returns top-k historical cases
    await engine.store_incident(record)         # saves new incident
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "eco_guard_learning.db")
_DB_PATH = os.path.abspath(_DB_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Pre-seeded historical incidents (the 10 demo cases + their resolved outcomes)
# ──────────────────────────────────────────────────────────────────────────────

_SEED_INCIDENTS: list[dict[str, Any]] = [
    {
        "id": "HIST-001",
        "incident_type": "Water Contamination",
        "location": "Tower B Water Tank",
        "severity": "HIGH",
        "hazard_score": 82.0,
        "sustainability_score": 74.0,
        "safety_score": 79.0,
        "regulatory_issues": json.dumps(["BIS 10500 pH violation", "CPCB water quality standards breach"]),
        "recommended_actions": json.dumps([
            "Emergency flush and chlorination of water tank",
            "Deploy water tankers for 1,200 residents",
            "Mandatory third-party water quality test before reopening",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Resolved — tank cleaned and re-certified within 48 hours",
        "lessons_learned": "Overdue tank cleaning is leading cause of pH spikes; enforce monthly schedule",
        "timestamp": "2025-11-15T08:30:00Z",
    },
    {
        "id": "HIST-002",
        "incident_type": "Fire Hazard",
        "location": "Community Waste Collection Area",
        "severity": "HIGH",
        "hazard_score": 78.0,
        "sustainability_score": 65.0,
        "safety_score": 81.0,
        "regulatory_issues": json.dumps(["Solid Waste Management Rules 2016", "NBC Fire Safety §4"]),
        "recommended_actions": json.dumps([
            "Immediate waste removal by municipal contractor",
            "Install fire suppression near waste area",
            "Enforce 48-hour collection SLA",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Resolved — emergency waste clearance completed; fire risk eliminated",
        "lessons_learned": "95% overflow is a critical fire threshold; trigger alert at 70%",
        "timestamp": "2025-10-03T14:15:00Z",
    },
    {
        "id": "HIST-003",
        "incident_type": "Gas Leak",
        "location": "Basement Utility Room",
        "severity": "CRITICAL",
        "hazard_score": 94.0,
        "sustainability_score": 45.0,
        "safety_score": 96.0,
        "regulatory_issues": json.dumps(["PNGRB Safety Regulations", "NBC Gas Safety §7", "NDMA Gas Leak SOP"]),
        "recommended_actions": json.dumps([
            "Immediate evacuation of all 2,000 residents",
            "Isolate gas supply at main valve",
            "Deploy fire brigade and NDRF hazmat team",
            "Do not re-enter until gas concentration < 5 ppm",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Critical — full evacuation executed; leak sealed by certified gas engineer in 6 hours",
        "lessons_learned": "9-month inspection gap is unacceptable for gas systems; quarterly mandatory",
        "timestamp": "2025-08-21T03:45:00Z",
    },
    {
        "id": "HIST-004",
        "incident_type": "Infrastructure Failure",
        "location": "Tower A Elevator",
        "severity": "MEDIUM",
        "hazard_score": 52.0,
        "sustainability_score": 30.0,
        "safety_score": 61.0,
        "regulatory_issues": json.dumps(["BIS 14665 Lift Safety", "NBC Vertical Transport §9"]),
        "recommended_actions": json.dumps([
            "Take elevator out of service immediately",
            "Schedule AMC-certified inspection within 24 hours",
            "Deploy stair assistance for senior residents",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Resolved — elevator repaired and re-certified; missed AMC schedule identified as root cause",
        "lessons_learned": "7+ sudden stops = imminent failure; automate AMC reminders",
        "timestamp": "2025-12-01T11:00:00Z",
    },
    {
        "id": "HIST-005",
        "incident_type": "Flooding",
        "location": "Basement Parking",
        "severity": "HIGH",
        "hazard_score": 76.0,
        "sustainability_score": 55.0,
        "safety_score": 74.0,
        "regulatory_issues": json.dumps(["Municipal Drainage Byelaws", "NBC Waterproofing §5"]),
        "recommended_actions": json.dumps([
            "Deploy submersible pumps within 1 hour",
            "Shut down all electrical systems in basement",
            "Relocate 150 vehicles to open ground",
            "Clear drain blockage within 2 hours",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Resolved — pumping completed in 3 hours; electrical systems undamaged",
        "lessons_learned": "Electrical shutdown must precede flooding response; add to SOP",
        "timestamp": "2025-09-07T18:30:00Z",
    },
    {
        "id": "HIST-006",
        "incident_type": "Electrical Fault",
        "location": "Main Distribution Panel",
        "severity": "CRITICAL",
        "hazard_score": 92.0,
        "sustainability_score": 40.0,
        "safety_score": 95.0,
        "regulatory_issues": json.dumps(["CEA Electrical Safety Regulations 2010", "BIS 732 Wiring", "NBC Electrical §4"]),
        "recommended_actions": json.dumps([
            "Isolate main breaker immediately",
            "Evacuate all 4 buildings as precaution",
            "Call DISCOM emergency: 1912",
            "Do not restore power until licensed electrician certifies panel",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Critical — arcing in distribution panel detected; full panel replacement required",
        "lessons_learned": "18-month inspection gap caused insulation failure; annual inspection is mandatory",
        "timestamp": "2025-07-14T22:10:00Z",
    },
    {
        "id": "HIST-007",
        "incident_type": "Resource Waste",
        "location": "Tower C Pipeline",
        "severity": "MEDIUM",
        "hazard_score": 44.0,
        "sustainability_score": 78.0,
        "safety_score": 35.0,
        "regulatory_issues": json.dumps(["Water Conservation Act", "Municipal Water Byelaws"]),
        "recommended_actions": json.dumps([
            "Close isolation valve for affected section",
            "Deploy repair team within 4 hours",
            "Provide temporary water supply to 34 affected residents",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Resolved — pipeline repaired; 21-day ticket delay identified as process failure",
        "lessons_learned": "Water loss > 5,000 L/day should auto-escalate repair SLA to 24 hours",
        "timestamp": "2026-01-09T09:00:00Z",
    },
    {
        "id": "HIST-008",
        "incident_type": "Sustainability Violation",
        "location": "Open Ground Near Compound Wall",
        "severity": "HIGH",
        "hazard_score": 71.0,
        "sustainability_score": 88.0,
        "safety_score": 68.0,
        "regulatory_issues": json.dumps(["Hazardous Waste Rules 2016", "Environment Protection Act 1986", "CPCB Guidelines"]),
        "recommended_actions": json.dumps([
            "File FIR under Environment Protection Act",
            "Engage CPCB-registered hazardous waste handler",
            "Cordon off 50m radius; restrict school access",
            "Notify State Pollution Control Board within 24 hours",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Resolved — waste removed by authorised contractor; FIR filed",
        "lessons_learned": "School proximity makes any hazardous waste a CRITICAL public health incident",
        "timestamp": "2025-10-22T07:45:00Z",
    },
    {
        "id": "HIST-009",
        "incident_type": "Environmental Hazard",
        "location": "Apartment Generator Zone",
        "severity": "HIGH",
        "hazard_score": 75.0,
        "sustainability_score": 82.0,
        "safety_score": 72.0,
        "regulatory_issues": json.dumps(["CPCB Air Quality Standards (NAAQS)", "Environment Protection Rules 1986", "Generator Emission Norms"]),
        "recommended_actions": json.dumps([
            "Restrict generator runtime to essential loads only",
            "Install DG emission scrubber or switch to grid",
            "Distribute N95 masks to asthma patients",
            "Report to State Pollution Control Board",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Partially resolved — generator hours reduced; solar backup procurement initiated",
        "lessons_learned": "PM2.5 > 150 with medical cases already reported warrants immediate HITL escalation",
        "timestamp": "2025-11-28T16:20:00Z",
    },
    {
        "id": "HIST-010",
        "incident_type": "Public Safety",
        "location": "Main Entrance",
        "severity": "HIGH",
        "hazard_score": 68.0,
        "sustainability_score": 20.0,
        "safety_score": 85.0,
        "regulatory_issues": json.dumps(["IPC Section 441 (Criminal Trespass)", "Private Security Agencies Act", "NDMA Community Safety Guidelines"]),
        "recommended_actions": json.dumps([
            "Alert local police station immediately",
            "Increase guard deployment at entrance",
            "Fix CCTV blind spot within 24 hours",
            "Implement biometric access control",
        ]),
        "hitl_decision": "APPROVED",
        "outcome": "Resolved — police patrol increased; CCTV upgraded; 2 trespassers apprehended",
        "lessons_learned": "CCTV blind spots are critical security vulnerabilities; monthly CCTV audit required",
        "timestamp": "2026-02-14T23:55:00Z",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# LearningEngine class
# ──────────────────────────────────────────────────────────────────────────────

class LearningEngine:
    """
    Async SQLite-backed learning engine for Eco Guard 3.0.

    Usage pattern (async context):
        engine = LearningEngine()
        await engine.initialize()
        similar = await engine.find_similar(raw_alert, top_k=3)
        await engine.store_incident(record_dict)
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self.db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        """Create the DB schema and seed demo incidents if not already present."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id                  TEXT PRIMARY KEY,
                    incident_type       TEXT NOT NULL,
                    location            TEXT,
                    severity            TEXT,
                    hazard_score        REAL,
                    sustainability_score REAL,
                    safety_score        REAL,
                    regulatory_issues   TEXT,
                    recommended_actions TEXT,
                    hitl_decision       TEXT,
                    outcome             TEXT,
                    lessons_learned     TEXT,
                    timestamp           TEXT
                )
            """)
            await db.commit()

            # Seed demo data only if table is empty
            cursor = await db.execute("SELECT COUNT(*) FROM incidents")
            row = await cursor.fetchone()
            if row and row[0] == 0:
                for inc in _SEED_INCIDENTS:
                    await db.execute(
                        """
                        INSERT OR IGNORE INTO incidents
                            (id, incident_type, location, severity, hazard_score,
                             sustainability_score, safety_score, regulatory_issues,
                             recommended_actions, hitl_decision, outcome, lessons_learned, timestamp)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            inc["id"], inc["incident_type"], inc["location"],
                            inc["severity"], inc["hazard_score"],
                            inc["sustainability_score"], inc["safety_score"],
                            inc["regulatory_issues"], inc["recommended_actions"],
                            inc["hitl_decision"], inc["outcome"],
                            inc["lessons_learned"], inc["timestamp"],
                        ),
                    )
                await db.commit()
                logger.info("LearningEngine: seeded %d demo incidents.", len(_SEED_INCIDENTS))
            else:
                logger.info("LearningEngine: DB already seeded (%d records).", row[0] if row else 0)

        self._initialized = True

    async def find_similar(
        self,
        raw_alert: dict[str, Any],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k most similar historical incidents.

        Similarity is scored by:
          +3  exact incident_type match
          +2  same severity
          +1  keyword overlap in location/description

        Returns list of (record_dict, similarity_score) sorted descending.
        """
        await self.initialize()

        incident_type = raw_alert.get("raw_payload", {}).get("incident_type", "")
        severity = raw_alert.get("severity", "")
        description = (raw_alert.get("description", "") + " " + raw_alert.get("title", "")).lower()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM incidents ORDER BY timestamp DESC")
            rows = await cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            rec = dict(row)
            score = 0.0

            # Exact incident type match
            if rec.get("incident_type", "").lower() == incident_type.lower():
                score += 3.0

            # Severity match
            if rec.get("severity", "").upper() == severity.upper():
                score += 2.0

            # Keyword overlap
            loc_lower = rec.get("location", "").lower()
            type_lower = rec.get("incident_type", "").lower()
            for kw in type_lower.split() + loc_lower.split():
                if len(kw) > 3 and kw in description:
                    score += 0.5

            if score > 0:
                # Parse JSON fields back
                try:
                    rec["regulatory_issues"] = json.loads(rec.get("regulatory_issues") or "[]")
                    rec["recommended_actions"] = json.loads(rec.get("recommended_actions") or "[]")
                except (json.JSONDecodeError, TypeError):
                    pass
                rec["similarity_score"] = round(score / 8.0, 3)  # normalise to ~[0,1]
                results.append(rec)

        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results[:top_k]

    async def store_incident(self, record: dict[str, Any]) -> None:
        """Persist a completed incident record to the learning DB."""
        await self.initialize()

        inc_id = record.get("id") or record.get("trace_id", f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
        regulatory_issues = record.get("regulatory_issues", [])
        recommended_actions = record.get("recommended_actions", [])

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO incidents
                    (id, incident_type, location, severity, hazard_score,
                     sustainability_score, safety_score, regulatory_issues,
                     recommended_actions, hitl_decision, outcome, lessons_learned, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    inc_id,
                    record.get("incident_type", "Unknown"),
                    record.get("location", "Unknown"),
                    record.get("severity", "UNKNOWN"),
                    record.get("hazard_score", 0.0),
                    record.get("sustainability_score", 0.0),
                    record.get("safety_score", 0.0),
                    json.dumps(regulatory_issues if isinstance(regulatory_issues, list) else [regulatory_issues]),
                    json.dumps(recommended_actions if isinstance(recommended_actions, list) else [recommended_actions]),
                    record.get("hitl_decision", "PENDING"),
                    record.get("outcome", ""),
                    record.get("lessons_learned", ""),
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            await db.commit()
        logger.info("LearningEngine: stored incident %s", inc_id)

    async def get_all(self) -> list[dict[str, Any]]:
        """Return all stored incidents (for debugging / export)."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM incidents ORDER BY timestamp DESC")
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton (shared across all agent nodes)
# ──────────────────────────────────────────────────────────────────────────────

_engine: LearningEngine | None = None


def get_learning_engine() -> LearningEngine:
    """Return the module-level singleton LearningEngine."""
    global _engine
    if _engine is None:
        _engine = LearningEngine()
    return _engine
