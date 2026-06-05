"""
demo_incidents.py
─────────────────────────────────────────────────────────────────────────────
10 pre-built demo incidents from the Eco Guard 3.0 master prompt.

Each incident is a dict compatible with the RawAlert schema (after planner
enrichment).  They are used by:
  - The Streamlit UI sidebar selector
  - The Learning Engine pre-seeder
  - Integration tests
"""

from __future__ import annotations

from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# The canonical 10 demo incidents
# ──────────────────────────────────────────────────────────────────────────────

DEMO_INCIDENTS: list[dict[str, Any]] = [
    # ── 001: Water Contamination ──────────────────────────────────────────────
    {
        "incident_id": "DEMO-001",
        "title": "Water Contamination – Tower B Water Tank",
        "description": (
            "pH Level detected at 11.2 (alkaline), turbidity is high, and chlorine levels "
            "are critically low in Tower B water tank. Tank cleaning is 14 days overdue. "
            "1,200 residents are affected including a nearby school."
        ),
        "severity": "HIGH",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [19.0760, 72.8777],  # Mumbai
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Water Contamination",
            "location": "Tower B Water Tank",
            "ph_level": 11.2,
            "turbidity": "High",
            "chlorine_level": "Low",
            "residents_affected": 1200,
            "nearby_school": True,
            "nearby_hospital": False,
            "maintenance_overdue_days": 14,
        },
    },

    # ── 002: Waste Dump Fire Hazard ──────────────────────────────────────────
    {
        "incident_id": "DEMO-002",
        "title": "Waste Dump Fire Hazard – Community Waste Collection Area",
        "description": (
            "Smoke observed at community waste collection area. Waste overflow at 95%. "
            "Temperature 12°C above normal. Waste collection delayed 5 days. "
            "600 residents affected; children's play area nearby."
        ),
        "severity": "HIGH",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [28.7041, 77.1025],  # Delhi
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Fire Hazard",
            "location": "Community Waste Collection Area",
            "smoke_observed": True,
            "waste_overflow_pct": 95,
            "temperature_rise_c": 12,
            "residents_affected": 600,
            "children_play_area_nearby": True,
            "collection_delayed_days": 5,
        },
    },

    # ── 003: Gas Leak ──────────────────────────────────────────────────────────
    {
        "incident_id": "DEMO-003",
        "title": "Gas Leak – Basement Utility Room",
        "description": (
            "Gas concentration detected at 78 ppm in basement utility room. "
            "Ventilation status is poor. 2,000 residents affected including 350 elderly. "
            "Last inspection was 9 months ago."
        ),
        "severity": "CRITICAL",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [12.9716, 77.5946],  # Bangalore
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Gas Leak",
            "location": "Basement Utility Room",
            "gas_concentration_ppm": 78,
            "ventilation_status": "Poor",
            "residents_affected": 2000,
            "elderly_residents": 350,
            "last_inspection_months_ago": 9,
        },
    },

    # ── 004: Elevator Failure ─────────────────────────────────────────────────
    {
        "incident_id": "DEMO-004",
        "title": "Elevator Failure – Tower A",
        "description": (
            "Tower A elevator has 7 sudden stop events. Motor temperature is elevated. "
            "850 residents affected with high dependency among senior citizens. "
            "Preventive maintenance was missed."
        ),
        "severity": "MEDIUM",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [18.5204, 73.8567],  # Pune
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Infrastructure Failure",
            "location": "Tower A Elevator",
            "sudden_stop_events": 7,
            "motor_temperature": "Elevated",
            "residents_affected": 850,
            "senior_citizen_dependency": "High",
            "maintenance_missed": True,
        },
    },

    # ── 005: Flooding Risk ────────────────────────────────────────────────────
    {
        "incident_id": "DEMO-005",
        "title": "Flooding Risk – Basement Parking",
        "description": (
            "Water level at 25 cm in basement parking. Drain blockage confirmed. "
            "150 vehicles at risk. Electrical systems are nearby. Drain cleaning delayed."
        ),
        "severity": "HIGH",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [22.5726, 88.3639],  # Kolkata
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Flooding",
            "location": "Basement Parking",
            "water_level_cm": 25,
            "drain_blockage": True,
            "vehicles_at_risk": 150,
            "electrical_systems_nearby": True,
            "drain_cleaning_delayed": True,
        },
    },

    # ── 006: Electrical Hazard ────────────────────────────────────────────────
    {
        "incident_id": "DEMO-006",
        "title": "Electrical Hazard – Main Distribution Panel",
        "description": (
            "Severe voltage fluctuation and burning smell detected at the main distribution "
            "panel. 4 buildings affected. Last inspection was 18 months ago."
        ),
        "severity": "CRITICAL",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [17.3850, 78.4867],  # Hyderabad
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Electrical Fault",
            "location": "Main Distribution Panel",
            "voltage_fluctuation": "Severe",
            "burning_smell": True,
            "buildings_affected": 4,
            "last_inspection_months_ago": 18,
        },
    },

    # ── 007: Water Leakage ────────────────────────────────────────────────────
    {
        "incident_id": "DEMO-007",
        "title": "Water Leakage – Tower C Pipeline",
        "description": (
            "Estimated water loss of 10,000 liters/day from Tower C pipeline. "
            "34 residents reporting low pressure. Repair ticket has been open for 21 days."
        ),
        "severity": "MEDIUM",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [13.0827, 80.2707],  # Chennai
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Resource Waste",
            "location": "Tower C Pipeline",
            "water_loss_liters_per_day": 10000,
            "residents_reporting_low_pressure": 34,
            "repair_ticket_open_days": 21,
        },
    },

    # ── 008: Illegal Waste Dumping ────────────────────────────────────────────
    {
        "incident_id": "DEMO-008",
        "title": "Illegal Waste Dumping – Open Ground Near Compound Wall",
        "description": (
            "Hazardous waste present at open ground near compound wall. "
            "Waste volume: 4 tons. School is within 300 meters."
        ),
        "severity": "HIGH",
        "source_system": "COMMUNITY_REPORT",
        "geo_coordinates": [26.8467, 80.9462],  # Lucknow
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Sustainability Violation",
            "location": "Open Ground Near Compound Wall",
            "hazardous_waste": True,
            "waste_volume_tons": 4,
            "school_within_300m": True,
        },
    },

    # ── 009: Air Quality Degradation ──────────────────────────────────────────
    {
        "incident_id": "DEMO-009",
        "title": "Air Quality Degradation – Apartment Generator Zone",
        "description": (
            "PM2.5 level at 180 (Hazardous). Generator runtime is excessive in the "
            "apartment generator zone. 23 asthma cases already reported by residents."
        ),
        "severity": "HIGH",
        "source_system": "IOT_SENSOR_MESH",
        "geo_coordinates": [23.0225, 72.5714],  # Ahmedabad
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Environmental Hazard",
            "location": "Apartment Generator Zone",
            "pm25_level": 180,
            "generator_runtime": "Excessive",
            "asthma_cases_reported": 23,
        },
    },

    # ── 010: Security Threat ──────────────────────────────────────────────────
    {
        "incident_id": "DEMO-010",
        "title": "Security Threat – Main Entrance",
        "description": (
            "12 unauthorized entry attempts at main entrance. CCTV blind spot detected. "
            "Entire community is impacted."
        ),
        "severity": "HIGH",
        "source_system": "SECURITY_SYSTEM",
        "geo_coordinates": [21.1458, 79.0882],  # Nagpur
        "timestamp_utc": "2026-06-04T12:00:00Z",
        "raw_payload": {
            "incident_type": "Public Safety",
            "location": "Main Entrance",
            "unauthorized_entry_attempts": 12,
            "cctv_blind_spot": True,
            "entire_community_affected": True,
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

INCIDENT_TITLES: list[str] = [inc["title"] for inc in DEMO_INCIDENTS]

INCIDENT_BY_ID: dict[str, dict] = {inc["incident_id"]: inc for inc in DEMO_INCIDENTS}


def get_incident_by_title(title: str) -> dict[str, Any] | None:
    """Return a demo incident dict by its title (exact match)."""
    return next((inc for inc in DEMO_INCIDENTS if inc["title"] == title), None)


def get_incident_type(incident: dict[str, Any]) -> str:
    """Extract the incident_type string from raw_payload."""
    return incident.get("raw_payload", {}).get("incident_type", "Unknown")
