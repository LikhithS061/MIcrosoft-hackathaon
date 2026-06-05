"""
api.py
─────────────────────────────────────────────────────────────────────────────
FastAPI Backend API for ECO GUARD.
Exposes endpoints for triggering the 7-agent pipeline, fetching state, 
handling Human-in-the-Loop decision overlays, and loading learning history.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from typing import Any, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from demo_incidents import INCIDENT_TITLES, get_incident_by_title
from graph import build_graph
from state import HITLDecision
from learning_engine import get_learning_engine

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="ECO GUARD",
    description="Trustworthy Multi-Agent Decision Intelligence API",
    version="3.0.0"
)

# Enable CORS for UI developers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangGraph Instance (compiled with HITL interrupts enabled)
graph = build_graph(interrupt_before_hitl=True)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas for API inputs
# ──────────────────────────────────────────────────────────────────────────────

class AlertIngestRequest(BaseModel):
    title: str
    description: str
    severity: str
    source_system: str
    geo_coordinates: list[float]  # [latitude, longitude]
    location: str = "Unknown"
    timestamp_utc: Optional[str] = None
    raw_payload: Optional[dict[str, Any]] = None

class HITLDecisionRequest(BaseModel):
    decision: str  # APPROVED, REJECTED, INVESTIGATE
    notes: Optional[str] = ""

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _extract_display_state(graph_state: Any) -> dict[str, Any]:
    """Helper to serialize graph state values into a clean UI JSON payload."""
    if not graph_state or not graph_state.values:
        return {}
    
    val = graph_state.values
    return {
        "trace_id": val.get("trace_id"),
        "raw_alert": val.get("raw_alert"),
        "historical_similar_cases": val.get("historical_similar_cases", []),
        "agent_scores": val.get("agent_scores", []),
        "sustainability_score": val.get("sustainability_score"),
        "environmental_impact_report": val.get("environmental_impact_report"),
        "safety_risk_score": val.get("safety_risk_score"),
        "emergency_response_plan": val.get("emergency_response_plan", []),
        "regulatory_report": val.get("regulatory_report"),
        "stakeholders_to_notify": val.get("stakeholders_to_notify", []),
        "prediction_report": val.get("prediction_report"),
        "verification_report": val.get("verification_report"),
        "aggregate_hazard_index": val.get("aggregate_hazard_index"),
        "aggregate_sustainability_score": val.get("aggregate_sustainability_score"),
        "aggregate_safety_score": val.get("aggregate_safety_score"),
        "consensus_variance": val.get("consensus_variance"),
        "agent_disagreement_level": val.get("agent_disagreement_level"),
        "hitl_required": val.get("hitl_required", False),
        "hitl_triggered_reason": val.get("hitl_triggered_reason"),
        "mitigation_brief": val.get("mitigation_brief"),
        "hitl_decision": val.get("hitl_decision", "PENDING"),
        "operational_mode": val.get("operational_mode", "COMMUNITY_INTELLIGENCE"),
        "circular_opportunity_report": val.get("circular_opportunity_report"),
        "symbiosis_index": val.get("symbiosis_index"),
        "node_execution_log": val.get("node_execution_log", []),
        "langfuse_trace_url": val.get("langfuse_trace_url"),
    }

# ──────────────────────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/incidents")
async def get_demo_incidents():
    """List the preloaded incident options from the demo dataset."""
    incidents = []
    for title in INCIDENT_TITLES:
        raw = get_incident_by_title(title)
        if raw:
            incidents.append({
                "title": title,
                "payload": raw
            })
    return incidents

@app.post("/api/alerts")
async def trigger_alert_pipeline(payload: AlertIngestRequest):
    """Ingest a new alert and run the 7-agent consensus pipeline."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    from datetime import datetime, timezone
    timestamp_utc = payload.timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    initial_state = {
        "raw_alert": {
            "title": payload.title,
            "description": payload.description,
            "severity": payload.severity,
            "source_system": payload.source_system,
            "geo_coordinates": tuple(payload.geo_coordinates),
            "location": payload.location,
            "timestamp_utc": timestamp_utc,
            "raw_payload": payload.raw_payload or {},
        },
        "hitl_decision": HITLDecision.PENDING.value,
    }

    try:
        # Run graph. It will pause at hitl_gate if hitl_required is True
        logger.info(f"Triggering graph execution for thread={thread_id}")
        live_feed_messages.insert(0, {"timestamp": datetime.now(timezone.utc).isoformat(), "message": f"Agent 0 (Evidence Collector) initiated thread {thread_id[:8]} for '{payload.title}'."})
        await graph.ainvoke(initial_state, config=config)
        
        # Check if it paused at hitl_gate
        state = graph.get_state(config)
        is_paused = "hitl_gate" in state.next

        # Save finished record to learning engine automatically if it didn't pause
        if not is_paused:
            engine = get_learning_engine()
            record = _extract_display_state(state)
            await engine.store_incident(record)

        return {
            "thread_id": thread_id,
            "is_paused": is_paused,
            "state": _extract_display_state(state)
        }
    except Exception as exc:
        logger.exception("Error running graph pipeline")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/state/{thread_id}")
async def get_pipeline_state(thread_id: str):
    """Retrieve the current state of a graph run by thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session state not found")
    
    return {
        "thread_id": thread_id,
        "is_paused": "hitl_gate" in state.next,
        "state": _extract_display_state(state)
    }

@app.post("/api/decide/{thread_id}")
async def submit_operator_decision(thread_id: str, request: HITLDecisionRequest):
    """Resume a paused pipeline by submitting the human approval decision."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session state not found")
    
    if "hitl_gate" not in state.next:
        raise HTTPException(status_code=400, detail="Pipeline is not currently paused for human approval")

    try:
        logger.info(f"Resuming graph for thread={thread_id} with decision={request.decision}")
        
        # Update graph state with human approval input
        graph.update_state(
            config,
            {
                "hitl_decision": request.decision,
                "human_feedback_notes": request.notes
            }
        )

        # Resume execution
        await graph.ainvoke(None, config=config)

        # Get final state
        final_state = graph.get_state(config)

        # Store finalized run in SQLite learning engine
        engine = get_learning_engine()
        record = _extract_display_state(final_state)
        await engine.store_incident(record)

        return {
            "thread_id": thread_id,
            "is_paused": False,
            "state": _extract_display_state(final_state)
        }
    except Exception as exc:
        logger.exception("Error resuming graph pipeline")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/history")
async def get_learning_history():
    """Retrieve historical logs stored in the SQLite Learning Engine."""
    try:
        engine = get_learning_engine()
        history = await engine.get_all()
        return history
    except Exception as exc:
        logger.exception("Error loading learning database history")
        raise HTTPException(status_code=500, detail=str(exc))

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Real-Time Monitoring
# ──────────────────────────────────────────────────────────────────────────────

monitoring_task = None
monitoring_enabled = False
live_feed_messages = []

async def _monitoring_loop():
    global monitoring_enabled
    while monitoring_enabled:
        await asyncio.sleep(45) # 45 seconds
        if not monitoring_enabled:
            break
        # Pick random demo incident to simulate synthetic telemetry
        inc_title = random.choice(INCIDENT_TITLES)
        raw_inc = get_incident_by_title(inc_title)
        if raw_inc:
            msg = f"Synthetic Stream: Detected potential anomaly matching {inc_title}. Initiating analysis..."
            live_feed_messages.insert(0, {"timestamp": datetime.now(timezone.utc).isoformat(), "message": msg})
            if len(live_feed_messages) > 50:
                live_feed_messages.pop()
            
            # Auto-trigger pipeline
            req = AlertIngestRequest(
                title=raw_inc["title"],
                description=raw_inc["description"],
                severity=raw_inc["severity"],
                source_system=raw_inc.get("source_system", "Synthetic"),
                location=raw_inc.get("location", "Unknown"),
                geo_coordinates=raw_inc.get("geo_coordinates", [0.0, 0.0]),
                raw_payload=raw_inc.get("raw_payload")
            )
            # Fire and forget
            asyncio.create_task(trigger_alert_pipeline(req))


@app.post("/api/monitoring/toggle")
async def toggle_monitoring():
    global monitoring_task, monitoring_enabled
    monitoring_enabled = not monitoring_enabled
    if monitoring_enabled:
        monitoring_task = asyncio.create_task(_monitoring_loop())
    elif monitoring_task:
        monitoring_task.cancel()
        monitoring_task = None
    return {"monitoring_enabled": monitoring_enabled}

@app.get("/api/live_feed")
async def get_live_feed():
    return {"messages": live_feed_messages}


# ──────────────────────────────────────────────────────────────────────────────
# Serve Static Frontend Files
# ──────────────────────────────────────────────────────────────────────────────

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    """Serve the web dashboard index page."""
    return FileResponse("static/index.html")
