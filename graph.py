"""
graph.py
─────────────────────────────────────────────────────────────────────────────
The ECO GUARD — LangGraph StateGraph Wiring.

Orchestration Topology:
Orchestration Topology:
                                                         ┌──────────────────┐
  ┌─────────┐    ┌──────────────────────┐   ┌──────────►│  telemetry_node  │──┐
  │  START  │───►│    planner_node      │───┤           └──────────────────┘  │
  └─────────┘    └──────────────────────┘   │           ┌──────────────────┐  │
                                            ├──────────►│ socio_impact_node│──┤
                                            │           └──────────────────┘  │
                                            │           ┌──────────────────┐  │
                                            ├──────────►│sustainability_node│─┤
                                            │           └──────────────────┘  │
                                            │           ┌──────────────────┐  │
                                            ├──────────►│safety_auditor_node│─┤
                                            │           └──────────────────┘  │
                                            │           ┌──────────────────┐  │
                                            └──────────►│ regulatory_node  │──┤
                                                        └──────────────────┘  │
                                                                               │
                                                                               ▼
  ┌─────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐  ┌──────────────────┐
  │   gatekeeper_node   │◄──│  consensus_node  │◄──│ verification_node    │◄─│ prediction_node  │
  └─────────────────────┘   └──────────────────┘   └──────────────────────┘  └──────────────────┘
          │
          │ interrupt_before=[hitl_gate]
          ▼
  ┌─────────────────────┐
  │     hitl_gate       │  ◄── Human approves / rejects via Streamlit UI
  └─────────────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
  APPROVED    REJECTED
    │            │
    ▼            ▼
  __end__    planner_node  (re-route loop)

Key LangGraph design decisions:
  - Parallel fan-out uses `Send` API so telemetry and socio branches
    run concurrently and their `agent_scores` are appended by the
    `operator.add` reducer on EcoGuardState.
  - HITL is implemented via `interrupt_before=["hitl_gate"]` on the
    compiled graph.  The Streamlit app resumes with `.invoke()` passing
    an updated state containing `hitl_decision`.
  - The graph is compiled with a `MemorySaver` checkpointer for
    persistence across the HITL pause.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from nodes.consensus_node import consensus_node
from nodes.gatekeeper_node import gatekeeper_node
from nodes.planner import planner_node
from nodes.socio_impact_agent import socio_impact_node
from nodes.telemetry_agent import telemetry_node
from nodes.verification_agent import verification_node
from nodes.sustainability_agent import sustainability_node
from nodes.safety_auditor import safety_auditor_node
from nodes.regulatory_agent import regulatory_node
from nodes.prediction_agent import prediction_node
from nodes.circular_economy_agent import circular_economy_node
from state import HITLDecision, EcoGuardState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Parallel fan-out router
# ──────────────────────────────────────────────────────────────────────────────

def _parallel_dispatch(state: EcoGuardState) -> list[Send]:
    """
    Router function called after the Planner node.

    Returns a list of `Send` objects that LangGraph uses to dispatch
    the 5 domain agents concurrently.

    This is LangGraph's idiomatic parallel fan-out pattern.
    """
    logger.debug("parallel_dispatch: fanning out to 6 domain agents.")
    return [
        Send("telemetry_node", state),
        Send("socio_impact_node", state),
        Send("sustainability_node", state),
        Send("safety_auditor_node", state),
        Send("regulatory_node", state),
        Send("circular_economy_node", state),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# HITL gate node
# ──────────────────────────────────────────────────────────────────────────────

async def hitl_gate_node(state: EcoGuardState) -> dict[str, Any]:
    """
    Human-in-the-Loop gate node.

    This node is a thin pass-through.  The actual human pause is enforced
    by LangGraph's `interrupt_before=["hitl_gate"]` at compile time.

    When the graph is resumed by the Streamlit UI, LangGraph re-enters here
    with the updated state (containing `hitl_decision` and optionally
    `human_feedback_notes`).  This node simply validates the decision and
    returns it unchanged so the conditional edge below can route correctly.

    In mock/CLI mode the state is pre-populated with APPROVED to skip the pause.
    """
    decision_str: str = state.get("hitl_decision", HITLDecision.PENDING.value)
    try:
        decision = HITLDecision(decision_str)
    except ValueError:
        logger.error("hitl_gate: invalid hitl_decision value '%s' — defaulting to PENDING.", decision_str)
        decision = HITLDecision.PENDING

    notes: str | None = state.get("human_feedback_notes")
    logger.info(
        "hitl_gate: decision=%s notes=%s",
        decision.value,
        f'"{notes}"' if notes else "None",
    )

    return {
        "hitl_decision": decision.value,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge: route after HITL gate
# ──────────────────────────────────────────────────────────────────────────────

def _hitl_route(state: EcoGuardState) -> Literal["__end__", "planner_node"]:
    """
    Conditional edge evaluated after hitl_gate_node completes.

      APPROVED  → __end__   (pipeline concludes)
      REJECTED  → planner_node  (re-analyse loop — operator wants a re-run)
      PENDING   → __end__   (safety fallback; should not normally occur here)
    """
    decision_str: str = state.get("hitl_decision", HITLDecision.PENDING.value)

    if decision_str == HITLDecision.REJECTED.value:
        logger.warning("hitl_route: REJECTED — routing back to planner for re-analysis.")
        return "planner_node"

    logger.info("hitl_route: %s — pipeline concluding.", decision_str)
    return "__end__"


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge: should HITL be forced before gatekeeper completes?
# ──────────────────────────────────────────────────────────────────────────────

def _should_force_hitl(state: EcoGuardState) -> Literal["hitl_gate", "__end__"]:
    """
    After gatekeeper_node, decide if we must route through hitl_gate.

    Always route through hitl_gate based on user request.
    """
    logger.info("_should_force_hitl: Always routing to hitl_gate for human review.")
    return "hitl_gate"


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(interrupt_before_hitl: bool = True) -> Any:
    """
    Construct and compile the Eco Guard StateGraph.

    Args:
        interrupt_before_hitl:
            If True (default), the compiled graph will pause execution
            at `hitl_gate` and surface state to the Streamlit HITL UI.
            Set to False in automated tests / CLI smoke-runs.

    Returns:
        A compiled LangGraph `CompiledGraph` ready for `.ainvoke()`.
    """
    builder = StateGraph(EcoGuardState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("planner_node",        planner_node)
    builder.add_node("telemetry_node",      telemetry_node)
    builder.add_node("socio_impact_node",   socio_impact_node)
    builder.add_node("sustainability_node", sustainability_node)
    builder.add_node("safety_auditor_node", safety_auditor_node)
    builder.add_node("regulatory_node",     regulatory_node)
    builder.add_node("circular_economy_node", circular_economy_node)
    builder.add_node("prediction_node",     prediction_node)
    builder.add_node("verification_node",   verification_node)
    builder.add_node("consensus_node",      consensus_node)
    builder.add_node("gatekeeper_node",     gatekeeper_node)
    builder.add_node("hitl_gate",           hitl_gate_node)

    # ── Define edges ──────────────────────────────────────────────────────────

    # 1. Entry edge
    builder.add_edge(START, "planner_node")

    # 2. Parallel fan-out (Send API — concurrent branch dispatch)
    builder.add_conditional_edges(
        "planner_node",
        _parallel_dispatch,
        # Explicit targets for static graph visualisation support
        [
            "telemetry_node", "socio_impact_node", "sustainability_node",
            "safety_auditor_node", "regulatory_node", "circular_economy_node"
        ],
    )

    # 3. All 6 parallel branches converge at prediction_node
    builder.add_edge("telemetry_node",      "prediction_node")
    builder.add_edge("socio_impact_node",   "prediction_node")
    builder.add_edge("sustainability_node", "prediction_node")
    builder.add_edge("safety_auditor_node", "prediction_node")
    builder.add_edge("regulatory_node",     "prediction_node")
    builder.add_edge("circular_economy_node", "prediction_node")

    # 4. Linear pipeline: prediction → verification → consensus → gatekeeper
    builder.add_edge("prediction_node",   "verification_node")
    builder.add_edge("verification_node", "consensus_node")
    builder.add_edge("consensus_node",    "gatekeeper_node")

    # 5. Conditional edge after gatekeeper: force HITL or autonomous end
    builder.add_conditional_edges(
        "gatekeeper_node",
        _should_force_hitl,
        {
            "hitl_gate": "hitl_gate",
            "__end__":   END,
        },
    )

    # 6. HITL gate → approve (end) or reject (re-analyse loop)
    builder.add_conditional_edges(
        "hitl_gate",
        _hitl_route,
        {
            "__end__":     END,
            "planner_node": "planner_node",
        },
    )

    # ── Compile with checkpoint (required for HITL pause/resume) ─────────────
    checkpointer = MemorySaver()

    interrupt_nodes = ["hitl_gate"] if interrupt_before_hitl else []

    compiled = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,
    )

    logger.info(
        "EcoGuardGraph compiled — nodes=%d interrupt_before=%s",
        len(builder.nodes),
        interrupt_nodes or "none",
    )
    return compiled


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point (smoke-test / development runner)
# ──────────────────────────────────────────────────────────────────────────────

_DEMO_ALERT: dict[str, Any] = {
    "raw_alert": {
        "alert_id": "demo-cascadia-001",
        "title": "Magnitude 8.2 Subduction Zone Rupture — Cascadia",
        "description": (
            "Full-margin rupture detected along the Cascadia Subduction Zone. "
            "DART buoy 56001 confirms tsunami generation. PNSN ShakeAlert activated. "
            "Estimated rupture length 900 km. PGA ≥ 0.6 g across coastal WA/OR."
        ),
        "severity": "CATASTROPHIC",
        "location": "Cascadia Coast",
        "source_system": "USGS_SEISMIC",
        "geo_coordinates": [47.5, -124.3],
        "timestamp_utc": "2024-06-04T12:00:00Z",
        "raw_payload": {
            "magnitude": 8.2,
            "depth_km": 15,
            "usgs_event_id": "us7000abcd",
            "pnsn_shake_alert_issued": True,
            "dart_buoy_amplitude_cm": 12.4,
        },
    }
}


async def _run_demo() -> None:
    """CLI smoke-run: execute the full graph with the demo alert."""
    import json
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]The ECO GUARD[/bold cyan]\n"
            "[dim]Local-first multi-agent disaster mitigation engine[/dim]",
            border_style="cyan",
        )
    )

    # Compile without HITL interrupt for CLI demo
    graph = build_graph(interrupt_before_hitl=False)

    # Inject APPROVED so the graph doesn't hang at the HITL gate
    initial_state = {
        **_DEMO_ALERT,
        "hitl_decision": HITLDecision.APPROVED.value,
    }

    config = {"configurable": {"thread_id": "demo-thread-001"}}

    console.print("[yellow]> Invoking graph...[/yellow]")
    final_state = await graph.ainvoke(initial_state, config=config)

    # ── Print key outputs ─────────────────────────────────────────────────────
    console.print(f"\n[bold green]v Graph completed[/bold green]")
    console.print(f"  trace_id:              {final_state.get('trace_id')}")
    console.print(f"  H (hazard index):      {final_state.get('aggregate_hazard_index', 'N/A'):.3f}")
    console.print(f"  Var_H (variance):      {final_state.get('consensus_variance', 'N/A'):.3f}")
    console.print(f"  hitl_required:         {final_state.get('hitl_required')}")
    console.print(f"  hitl_decision:         {final_state.get('hitl_decision')}")

    # Print mitigation brief
    brief: dict = final_state.get("mitigation_brief", {})
    if brief:
        console.print("\n" + "-" * 60)
        console.print("[bold]MITIGATION BRIEF — EXECUTIVE SUMMARY[/bold]\n")
        console.print(brief.get("executive_summary", ""))
        console.print("\n[bold]RECOMMENDED ACTIONS:[/bold]")
        for i, action in enumerate(brief.get("recommended_actions", []), 1):
            console.print(f"  {i:02d}. {action}")
        console.print("\n[bold]CHAIN OF EVIDENCE:[/bold]")
        for cit in brief.get("chain_of_evidence", [])[:6]:  # cap display
            console.print(
                f"  [{cit['source_type']}] {cit['label']} "
                f"(confidence={cit['confidence_score']:.2f})"
            )

    # Print agent scores summary
    console.print("\n[bold]AGENT SCORES:[/bold]")
    for score in final_state.get("agent_scores", []):
        console.print(
            f"  {score['agent_id']}: C_i={score['hazard_score']:.1f}  "
            f"α_i={score['confidence']:.2f}  w_i={score['domain_weight']:.2f}"
        )

    # Dump full state to JSON for inspection
    output_path = os.path.join(os.path.dirname(__file__), "eco_guard_demo_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        import json
        # Serialise — state dicts are plain JSON-compatible
        json.dump(final_state, f, indent=2, default=str)
    console.print(f"\n[dim]Full state dumped to: {os.path.abspath(output_path)}[/dim]")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(_run_demo())
