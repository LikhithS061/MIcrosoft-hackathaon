# ECO GUARD

A resilient, **local-first**, Trustworthy Multi-Agent Decision Intelligence Platform built on LangGraph.

## Stack
- **Orchestration:** LangGraph `StateGraph` (7-agent topology)
- **LLM:** Groq (llama-3.3-70b-versatile) / Local Ollama Fallback
- **Regulatory RAG:** ChromaDB + Sentence Transformers (Zero Hallucination)
- **Learning Engine:** SQLite (Historical Case Pattern Matching)
- **GraphRAG:** Neo4j + Qdrant (Verification Agent)
- **HITL UI:** Streamlit (Premium Dark Mode Dashboard)

## Quick Start

```bash
# 1. Copy env template
cp .env.example .env

# 2. Install
pip install -e ".[dev]"

# 3. Run CLI demo (mock mode — no Ollama/Neo4j/Qdrant required)
python -m eco_guard.graph

# 4. Run tests
pytest tests/ -v
```

## Architecture
```
                                        ┌──► TelemetryAgent (A1) ──┐
                                        ├──► SocioImpactAgent (A2) ┤
Planner ──► Parallel Dispatch (Send) ───┼──► Sustainability (A3) ──┼──► Prediction (A6) ──► Verification (A7) ──► Consensus ──► Gatekeeper ──► [HITL]
                                        ├──► Safety Auditor (A4) ──┤
                                        └──► Regulatory Agent (A5) ┘
```

## Features
- **7 Specialized Agents:** Telemetry, Socio-Impact, Sustainability, Safety, Regulatory, Prediction, Verification
- **Zero Hallucination Regulations:** ChromaDB RAG against actual Indian government PDFs (NDMA, CPCB, BIS)
- **Pattern Learning:** SQLite-backed historical incident learning engine
- **Mathematical Consensus:** H-Index and variance (σ²) cross-checks
- **10 Demo Incidents:** Pre-built dataset covering fire, flood, water contamination, gas leaks, etc.
