# ALLIANCE UNIVERSITY AI FOR GOOD HACKATHON 2026
## ECO GUARD: Resilient Local-First Trustworthy Multi-Agent Decision Intelligence Platform
### Project Specification & System Architecture Document
**Team:** Innovation Alpha | **Date:** June 5, 2026 | **Status:** Technical Spec

---

## Abstract
Modern smart-city and industrial infrastructure deployments generate massive telemetry arrays from IoT sensor meshes, yet they remain vulnerable to critical incident blind spots and alarm fatigue. Traditional disaster-response software operates in a reactive "log-and-alert" loop, heavily relying on general-purpose Large Language Models (LLMs) that are prone to hallucinating regulatory thresholds or physical boundaries. 

**Eco Guard** addresses these architectural challenges by introducing a local-first, trust-orchestrated multi-agent decision intelligence platform built on LangGraph. By combining a parallel 7-agent topology, a mathematically rigorous weighted consensus engine, an SQLite-backed historic pattern learning database, and a zero-hallucination GraphRAG compliance verifier, Eco Guard bridges the gap between raw physical telemetry and human-in-the-loop incident mitigation.

---

## 1. Executive Summary
In high-density residential and complex industrial zones, infrastructure anomalies—including water contamination, structural elevator failures, basement flooding, gas leaks, and electrical distribution panel faults—must be identified, analyzed, and mitigated immediately. Relying on isolated sensor thresholds or unverified single-LLM pipelines poses severe safety and reliability risks.

Eco Guard introduces a generative, proactive multi-agent orchestration pipeline. Running locally on edge gateways and utilizing ChatGroq (llama-3.3-70b-versatile) with local Ollama fallback, the system processes raw telemetry streams, evaluates demographic vulnerability, consults national regulations, references historic incidents, and cross-checks agent claims against factual knowledge bases.

To guarantee operational safety in high-stakes settings, Eco Guard incorporates a dual-layer safety defense:
1. **The Mathematical Consensus Engine:** A weighted algebraic evaluation layer that aggregates agent-specific hazard scores ($C_i$) and confidence levels ($\alpha_i$) to calculate a centralized Aggregate Hazard Index ($H$) and a Consensus Variance ($\sigma^2_H$).
2. **The "Glass-Box" HITL Action Gate:** A web control center served via FastAPI and a custom reactive frontend dashboard. It pauses execution via LangGraph state checkpoints (interrupts) to stream live agent thought traces and holds warning/mitigation dispatches until an authorized human operator clicks `[ Approve Action ]`.

---

## 2. The Core Paradigm: Multi-Agent Collaborative Task Force
Eco Guard replaces monolithic AI models with a specialized 7-agent task force. To understand this paradigm, consider the **Medical Task Force Analogy** contrasted with traditional systems in Table 1.

### Table 1: Systemic Comparison: Reactive Monitoring vs. Eco Guard

| Feature Dimension | Traditional Reactive Monitoring | Eco Guard Decision Intelligence |
| :--- | :--- | :--- |
| **Primary Objective** | Passive hazard detection, logging, and raw alert broadcasts. | Active threat reasoning, claims verification, and automated mitigation planning. |
| **Data Trigger** | Anomaly boundary breach (e.g., sensor threshold exceeding). | Steady-state telemetry streams + synthetic real-time event ingestion. |
| **Target Sinks** | Raw notification payloads to emergency operators. | Human-in-the-loop dashboards, local authorities, and safety command boards. |
| **Economic & Operational Model** | Cost center (manual verification, slower incident response). | Profit center (reduced liability, rapid risk mitigation, minimized infrastructure downtime). |
| **Control Loop** | Manual operator intervention and offline emergency execution. | Semi-autonomous flow via LangGraph checkpoints with a physical operator interrupt. |

Rather than delegating the entire risk assessment to a single general-purpose prompt, Eco Guard deploys specialized agents that evaluate raw incident inputs from distinct domain perspectives. The agents debate, score individual risks, and compile a structured, verified mitigation brief.

---

## 3. System Architecture & Agent Topologies
Eco Guard operates an event-driven LangGraph `StateGraph` microservices workflow. The runtime environment executes a parallel dispatch fan-out using LangGraph's Send API, converging at sequential prediction, verification, and consensus nodes.

```
                                        ┌──► TelemetryAgent (A1) ──┐
                                        ├──► SocioImpactAgent (A2) ┤
Planner ──► Parallel Dispatch (Send) ───┼──► Sustainability (A3) ──┼──► Prediction (A6) ──► Verification (A7) ──► Consensus ──► Gatekeeper ──► [HITL]
                                        ├──► Safety Auditor (A4) ──┤
                                        └──► Regulatory Agent (A5) ┘
```

### 3.1 Agent 1: The Telemetry Agent (Physical Sensor & Telemetry Expert)
* **Focus:** Real-time parsing of raw physical sensor arrays (seismic, meteorological, chemical, IoT mesh).
* **Inputs:** Raw alert payloads containing parameters such as seismic Peak Ground Acceleration (PGA), magnitude, temperature rise, pH level, turbidity, gas concentration (ppm), water loss (liters/day), or voltage fluctuations.
* **Mechanism:** Ingests raw metrics. Queries Qdrant semantic databases and Neo4j relational mocks to construct sensor context.
* **Outputs:** Localized hazard score $C_1 \in [0, 100]$ indicating physical severity, and a confidence score $\alpha_1 \in [0, 1]$ reflecting data integrity.

### 3.2 Agent 2: The Socio-Impact Agent (Demographic & Vulnerability Specialist)
* **Focus:** Human vulnerability scoring, demographic density evaluation, and geographic proximity checks.
* **Inputs:** Alert geocoordinates, residents affected, presence of sensitive structures (schools, hospitals, care homes).
* **Mechanism:** Analyzes the population footprint surrounding the anomaly center to assess social exposure.
* **Outputs:** Social hazard score $C_2 \in [0, 100]$ and confidence score $\alpha_2 \in [0, 1]$.

### 3.3 Agent 3: The Sustainability Agent (Environmental Sustainability Analyst)
* **Focus:** Environmental damage, resource wastage, and ecological policy compliance.
* **Inputs:** Water loss rates, air quality indices (PM2.5, NOx), carbon emissions, and solid waste volumes.
* **Mechanism:** Gauges environmental decay and resource footprint. Cross-references CPCB guidelines and waste management regulations.
* **Outputs:** Environmental hazard score $C_3 \in [0, 100]$, confidence score $\alpha_3 \in [0, 1]$, and a detailed resource loss estimate.

### 3.4 Agent 4: The Safety Auditor (Structural & Physical Safety Expert)
* **Focus:** Fire hazards, structural integrity breaches, electrical faults, and emergency response planning.
* **Inputs:** Smoke sensors, motor temperatures, flood water levels, electrical arcing, and inspector reports.
* **Mechanism:** Evaluates primary physical safety hazards and drafts actionable NIMS/ICS-compliant emergency response plans.
* **Outputs:** Safety hazard score $C_4 \in [0, 100]$, confidence score $\alpha_4 \in [0, 1]$, and recommended emergency actions.

### 3.5 Agent 5: The Regulatory Agent (Government Regulatory Compliance Expert)
* **Focus:** Statutory compliance and compliance violations.
* **Inputs:** Raw incident properties. Queries a local **ChromaDB vector store** preloaded with regulatory documents.
* **Mechanism:** Cross-references alert parameters against official Indian guidelines (BIS, CPCB, NDMA, CEE). Falls back to a curated compliance knowledge base in offline/mock modes.
* **Outputs:** Regulatory compliance score $C_5 \in [0, 100]$ (reflecting compliance risk), confidence score $\alpha_5 \in [0, 1]$, a list of violations, and legal implication briefs.

### 3.6 Agent 6: The Prediction Agent (Future Risk Forecasting & SQLite Learning Engine)
* **Focus:** Forecasting cascading risk sequences and secondary failures.
* **Inputs:** Aggregated rationales from upstream agents, plus historical similar cases queried from the **SQLite Learning Engine database**.
* **Mechanism:** Analyzes keyword similarity, severity, and historic case outcomes to predict follow-on events, probabilities, and time horizons.
* **Outputs:** Predicted risk index $C_6 \in [0, 100]$, confidence score $\alpha_6 \in [0, 1]$, and a list of structured `PredictedRisk` forecasts.

### 3.7 Agent 7: The Verification Agent (Adversarial Red-Team Verifier)
* **Focus:** Claims cross-referencing and hallucination mitigation.
* **Inputs:** Scores, rationales, and citations generated by Agents 1–6.
* **Mechanism:** Adopts an adversarial posture. Concurrently cross-checks numeric assertions (e.g. `entity:property=value` assertions in rationales) against properties in the Neo4j Graph DB and validates the existence of cited vector chunk IDs in Qdrant.
* **Outputs:** Verification report containing verified claim registries, flagged claims (schema violations or contradictions), and an overall trustworthiness score.

---

## 4. The Mathematical Consensus Engine
To prevent unverified AI outputs and secure deterministic decision gates, Eco Guard routes all domain scores through a weighted mathematical consensus layer in [sentinel/math_consensus.py](file:///c:/THE%20FILE%20OF%20FILES/Programing%20files/Code%20files/sentinel_guard/sentinel_guard/sentinel/math_consensus.py).

### 4.1 Aggregate Hazard Index ($H$)
Let $N$ be the number of scoring agents ($N=6$). Each agent $i$ outputs its localized hazard score $C_i \in [0, 100]$ and confidence score $\alpha_i \in [0, 1]$. The system is pre-configured with static domain weights $w_i$, reflecting the evidentiary priority of each agent's domain:

$$\text{Telemetry (} w_1=0.55\text{), Socio-Impact (} w_2=0.45\text{), Sustainability (} w_3=0.35\text{), Safety (} w_4=0.50\text{), Regulatory (} w_5=0.40\text{), Prediction (} w_6=0.30\text{)}$$

The **Aggregate Hazard Index ($H$)** is calculated as the weighted confidence average of agent hazard scores:

$$H = \frac{\sum_{i=1}^N w_i \cdot \alpha_i \cdot C_i}{\sum_{i=1}^N w_i \cdot \alpha_i}$$

Clamped to $[0.0, 100.0]$, the index triggers a mandatory Human-in-the-Loop interrupt if $H \ge H_{\text{critical}}$ (where $H_{\text{critical}} = 75.0$).

### 4.2 Consensus Variance ($\sigma^2_H$)
To capture internal agent disagreements, LLM hallucinations, or sensor drift conflicts, the engine calculates the weighted **Consensus Variance ($\sigma^2_H$)**:

$$\sigma^2_H = \frac{\sum_{i=1}^N w_i \cdot \alpha_i \cdot (C_i - H)^2}{\sum_{i=1}^N w_i \cdot \alpha_i}$$

If $\sigma^2_H \ge \sigma^2_{\text{critical}}$ (where $\sigma^2_{\text{critical}} = 150.0$), the system flags a "Critical Internal Disagreement", bypasses autonomous logic, forces the decision status to `HUMAN REVIEW REQUIRED`, and triggers the human confirmation gate.

### 4.3 Consensus Engine Decision Matrix
Table 2 outlines the consensus engine behavior under nominal, high-risk, and conflict conditions.

#### Table 2: Mathematical Consensus Engine Validation Matrix
| Scenario | $w$ Vector | $\alpha$ Vector | $C$ Vector | Engine Metrics | Decision Status & Actions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Nominal Low Risk** | $[0.55, 0.45, 0.35, 0.50, 0.40, 0.30]$ | $[0.90, 0.85, 0.80, 0.90, 0.90, 0.70]$ | $[20, 15, 10, 25, 20, 15]$ | $H \approx 18.25$, $\sigma^2_H \approx 20.40$ | **AUTO APPROVED** (Low risk, strong consensus). |
| **2. Severe High Risk** | $[0.55, 0.45, 0.35, 0.50, 0.40, 0.30]$ | $[0.95, 0.90, 0.85, 0.95, 0.95, 0.80]$ | $[85, 80, 75, 90, 88, 80]$ | $H \approx 83.40$, $\sigma^2_H \approx 25.60$ | **HUMAN REVIEW REQUIRED** ($H \ge 75.0$ critical breach). |
| **3. Agent Conflict** | $[0.55, 0.45, 0.35, 0.50, 0.40, 0.30]$ | $[0.95, 0.90, 0.80, 0.95, 0.95, 0.85]$ | $[98, 10, 85, 95, 12, 90]$ | $H \approx 62.45$, $\sigma^2_H \approx 165.20$ | **HUMAN REVIEW REQUIRED** (Variance $\sigma^2_H \ge 150.0$ threshold breach). |

---

## 5. The "Glass-Box" Safety Protocol & HITL UI
The system architecture prioritizes absolute explainability and operator oversight before executing any public-safety dispatches or hardware relays.

### 5.1 Thought Trace Logs
Every reasoning step, vector lookup, relational cross-check, and numerical calculation is preserved in structured JSON blocks inside the LangGraph state registry. This trace is rendered sequentially on the operator dashboard, displaying exactly how each agent reasoned about the incident.

### 5.2 The Human-in-the-Loop Action Gate
Eco Guard compiles the StateGraph with a persistent `MemorySaver` checkpointer. When the consensus node or verification agent flags a critical threshold breach:
1. The graph execution pauses at the `hitl_gate` node, saving the state under a unique `thread_id`.
2. The FastAPI backend exposes the paused state via the `/api/state/{thread_id}` endpoint.
3. The custom reactive dashboard displays the paused alert, risk gauges, agent thought traces, and the red-bordered **CRITICAL DECISION REQUIRED** panel.
4. The operator enters assessment notes and clicks `[ Approve Action ]`, `[ Reject & Re-evaluate ]`, or `[ Order Further Investigation ]`.
5. The frontend POSTs the decision to `/api/decide/{thread_id}`, updates the LangGraph checkpointer, resumes execution, and saves the final audit trail.

---

## 6. Persistence & Learning Infrastructure

### 6.1 SQLite Learning Engine
Eco Guard implements an asynchronous SQLite-backed learning engine in [sentinel/learning_engine.py](file:///c:/THE%20FILE%20OF%20FILES/Programing%20files/Code%20files/sentinel_guard/sentinel_guard/sentinel/learning_engine.py).

#### Database Schema:
```sql
CREATE TABLE IF NOT EXISTS incidents (
    id                   TEXT PRIMARY KEY,
    incident_type        TEXT NOT NULL,
    location             TEXT,
    severity             TEXT,
    hazard_score         REAL,
    sustainability_score REAL,
    safety_score         REAL,
    regulatory_issues    TEXT, -- JSON list
    recommended_actions  TEXT, -- JSON list
    hitl_decision        TEXT,
    outcome              TEXT,
    lessons_learned      TEXT,
    timestamp            TEXT
);
```

#### Similarity Scoring Model:
When a new alert is ingested, historical cases are queried and ranked using a custom similarity score:

$$\text{Similarity Score} = \frac{3 \cdot \mathbb{I}(\text{Type Match}) + 2 \cdot \mathbb{I}(\text{Severity Match}) + 0.5 \cdot \sum_{w \in \text{Keywords}} \mathbb{I}(w \in \text{Description})}{8.0}$$

### 6.2 GraphRAG & Claims Verification
* **Vector DB Client (Qdrant):** Re-runs semantic searches to verify that the text inside the agent's cited chunk IDs (e.g. `QDRANT_CHUNK`) contains terms relevant to the agent's rationale.
* **Graph DB Client (Neo4j):** Queries specific entity relationships and properties (e.g. `NEO4J_NODE`) to cross-reference numeric values and physical parameters, ensuring the LLM has not hallucinated sensor metadata.

---

## 7. Sustainability and ESG Impact / Future Roadmap
Eco Guard integrates with sustainability objectives by providing:
1. **Sustainability Tracking:** Agent 3 (Sustainability Analyst) isolates resource loss rates and compiles environmental damage levels, mapping them to ESG reporting.
2. **Modular Extension Interface:** The LangGraph `StateGraph` allows developers to plug in new agents (e.g., a financial impact estimator or smart contract auditor) by adding a node to the graph compilation in [sentinel/graph.py](file:///c:/THE%20FILE%20OF%20FILES/Programing%20files/Code%20files/sentinel_guard/sentinel_guard/sentinel/graph.py) and extending the `SentinelState` dictionary in [sentinel/state.py](file:///c:/THE%20FILE%20OF%20FILES/Programing%20files/Code%20files/sentinel_guard/sentinel_guard/sentinel/state.py).

---

## 8. Conclusion
Eco Guard transitions public safety and environmental monitoring from passive alarm boards to proactive, verifiable multi-agent intelligence. By combining LangGraph state orchestration with mathematical consensus cross-checks, zero-hallucination vector/graph verifications, and an interactive human-in-the-loop dashboard, the platform provides a robust framework for high-stakes municipal and industrial safety assurance.
