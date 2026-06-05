'use strict';

let activeThreadId = null;
let activeState = null;
let demoIncidents = [];

const systemStatusPill = document.getElementById('system-status-pill');
const systemStatusText = document.getElementById('system-status-text');
const systemStatusDot = document.getElementById('system-status-dot');
const decisionStatusBadge = document.getElementById('decision-status-badge');

const uploadZone = document.getElementById('upload-zone');
const incidentSelect = document.getElementById('incident-select');
const injectBtn = document.getElementById('inject-btn');
const scenarioButtons = document.querySelectorAll('.scenario-btn');

const hazardValue = document.getElementById('hazard-value');
const hazardProgressFill = document.getElementById('hazard-progress-fill');
const hazardStatusLabel = document.getElementById('hazard-status-label');
const varianceValue = document.getElementById('variance-value');
const varianceStatusLabel = document.getElementById('variance-status-label');
const varianceStatePill = document.getElementById('variance-state-pill');
const opportunityValue = document.getElementById('opportunity-value');
const opportunityStatusLabel = document.getElementById('opportunity-status-label');
const opportunityStatePill = document.getElementById('opportunity-state-pill');
const decisionReasonText = document.getElementById('decision-reason-text');

const btnApprove = document.getElementById('btn-approve');
const btnReject = document.getElementById('btn-reject');
const btnApproveModal = document.getElementById('btn-approve-modal');
const btnRejectModal = document.getElementById('btn-reject-modal');
const btnInvestigate = document.getElementById('btn-investigate');
const hitlOverlay = document.getElementById('hitl-overlay');
const hitlReasonText = document.getElementById('hitl-reason-text');

const toggleAdvancedInput = document.getElementById('toggle-advanced-input');
const advancedTabs = document.getElementById('advanced-tabs');
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

const agentTimeline = document.getElementById('agent-timeline');
const agentsReportGrid = document.getElementById('agents-report-grid');
const mathLedgerBody = document.getElementById('math-ledger-body');
const regulatoryTableBody = document.getElementById('regulatory-table-body');
const historyTableBody = document.getElementById('history-table-body');

const toastContainer = document.getElementById('toast-container');

const STATUS_COLORS = {
  nominal: 'var(--accent-green)',
  warning: 'var(--accent-amber)',
  critical: 'var(--accent-red)',
  info: 'var(--accent-blue)',
};

const TAG_CLASS = {
  nominal: 'tag-green',
  warning: 'tag-amber',
  critical: 'tag-red',
  info: 'tag-blue',
  muted: 'tag-muted',
};

document.addEventListener('DOMContentLoaded', () => {
  setUiState('idle');
  loadDemoIncidents();
  loadHistoryLogs();
  setupTabListeners();
  setupEventListeners();
  setupDragAndDrop();
  setupAdvancedToggle();
});

function setupEventListeners() {
  if (injectBtn) {
    injectBtn.addEventListener('click', () => {
      const idx = incidentSelect?.value;
      if (!idx) {
        showToast('Select a preset scenario first.', 'warning');
        return;
      }
      const incident = demoIncidents[Number(idx)];
      if (incident?.payload) {
        injectTelemetryStream(incident.payload);
      }
    });
  }

  scenarioButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const scenario = btn.getAttribute('data-scenario');
      if (!scenario) return;
      const incident = pickScenarioIncident(scenario);
      if (!incident) {
        showToast('No matching scenario found.', 'warning');
        return;
      }
      injectTelemetryStream(incident.payload);
    });
  });

  if (btnApprove) btnApprove.addEventListener('click', () => submitOperatorDecision(activeThreadId, 'APPROVED'));
  if (btnReject) btnReject.addEventListener('click', () => submitOperatorDecision(activeThreadId, 'REJECTED'));
  if (btnApproveModal) btnApproveModal.addEventListener('click', () => submitOperatorDecision(activeThreadId, 'APPROVED'));
  if (btnRejectModal) btnRejectModal.addEventListener('click', () => submitOperatorDecision(activeThreadId, 'REJECTED'));
  if (btnInvestigate) btnInvestigate.addEventListener('click', () => submitOperatorDecision(activeThreadId, 'INVESTIGATE'));
}

function pickScenarioIncident(scenario) {
  if (!demoIncidents.length) return null;
  const bySeverity = (sevList) => demoIncidents.find(i => sevList.includes((i.payload?.severity || '').toUpperCase()));

  if (scenario === 'A') {
    return bySeverity(['LOW', 'MEDIUM']) || demoIncidents[0];
  }
  if (scenario === 'B') {
    return bySeverity(['CRITICAL', 'HIGH']) || demoIncidents[0];
  }
  if (scenario === 'C') {
    return bySeverity(['HIGH', 'MEDIUM']) || demoIncidents[0];
  }
  return demoIncidents[0];
}

function setupAdvancedToggle() {
  if (!toggleAdvancedInput || !advancedTabs) return;
  advancedTabs.classList.add('hidden');
  toggleAdvancedInput.addEventListener('change', () => {
    if (toggleAdvancedInput.checked) {
      advancedTabs.classList.remove('hidden');
      advancedTabs.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      advancedTabs.classList.add('hidden');
    }
  });
}

function setupTabListeners() {
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      const targetId = btn.getAttribute('data-tab');
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');
    });
  });
}

function setupDragAndDrop() {
  if (!uploadZone) return;

  ['dragenter', 'dragover'].forEach(evt => {
    uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    uploadZone.addEventListener(evt, () => {
      uploadZone.classList.remove('dragover');
    });
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      const f = files[0];
      const isValid = f.name.endsWith('.csv') || f.name.endsWith('.xlsx');
      if (isValid) {
        showToast(`"${f.name}" detected — select a preset to ingest.`, 'info', 5000);
      } else {
        showToast('Unsupported format. Use .csv or .xlsx files.', 'warning');
      }
    }
  });

  uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      showToast('Use the preset selector to ingest a scenario.', 'info');
    }
  });
}

async function loadDemoIncidents() {
  try {
    const res = await fetch('/api/incidents');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    demoIncidents = await res.json();

    incidentSelect.innerHTML = '<option value="" disabled selected>Select scenario...</option>';
    demoIncidents.forEach((inc, index) => {
      const opt = document.createElement('option');
      opt.value = index;
      opt.textContent = inc.title;
      incidentSelect.appendChild(opt);
    });
  } catch (err) {
    showToast('Could not load incident presets.', 'danger');
  }
}

async function injectTelemetryStream(payload) {
  setUiState('processing');
  showToast('Pipeline initiated — agents dispatched.', 'info');

  try {
    const response = await fetch('/api/alerts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server error ${response.status}`);
    }

    const data = await response.json();
    activeThreadId = data.thread_id;
    activeState = data.state;

    renderDashboard(data.is_paused);
  } catch (err) {
    setUiState('idle');
    showToast(`Pipeline error: ${err.message}`, 'danger');
  }
}

function renderDashboard(isPaused) {
  updateMetrics();
  populateAllTabs();

  if (isPaused) {
    setUiState('hitl');
  } else {
    setUiState('resolved');
  }
}

function setUiState(state) {
  if (!systemStatusPill || !systemStatusText) return;

  systemStatusPill.classList.remove('nominal', 'warning', 'critical');
  if (hitlOverlay) hitlOverlay.classList.add('hidden');

  if (state === 'idle') {
    updateStatusPill('System idle — awaiting ingestion', 'info');
    if (decisionStatusBadge) {
      decisionStatusBadge.textContent = 'Awaiting';
      decisionStatusBadge.className = `tag ${TAG_CLASS.muted}`;
    }
  }

  if (state === 'processing') {
    updateStatusPill('Processing telemetry and fanning out to agents...', 'info');
    if (decisionStatusBadge) {
      decisionStatusBadge.textContent = 'Processing';
      decisionStatusBadge.className = `tag ${TAG_CLASS.info}`;
    }
  }

  if (state === 'hitl') {
    updateStatusPill('Operator action required', 'critical');
    if (hitlOverlay) hitlOverlay.classList.remove('hidden');
    if (decisionStatusBadge) {
      decisionStatusBadge.textContent = 'Review Required';
      decisionStatusBadge.className = `tag ${TAG_CLASS.critical}`;
    }
  }

  if (state === 'resolved') {
    updateStatusPill('Decision recorded — monitoring stabilized', 'nominal');
    if (decisionStatusBadge) {
      const dec = activeState?.hitl_decision || 'APPROVED';
      decisionStatusBadge.textContent = dec;
      decisionStatusBadge.className = `tag ${dec === 'REJECTED' ? TAG_CLASS.critical : TAG_CLASS.nominal}`;
    }
  }
}

function updateStatusPill(text, tone) {
  systemStatusText.textContent = text;
  if (tone && STATUS_COLORS[tone]) {
    systemStatusDot.style.background = STATUS_COLORS[tone];
  }
  if (tone === 'nominal') systemStatusPill.classList.add('nominal');
  if (tone === 'warning') systemStatusPill.classList.add('warning');
  if (tone === 'critical') systemStatusPill.classList.add('critical');
}

function updateMetrics() {
  const H = Number(activeState?.aggregate_hazard_index ?? 0);
  const variance = Number(activeState?.consensus_variance ?? 0);
  const opportunity = Number(activeState?.symbiosis_index ?? 0);

  if (hazardValue) hazardValue.textContent = `${H.toFixed(1)}%`;
  if (hazardProgressFill) hazardProgressFill.style.width = `${Math.min(100, H)}%`;

  if (H >= 75) {
    hazardProgressFill.style.background = STATUS_COLORS.critical;
    setTag(hazardStatusLabel, 'Critical', 'critical');
  } else if (H >= 40) {
    hazardProgressFill.style.background = STATUS_COLORS.warning;
    setTag(hazardStatusLabel, 'Elevated', 'warning');
  } else {
    hazardProgressFill.style.background = STATUS_COLORS.nominal;
    setTag(hazardStatusLabel, 'Nominal', 'nominal');
  }

  if (varianceValue) varianceValue.textContent = variance.toFixed(1);
  if (variance >= 150) {
    setTag(varianceStatusLabel, 'Conflict', 'critical');
    varianceStatePill.textContent = 'Logic Contradiction Flagged';
  } else if (variance >= 75) {
    setTag(varianceStatusLabel, 'Warning', 'warning');
    varianceStatePill.textContent = 'Moderate Divergence';
  } else {
    setTag(varianceStatusLabel, 'Stable', 'nominal');
    varianceStatePill.textContent = 'High Agreement';
  }

  if (opportunityValue) opportunityValue.textContent = opportunity.toFixed(1);
  if (opportunity >= 70) {
    setTag(opportunityStatusLabel, 'High', 'nominal');
    opportunityStatePill.textContent = 'High Symbiosis Potential';
  } else if (opportunity >= 40) {
    setTag(opportunityStatusLabel, 'Moderate', 'info');
    opportunityStatePill.textContent = 'Moderate Opportunity';
  } else if (opportunity > 0) {
    setTag(opportunityStatusLabel, 'Low', 'warning');
    opportunityStatePill.textContent = 'Low Opportunity';
  } else {
    setTag(opportunityStatusLabel, 'None', 'muted');
    opportunityStatePill.textContent = 'No Opportunity';
  }

  if (decisionReasonText) {
    decisionReasonText.textContent = activeState?.mitigation_brief?.executive_summary
      || activeState?.hitl_triggered_reason
      || 'Awaiting consensus analysis...';
  }

  if (hitlReasonText) {
    hitlReasonText.textContent = activeState?.hitl_triggered_reason
      || 'Pipeline paused: operator decision required.';
  }
}

function setTag(el, text, tone) {
  if (!el) return;
  const cls = TAG_CLASS[tone] || TAG_CLASS.muted;
  el.textContent = text;
  el.className = `tag ${cls}`;
}

async function submitOperatorDecision(threadId, decisionType) {
  if (!threadId) {
    showToast('No active pipeline thread.', 'warning');
    return;
  }

  setUiState('processing');
  showToast(`Submitting decision: ${decisionType}`, 'info');

  try {
    const response = await fetch(`/api/decide/${threadId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision: decisionType, notes: 'Operator reviewed and confirmed.' }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server error ${response.status}`);
    }

    const finalizedState = await response.json();
    activeState = finalizedState.state;
    renderDashboard(false);
    refreshHistoryTable();
    showToast(`Decision submitted: ${decisionType}`, 'success');
  } catch (err) {
    setUiState('hitl');
    showToast(`Error submitting decision: ${err.message}`, 'danger');
  }
}

function populateAllTabs() {
  renderAgentMapTab();
  renderMathLedgerTab();
  renderRegulatoryAuditTab();
}

function renderAgentMapTab() {
  agentTimeline.innerHTML = '';
  agentsReportGrid.innerHTML = '';

  const scores = activeState?.agent_scores || [];
  if (scores.length === 0) {
    agentsReportGrid.innerHTML = '<div class="text-center opacity-60">No agent data yet.</div>';
    return;
  }

  scores.slice(0, 7).forEach(score => {
    const node = document.createElement('div');
    node.className = 'agent-node';
    node.innerHTML = `
      <div class="agent-node-label">${score.agent_id.replace(/_/g, ' ')}</div>
      <div class="agent-node-score">${score.hazard_score.toFixed(1)}</div>
    `;
    agentTimeline.appendChild(node);
  });

  scores.forEach(score => {
    const card = document.createElement('div');
    card.className = 'agent-card';
    card.innerHTML = `
      <h4>${score.agent_id.replace(/_/g, ' ')}</h4>
      <p><strong>Score:</strong> ${score.hazard_score.toFixed(1)} · <strong>Confidence:</strong> ${(score.confidence * 100).toFixed(0)}% · <strong>Weight:</strong> ${score.domain_weight.toFixed(2)}</p>
      <p>${score.rationale || 'No rationale provided.'}</p>
    `;
    agentsReportGrid.appendChild(card);
  });
}

function renderMathLedgerTab() {
  mathLedgerBody.innerHTML = '';
  const scores = activeState?.agent_scores || [];
  if (scores.length === 0) {
    mathLedgerBody.innerHTML = '<tr><td colspan="6" class="text-center opacity-60">No data yet.</td></tr>';
    return;
  }

  scores.forEach(score => {
    const effWeight = score.domain_weight * score.confidence;
    const contribution = effWeight * score.hazard_score;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${score.agent_id}</strong></td>
      <td>${score.hazard_score.toFixed(2)}</td>
      <td>${score.confidence.toFixed(3)}</td>
      <td>${score.domain_weight.toFixed(3)}</td>
      <td>${effWeight.toFixed(3)}</td>
      <td>${contribution.toFixed(3)}</td>
    `;
    mathLedgerBody.appendChild(tr);
  });
}

function renderRegulatoryAuditTab() {
  regulatoryTableBody.innerHTML = '';
  const report = activeState?.regulatory_report;
  if (!report || !report.applicable_regulations?.length) {
    regulatoryTableBody.innerHTML = '<tr><td colspan="6" class="text-center opacity-60">No regulatory citations loaded.</td></tr>';
    return;
  }

  report.applicable_regulations.forEach(reg => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${reg.regulation_id}</strong></td>
      <td>${reg.title}</td>
      <td>${reg.section}</td>
      <td>${reg.source}</td>
      <td>${reg.compliance_status}</td>
      <td>${reg.legal_implications || 'N/A'}</td>
    `;
    regulatoryTableBody.appendChild(tr);
  });
}

async function loadHistoryLogs() {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const logs = await res.json();
    renderHistoryTable(logs);
  } catch (err) {
    historyTableBody.innerHTML = '<tr><td colspan="7" class="text-center opacity-60">Error loading history logs.</td></tr>';
  }
}

async function refreshHistoryTable() {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const logs = await res.json();
    renderHistoryTable(logs);
  } catch (_) {
    // ignore
  }
}

function renderHistoryTable(logs) {
  historyTableBody.innerHTML = '';
  if (!logs || logs.length === 0) {
    historyTableBody.innerHTML = '<tr><td colspan="7" class="text-center opacity-60">No logs found.</td></tr>';
    return;
  }

  logs.forEach(log => {
    const tr = document.createElement('tr');
    const dateStr = log.timestamp ? new Date(log.timestamp).toLocaleString() : 'N/A';
    const sev = (log.severity || 'LOW').toUpperCase();
    const dec = (log.hitl_decision || 'PENDING').toUpperCase();

    tr.innerHTML = `
      <td><strong>${log.id}</strong></td>
      <td>${log.incident_type || 'Unknown'}</td>
      <td>${log.location || 'N/A'}</td>
      <td>${sev}</td>
      <td>${(log.hazard_score || 0).toFixed(1)}</td>
      <td>${dec}</td>
      <td>${dateStr}</td>
    `;
    historyTableBody.appendChild(tr);
  });
}

function showToast(message, type = 'info', duration = 3000) {
  if (!toastContainer) return;
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}
