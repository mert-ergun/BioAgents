/**
 * Drug Discovery scenario runner — one-form launchers for the four agentic
 * small-molecule workflows exposed by /api/drug-discovery/scenarios.
 *
 * Renders:
 *   • a grid of scenario cards (Scenarios A / B / C / D)
 *   • a detail runner panel with inputs + options + per-gate threshold editor
 *   • a results panel with decision log, gate verdicts, dossier, and raw JSON
 */

const DD_SCENARIO_ICONS = {
    dd_scenario_disease_first: 'coronavirus',
    dd_scenario_target_first: 'target',
    dd_scenario_molecule_first: 'science',
    dd_scenario_optimization_loop: 'autorenew',
};

const DD_GATE_ORDER = [
    'structure_readiness',
    'docking_validation',
    'boltz_hit_screen',
    'consensus_structural_review',
    'early_admet',
    'off_target_tier1',
    'off_target_tier2',
    'retrosynthesis',
];

const DD_GATE_LABEL = {
    structure_readiness: 'Structure readiness',
    docking_validation: 'Docking validation',
    boltz_hit_screen: 'Boltz hit screen',
    consensus_structural_review: 'Consensus structural',
    early_admet: 'Early ADMET',
    off_target_tier1: 'Off-target Tier 1',
    off_target_tier2: 'Off-target Tier 2',
    retrosynthesis: 'Retrosynthesis gate',
};

const ddState = {
    scenarios: [],
    defaultThresholds: {},
    active: null, // active scenario object
    thresholdOverrides: {}, // per-gate overrides for active scenario
    isRunning: false,
    lastResult: null,
};

function ddSetStatus(text, kind) {
    const el = document.getElementById('dd-status');
    if (!el) return;
    const base = 'text-xs font-mono px-2 py-1 rounded border';
    const styles = {
        idle: 'bg-white/5 text-text-subtle border-white/10',
        running: 'bg-primary/20 text-primary border-primary/30',
        success: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
        error: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
    };
    el.textContent = text;
    el.className = `${base} ${styles[kind] || styles.idle}`;
}

async function initDrugDiscovery() {
    const grid = document.getElementById('dd-scenario-grid');
    if (!grid) return;
    if (grid.dataset.bound === '1') return;
    grid.dataset.bound = '1';

    document.getElementById('dd-runner-close')?.addEventListener('click', () => {
        document.getElementById('dd-runner-panel')?.classList.add('hidden');
        document.getElementById('dd-results-panel')?.classList.add('hidden');
        ddState.active = null;
    });
    document.getElementById('dd-run-btn')?.addEventListener('click', () => {
        runDrugDiscoveryScenario();
    });
    document.getElementById('dd-thresholds-reset')?.addEventListener('click', () => {
        ddState.thresholdOverrides = {};
        renderThresholdEditor();
    });
    document.getElementById('dd-open-in-builder')?.addEventListener('click', () => {
        if (!ddState.active) return;
        openScenarioInBuilder(ddState.active.id);
    });

    try {
        const res = await fetch('/api/drug-discovery/scenarios');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        ddState.scenarios = Array.isArray(data.scenarios) ? data.scenarios : [];
        ddState.defaultThresholds = data.default_thresholds || {};
        renderScenarioGrid();
    } catch (e) {
        console.error('Failed to load drug-discovery scenarios:', e);
        grid.innerHTML = `<div class="col-span-full text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3">
            Failed to load scenarios (${String(e)}). Is the BioAgents API running?
        </div>`;
    }
}

function renderScenarioGrid() {
    const grid = document.getElementById('dd-scenario-grid');
    if (!grid) return;
    grid.innerHTML = '';
    ddState.scenarios.forEach((sc) => {
        const icon = DD_SCENARIO_ICONS[sc.id] || 'medication';
        const card = document.createElement('button');
        card.type = 'button';
        card.className =
            'dd-scenario-card text-left rounded-xl border border-white/10 bg-surface-card/50 hover:border-primary/40 hover:bg-primary/5 p-4 flex flex-col gap-2 transition-all';
        card.innerHTML = `
            <div class="flex items-center gap-2">
                <span class="material-symbols-outlined text-primary text-[22px]">${icon}</span>
                <h3 class="text-sm font-bold text-white">${sc.name || sc.id}</h3>
            </div>
            <p class="text-[11px] text-text-subtle leading-relaxed">${sc.description || ''}</p>
            <div class="flex items-center gap-1.5 mt-auto pt-2 text-[10px] text-text-subtle">
                <span class="material-symbols-outlined text-[14px]">tune</span>
                <span>${(sc.form_fields || []).length} inputs · ${Object.keys(sc.options || {}).length} options</span>
            </div>
        `;
        card.addEventListener('click', () => openScenarioRunner(sc.id));
        grid.appendChild(card);
    });
}

async function openScenarioInBuilder(scenarioId) {
    const navBtn = document.getElementById('nav-workflow-builder');
    if (!navBtn) {
        console.warn('Workflow Builder nav button not found');
        return;
    }
    navBtn.click();
    // Wait for the builder to initialize and populate its template dropdown.
    const sel = await waitForElement('#wb-template-select', 3000);
    if (!sel) return;
    // Wait until the dropdown actually contains the scenario option. The
    // builder loads presets asynchronously on first activation.
    const deadline = Date.now() + 4000;
    const hasOption = () => Array.from(sel.options).some((o) => o.value === scenarioId);
    while (!hasOption() && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 100));
    }
    if (!hasOption()) {
        console.warn(`Scenario "${scenarioId}" not found in builder template list`);
        if (typeof showToast === 'function') {
            showToast('Scenario not yet available in builder', 'error');
        }
        return;
    }
    sel.value = scenarioId;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
}

function waitForElement(selector, timeoutMs) {
    return new Promise((resolve) => {
        const existing = document.querySelector(selector);
        if (existing) {
            resolve(existing);
            return;
        }
        const start = Date.now();
        const poll = () => {
            const el = document.querySelector(selector);
            if (el) return resolve(el);
            if (Date.now() - start > timeoutMs) return resolve(null);
            setTimeout(poll, 80);
        };
        poll();
    });
}

function openScenarioRunner(scenarioId) {
    const sc = ddState.scenarios.find((s) => s.id === scenarioId);
    if (!sc) return;
    ddState.active = sc;
    ddState.thresholdOverrides = {};
    ddState.lastResult = null;

    document.getElementById('dd-runner-panel')?.classList.remove('hidden');
    document.getElementById('dd-results-panel')?.classList.add('hidden');
    document.getElementById('dd-runner-title').textContent = sc.name || sc.id;
    document.getElementById('dd-runner-description').textContent = sc.description || '';
    document.getElementById('dd-run-hint').textContent = '';

    renderScenarioInputs();
    renderScenarioOptions();
    renderThresholdEditor();
    ddSetStatus('Idle', 'idle');
}

function renderScenarioInputs() {
    const wrap = document.getElementById('dd-runner-inputs');
    if (!wrap) return;
    wrap.innerHTML = '';
    const fields = ddState.active?.form_fields || [];
    fields.forEach((f) => {
        const row = document.createElement('div');
        row.className = 'w-full';
        const label = document.createElement('label');
        label.className = 'text-[10px] text-text-subtle uppercase tracking-wider block mb-1';
        label.textContent = f.required ? `${f.label} *` : f.label;
        const input = document.createElement('input');
        input.type = 'text';
        input.id = `dd-input-${f.key}`;
        input.dataset.ddKey = f.key;
        input.placeholder = f.example || '';
        input.className =
            'w-full rounded-lg bg-surface-dark border border-white/10 text-sm text-white px-3 py-2 focus:border-primary/50 focus:ring-1 focus:ring-primary/30';
        if (f.example && !input.value) input.value = f.example;
        row.append(label, input);
        wrap.appendChild(row);
    });
}

function renderScenarioOptions() {
    const wrap = document.getElementById('dd-runner-options');
    if (!wrap) return;
    wrap.innerHTML = '';
    const opts = ddState.active?.options || {};
    Object.entries(opts).forEach(([key, meta]) => {
        const row = document.createElement('div');
        row.className = 'w-full';
        const label = document.createElement('label');
        label.className = 'text-[10px] text-text-subtle uppercase tracking-wider block mb-1';
        const minmax = meta.min !== undefined ? ` (${meta.min}–${meta.max})` : '';
        label.textContent = `${key}${minmax}`;
        const input = document.createElement('input');
        input.id = `dd-option-${key}`;
        input.dataset.ddOption = key;
        if (meta.type === 'str') {
            input.type = 'text';
            input.value = meta.default ?? '';
        } else {
            input.type = 'number';
            if (meta.min !== undefined) input.min = String(meta.min);
            if (meta.max !== undefined) input.max = String(meta.max);
            input.value = String(meta.default ?? 1);
        }
        input.className =
            'w-full max-w-xs rounded-lg bg-surface-dark border border-white/10 text-sm text-white px-3 py-2 focus:border-primary/50 focus:ring-1 focus:ring-primary/30';
        row.append(label, input);
        wrap.appendChild(row);
    });
}

function renderThresholdEditor() {
    const wrap = document.getElementById('dd-threshold-editor');
    if (!wrap) return;
    wrap.innerHTML = '';
    const defaults = ddState.defaultThresholds || {};
    DD_GATE_ORDER.forEach((gateKey) => {
        const base = defaults[gateKey];
        if (!base) return;
        const override = ddState.thresholdOverrides[gateKey] || {};
        const block = document.createElement('div');
        block.className = 'rounded-lg border border-white/10 bg-black/20 p-3';
        const head = document.createElement('div');
        head.className = 'text-[11px] font-bold text-white mb-2 flex items-center gap-2';
        head.innerHTML = `<span class="material-symbols-outlined text-[14px] text-primary">rule</span>${DD_GATE_LABEL[gateKey] || gateKey}`;
        block.appendChild(head);
        Object.entries(base).forEach(([k, v]) => {
            const row = document.createElement('div');
            row.className = 'flex items-center gap-2 mb-1 last:mb-0';
            const label = document.createElement('label');
            label.className = 'text-[10px] text-text-subtle font-mono flex-1';
            label.textContent = k;
            const input = document.createElement('input');
            input.dataset.ddGate = gateKey;
            input.dataset.ddThreshKey = k;
            const current = override[k] !== undefined ? override[k] : v;
            if (typeof v === 'boolean') {
                input.type = 'checkbox';
                input.checked = Boolean(current);
                input.className = 'rounded border-white/20 bg-surface-dark text-primary focus:ring-primary';
            } else if (Array.isArray(v)) {
                input.type = 'text';
                input.value = (current || []).join(', ');
                input.className =
                    'rounded-md bg-surface-dark border border-white/10 text-[11px] font-mono text-cyan-100 px-2 py-1 flex-[2]';
            } else if (typeof v === 'number') {
                input.type = 'number';
                input.step = Number.isInteger(v) ? '1' : '0.01';
                input.value = String(current);
                input.className =
                    'rounded-md bg-surface-dark border border-white/10 text-[11px] font-mono text-white px-2 py-1 w-24';
            } else {
                input.type = 'text';
                input.value = String(current ?? '');
                input.className =
                    'rounded-md bg-surface-dark border border-white/10 text-[11px] font-mono text-white px-2 py-1 flex-[2]';
            }
            input.addEventListener('change', () => captureThresholdChange(gateKey, k, v, input));
            row.append(label, input);
            block.appendChild(row);
        });
        wrap.appendChild(block);
    });
}

function captureThresholdChange(gateKey, thresholdKey, defaultValue, inputEl) {
    const target = (ddState.thresholdOverrides[gateKey] ??= {});
    let value;
    if (typeof defaultValue === 'boolean') {
        value = inputEl.checked;
    } else if (Array.isArray(defaultValue)) {
        value = inputEl.value.split(',').map((s) => s.trim()).filter(Boolean);
    } else if (typeof defaultValue === 'number') {
        const parsed = Number(inputEl.value);
        value = Number.isNaN(parsed) ? defaultValue : parsed;
    } else {
        value = inputEl.value;
    }
    target[thresholdKey] = value;
}

function collectInputs() {
    const out = {};
    document.querySelectorAll('#dd-runner-inputs [data-dd-key]').forEach((el) => {
        const k = el.dataset.ddKey;
        const v = (el.value || '').trim();
        if (v) out[k] = v;
    });
    return out;
}

function collectOptions() {
    const out = {};
    document.querySelectorAll('#dd-runner-options [data-dd-option]').forEach((el) => {
        const k = el.dataset.ddOption;
        if (el.type === 'number') {
            const n = Number(el.value);
            if (!Number.isNaN(n)) out[k] = n;
        } else {
            const v = (el.value || '').trim();
            if (v) out[k] = v;
        }
    });
    return out;
}

async function runDrugDiscoveryScenario() {
    if (!ddState.active || ddState.isRunning) return;
    const inputs = collectInputs();
    const missing = (ddState.active.form_fields || [])
        .filter((f) => f.required && !inputs[f.key])
        .map((f) => f.key);
    if (missing.length > 0) {
        document.getElementById('dd-run-hint').textContent =
            `Missing required inputs: ${missing.join(', ')}`;
        return;
    }

    const body = {
        scenario_id: ddState.active.id,
        inputs,
        options: collectOptions(),
        thresholds: ddState.thresholdOverrides,
    };

    const runBtn = document.getElementById('dd-run-btn');
    const hint = document.getElementById('dd-run-hint');
    runBtn.disabled = true;
    ddState.isRunning = true;
    ddSetStatus('Running…', 'running');
    hint.textContent = 'Executing scenario graph (this uses stubbed heavy-ML nodes — see README for real-tool hookup).';

    try {
        const res = await fetch('/api/drug-discovery/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        let data;
        try {
            data = await res.json();
        } catch {
            ddSetStatus(`Invalid response (${res.status})`, 'error');
            return;
        }
        if (!res.ok) {
            const detail = data.detail;
            const msg =
                typeof detail === 'string'
                    ? detail
                    : Array.isArray(detail)
                      ? detail.map((d) => d.msg || d).join('; ')
                      : res.statusText;
            ddSetStatus(msg || `HTTP ${res.status}`, 'error');
            hint.textContent = msg || '';
            return;
        }
        ddState.lastResult = data;
        if (data.success) {
            ddSetStatus('Completed', 'success');
            hint.textContent = '';
            renderResults(data);
        } else {
            ddSetStatus(data.error || 'Failed', 'error');
            hint.textContent = data.error || '';
        }
    } catch (e) {
        console.error(e);
        ddSetStatus('Network error', 'error');
        hint.textContent = String(e);
    } finally {
        runBtn.disabled = false;
        ddState.isRunning = false;
    }
}

function renderResults(data) {
    document.getElementById('dd-results-panel')?.classList.remove('hidden');
    renderDecisionLog(data.decision_log || []);
    renderVerdicts(data.verdicts || {});
    renderDossier(data.target_dossier || null);
    const nodesEl = document.getElementById('dd-nodes-json');
    if (nodesEl) {
        try {
            nodesEl.textContent = JSON.stringify(data.node_outputs || {}, null, 2);
        } catch {
            nodesEl.textContent = String(data.node_outputs);
        }
    }
}

function renderDecisionLog(entries) {
    const wrap = document.getElementById('dd-decision-log');
    const summary = document.getElementById('dd-decision-summary');
    if (!wrap) return;
    wrap.innerHTML = '';
    if (!Array.isArray(entries) || entries.length === 0) {
        wrap.innerHTML = `<div class="text-[11px] text-text-subtle italic">Decision log is empty — no candidates produced verdicts.</div>`;
        if (summary) summary.textContent = '';
        return;
    }
    const counts = {};
    entries.forEach((e) => {
        counts[e.tier] = (counts[e.tier] || 0) + 1;
    });
    if (summary) {
        summary.textContent = Object.entries(counts)
            .map(([tier, n]) => `${n} ${tier}`)
            .join(' · ');
    }
    entries.forEach((entry) => {
        const row = document.createElement('div');
        row.className = 'dd-decision-row rounded-lg border border-white/10 bg-black/20 p-3';
        const header = document.createElement('div');
        header.className = 'flex items-center gap-2 mb-2';
        const tierBadge = document.createElement('span');
        tierBadge.className = `dd-tier-badge dd-tier-${entry.tier || 'reject'}`;
        tierBadge.textContent = entry.tier || 'reject';
        const cid = document.createElement('span');
        cid.className = 'text-[11px] font-mono text-white';
        cid.textContent = entry.candidate_id || '—';
        header.append(tierBadge, cid);
        row.appendChild(header);
        if (entry.summary) {
            const p = document.createElement('p');
            p.className = 'text-[11px] text-text-subtle mb-2 leading-relaxed';
            p.textContent = entry.summary;
            row.appendChild(p);
        }
        const pillRow = document.createElement('div');
        pillRow.className = 'flex flex-wrap gap-1.5';
        (entry.gates || []).forEach((g) => {
            pillRow.appendChild(makeVerdictPill(g));
        });
        row.appendChild(pillRow);
        wrap.appendChild(row);
    });
}

function makeVerdictPill(gateVerdict) {
    const verdict = gateVerdict.verdict || 'borderline';
    const pill = document.createElement('span');
    pill.className = `dd-gate-pill dd-verdict-${verdict}`;
    pill.innerHTML = `
        <span class="dd-pill-label">${DD_GATE_LABEL[gateVerdict.gate] || gateVerdict.gate}</span>
        <span class="dd-pill-verdict">${verdict}</span>
    `;
    const tooltipParts = [];
    if (gateVerdict.reason_code) tooltipParts.push(`code: ${gateVerdict.reason_code}`);
    if (gateVerdict.reason_text) tooltipParts.push(gateVerdict.reason_text);
    if (tooltipParts.length > 0) pill.title = tooltipParts.join(' — ');
    return pill;
}

function renderVerdicts(verdictsByGate) {
    const grid = document.getElementById('dd-verdicts-grid');
    if (!grid) return;
    grid.innerHTML = '';
    DD_GATE_ORDER.forEach((gateKey) => {
        const list = verdictsByGate[gateKey] || [];
        const block = document.createElement('div');
        block.className = 'rounded-lg border border-white/10 bg-black/20 p-3';
        const title = document.createElement('div');
        title.className = 'text-[11px] font-bold text-white flex items-center gap-2 mb-2';
        title.innerHTML = `<span class="material-symbols-outlined text-[14px] text-primary">rule</span>${DD_GATE_LABEL[gateKey] || gateKey} <span class="text-[10px] text-text-subtle">(${list.length})</span>`;
        block.appendChild(title);
        if (list.length === 0) {
            const p = document.createElement('p');
            p.className = 'text-[10px] text-text-subtle italic';
            p.textContent = 'No verdicts emitted for this gate.';
            block.appendChild(p);
        } else {
            list.slice(0, 10).forEach((v) => {
                const row = document.createElement('div');
                row.className = 'flex items-center justify-between gap-2 py-0.5';
                const cid = document.createElement('span');
                cid.className = 'text-[10px] font-mono text-text-subtle truncate';
                cid.textContent = v.candidate_id || '—';
                const pill = document.createElement('span');
                pill.className = `dd-gate-pill dd-verdict-${v.verdict || 'borderline'} dd-pill-compact`;
                pill.textContent = v.verdict || 'borderline';
                row.append(cid, pill);
                block.appendChild(row);
            });
            if (list.length > 10) {
                const more = document.createElement('div');
                more.className = 'text-[10px] text-text-subtle italic mt-1';
                more.textContent = `…and ${list.length - 10} more`;
                block.appendChild(more);
            }
        }
        grid.appendChild(block);
    });
}

function renderDossier(dossier) {
    const pre = document.getElementById('dd-dossier-json');
    const meta = document.getElementById('dd-dossier-meta');
    if (!pre) return;
    if (!dossier) {
        pre.textContent = '// Scenario did not produce a target dossier';
        if (meta) meta.textContent = '';
        return;
    }
    try {
        pre.textContent = JSON.stringify(dossier, null, 2);
    } catch {
        pre.textContent = String(dossier);
    }
    if (meta) {
        const parts = [];
        if (dossier.uniprot_id) parts.push(dossier.uniprot_id);
        if (dossier.gene_symbol) parts.push(dossier.gene_symbol);
        if (dossier.organism) parts.push(dossier.organism);
        meta.textContent = parts.join(' · ');
    }
}

// Expose init fn so index.html nav wiring can call it.
window.initDrugDiscovery = initDrugDiscovery;
