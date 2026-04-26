/**
 * BioAgents Experiments UI
 * Handles the experiment runner view: use case selection, config, run, results.
 */

// ─────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────
const ExperimentsState = {
    useCases: [],
    configs: [],
    runs: [],
    selectedUseCaseIds: new Set(),
    selectedConfigId: null,
    activeRunId: null,
    activeRunData: null,
    pollingTimer: null,
    /** Set when POST /run succeeds; cleared when the run appears in list_runs (persisted on completion). */
    pendingRun: null,
    filterCategory: null,
    filterTags: [],
};

// ─────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────
async function initExperiments() {
    await Promise.all([loadUseCases(), loadConfigs(), loadRuns()]);
    renderUseCaseList();
    renderConfigSelector();
    renderRunsTable();
    attachExperimentsListeners();
}

// ─────────────────────────────────────────────────────────
// Data fetching
// ─────────────────────────────────────────────────────────
async function loadUseCases() {
    try {
        const res = await fetch('/api/experiments/use-cases');
        const data = await res.json();
        ExperimentsState.useCases = data.use_cases || [];
    } catch (e) {
        console.error('Failed to load use cases:', e);
    }
}

async function loadConfigs() {
    try {
        const res = await fetch('/api/experiments/configs');
        const data = await res.json();
        ExperimentsState.configs = data.configs || [];
        if (ExperimentsState.configs.length > 0) {
            ExperimentsState.selectedConfigId = ExperimentsState.configs[0].id;
        }
    } catch (e) {
        console.error('Failed to load configs:', e);
    }
}

async function loadRuns() {
    try {
        const res = await fetch('/api/experiments/runs?limit=30');
        const data = await res.json();
        ExperimentsState.runs = data.runs || [];
    } catch (e) {
        console.error('Failed to load runs:', e);
    }
}

async function loadRunDetails(runId) {
    try {
        const res = await fetch(`/api/experiments/runs/${runId}`);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.error('Failed to load run details:', e);
        return null;
    }
}

// ─────────────────────────────────────────────────────────
// Rendering helpers
// ─────────────────────────────────────────────────────────
function getCategories() {
    const cats = new Set(ExperimentsState.useCases.map(uc => uc.category).filter(Boolean));
    return Array.from(cats);
}

function scoreColor(score) {
    if (score === null || score === undefined) return 'text-slate-400';
    if (score >= 4.0) return 'text-green-400';
    if (score >= 3.0) return 'text-yellow-400';
    if (score >= 2.0) return 'text-orange-400';
    return 'text-red-400';
}

const AGENT_COLORS = {
    supervisor:      { bg: 'bg-purple-500/20',  text: 'text-purple-400',  border: 'border-purple-500/30' },
    analysis:        { bg: 'bg-cyan-500/20',     text: 'text-cyan-400',    border: 'border-cyan-500/30' },
    analysis_tools:  { bg: 'bg-slate-500/20',    text: 'text-slate-300',   border: 'border-slate-500/30' },
    research:        { bg: 'bg-blue-500/20',      text: 'text-blue-400',    border: 'border-blue-500/30' },
    report:          { bg: 'bg-yellow-500/20',    text: 'text-yellow-400',  border: 'border-yellow-500/30' },
    summary:         { bg: 'bg-yellow-500/20',    text: 'text-yellow-400',  border: 'border-yellow-500/30' },
    protein_design:  { bg: 'bg-teal-500/20',      text: 'text-teal-400',    border: 'border-teal-500/30' },
    critic:          { bg: 'bg-red-500/20',        text: 'text-red-400',     border: 'border-red-500/30' },
    coder:           { bg: 'bg-orange-500/20',    text: 'text-orange-400',  border: 'border-orange-500/30' },
    tool_builder:    { bg: 'bg-green-500/20',     text: 'text-green-400',   border: 'border-green-500/30' },
    ml:              { bg: 'bg-indigo-500/20',    text: 'text-indigo-400',  border: 'border-indigo-500/30' },
    dl:              { bg: 'bg-violet-500/20',    text: 'text-violet-400',  border: 'border-violet-500/30' },
    __end__:         { bg: 'bg-green-500/20',     text: 'text-green-400',   border: 'border-green-500/30' },
};

function agentColor(agent) {
    return AGENT_COLORS[agent] || { bg: 'bg-white/10', text: 'text-slate-300', border: 'border-white/10' };
}

function scoreBar(score, max = 5) {
    const pct = score != null ? Math.round((score / max) * 100) : 0;
    const color = score >= 4 ? 'bg-green-500' : score >= 3 ? 'bg-yellow-500' : score >= 2 ? 'bg-orange-500' : 'bg-red-500';
    return `<div class="w-full bg-white/10 rounded-full h-1.5 mt-1">
        <div class="${color} h-1.5 rounded-full transition-all" style="width:${pct}%"></div>
    </div>`;
}

function failureBadge(mode) {
    const map = {
        completed: ['bg-green-500/20 text-green-400 border-green-500/30', 'check_circle'],
        timeout:   ['bg-yellow-500/20 text-yellow-400 border-yellow-500/30', 'timer_off'],
        max_steps: ['bg-orange-500/20 text-orange-400 border-orange-500/30', 'repeat'],
        exception: ['bg-red-500/20 text-red-400 border-red-500/30', 'error'],
        incomplete:['bg-slate-500/20 text-slate-400 border-slate-500/30', 'help'],
    };
    const [cls, icon] = map[mode] || map.incomplete;
    return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${cls}">
        <span class="material-symbols-outlined text-[12px]">${icon}</span>${mode}
    </span>`;
}

function renderCategoryFilters() {
    const container = document.getElementById('exp-category-filters');
    if (!container) return;
    const cats = getCategories();
    container.innerHTML = `
        <button class="category-filter px-2.5 py-1 rounded-full text-[11px] font-medium border transition-all
            ${!ExperimentsState.filterCategory ? 'bg-primary text-background-dark border-primary' : 'border-white/10 text-text-subtle hover:border-primary/50 hover:text-white'}"
            data-cat="">All</button>
        ${cats.map(cat => `
        <button class="category-filter px-2.5 py-1 rounded-full text-[11px] font-medium border transition-all
            ${ExperimentsState.filterCategory === cat ? 'bg-primary text-background-dark border-primary' : 'border-white/10 text-text-subtle hover:border-primary/50 hover:text-white'}"
            data-cat="${cat}">${cat.replace(/_/g, ' ')}</button>`).join('')}
    `;
    container.querySelectorAll('.category-filter').forEach(btn => {
        btn.addEventListener('click', () => {
            ExperimentsState.filterCategory = btn.dataset.cat || null;
            renderCategoryFilters();
            renderUseCaseList();
        });
    });
}

function renderUseCaseList() {
    renderCategoryFilters();
    const container = document.getElementById('exp-use-case-list');
    if (!container) return;

    let useCases = ExperimentsState.useCases;
    if (ExperimentsState.filterCategory) {
        useCases = useCases.filter(uc => uc.category === ExperimentsState.filterCategory);
    }

    if (useCases.length === 0) {
        container.innerHTML = `<div class="text-center py-10 text-text-subtle text-sm">
            <span class="material-symbols-outlined text-3xl block mb-2">science_off</span>
            No use cases found. Add YAML files to the <code>use_cases/</code> directory.
        </div>`;
        return;
    }

    container.innerHTML = useCases.map(uc => {
        const isSelected = ExperimentsState.selectedUseCaseIds.has(uc.id);
        return `
        <div class="exp-use-case-card group flex items-start gap-3 p-3 rounded-lg border transition-all cursor-pointer
            ${isSelected ? 'bg-primary/10 border-primary/40' : 'border-white/5 hover:border-white/15 bg-surface-card/30 hover:bg-surface-card/60'}"
            data-id="${uc.id}">
            <div class="mt-0.5 flex-shrink-0 w-4 h-4 rounded border-2 flex items-center justify-center transition-all
                ${isSelected ? 'bg-primary border-primary' : 'border-white/20 group-hover:border-primary/60'}">
                ${isSelected ? '<span class="material-symbols-outlined text-[12px] text-background-dark">check</span>' : ''}
            </div>
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap mb-0.5">
                    <span class="text-sm font-semibold text-white truncate">${uc.name}</span>
                    ${uc.category ? `<span class="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-text-subtle">${uc.category.replace(/_/g,' ')}</span>` : ''}
                </div>
                <p class="text-xs text-text-subtle line-clamp-2">${uc.description || uc.prompt.substring(0, 120)}</p>
                ${uc.tags && uc.tags.length ? `<div class="flex gap-1 mt-1.5 flex-wrap">${uc.tags.map(t=>`<span class="text-[10px] px-1.5 py-0.5 rounded-full border border-white/10 text-slate-500">${t}</span>`).join('')}</div>` : ''}
            </div>
        </div>`;
    }).join('');

    container.querySelectorAll('.exp-use-case-card').forEach(card => {
        card.addEventListener('click', () => {
            const id = card.dataset.id;
            if (ExperimentsState.selectedUseCaseIds.has(id)) {
                ExperimentsState.selectedUseCaseIds.delete(id);
            } else {
                ExperimentsState.selectedUseCaseIds.add(id);
            }
            updateSelectionCount();
            renderUseCaseList();
        });
    });

    updateSelectionCount();
}

function updateSelectionCount() {
    const el = document.getElementById('exp-selection-count');
    if (!el) return;
    const n = ExperimentsState.selectedUseCaseIds.size;
    el.textContent = n === 0 ? 'Select use cases to run' : `${n} use case${n > 1 ? 's' : ''} selected`;
}

function renderConfigSelector() {
    const container = document.getElementById('exp-config-selector');
    if (!container) return;
    container.innerHTML = ExperimentsState.configs.map(c => `
        <button class="config-btn flex-1 px-3 py-2 rounded-lg text-xs font-medium border transition-all text-left
            ${ExperimentsState.selectedConfigId === c.id ? 'bg-primary/20 border-primary/50 text-white' : 'border-white/10 text-text-subtle hover:border-white/25 hover:text-white'}"
            data-id="${c.id}" title="${c.description || ''}">
            <div class="font-semibold">${c.name}</div>
            <div class="text-[10px] opacity-70 mt-0.5">${c.description || `steps: ${c.max_steps}, timeout: ${c.timeout}s`}</div>
        </button>`).join('');

    container.querySelectorAll('.config-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            ExperimentsState.selectedConfigId = btn.dataset.id;
            renderConfigSelector();
        });
    });
}

function updateRunsLiveBanner() {
    const wrap = document.getElementById('exp-runs-live-status');
    const textEl = document.getElementById('exp-runs-live-status-text');
    if (!wrap || !textEl) return;
    const p = ExperimentsState.pendingRun;
    if (!p) {
        wrap.classList.add('hidden');
        return;
    }
    const short = (p.runId || '').substring(0, 8);
    wrap.classList.remove('hidden');
    textEl.textContent =
        `Run ${short}… is executing — results stay hidden until the run finishes and is saved. Watch this row and the server terminal for activity.`;
}

function renderRunsTable() {
    const container = document.getElementById('exp-runs-table-body');
    if (!container) return;

    if (ExperimentsState.pendingRun) {
        const pid = ExperimentsState.pendingRun.runId;
        if (ExperimentsState.runs.some(r => r.run_id === pid)) {
            ExperimentsState.pendingRun = null;
        }
    }
    updateRunsLiveBanner();

    const pendingRow = () => {
        const p = ExperimentsState.pendingRun;
        if (!p) return '';
        const shortId = (p.runId || '').substring(0, 8);
        const dt = p.startedAt ? new Date(p.startedAt).toLocaleString() : '—';
        return `
        <tr class="exp-pending-run-row border-b border-primary/30 bg-primary/10">
            <td class="px-4 py-3 font-mono text-[11px] text-primary">
                <span class="inline-flex items-center gap-2">
                    <span class="material-symbols-outlined animate-spin text-base">progress_activity</span>
                    ${shortId}…
                </span>
            </td>
            <td class="px-4 py-3 text-sm font-medium text-white">${p.configName || '—'}</td>
            <td class="px-4 py-3 text-xs text-text-subtle">${dt}</td>
            <td class="px-4 py-3 text-xs text-amber-300/90" colspan="4">Running… (saved when finished)</td>
        </tr>`;
    };

    const completedRows = ExperimentsState.runs.map(run => {
        const score = run.mean_score != null ? run.mean_score.toFixed(2) : '—';
        const rate = run.success_rate != null ? `${Math.round(run.success_rate * 100)}%` : '—';
        const time = run.total_execution_time != null ? `${run.total_execution_time.toFixed(0)}s` : '—';
        const tokens = run.total_tokens != null ? run.total_tokens.toLocaleString() : '—';
        const dt = run.started_at ? new Date(run.started_at).toLocaleString() : '—';
        const shortId = (run.run_id || '').substring(0, 8);
        return `
        <tr class="runs-table-row border-b border-white/5 hover:bg-white/3 cursor-pointer transition-all" data-run-id="${run.run_id}">
            <td class="px-4 py-3 font-mono text-[11px] text-text-subtle">${shortId}…</td>
            <td class="px-4 py-3 text-sm font-medium text-white">${run.config_name || '—'}</td>
            <td class="px-4 py-3 text-xs text-text-subtle">${dt}</td>
            <td class="px-4 py-3">
                <span class="font-mono font-bold ${scoreColor(run.mean_score)}">${score}</span>
                ${run.mean_score != null ? scoreBar(run.mean_score) : ''}
            </td>
            <td class="px-4 py-3 text-sm text-text-subtle">${rate}</td>
            <td class="px-4 py-3 text-sm text-text-subtle">${time}</td>
            <td class="px-4 py-3 text-xs text-text-subtle">${tokens}</td>
        </tr>`;
    }).join('');

    if (ExperimentsState.runs.length === 0 && !ExperimentsState.pendingRun) {
        container.innerHTML = `<tr><td colspan="7" class="text-center py-10 text-text-subtle text-sm">
            <span class="material-symbols-outlined block text-3xl mb-2 text-border-dark">history_toggle_off</span>
            No experiment runs yet. Run an experiment to see results here.
        </td></tr>`;
        return;
    }

    container.innerHTML = pendingRow() + completedRows;

    container.querySelectorAll('.runs-table-row').forEach(row => {
        row.addEventListener('click', () => openRunDetails(row.dataset.runId));
    });
}

// ─────────────────────────────────────────────────────────
// Run details panel
// ─────────────────────────────────────────────────────────
async function openRunDetails(runId) {
    ExperimentsState.activeRunId = runId;
    showDetailPanel(null); // show loading
    const data = await loadRunDetails(runId);
    ExperimentsState.activeRunData = data;
    showDetailPanel(data);
}

function showDetailPanel(data) {
    const panel = document.getElementById('exp-detail-panel');
    const overlay = document.getElementById('exp-detail-overlay');
    if (!panel || !overlay) return;

    panel.classList.remove('hidden');
    overlay.classList.remove('hidden');

    const content = document.getElementById('exp-detail-content');
    if (!data) {
        content.innerHTML = `<div class="flex items-center justify-center h-64 text-text-subtle">
            <span class="material-symbols-outlined animate-spin text-3xl text-primary mr-3">progress_activity</span>Loading…</div>`;
        return;
    }

    const agg = data.aggregates || {};
    const results = data.results || [];
    const cfg = data.experiment_config || {};

    content.innerHTML = `
        <!-- Header -->
        <div class="mb-6">
            <div class="flex items-center gap-3 mb-1">
                <h2 class="text-lg font-bold text-white">Run: <span class="font-mono text-primary">${(data.run_id||'').substring(0,12)}…</span></h2>
            </div>
            <div class="flex flex-wrap gap-4 text-xs text-text-subtle">
                <span><span class="text-white font-medium">Config:</span> ${cfg.name || '—'}</span>
                <span><span class="text-white font-medium">Started:</span> ${data.started_at ? new Date(data.started_at).toLocaleString() : '—'}</span>
                <span><span class="text-white font-medium">Duration:</span> ${agg.total_execution_time != null ? agg.total_execution_time.toFixed(1)+'s' : '—'}</span>
            </div>
        </div>

        <!-- Aggregates -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            ${metricCard('Score', agg.mean_score != null ? agg.mean_score.toFixed(2)+'/5' : '—', 'star', scoreColor(agg.mean_score))}
            ${metricCard('Success Rate', agg.success_rate != null ? Math.round(agg.success_rate*100)+'%' : '—', 'check_circle', 'text-green-400')}
            ${metricCard('Cases', agg.total_use_cases || results.length, 'science', 'text-primary')}
            ${metricCard('Tokens', agg.total_tokens != null ? agg.total_tokens.toLocaleString() : '—', 'token', 'text-purple-400')}
        </div>

        <!-- Failure modes -->
        ${renderFailureModeBar(agg.failure_mode_counts || {})}

        <!-- Results table -->
        <h3 class="text-sm font-bold text-white mb-3 mt-6">Per-Case Results</h3>
        <div class="flex flex-col gap-3">
            ${results.map(r => renderResultCard(r)).join('')}
        </div>
    `;

    // Attach expand listeners
    content.querySelectorAll('.result-card-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = document.getElementById(btn.dataset.target);
            if (!target) return;
            const hidden = target.classList.toggle('hidden');
            btn.querySelector('.expand-icon').textContent = hidden ? 'expand_more' : 'expand_less';
        });
    });
}

function metricCard(label, value, icon, colorCls) {
    return `<div class="bg-surface-card/50 border border-white/5 rounded-lg p-3">
        <div class="flex items-center gap-2 mb-1">
            <span class="material-symbols-outlined text-[16px] ${colorCls}">${icon}</span>
            <span class="text-[10px] text-text-subtle uppercase tracking-wider">${label}</span>
        </div>
        <div class="text-lg font-bold text-white">${value}</div>
    </div>`;
}

function renderFailureModeBar(counts) {
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (total === 0) return '';
    const colors = { completed: '#22c55e', timeout: '#eab308', max_steps: '#f97316', exception: '#ef4444', incomplete: '#64748b' };
    const segments = Object.entries(counts).map(([mode, n]) => {
        const pct = (n / total * 100).toFixed(1);
        return `<div class="h-full rounded" style="width:${pct}%;background:${colors[mode]||'#64748b'}" title="${mode}: ${n}"></div>`;
    }).join('');
    const labels = Object.entries(counts).map(([mode, n]) =>
        `<span class="flex items-center gap-1 text-[10px] text-text-subtle">
            <span class="w-2 h-2 rounded-sm inline-block" style="background:${colors[mode]||'#64748b'}"></span>${mode}: ${n}
        </span>`).join('');
    return `<div>
        <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-1">Failure Modes</p>
        <div class="flex h-2 rounded-full overflow-hidden gap-0.5 mb-2">${segments}</div>
        <div class="flex flex-wrap gap-3">${labels}</div>
    </div>`;
}

function renderResultCard(result) {
    const cardId = `result-detail-${result.use_case_id}-${Math.random().toString(36).slice(2,6)}`;
    const tools = (result.tool_calls || []).map(t => t.tool_name);
    const uniqueTools = [...new Set(tools)];
    const agents = [...new Set(result.agent_flow || [])].filter(a => !a.startsWith('__'));
    const scoreStr = result.judge_score != null ? result.judge_score.toFixed(2) : '—';
    const tu = result.token_usage;

    return `
    <div class="border border-white/5 rounded-xl overflow-hidden bg-surface-card/30">
        <div class="flex items-center gap-3 px-4 py-3 cursor-pointer result-card-toggle" data-target="${cardId}">
            <div class="w-2 h-2 rounded-full flex-shrink-0 ${result.workflow_completed ? 'bg-green-500' : 'bg-red-500'}"></div>
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                    <span class="text-sm font-semibold text-white truncate">${result.use_case_name}</span>
                    ${failureBadge(result.failure_mode)}
                </div>
                <div class="flex items-center gap-3 mt-0.5 text-[11px] text-text-subtle flex-wrap">
                    <span>${result.total_steps} steps</span>
                    <span>${result.execution_time != null ? result.execution_time.toFixed(1)+'s' : '—'}</span>
                    ${tu ? `<span>${tu.total_tokens != null ? tu.total_tokens.toLocaleString() : '—'} tokens</span>` : '<span class="text-text-subtle/50">no token data</span>'}
                </div>
            </div>
            <div class="text-right flex-shrink-0">
                <div class="text-lg font-bold font-mono ${scoreColor(result.judge_score)}">${scoreStr}</div>
                <div class="text-[10px] text-text-subtle">/ 5.0</div>
                ${result.judge_score != null ? scoreBar(result.judge_score) : ''}
            </div>
            <span class="material-symbols-outlined text-slate-500 text-[20px] flex-shrink-0 expand-icon">expand_more</span>
        </div>

        <div id="${cardId}" class="hidden border-t border-white/5 px-4 pb-4 pt-3 space-y-4">
            <!-- Prompt -->
            <div>
                <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-1">Prompt</p>
                <p class="text-xs text-white/80 bg-black/20 rounded p-2 font-mono whitespace-pre-wrap">${escHtml(result.prompt || '')}</p>
            </div>

            <!-- Agents & Tools row -->
            <div class="grid grid-cols-2 gap-3">
                <div>
                    <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-1">Agents Used</p>
                    <div class="flex flex-wrap gap-1">
                        ${agents.length ? agents.map(a => {
                            const c = agentColor(a);
                            return `<span class="text-[10px] px-2 py-0.5 rounded-full ${c.bg} border ${c.border} ${c.text}">${a}</span>`;
                        }).join('') : '<span class="text-xs text-text-subtle">—</span>'}
                    </div>
                </div>
                <div>
                    <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-1">Tools Called (${tools.length})</p>
                    <div class="flex flex-wrap gap-1">
                        ${uniqueTools.length ? uniqueTools.map(t => `<span class="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-text-subtle">${t}</span>`).join('') : '<span class="text-xs text-text-subtle">—</span>'}
                    </div>
                </div>
            </div>

            <!-- Agent Call Flow -->
            ${renderAgentCallFlow(result.agent_steps)}

            <!-- Tool Call Details -->
            ${renderToolCallDetails(result.tool_calls)}

            <!-- Token Usage -->
            ${renderTokenUsage(result.token_usage)}

            <!-- Judge reasoning -->
            ${result.judge_reasoning ? `
            <div>
                <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-1">Judge Reasoning</p>
                <p class="text-xs text-white/80 bg-black/20 rounded p-2 italic leading-relaxed">${escHtml(result.judge_reasoning)}</p>
            </div>` : ''}

            <!-- Judge breakdown -->
            ${renderJudgeBreakdown(result.judge_breakdown)}

            <!-- Full Output -->
            ${result.raw_output ? `
            <div>
                <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-1">Full Output</p>
                <pre class="text-[11px] text-white/70 bg-black/30 rounded p-2 overflow-auto max-h-64 whitespace-pre-wrap font-mono">${escHtml(result.raw_output)}</pre>
            </div>` : ''}

            <!-- Error -->
            ${result.error_message ? `
            <div class="bg-red-500/10 border border-red-500/30 rounded p-2">
                <p class="text-[10px] font-bold text-red-400 mb-1">Error</p>
                <p class="text-xs text-red-300 font-mono">${escHtml(result.error_message)}</p>
            </div>` : ''}
        </div>
    </div>`;
}

function renderAgentCallFlow(steps) {
    if (!steps || steps.length === 0) return '';

    const rows = steps.map((step, idx) => {
        const c = agentColor(step.agent);
        const isLast = idx === steps.length - 1;
        const msgs = (step.messages || []).filter(m => m.content && m.content.trim());

        const msgHtml = msgs.map(m => {
            const isToolMsg = m.type === 'ToolMessage';
            const isTool = !!(m.tool_calls && m.tool_calls.length);
            let typeLabel = m.type || 'Message';
            if (isTool) typeLabel = 'Tool Request';
            if (isToolMsg) typeLabel = 'Tool Result';
            const contentCls = isToolMsg
                ? 'text-amber-300/80 font-mono'
                : isTool
                ? 'text-primary/80'
                : 'text-white/75';
            const preview = m.content.length > 400 ? m.content.substring(0, 400) + '…' : m.content;
            return `<div class="bg-black/20 rounded p-2 mt-1.5 text-[11px]">
                <span class="text-[9px] uppercase tracking-wider text-text-subtle mr-1.5">${typeLabel}</span>
                <span class="${contentCls} whitespace-pre-wrap">${escHtml(preview)}</span>
            </div>`;
        }).join('');

        const routeHtml = step.routing_decision
            ? `<span class="ml-2 text-[10px] text-text-subtle">→ <span class="text-white/60">${escHtml(step.routing_decision)}</span></span>`
            : '';

        return `<div class="flex gap-3">
            <div class="flex flex-col items-center flex-shrink-0">
                <div class="w-6 h-6 rounded-full ${c.bg} border ${c.border} flex items-center justify-center text-[9px] font-bold ${c.text}">${step.step}</div>
                ${!isLast ? `<div class="w-px flex-1 bg-white/8 mt-1"></div>` : ''}
            </div>
            <div class="flex-1 pb-3 min-w-0">
                <div class="flex items-center gap-1.5 flex-wrap">
                    <span class="text-xs font-semibold ${c.text}">${step.agent}</span>
                    <span class="text-[10px] text-text-subtle/60">${step.elapsed_ms != null ? step.elapsed_ms.toFixed(0)+'ms' : ''}</span>
                    ${routeHtml}
                </div>
                ${msgHtml}
            </div>
        </div>`;
    }).join('');

    return `<div>
        <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-2">Agent Call Flow</p>
        <div class="bg-black/10 rounded-lg p-3 space-y-0">${rows}</div>
    </div>`;
}

function renderToolCallDetails(toolCalls) {
    if (!toolCalls || toolCalls.length === 0) return '';
    return `<div>
        <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-2">Tool Call Details</p>
        <div class="space-y-2">
            ${toolCalls.map((tc, i) => {
                const argsStr = JSON.stringify(tc.args || {}, null, 2);
                const result = tc.result_preview || '';
                return `<div class="bg-black/20 rounded-lg p-3 space-y-2">
                    <div class="flex items-center gap-2">
                        <span class="material-symbols-outlined text-[14px] text-amber-400">build</span>
                        <span class="text-xs font-semibold text-amber-400 font-mono">${escHtml(tc.tool_name)}</span>
                        ${tc.tool_call_id ? `<span class="text-[9px] text-text-subtle font-mono ml-auto">${escHtml(tc.tool_call_id.substring(0,12))}…</span>` : ''}
                    </div>
                    ${argsStr && argsStr !== '{}' ? `
                    <div>
                        <p class="text-[9px] text-text-subtle uppercase mb-0.5">Arguments</p>
                        <pre class="text-[10px] text-white/60 bg-black/30 rounded p-1.5 overflow-auto max-h-24 font-mono">${escHtml(argsStr)}</pre>
                    </div>` : ''}
                    ${result ? `
                    <div>
                        <p class="text-[9px] text-text-subtle uppercase mb-0.5">Result</p>
                        <pre class="text-[10px] text-amber-200/70 bg-black/30 rounded p-1.5 overflow-auto max-h-48 whitespace-pre-wrap font-mono">${escHtml(result)}</pre>
                    </div>` : ''}
                </div>`;
            }).join('')}
        </div>
    </div>`;
}

function renderTokenUsage(tokenUsage) {
    if (!tokenUsage) {
        return `<div class="flex items-center gap-2 px-3 py-2 bg-black/10 border border-white/5 rounded-lg">
            <span class="material-symbols-outlined text-[14px] text-slate-500">token</span>
            <span class="text-[10px] text-text-subtle uppercase tracking-wider">Token Usage</span>
            <span class="text-[11px] text-text-subtle ml-auto">Not available — LLM did not return usage metadata</span>
        </div>`;
    }
    const { input_tokens, output_tokens, total_tokens } = tokenUsage;
    const fmt = n => (n != null ? n.toLocaleString() : '—');
    return `<div>
        <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-2">Token Usage</p>
        <div class="grid grid-cols-3 gap-2">
            <div class="bg-black/20 rounded p-2 text-center">
                <div class="text-[10px] text-text-subtle mb-0.5">Input</div>
                <div class="text-sm font-bold font-mono text-blue-400">${fmt(input_tokens)}</div>
            </div>
            <div class="bg-black/20 rounded p-2 text-center">
                <div class="text-[10px] text-text-subtle mb-0.5">Output</div>
                <div class="text-sm font-bold font-mono text-green-400">${fmt(output_tokens)}</div>
            </div>
            <div class="bg-black/20 rounded p-2 text-center">
                <div class="text-[10px] text-text-subtle mb-0.5">Total</div>
                <div class="text-sm font-bold font-mono text-purple-400">${fmt(total_tokens)}</div>
            </div>
        </div>
    </div>`;
}

function renderJudgeBreakdown(breakdown) {
    if (!breakdown) return '';
    const dims = ['correctness', 'tool_use', 'clarity', 'completeness'];
    const hasScores = dims.some(d => breakdown[d] != null);
    if (!hasScores) return '';
    const dimReasons = breakdown.dimensions || {};
    return `<div>
        <p class="text-[10px] text-text-subtle uppercase tracking-wider mb-2">Score Breakdown</p>
        <div class="grid grid-cols-2 gap-2">
            ${dims.map(d => {
                const score = breakdown[d];
                if (score == null) return '';
                const reasonKey = `${d}_reasoning`;
                const reason = dimReasons[reasonKey];
                return `<div class="bg-black/20 rounded p-2">
                    <div class="flex justify-between items-center">
                        <span class="text-[10px] text-text-subtle capitalize">${d.replace('_',' ')}</span>
                        <span class="text-xs font-bold font-mono ${scoreColor(score)}">${score.toFixed(1)}</span>
                    </div>
                    ${scoreBar(score)}
                    ${reason ? `<p class="text-[10px] text-white/50 mt-1.5 leading-relaxed">${escHtml(reason)}</p>` : ''}
                </div>`;
            }).join('')}
        </div>
    </div>`;
}

function closeDetailPanel() {
    const panel = document.getElementById('exp-detail-panel');
    const overlay = document.getElementById('exp-detail-overlay');
    if (panel) panel.classList.add('hidden');
    if (overlay) overlay.classList.add('hidden');
    ExperimentsState.activeRunId = null;
    ExperimentsState.activeRunData = null;
}

// ─────────────────────────────────────────────────────────
// Run experiment
// ─────────────────────────────────────────────────────────
async function startExperimentRun() {
    const selectedIds = Array.from(ExperimentsState.selectedUseCaseIds);
    if (selectedIds.length === 0) {
        showExpToast('Select at least one use case.', 'warning');
        return;
    }

    const config = ExperimentsState.configs.find(c => c.id === ExperimentsState.selectedConfigId);
    const skipJudge = document.getElementById('exp-skip-judge')?.checked ?? false;
    const useLlmJudge = document.getElementById('exp-llm-judge')?.checked ?? false;

    const runBtn = document.getElementById('exp-run-btn');
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.innerHTML = `<span class="material-symbols-outlined animate-spin text-[18px]">progress_activity</span><span>Starting…</span>`;
    }

    try {
        const res = await fetch('/api/experiments/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                use_case_ids: selectedIds,
                config: config ? config : null,
                skip_judge: skipJudge,
                use_llm_judge: useLlmJudge,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Run failed');
        }

        const data = await res.json();
        ExperimentsState.pendingRun = {
            runId: data.run_id,
            configName: config?.name || '—',
            startedAt: new Date().toISOString(),
        };
        await loadRuns();
        renderRunsTable();
        showExpToast(
            `Experiment started (${data.use_case_count} case(s)). A “Running…” row appears until results are saved.`,
            'success',
        );
        startPollingRuns(data.run_id);
    } catch (err) {
        showExpToast(`Error: ${err.message}`, 'error');
    } finally {
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerHTML = `<span class="material-symbols-outlined text-[18px]">play_arrow</span><span>Run Experiment</span>`;
        }
    }
}

function startPollingRuns(runId) {
    const POLL_INIT_MS = 3000;
    const POLL_MAX_MS = 30000;
    const POLL_MAX_DURATION_MS = 30 * 60 * 1000; // 30 minutes
    let interval = POLL_INIT_MS;
    const startedAt = Date.now();

    function stopPolling() {
        if (ExperimentsState.pollingTimer) {
            clearTimeout(ExperimentsState.pollingTimer);
            ExperimentsState.pollingTimer = null;
        }
    }

    async function poll() {
        if (Date.now() - startedAt > POLL_MAX_DURATION_MS) {
            stopPolling();
            showExpToast('Polling timed out after 30 minutes — refresh to check status.', 'warning');
            return;
        }

        await loadRuns();
        renderRunsTable();

        const found = ExperimentsState.runs.find(r => r.run_id === runId);
        if (found) {
            stopPolling();
            showExpToast('Experiment run complete — click the row to view results.', 'success');
            return;
        }

        interval = Math.min(interval * 1.5, POLL_MAX_MS);
        ExperimentsState.pollingTimer = setTimeout(poll, interval);
    }

    stopPolling();
    void poll();
}

// ─────────────────────────────────────────────────────────
// Misc helpers
// ─────────────────────────────────────────────────────────
function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function showExpToast(message, type = 'info') {
    const colors = { success: 'bg-green-500/20 border-green-500/50 text-green-300',
                     warning: 'bg-yellow-500/20 border-yellow-500/50 text-yellow-300',
                     error:   'bg-red-500/20 border-red-500/50 text-red-300',
                     info:    'bg-primary/20 border-primary/50 text-primary' };
    const icons  = { success: 'check_circle', warning: 'warning', error: 'error', info: 'info' };
    const toast = document.createElement('div');
    toast.className = `flex items-center gap-2 px-4 py-3 rounded-lg border text-sm animate-fadeIn ${colors[type] || colors.info}`;
    toast.innerHTML = `<span class="material-symbols-outlined text-[18px]">${icons[type] || 'info'}</span><span>${escHtml(message)}</span>`;
    const container = document.getElementById('toastContainer');
    if (container) {
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }
}

function attachExperimentsListeners() {
    document.getElementById('exp-run-btn')?.addEventListener('click', startExperimentRun);
    document.getElementById('exp-select-all')?.addEventListener('click', () => {
        const filtered = ExperimentsState.filterCategory
            ? ExperimentsState.useCases.filter(uc => uc.category === ExperimentsState.filterCategory)
            : ExperimentsState.useCases;
        filtered.forEach(uc => ExperimentsState.selectedUseCaseIds.add(uc.id));
        renderUseCaseList();
    });
    document.getElementById('exp-deselect-all')?.addEventListener('click', () => {
        ExperimentsState.selectedUseCaseIds.clear();
        renderUseCaseList();
    });
    document.getElementById('exp-refresh-runs')?.addEventListener('click', async () => {
        await loadRuns();
        renderRunsTable();
    });
    document.getElementById('exp-detail-overlay')?.addEventListener('click', closeDetailPanel);
    document.getElementById('exp-detail-close')?.addEventListener('click', closeDetailPanel);
}
