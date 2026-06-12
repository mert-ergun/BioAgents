/**
 * BioAgents Admin Dashboard — Ultra Detailed
 * Shows everything: full chats, tool calls, agent decisions, engagement events,
 * artifacts, and session timelines.
 */

const adminState = {
    token: localStorage.getItem('admin_token') || null,
    currentTab: 'overview',
    currentPage: 1,
    filters: {},
    refreshInterval: null,
    initialized: false,
};

// =====================================================
// API HELPERS
// =====================================================

async function adminFetch(endpoint, params = {}) {
    const url = new URL(endpoint, window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null && v !== '') {
            url.searchParams.set(k, v);
        }
    });
    const headers = { 'Authorization': `Bearer ${adminState.token}` };
    const resp = await fetch(url, { headers });
    if (resp.status === 401) { adminLogout(); throw new Error('Session expired'); }
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.json();
}

// =====================================================
// AUTH
// =====================================================

async function adminLogin(password) {
    const resp = await fetch('/api/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
    });
    if (!resp.ok) { const data = await resp.json(); throw new Error(data.detail || 'Login failed'); }
    const data = await resp.json();
    adminState.token = data.token;
    localStorage.setItem('admin_token', data.token);
}

function adminLogout() {
    adminState.token = null;
    localStorage.removeItem('admin_token');
    if (adminState.refreshInterval) { clearInterval(adminState.refreshInterval); adminState.refreshInterval = null; }
    const loginEl = document.getElementById('admin-login');
    const dashEl = document.getElementById('admin-dashboard');
    if (loginEl) loginEl.classList.remove('hidden');
    if (dashEl) dashEl.classList.add('hidden');
}

// =====================================================
// FORMATTING
// =====================================================

function formatTimestamp(iso) {
    if (!iso) return '—';
    try {
        const d = new Date(iso);
        const now = new Date();
        const isToday = d.toDateString() === now.toDateString();
        return d.toLocaleString(undefined, {
            month: 'short', day: isToday ? undefined : 'numeric',
            hour: '2-digit', minute: '2-digit', second: '2-digit',
        });
    } catch { return iso; }
}

function formatTimestampFull(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toISOString().replace('T', ' ').replace(/\.\d+Z$/, ''); } catch { return iso; }
}

function truncate(str, len = 80) {
    if (!str) return '—';
    return str.length > len ? str.slice(0, len) + '…' : str;
}

function relativeTime(iso) {
    if (!iso) return '';
    const diff = Math.max(0, Date.now() - new Date(iso).getTime());
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatJson(jsonStr) {
    if (!jsonStr) return '';
    try { return JSON.stringify(JSON.parse(jsonStr), null, 2); } catch { return jsonStr; }
}

function formatBytes(bytes) {
    if (!bytes) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
}

// =====================================================
// INITIALIZATION
// =====================================================

function initAdmin() {
    if (adminState.initialized) { showAdminView(); return; }
    adminState.initialized = true;

    const form = document.getElementById('admin-login-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const pwd = document.getElementById('admin-password');
            const errEl = document.getElementById('admin-login-error');
            if (!pwd) return;
            try { errEl.classList.add('hidden'); await adminLogin(pwd.value); pwd.value = ''; showAdminView(); }
            catch (err) { errEl.textContent = err.message || 'Login failed'; errEl.classList.remove('hidden'); }
        });
    }
    document.getElementById('admin-logout')?.addEventListener('click', () => { adminLogout(); });
    document.getElementById('admin-refresh')?.addEventListener('click', () => { loadCurrentTab(); });

    document.querySelectorAll('.admin-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.admin-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            adminState.currentTab = btn.dataset.tab;
            adminState.currentPage = 1;
            adminState.filters = {};
            loadCurrentTab();
        });
    });
    showAdminView();
}

function showAdminView() {
    const loginEl = document.getElementById('admin-login');
    const dashEl = document.getElementById('admin-dashboard');
    if (adminState.token) {
        if (loginEl) loginEl.classList.add('hidden');
        if (dashEl) dashEl.classList.remove('hidden');
        loadCurrentTab();
        if (adminState.refreshInterval) clearInterval(adminState.refreshInterval);
        adminState.refreshInterval = setInterval(() => { if (adminState.currentTab === 'overview') loadCurrentTab(); }, 30000);
    } else {
        if (loginEl) loginEl.classList.remove('hidden');
        if (dashEl) dashEl.classList.add('hidden');
    }
}

async function loadCurrentTab() {
    const content = document.getElementById('admin-tab-content');
    if (!content) return;
    content.innerHTML = '<div class="admin-loading"><span class="material-symbols-outlined">progress_activity</span></div>';
    try {
        switch (adminState.currentTab) {
            case 'overview': await renderOverview(content); break;
            case 'clients': await renderClients(content); break;
            case 'sessions': await renderSessions(content); break;
            case 'chats': await renderChats(content); break;
            case 'experiments': await renderExperiments(content); break;
            case 'workflows': await renderWorkflows(content); break;
            case 'tools': await renderToolEvents(content); break;
            case 'decisions': await renderAgentDecisions(content); break;
            case 'logs': await renderLogs(content); break;
            case 'timeline': await renderSessionTimeline(content); break;
        }
    } catch (err) {
        content.innerHTML = `<div class="admin-empty"><p>Error loading data${err.message === 'Session expired' ? ' — please log in again' : ''}</p></div>`;
    }
}

// =====================================================
// EXPANDABLE ROW HELPER
// =====================================================

function wireExpandableRows(container) {
    container.querySelectorAll('.admin-chat-row').forEach(row => {
        row.addEventListener('click', () => {
            const detail = container.querySelector(`[data-detail-for="${row.dataset.msgId}"]`);
            if (detail) detail.classList.toggle('hidden');
            row.classList.toggle('admin-row-expanded');
        });
    });
    container.querySelectorAll('.admin-copy-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const text = decodeURIComponent(atob(btn.dataset.copy || ''));
            navigator.clipboard.writeText(text).catch(() => {});
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
        });
    });
}

// =====================================================
// TAB RENDERERS
// =====================================================

async function renderOverview(container) {
    const stats = await adminFetch('/api/admin/dashboard');
    const agents = await adminFetch('/api/admin/stats/agents');

    document.getElementById('admin-stats-cards').innerHTML = `
        <div class="admin-stat-card">
            <div class="flex items-center gap-3 mb-3">
                <div class="admin-stat-icon" style="background:rgba(6,182,212,0.12);color:var(--accent-cyan)"><span class="material-symbols-outlined text-lg">group</span></div>
            </div>
            <div class="admin-stat-value">${stats.total_clients}</div>
            <div class="admin-stat-label">Clients</div>
        </div>
        <div class="admin-stat-card">
            <div class="flex items-center gap-3 mb-3">
                <div class="admin-stat-icon" style="background:rgba(139,92,246,0.12);color:var(--accent-violet)"><span class="material-symbols-outlined text-lg">forum</span></div>
            </div>
            <div class="admin-stat-value">${stats.total_queries}</div>
            <div class="admin-stat-label">Queries <span style="font-size:0.625rem;color:var(--text-muted)">${stats.queries_today} today</span></div>
        </div>
        <div class="admin-stat-card">
            <div class="flex items-center gap-3 mb-3">
                <div class="admin-stat-icon" style="background:rgba(16,185,129,0.12);color:var(--accent-emerald)"><span class="material-symbols-outlined text-lg">science</span></div>
            </div>
            <div class="admin-stat-value">${stats.total_experiments}</div>
            <div class="admin-stat-label">Experiments</div>
        </div>
        <div class="admin-stat-card">
            <div class="flex items-center gap-3 mb-3">
                <div class="admin-stat-icon" style="background:rgba(245,158,11,0.12);color:var(--accent-amber)"><span class="material-symbols-outlined text-lg">account_tree</span></div>
            </div>
            <div class="admin-stat-value">${stats.total_workflows}</div>
            <div class="admin-stat-label">Workflows</div>
        </div>
    `;

    const agentData = agents.agents || [];
    const totalAgentCalls = agentData.reduce((s, a) => s + a.count, 0) || 1;
    const colorMap = { slate:'#64748b', blue:'#3b82f6', emerald:'#10b981', amber:'#f59e0b', pink:'#ec4899', indigo:'#6366f1', primary:'#06b6d4', rose:'#f43f5e', purple:'#a855f7', cyan:'#06b6d4' };

    let agentHtml = agentData.length > 0 ? '<div class="space-y-2">' + agentData.slice(0, 8).map(a => {
        const pct = Math.round((a.count / totalAgentCalls) * 100);
        const ag = (typeof AGENTS !== 'undefined' && AGENTS[a.agent]) || { label: a.agent, color: 'slate' };
        const clr = colorMap[ag.color] || '#64748b';
        return `<div class="flex items-center gap-3"><span class="text-xs font-medium w-24 truncate" style="color:var(--text-secondary)">${ag.label || a.agent}</span><div class="flex-1 h-2 rounded-full" style="background:var(--bg-tertiary)"><div class="h-2 rounded-full" style="width:${pct}%;background:${clr}"></div></div><span class="text-xs font-mono" style="color:var(--text-muted)">${a.count}</span></div>`;
    }).join('') + '</div>' : '<div class="admin-empty"><p>No agent usage data yet</p></div>';

    container.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div><h3 class="text-sm font-semibold mb-3" style="color:var(--text-primary)">Agent Usage</h3>${agentHtml}</div>
            <div><h3 class="text-sm font-semibold mb-3" style="color:var(--text-primary)">Quick Stats</h3>
                <div class="space-y-2 text-sm" style="color:var(--text-secondary)">
                    <div class="flex justify-between py-1.5 border-b" style="border-color:var(--border-subtle)"><span>Active Sessions</span><span class="font-mono">${stats.active_sessions} / ${stats.total_sessions}</span></div>
                    <div class="flex justify-between py-1.5 border-b" style="border-color:var(--border-subtle)"><span>Total Activities</span><span class="font-mono">${stats.total_activities}</span></div>
                    <div class="flex justify-between py-1.5 border-b" style="border-color:var(--border-subtle)"><span>Activities Today</span><span class="font-mono">${stats.activities_today}</span></div>
                </div>
            </div>
        </div>`;
}

async function renderClients(container) {
    const data = await adminFetch('/api/admin/clients', { page: adminState.currentPage, limit: 20 });
    if (data.items.length === 0) { container.innerHTML = '<div class="admin-empty"><span class="material-symbols-outlined admin-empty-icon">group_off</span><p>No clients yet</p></div>'; return; }
    container.innerHTML = `<div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Client ID</th><th>First Seen</th><th>Last Seen</th><th>Requests</th></tr></thead><tbody>${data.items.map(c => `<tr><td><span class="admin-mono">${c.client_id}</span></td><td title="${c.first_seen}">${formatTimestamp(c.first_seen)}</td><td title="${c.last_seen}">${relativeTime(c.last_seen)}</td><td class="font-mono">${c.total_requests}</td></tr>`).join('')}</tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`;
    wirePagination(container);
}

async function renderSessions(container) {
    const data = await adminFetch('/api/admin/sessions', { page: adminState.currentPage, limit: 20 });
    if (data.items.length === 0) { container.innerHTML = '<div class="admin-empty"><span class="material-symbols-outlined admin-empty-icon">timer_off</span><p>No sessions yet</p></div>'; return; }
    container.innerHTML = `<div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Session</th><th>Client</th><th>Started</th><th>Queries</th><th>Experiments</th><th>Status</th></tr></thead><tbody>${data.items.map(s => {
        const statusBadge = !s.ended_at ? '<span class="admin-badge admin-badge--success">Active</span>' : '<span class="admin-badge admin-badge--api">Ended</span>';
        return `<tr><td><span class="admin-mono">${s.session_id.slice(0, 8)}…</span></td><td><span class="admin-mono">${s.client_id.slice(0, 8)}</span></td><td title="${s.started_at}">${formatTimestamp(s.started_at)}</td><td class="font-mono">${s.total_queries}</td><td class="font-mono">${s.total_experiments}</td><td>${statusBadge}</td></tr>`;
    }).join('')}</tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`;
    wirePagination(container);
}

async function renderChats(container) {
    const search = adminState.filters.search || '';
    const data = await adminFetch('/api/admin/chats', { page: adminState.currentPage, limit: 50, search: search || undefined });

    container.innerHTML = `
        <div class="admin-search-bar">
            <input type="text" class="admin-search-input" placeholder="Search messages…" value="${escapeHtml(search)}" id="admin-chat-search">
            <button class="admin-pagination-btn" id="admin-chat-search-btn">Search</button>
        </div>
        ${data.items.length === 0 ? '<div class="admin-empty"><span class="material-symbols-outlined admin-empty-icon">chat_bubble_outline</span><p>No chat messages yet</p></div>' : `
        <div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Time</th><th>Role</th><th>Agent</th><th>Content</th><th>Session</th></tr></thead><tbody>
        ${data.items.map(m => {
            const roleBadge = m.role === 'user' ? '<span class="admin-badge admin-badge--query">User</span>' : m.role === 'assistant' ? '<span class="admin-badge admin-badge--success">Assistant</span>' : '<span class="admin-badge admin-badge--api">Tool</span>';
            const ag = (typeof AGENTS !== 'undefined' && AGENTS[m.agent]) || null;
            const agentLabel = ag ? ag.label : (m.agent || '—');
            const hasToolCalls = m.tool_calls && m.tool_calls !== 'null';
            const hasArtifacts = m.artifacts && m.artifacts !== 'null';
            return `<tr class="admin-chat-row" data-msg-id="${m.id}">
                <td class="admin-activity-time" title="${m.created_at}"><div>${relativeTime(m.created_at)}</div><div class="admin-mono" style="font-size:0.6rem;color:var(--text-muted)">${formatTimestamp(m.created_at)}</div></td>
                <td>${roleBadge}${hasToolCalls ? '<br><span class="admin-badge admin-badge--warning" style="margin-top:2px">Tool Calls</span>' : ''}</td>
                <td><span class="text-xs font-medium">${agentLabel}</span></td>
                <td>${truncate(m.content, 200)}</td>
                <td><span class="admin-mono">${(m.session_id || '').slice(0, 8)}</span></td>
            </tr>
            <tr class="admin-chat-detail hidden" data-detail-for="${m.id}"><td colspan="5">
                <div class="admin-detail-panel">
                    <div class="admin-detail-header"><span style="font-weight:600;font-size:0.8125rem;color:var(--text-primary)">Full Message</span><button class="admin-copy-btn" data-copy="${btoa(encodeURIComponent(m.content || ''))}">Copy</button></div>
                    <pre class="admin-detail-content">${escapeHtml(m.content || '')}</pre>
                    ${hasToolCalls ? `<div class="admin-detail-section"><h4>Tool Calls</h4><pre class="admin-json-block">${escapeHtml(formatJson(m.tool_calls))}</pre></div>` : ''}
                    ${hasArtifacts ? `<div class="admin-detail-section"><h4>Artifacts</h4><pre class="admin-json-block">${escapeHtml(formatJson(m.artifacts))}</pre></div>` : ''}
                    ${m.tokens_used ? `<div class="admin-detail-section" style="font-size:0.75rem;color:var(--text-tertiary)">Tokens: ${m.tokens_used} · References: ${m.references_count || 0}</div>` : ''}
                </div>
            </td></tr>`;
        }).join('')}
        </tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`} `;

    const searchInput = document.getElementById('admin-chat-search');
    const searchBtn = document.getElementById('admin-chat-search-btn');
    if (searchBtn && searchInput) {
        const doSearch = () => { adminState.filters.search = searchInput.value; adminState.currentPage = 1; renderChats(container); };
        searchBtn.addEventListener('click', doSearch);
        searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
    }
    wireExpandableRows(container);
    wirePagination(container);
}

async function renderExperiments(container) {
    const data = await adminFetch('/api/admin/experiments', { page: adminState.currentPage, limit: 20 });
    if (data.items.length === 0) { container.innerHTML = '<div class="admin-empty"><span class="material-symbols-outlined admin-empty-icon">science</span><p>No experiments yet</p></div>'; return; }
    container.innerHTML = `<div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Run ID</th><th>Client</th><th>Status</th><th>Started</th><th>Duration</th></tr></thead><tbody>${data.items.map(e => {
        const statusBadge = e.status === 'completed' ? '<span class="admin-badge admin-badge--success">Completed</span>' : e.status === 'failed' ? '<span class="admin-badge admin-badge--error">Failed</span>' : '<span class="admin-badge admin-badge--running">Running</span>';
        return `<tr><td><span class="admin-mono">${e.run_id.slice(0, 12)}…</span></td><td><span class="admin-mono">${e.client_id.slice(0, 8)}</span></td><td>${statusBadge}</td><td title="${e.created_at}">${formatTimestamp(e.created_at)}</td><td class="font-mono">${e.duration_ms ? `${(e.duration_ms / 1000).toFixed(1)}s` : '—'}</td></tr>`;
    }).join('')}</tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`;
    wirePagination(container);
}

async function renderWorkflows(container) {
    const data = await adminFetch('/api/admin/workflows', { page: adminState.currentPage, limit: 20 });
    if (data.items.length === 0) { container.innerHTML = '<div class="admin-empty"><span class="material-symbols-outlined admin-empty-icon">account_tree</span><p>No workflows yet</p></div>'; return; }
    container.innerHTML = `<div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Type</th><th>Preset</th><th>Client</th><th>Status</th><th>Started</th><th>Duration</th></tr></thead><tbody>${data.items.map(w => {
        const statusBadge = w.status === 'completed' ? '<span class="admin-badge admin-badge--success">Completed</span>' : w.status === 'failed' ? '<span class="admin-badge admin-badge--error">Failed</span>' : '<span class="admin-badge admin-badge--running">Running</span>';
        return `<tr><td><span class="admin-badge admin-badge--workflow">${w.workflow_type}</span></td><td>${w.preset_id || '—'}</td><td><span class="admin-mono">${w.client_id.slice(0, 8)}</span></td><td>${statusBadge}</td><td title="${w.created_at}">${formatTimestamp(w.created_at)}</td><td class="font-mono">${w.duration_ms ? `${(w.duration_ms / 1000).toFixed(1)}s` : '—'}</td></tr>`;
    }).join('')}</tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`;
    wirePagination(container);
}

// =====================================================
// TOOL EVENTS TAB
// =====================================================

async function renderToolEvents(container) {
    const data = await adminFetch('/api/admin/tool-events', { page: adminState.currentPage, limit: 50 });
    if (data.items.length === 0) { container.innerHTML = '<div class="admin-empty"><span class="material-symbols-outlined admin-empty-icon">build</span><p>No tool events yet — send a query to see tool calls</p></div>'; return; }

    container.innerHTML = `<div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Time</th><th>Agent</th><th>Tool</th><th>Type</th><th>Arguments</th><th>Result</th><th>Duration</th></tr></thead><tbody>
    ${data.items.map(t => {
        const isCall = t.event_type === 'call';
        const typeBadge = isCall ? '<span class="admin-badge admin-badge--warning">Call</span>' : '<span class="admin-badge admin-badge--success">Result</span>';
        const argPreview = t.arguments ? truncate(formatJson(t.arguments), 100) : '—';
        const resultPreview = t.result ? (t.result_truncated ? truncate(t.result, 100) + ' [truncated]' : truncate(t.result, 100)) : '—';
        return `<tr class="admin-chat-row" data-msg-id="tool-${t.id}">
            <td class="admin-activity-time" title="${t.created_at}"><div>${relativeTime(t.created_at)}</div><div class="admin-mono" style="font-size:0.6rem">${formatTimestamp(t.created_at)}</div></td>
            <td><span class="text-xs font-medium">${t.agent}</span></td>
            <td><span class="text-xs font-mono font-semibold">${t.tool_name}</span></td>
            <td>${typeBadge} ${t.status === 'error' ? '<span class="admin-badge admin-badge--error">Err</span>' : ''}</td>
            <td>${escapeHtml(argPreview)}</td>
            <td>${escapeHtml(resultPreview)}</td>
            <td class="font-mono">${t.duration_ms ? `${t.duration_ms.toFixed(0)}ms` : '—'}</td>
        </tr>
        <tr class="admin-chat-detail hidden" data-detail-for="tool-${t.id}"><td colspan="7">
            <div class="admin-detail-panel">
                ${t.arguments ? `<div class="admin-detail-section"><h4>Arguments</h4><pre class="admin-json-block">${escapeHtml(formatJson(t.arguments))}</pre></div>` : ''}
                ${t.result ? `<div class="admin-detail-section"><h4>Result${t.result_truncated ? ' (truncated at 100KB)' : ''}</h4><pre class="admin-detail-content">${escapeHtml(t.result)}</pre><button class="admin-copy-btn" data-copy="${btoa(encodeURIComponent(t.result))}">Copy</button></div>` : ''}
            </div>
        </td></tr>`;
    }).join('')}
    </tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`;
    wireExpandableRows(container);
    wirePagination(container);
}

// =====================================================
// AGENT DECISIONS TAB
// =====================================================

async function renderAgentDecisions(container) {
    const data = await adminFetch('/api/admin/agent-decisions', { page: adminState.currentPage, limit: 100 });
    if (data.items.length === 0) { container.innerHTML = '<div class="admin-empty"><span class="material-symbols-outlined admin-empty-icon">route</span><p>No decisions yet</p></div>'; return; }

    container.innerHTML = `<div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Time</th><th>Agent</th><th>Decision</th><th>Reasoning</th><th>Step</th><th>Session</th></tr></thead><tbody>
    ${data.items.map(d => `<tr class="admin-chat-row" data-msg-id="dec-${d.id}">
        <td class="admin-activity-time" title="${d.created_at}"><div>${relativeTime(d.created_at)}</div><div class="admin-mono" style="font-size:0.6rem">${formatTimestamp(d.created_at)}</div></td>
        <td><span class="text-xs font-medium">${d.agent}</span></td>
        <td><span class="admin-badge ${d.decision === 'FINISH' ? 'admin-badge--success' : 'admin-badge--warning'}">${d.decision || '—'}</span></td>
        <td>${truncate(d.reasoning, 120)}</td>
        <td class="font-mono">#${d.step_index}</td>
        <td><span class="admin-mono">${(d.session_id || '').slice(0, 8)}</span></td>
    </tr>
    <tr class="admin-chat-detail hidden" data-detail-for="dec-${d.id}"><td colspan="6">
        <div class="admin-detail-panel">
            ${d.reasoning ? `<div class="admin-detail-section"><h4>Full Reasoning</h4><pre class="admin-detail-content">${escapeHtml(d.reasoning)}</pre></div>` : ''}
            ${d.step_messages ? `<div class="admin-detail-section"><h4>Step Messages</h4><pre class="admin-json-block">${escapeHtml(formatJson(d.step_messages))}</pre></div>` : ''}
        </div>
    </td></tr>`).join('')}
    </tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`;
    wireExpandableRows(container);
    wirePagination(container);
}

// =====================================================
// ACTIVITY LOGS TAB
// =====================================================

async function renderLogs(container) {
    const { search, action, status: statusFilter } = adminState.filters;
    container.innerHTML = `
        <div class="admin-search-bar">
            <input type="text" class="admin-search-input" placeholder="Search logs…" value="${escapeHtml(search || '')}" id="admin-log-search">
            <select class="admin-filter-select" id="admin-log-action">
                <option value="">All Actions</option><option value="query" ${action === 'query' ? 'selected' : ''}>Query</option>
                <option value="experiment_run" ${action === 'experiment_run' ? 'selected' : ''}>Experiment</option>
                <option value="workflow_run" ${action === 'workflow_run' ? 'selected' : ''}>Workflow</option>
                <option value="upload" ${action === 'upload' ? 'selected' : ''}>Upload</option>
                <option value="api_call" ${action === 'api_call' ? 'selected' : ''}>API Call</option>
            </select>
            <select class="admin-filter-select" id="admin-log-status">
                <option value="">All Status</option><option value="success" ${statusFilter === 'success' ? 'selected' : ''}>Success</option>
                <option value="error" ${statusFilter === 'error' ? 'selected' : ''}>Error</option>
            </select>
            <button class="admin-pagination-btn" id="admin-log-filter-btn">Filter</button>
        </div><div id="admin-log-results"></div>`;

    const fetchAndRender = async () => {
        const resultEl = document.getElementById('admin-log-results');
        if (!resultEl) return;
        resultEl.innerHTML = '<div class="admin-loading"><span class="material-symbols-outlined">progress_activity</span></div>';
        const data = await adminFetch('/api/admin/logs', { page: adminState.currentPage, limit: 50, query: adminState.filters.search || undefined, action: adminState.filters.action || undefined, status: adminState.filters.status || undefined });
        if (data.items.length === 0) { resultEl.innerHTML = '<div class="admin-empty"><p>No log entries found</p></div>'; return; }
        resultEl.innerHTML = `<div class="admin-table-wrap"><table class="admin-table"><thead><tr><th>Time</th><th>Action</th><th>Client</th><th>Details</th><th>Status</th><th>Duration</th></tr></thead><tbody>${data.items.map(l => {
            const actionBadge = `admin-badge--${['query','experiment','workflow','upload'].includes(l.action) ? l.action.replace('experiment_run','experiment').replace('workflow_run','workflow') : 'api'}`;
            const statusBadge = l.status === 'success' ? '<span class="admin-badge admin-badge--success">OK</span>' : '<span class="admin-badge admin-badge--error">Err</span>';
            let detail = ''; try { const d = JSON.parse(l.details || '{}'); detail = `${d.method || ''} ${d.path || ''}`.trim(); } catch {}
            return `<tr><td class="admin-activity-time" title="${l.created_at}"><div>${relativeTime(l.created_at)}</div><div class="admin-mono" style="font-size:0.6rem">${formatTimestamp(l.created_at)}</div></td><td><span class="admin-badge ${actionBadge}">${l.action}</span></td><td><span class="admin-mono">${l.client_id.slice(0, 8)}</span></td><td title="${escapeHtml(l.details || '')}">${truncate(detail, 80)}</td><td>${statusBadge}</td><td class="font-mono">${l.duration_ms ? `${l.duration_ms.toFixed(0)}ms` : '—'}</td></tr>`;
        }).join('')}</tbody></table></div>${renderPagination(data.total, data.page, data.limit)}`;
        wirePagination(resultEl);
    };

    document.getElementById('admin-log-filter-btn')?.addEventListener('click', () => {
        adminState.filters.search = document.getElementById('admin-log-search')?.value || '';
        adminState.filters.action = document.getElementById('admin-log-action')?.value || '';
        adminState.filters.status = document.getElementById('admin-log-status')?.value || '';
        adminState.currentPage = 1;
        fetchAndRender();
    });
    document.getElementById('admin-log-search')?.addEventListener('keydown', (e) => { if (e.key === 'Enter') document.getElementById('admin-log-filter-btn')?.click(); });
    fetchAndRender();
}

// =====================================================
// SESSION TIMELINE TAB
// =====================================================

async function renderSessionTimeline(container) {
    // Fetch sessions for dropdown
    const sessions = await adminFetch('/api/admin/sessions', { limit: 100, order: 'desc' });
    const selectedSession = adminState.filters.timeline_session || '';

    container.innerHTML = `
        <div style="margin-bottom:1rem">
            <label style="display:block;font-size:0.8125rem;font-weight:600;color:var(--text-secondary);margin-bottom:0.5rem">Select a session to view its complete timeline:</label>
            <select class="admin-session-select" id="admin-timeline-session">
                <option value="">— Choose a session —</option>
                ${sessions.items.map(s => `<option value="${s.session_id}" ${s.session_id === selectedSession ? 'selected' : ''}>${s.session_id.slice(0, 12)}… | ${formatTimestamp(s.started_at)} | ${s.total_queries} queries | ${!s.ended_at ? 'Active' : 'Ended'}</option>`).join('')}
            </select>
        </div>
        <div id="admin-timeline-content"></div>`;

    document.getElementById('admin-timeline-session')?.addEventListener('change', (e) => {
        adminState.filters.timeline_session = e.target.value;
        adminState.currentPage = 1;
        renderSessionTimeline(container);
    });

    if (!selectedSession) return;

    const tlContent = document.getElementById('admin-timeline-content');
    if (!tlContent) return;
    tlContent.innerHTML = '<div class="admin-loading"><span class="material-symbols-outlined">progress_activity</span></div>';

    const data = await adminFetch(`/api/admin/sessions/${selectedSession}/timeline`);
    const events = data.timeline?.items || [];

    if (events.length === 0) { tlContent.innerHTML = '<div class="admin-empty"><p>No events found for this session</p></div>'; return; }

    // Render timeline
    let html = `<div class="admin-timeline">`;
    for (const ev of events) {
        const time = formatTimestampFull(ev.created_at);
        switch (ev.event_type) {
            case 'message':
                const roleColor = ev.role === 'user' ? 'var(--accent-cyan)' : ev.role === 'assistant' ? 'var(--accent-emerald)' : 'var(--text-tertiary)';
                const agentLabel = (typeof AGENTS !== 'undefined' && AGENTS[ev.agent])?.label || ev.agent || '';
                html += `<div class="admin-tl-item admin-tl-item--message">
                    <div class="admin-tl-time">${time}</div>
                    <div class="admin-tl-type"><span class="admin-tl-label" style="color:${roleColor}">${ev.role.toUpperCase()}</span>${agentLabel ? `<span class="admin-tl-label" style="color:var(--accent-violet)">${agentLabel}</span>` : ''}</div>
                    <div class="admin-tl-body"><pre>${escapeHtml(ev.content || '')}</pre>
                    ${ev.tool_calls && ev.tool_calls !== 'null' ? `<div style="margin-top:0.25rem"><span style="font-size:0.6875rem;font-weight:600;color:var(--accent-amber)">Tool Calls:</span><pre class="admin-json-block">${escapeHtml(formatJson(ev.tool_calls))}</pre></div>` : ''}
                    </div></div>`;
                break;
            case 'tool':
                const isCall = ev.tool_event_type === 'call';
                html += `<div class="admin-tl-item admin-tl-item--tool">
                    <div class="admin-tl-time">${time}</div>
                    <div class="admin-tl-type"><span class="admin-tl-label" style="color:var(--accent-amber)">${isCall ? 'TOOL CALL' : 'TOOL RESULT'}</span><span class="admin-tl-label">${ev.agent}</span><span class="admin-tl-label" style="color:var(--accent-emerald)">${ev.tool_name}</span></div>
                    <div class="admin-tl-body">
                    ${ev.arguments ? `<details><summary style="cursor:pointer;font-size:0.75rem;color:var(--accent-cyan)">Arguments</summary><pre class="admin-json-block">${escapeHtml(formatJson(ev.arguments))}</pre></details>` : ''}
                    ${ev.result ? `<details><summary style="cursor:pointer;font-size:0.75rem;color:var(--accent-emerald)">Result${ev.result_truncated ? ' (truncated)' : ''}</summary><pre class="admin-detail-content">${escapeHtml(ev.result)}</pre></details>` : ''}
                    ${ev.duration_ms ? `<span style="font-size:0.6875rem;color:var(--text-muted)">${ev.duration_ms.toFixed(0)}ms</span>` : ''}
                    </div></div>`;
                break;
            case 'decision':
                html += `<div class="admin-tl-item admin-tl-item--decision">
                    <div class="admin-tl-time">${time}</div>
                    <div class="admin-tl-type"><span class="admin-tl-label" style="color:var(--accent-violet)">DECISION</span><span class="admin-tl-label">${ev.agent}</span></div>
                    <div class="admin-tl-body">
                    <div style="font-weight:600;font-size:0.75rem">Route: <span style="color:var(--accent-cyan)">${ev.decision || '—'}</span></div>
                    ${ev.reasoning ? `<details><summary style="cursor:pointer;font-size:0.75rem;color:var(--accent-violet)">Reasoning</summary><pre style="white-space:pre-wrap;font-size:0.75rem;color:var(--text-secondary)">${escapeHtml(ev.reasoning)}</pre></details>` : ''}
                    </div></div>`;
                break;
            case 'engagement':
                html += `<div class="admin-tl-item admin-tl-item--engagement">
                    <div class="admin-tl-time">${time}</div>
                    <div class="admin-tl-type"><span class="admin-tl-label" style="color:var(--accent-rose)">ENGAGEMENT</span><span class="admin-tl-label">${ev.engagement_type || ''}</span></div>
                    <div class="admin-tl-body">
                    <div style="font-weight:600;font-size:0.8125rem">${escapeHtml(ev.question || '')}</div>
                    ${ev.options ? `<div style="font-size:0.75rem;color:var(--text-tertiary)">Options: ${ev.options}</div>` : ''}
                    ${ev.response_content ? `<div style="margin-top:0.25rem;font-size:0.8125rem;color:var(--accent-emerald)">Response: ${escapeHtml(ev.response_content)}${ev.selected_option ? ` (${ev.selected_option})` : ''}</div>` : ''}
                    ${ev.timed_out ? '<div style="font-size:0.75rem;color:var(--accent-rose)">Timed out</div>' : ''}
                    </div></div>`;
                break;
            case 'artifact':
                html += `<div class="admin-tl-item admin-tl-item--artifact">
                    <div class="admin-tl-time">${time}</div>
                    <div class="admin-tl-type"><span class="admin-tl-label" style="color:var(--accent-emerald)">ARTIFACT</span>${ev.source_agent ? `<span class="admin-tl-label">${ev.source_agent}</span>` : ''}</div>
                    <div class="admin-tl-body"><span style="font-weight:600">${escapeHtml(ev.artifact_name)}</span> <span style="color:var(--text-muted);font-size:0.75rem">${ev.artifact_type || ''} ${ev.artifact_size ? formatBytes(ev.artifact_size) : ''}</span></div></div>`;
                break;
            case 'approval':
                const outColor = ev.outcome === 'approved' ? 'var(--accent-emerald)' : ev.outcome === 'blocked' ? 'var(--accent-rose)' : 'var(--accent-amber)';
                html += `<div class="admin-tl-item admin-tl-item--approval">
                    <div class="admin-tl-time">${time}</div>
                    <div class="admin-tl-type"><span class="admin-tl-label" style="color:var(--accent-pink)">APPROVAL</span><span class="admin-tl-label" style="color:${outColor}">${ev.outcome.toUpperCase()}</span></div>
                    <div class="admin-tl-body"><span style="font-weight:600">${escapeHtml(ev.tool_name)}</span>${ev.reason ? ` — ${escapeHtml(ev.reason)}` : ''}${ev.risk_level ? ` <span class="admin-badge admin-badge--warning">Risk: ${ev.risk_level}</span>` : ''}</div></div>`;
                break;
        }
    }
    html += '</div>';
    tlContent.innerHTML = html;
}

// =====================================================
// PAGINATION
// =====================================================

function renderPagination(total, page, limit) {
    const totalPages = Math.ceil(total / limit);
    if (totalPages <= 1) return '';
    const start = (page - 1) * limit + 1;
    const end = Math.min(page * limit, total);
    return `<div class="admin-pagination"><span>Showing ${start}–${end} of ${total}</span><div class="flex gap-2"><button class="admin-pagination-btn" data-page="${page - 1}" ${page <= 1 ? 'disabled' : ''}>Previous</button><button class="admin-pagination-btn" data-page="${page + 1}" ${page >= totalPages ? 'disabled' : ''}>Next</button></div></div>`;
}

function wirePagination(container) {
    container.querySelectorAll('.admin-pagination-btn[data-page]').forEach(btn => {
        btn.addEventListener('click', () => {
            const p = parseInt(btn.dataset.page, 10);
            if (p >= 1) { adminState.currentPage = p; loadCurrentTab(); }
        });
    });
}
