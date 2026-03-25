/**
 * Scientific workflow DAG UI (bioagents.workflows) — preset run + JSON results.
 */

let wfPresetsCache = [];

function syncWorkflowPresetUi() {
    const sel = document.getElementById('wf-preset-select');
    const esmRow = document.getElementById('wf-esm-model-row');
    const dimRow = document.getElementById('wf-dummy-dim-row');
    const kmerRow = document.getElementById('wf-kmer-k-row');
    const descEl = document.getElementById('wf-preset-description');
    if (!sel) return;

    const id = sel.value;
    const preset = wfPresetsCache.find((p) => p.id === id);
    if (preset && descEl) {
        descEl.textContent = preset.description || '';
    }

    if (esmRow) esmRow.classList.toggle('hidden', id !== 'protein_embedding');
    if (dimRow) dimRow.classList.toggle('hidden', id !== 'protein_embedding_dummy');
    if (kmerRow) kmerRow.classList.toggle('hidden', id !== 'uniprot_kmer_profile');
}

function populatePresetSelect(presets) {
    const sel = document.getElementById('wf-preset-select');
    const cnt = document.getElementById('wf-preset-count');
    if (!sel) return;
    sel.innerHTML = '';
    presets.forEach((p) => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name || p.id;
        sel.appendChild(opt);
    });
    if (cnt) cnt.textContent = String(presets.length);
}

async function initWorkflows() {
    const runBtn = document.getElementById('wf-run-btn');
    const presetSel = document.getElementById('wf-preset-select');
    if (!runBtn || runBtn.dataset.bound === '1') return;
    runBtn.dataset.bound = '1';

    runBtn.addEventListener('click', () => runWorkflowFromUI());
    presetSel?.addEventListener('change', () => syncWorkflowPresetUi());

    try {
        const res = await fetch('/api/workflows/presets');
        const data = await res.json();
        wfPresetsCache = data.presets || [];
        populatePresetSelect(wfPresetsCache);
        const descEl = document.getElementById('wf-preset-description');
        const first = wfPresetsCache[0];
        if (first && descEl && presetSel) {
            presetSel.value = first.id;
            descEl.textContent = first.description || '';
        }
    } catch (e) {
        console.error('Failed to load workflow presets:', e);
    }

    syncWorkflowPresetUi();
}

function renderWorkflowResult(data) {
    const statusEl = document.getElementById('wf-status');
    const sinkEl = document.getElementById('wf-sink-json');
    const nodesEl = document.getElementById('wf-nodes-json');

    if (statusEl) {
        if (data.success) {
            statusEl.textContent = 'Completed';
            statusEl.className =
                'text-xs font-mono px-2 py-1 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
        } else {
            statusEl.textContent = data.error || 'Failed';
            statusEl.className =
                'text-xs font-mono px-2 py-1 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30';
        }
    }

    const format = (obj) => {
        try {
            return JSON.stringify(obj, null, 2);
        } catch {
            return String(obj);
        }
    };

    if (sinkEl) {
        sinkEl.textContent = data.success ? format(data.sink_outputs || {}) : '';
    }
    if (nodesEl) {
        nodesEl.textContent = data.success ? format(data.node_outputs || {}) : format(data);
    }
}

function buildOptionsPayload(presetId) {
    const opts = {};
    if (presetId === 'uniprot_kmer_profile') {
        const el = document.getElementById('wf-kmer-k');
        let kk = parseInt(el?.value || '3', 10);
        if (Number.isNaN(kk)) kk = 3;
        opts.kmer_k = Math.max(1, Math.min(12, kk));
    }
    return opts;
}

async function runWorkflowFromUI() {
    const proteinInput = document.getElementById('wf-protein-id');
    const dimInput = document.getElementById('wf-embedding-dim');
    const presetSelect = document.getElementById('wf-preset-select');
    const esmSelect = document.getElementById('wf-esm-model');
    const runBtn = document.getElementById('wf-run-btn');
    const statusEl = document.getElementById('wf-status');

    const proteinId = (proteinInput?.value || '').trim();
    if (!proteinId) {
        if (typeof showToast === 'function') {
            showToast('Enter a UniProt accession or PDB ID', 'error');
        }
        return;
    }

    let dim = parseInt(dimInput?.value || '8', 10);
    if (Number.isNaN(dim)) dim = 8;
    dim = Math.max(1, Math.min(1280, dim));

    const presetId = presetSelect?.value || 'protein_embedding';
    const esm2ModelName = esmSelect?.value || 'esm2_t6_8M_UR50D';
    const options = buildOptionsPayload(presetId);

    if (statusEl) {
        statusEl.textContent = 'Running…';
        statusEl.className =
            'text-xs font-mono px-2 py-1 rounded bg-primary/20 text-primary border border-primary/30';
    }

    runBtn.disabled = true;
    try {
        const body = {
            preset_id: presetId,
            protein_id: proteinId,
            embedding_dim: dim,
            esm2_model_name: esm2ModelName,
            options,
        };

        const res = await fetch('/api/workflows/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        let data;
        try {
            data = await res.json();
        } catch {
            renderWorkflowResult({
                success: false,
                error: `Invalid response (${res.status})`,
            });
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
            renderWorkflowResult({ success: false, error: msg || `HTTP ${res.status}` });
            return;
        }

        renderWorkflowResult(data);

        if (data.success && typeof showToast === 'function') {
            showToast('Workflow finished', 'success');
        } else if (!data.success && typeof showToast === 'function') {
            showToast(data.error || 'Workflow failed', 'error');
        }
    } catch (e) {
        console.error(e);
        renderWorkflowResult({ success: false, error: String(e) });
        if (typeof showToast === 'function') {
            showToast('Network error', 'error');
        }
    } finally {
        runBtn.disabled = false;
    }
}
