/**
 * Custom workflow builder: compose nodes/edges, set source inputs, run via POST /api/workflows/run-custom.
 */

let wbNodeTypes = [];
let wbNodeIdCounter = 0;

function wbNextNodeId() {
    wbNodeIdCounter += 1;
    return `n${wbNodeIdCounter}`;
}

/** Order in the template dropdown (first is the default). */
const WB_TEMPLATE_KEYS = [
    'uniprot_mw_text',
    'uniprot_record_dict',
    'dummy_embed_export',
    'pdb_summary',
    'blank',
];

const WB_TEMPLATES = {
    blank: {
        label: 'Blank',
        nodes: [],
        edges: [],
        initial_inputs: {},
    },
    uniprot_mw_text: {
        label: 'UniProt → preprocess → MW → text JSON',
        nodes: [
            { id: 'fetch', type: 'uniprot_fasta', params: {} },
            { id: 'prep', type: 'fasta_preprocess', params: {} },
            { id: 'mw', type: 'molecular_weight', params: {} },
            { id: 'out', type: 'export_text_json', params: {} },
        ],
        edges: [
            { source: 'fetch', target: 'prep' },
            { source: 'prep', target: 'mw' },
            { source: 'mw', target: 'out', port_map: { report: 'text' } },
        ],
        initial_inputs: { fetch: { protein_id: 'P04637' } },
    },
    uniprot_record_dict: {
        label: 'UniProt → preprocess → biochem record → dict JSON',
        nodes: [
            { id: 'fetch', type: 'uniprot_fasta', params: {} },
            { id: 'prep', type: 'fasta_preprocess', params: {} },
            { id: 'rec', type: 'compact_biochem_record', params: {} },
            { id: 'out', type: 'export_dict_json', params: {} },
        ],
        edges: [
            { source: 'fetch', target: 'prep' },
            { source: 'prep', target: 'rec' },
            { source: 'rec', target: 'out', port_map: { record: 'record' } },
        ],
        initial_inputs: { fetch: { protein_id: 'P04637' } },
    },
    pdb_summary: {
        label: 'RCSB entry summary (PDB ID)',
        nodes: [{ id: 'pdb', type: 'rcsb_entry_summary', params: {} }],
        edges: [],
        initial_inputs: { pdb: { pdb_id: '1YCR' } },
    },
    dummy_embed_export: {
        label: 'UniProt → preprocess → dummy embed + export JSON',
        nodes: [
            { id: 'fetch', type: 'uniprot_fasta', params: {} },
            { id: 'prep', type: 'fasta_preprocess', params: {} },
            { id: 'emb', type: 'dummy_embedder', params: { dim: 8 } },
            { id: 'out', type: 'export_json', params: {} },
        ],
        edges: [
            { source: 'fetch', target: 'prep' },
            { source: 'prep', target: 'emb', port_map: { sequence: 'sequence' } },
            { source: 'prep', target: 'out', port_map: { residue_count: 'residue_count' } },
            { source: 'emb', target: 'out', port_map: { embedding: 'embedding' } },
        ],
        initial_inputs: { fetch: { protein_id: 'P04637' } },
    },
};

function wbGetState() {
    const nodes = [];
    document.querySelectorAll('#wb-nodes-tbody tr[data-wb-node]').forEach((row) => {
        const id = row.querySelector('.wb-node-id')?.value?.trim() || '';
        const type = row.querySelector('.wb-node-type')?.value || '';
        let paramsText = row.querySelector('.wb-node-params')?.value?.trim() || '{}';
        let params = {};
        try {
            params = paramsText ? JSON.parse(paramsText) : {};
        } catch {
            params = {};
        }
        if (id && type) nodes.push({ id, type, params });
    });

    const edges = [];
    document.querySelectorAll('#wb-edges-tbody tr[data-wb-edge]').forEach((row) => {
        const source = row.querySelector('.wb-edge-source')?.value?.trim() || '';
        const target = row.querySelector('.wb-edge-target')?.value?.trim() || '';
        const pmText = row.querySelector('.wb-edge-portmap')?.value?.trim() || '';
        if (!source || !target) return;
        const e = { source, target };
        if (pmText) {
            try {
                e.port_map = JSON.parse(pmText);
            } catch {
                /* skip invalid port map */
            }
        }
        edges.push(e);
    });

    let initial_inputs = {};
    const ta = document.getElementById('wb-initial-inputs');
    if (ta && ta.value.trim()) {
        try {
            initial_inputs = JSON.parse(ta.value);
        } catch {
            initial_inputs = {};
        }
    }
    return { nodes, edges, initial_inputs };
}

function wbSetInitialInputs(obj) {
    const ta = document.getElementById('wb-initial-inputs');
    if (ta) ta.value = JSON.stringify(obj, null, 2);
}

function wbClearTables() {
    const nt = document.getElementById('wb-nodes-tbody');
    const et = document.getElementById('wb-edges-tbody');
    if (nt) nt.innerHTML = '';
    if (et) et.innerHTML = '';
}

function wbPopulateTypeSelect(selectEl, selectedType) {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    wbNodeTypes.forEach((t) => {
        const opt = document.createElement('option');
        opt.value = t.id;
        opt.textContent = `${t.name} (${t.id})`;
        selectEl.appendChild(opt);
    });
    if (selectedType && wbNodeTypes.some((t) => t.id === selectedType)) {
        selectEl.value = selectedType;
    }
}

function wbAddNodeRow(node) {
    const tbody = document.getElementById('wb-nodes-tbody');
    if (!tbody) return;
    const tr = document.createElement('tr');
    tr.dataset.wbNode = '1';
    tr.className = 'border-b border-white/5';
    const id = node?.id || wbNextNodeId();
    const type = node?.type || (wbNodeTypes[0]?.id || '');
    const paramsJson = JSON.stringify(node?.params || {}, null, 2);

    const tdId = document.createElement('td');
    tdId.className = 'py-2 pr-2 align-top';
    const idInp = document.createElement('input');
    idInp.type = 'text';
    idInp.className =
        'wb-node-id w-full rounded bg-surface-dark border border-white/10 text-xs text-white px-2 py-1.5 font-mono';
    idInp.value = id;
    tdId.appendChild(idInp);

    const tdType = document.createElement('td');
    tdType.className = 'py-2 pr-2 align-top';
    const sel = document.createElement('select');
    sel.className =
        'wb-node-type w-full rounded bg-surface-dark border border-white/10 text-xs text-white px-2 py-1.5';
    tdType.appendChild(sel);

    const tdParams = document.createElement('td');
    tdParams.className = 'py-2 pr-2 align-top';
    const ta = document.createElement('textarea');
    ta.rows = 3;
    ta.className =
        'wb-node-params w-full rounded bg-black/40 border border-white/10 text-[11px] text-cyan-100/90 px-2 py-1 font-mono resize-y min-h-[3.5rem]';
    ta.value = paramsJson;
    tdParams.appendChild(ta);

    const tdRm = document.createElement('td');
    tdRm.className = 'py-2 align-top w-10';
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'wb-remove-node text-rose-400 hover:text-rose-300 p-1 rounded hover:bg-white/5';
    rm.title = 'Remove node';
    rm.innerHTML = '<span class="material-symbols-outlined text-[18px]">close</span>';
    tdRm.appendChild(rm);

    tr.append(tdId, tdType, tdParams, tdRm);
    tbody.appendChild(tr);

    wbPopulateTypeSelect(sel, type);
    rm.addEventListener('click', () => {
        tr.remove();
        wbRefreshEdgeSelectOptions();
    });
    idInp.addEventListener('input', () => wbRefreshEdgeSelectOptions());
}

function wbEdgeNodeOptions() {
    const ids = [];
    document.querySelectorAll('#wb-nodes-tbody .wb-node-id').forEach((inp) => {
        const v = inp.value?.trim();
        if (v) ids.push(v);
    });
    return ids;
}

function wbRefreshEdgeSelectOptions() {
    const opts = wbEdgeNodeOptions();
    document.querySelectorAll('#wb-edges-tbody tr[data-wb-edge]').forEach((row) => {
        ['.wb-edge-source', '.wb-edge-target'].forEach((sel) => {
            const select = row.querySelector(sel);
            if (!select || select.tagName !== 'SELECT') return;
            const cur = select.value;
            wbFillNodeSelect(select, opts, opts.includes(cur) ? cur : '');
        });
    });
}

function wbFillNodeSelect(select, options, selected) {
    select.innerHTML = '';
    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = '—';
    select.appendChild(empty);
    options.forEach((id) => {
        const o = document.createElement('option');
        o.value = id;
        o.textContent = id;
        if (id === selected) o.selected = true;
        select.appendChild(o);
    });
}

function wbAddEdgeRow(edge) {
    const tbody = document.getElementById('wb-edges-tbody');
    if (!tbody) return;
    const tr = document.createElement('tr');
    tr.dataset.wbEdge = '1';
    tr.className = 'border-b border-white/5';
    const opts = wbEdgeNodeOptions();
    const src = edge?.source || '';
    const tgt = edge?.target || '';
    const pm =
        edge?.port_map && Object.keys(edge.port_map).length
            ? JSON.stringify(edge.port_map, null, 2)
            : '';

    const tdS = document.createElement('td');
    tdS.className = 'py-2 pr-2 align-top';
    const selS = document.createElement('select');
    selS.className =
        'wb-edge-source w-full rounded bg-surface-dark border border-white/10 text-xs text-white px-2 py-1.5 font-mono';
    wbFillNodeSelect(selS, opts, src);
    tdS.appendChild(selS);

    const tdT = document.createElement('td');
    tdT.className = 'py-2 pr-2 align-top';
    const selT = document.createElement('select');
    selT.className =
        'wb-edge-target w-full rounded bg-surface-dark border border-white/10 text-xs text-white px-2 py-1.5 font-mono';
    wbFillNodeSelect(selT, opts, tgt);
    tdT.appendChild(selT);

    const tdPm = document.createElement('td');
    tdPm.className = 'py-2 pr-2 align-top';
    const ta = document.createElement('textarea');
    ta.rows = 2;
    ta.placeholder = '{"source_out":"target_in"}';
    ta.className =
        'wb-edge-portmap w-full rounded bg-black/40 border border-white/10 text-[11px] text-slate-300 px-2 py-1 font-mono resize-y min-h-[2.5rem]';
    ta.value = pm;
    tdPm.appendChild(ta);

    const tdRm = document.createElement('td');
    tdRm.className = 'py-2 align-top w-10';
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'wb-remove-edge text-rose-400 hover:text-rose-300 p-1 rounded hover:bg-white/5';
    rm.title = 'Remove edge';
    rm.innerHTML = '<span class="material-symbols-outlined text-[18px]">close</span>';
    tdRm.appendChild(rm);

    tr.append(tdS, tdT, tdPm, tdRm);
    tbody.appendChild(tr);
    rm.addEventListener('click', () => tr.remove());
}

function wbApplyTemplate(key) {
    const t = WB_TEMPLATES[key];
    if (!t) return;
    wbClearTables();
    t.nodes.forEach((n) => wbAddNodeRow(n));
    t.edges.forEach((e) => wbAddEdgeRow(e));
    wbSetInitialInputs(t.initial_inputs || {});
}

function wbRenderResult(data) {
    const statusEl = document.getElementById('wb-status');
    const sinkEl = document.getElementById('wb-sink-json');
    const nodesEl = document.getElementById('wb-builder-nodes-json');

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

    const fmt = (obj) => {
        try {
            return JSON.stringify(obj, null, 2);
        } catch {
            return String(obj);
        }
    };

    if (sinkEl) sinkEl.textContent = data.success ? fmt(data.sink_outputs || {}) : '';
    if (nodesEl) nodesEl.textContent = data.success ? fmt(data.node_outputs || {}) : fmt(data);
}

async function wbRun() {
    const runBtn = document.getElementById('wb-run-btn');
    const statusEl = document.getElementById('wb-status');
    const { nodes, edges, initial_inputs } = wbGetState();

    if (!nodes.length) {
        if (typeof showToast === 'function') showToast('Add at least one node', 'error');
        return;
    }

    const definition = { nodes, edges };

    if (statusEl) {
        statusEl.textContent = 'Running…';
        statusEl.className =
            'text-xs font-mono px-2 py-1 rounded bg-primary/20 text-primary border border-primary/30';
    }
    if (runBtn) runBtn.disabled = true;

    try {
        const res = await fetch('/api/workflows/run-custom', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ definition, initial_inputs }),
        });
        let data;
        try {
            data = await res.json();
        } catch {
            wbRenderResult({ success: false, error: `Invalid response (${res.status})` });
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
            wbRenderResult({ success: false, error: msg || `HTTP ${res.status}` });
            if (typeof showToast === 'function') showToast(msg || 'Request failed', 'error');
            return;
        }
        wbRenderResult(data);
        if (data.success && typeof showToast === 'function') showToast('Workflow finished', 'success');
        else if (!data.success && typeof showToast === 'function')
            showToast(data.error || 'Workflow failed', 'error');
    } catch (e) {
        console.error(e);
        wbRenderResult({ success: false, error: String(e) });
        if (typeof showToast === 'function') showToast('Network error', 'error');
    } finally {
        if (runBtn) runBtn.disabled = false;
    }
}

function wbWireTemplateSelect() {
    const sel = document.getElementById('wb-template-select');
    if (!sel || sel.dataset.bound === '1') return;
    sel.dataset.bound = '1';
    sel.innerHTML = '';
    WB_TEMPLATE_KEYS.forEach((k) => {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = WB_TEMPLATES[k].label;
        sel.appendChild(opt);
    });
    sel.value = 'uniprot_mw_text';
    sel.addEventListener('change', () => wbApplyTemplate(sel.value));
}

async function initWorkflowBuilder() {
    wbWireTemplateSelect();

    const addNodeBtn = document.getElementById('wb-add-node-btn');
    const addEdgeBtn = document.getElementById('wb-add-edge-btn');
    const paletteType = document.getElementById('wb-palette-type');
    const runBtn = document.getElementById('wb-run-btn');

    if (addNodeBtn && addNodeBtn.dataset.bound !== '1') {
        addNodeBtn.dataset.bound = '1';
        addNodeBtn.addEventListener('click', () => {
            const t = paletteType?.value || wbNodeTypes[0]?.id;
            wbAddNodeRow({ id: wbNextNodeId(), type: t, params: {} });
            wbRefreshEdgeSelectOptions();
        });
    }
    if (addEdgeBtn && addEdgeBtn.dataset.bound !== '1') {
        addEdgeBtn.dataset.bound = '1';
        addEdgeBtn.addEventListener('click', () => {
            wbAddEdgeRow({});
        });
    }
    if (runBtn && runBtn.dataset.bound !== '1') {
        runBtn.dataset.bound = '1';
        runBtn.addEventListener('click', () => wbRun());
    }

    try {
        const res = await fetch('/api/workflows/node-types');
        const data = await res.json();
        wbNodeTypes = data.node_types || [];
        const hint = document.getElementById('wb-node-type-count');
        if (hint) hint.textContent = String(wbNodeTypes.length);

        wbPopulateTypeSelect(document.getElementById('wb-palette-type'), wbNodeTypes[0]?.id);

        const tbody = document.getElementById('wb-nodes-tbody');
        if (tbody && !tbody.querySelector('tr[data-wb-node]')) {
            wbApplyTemplate(document.getElementById('wb-template-select')?.value || 'uniprot_mw_text');
        } else {
            wbRefreshEdgeSelectOptions();
        }
    } catch (e) {
        console.error('Failed to load node types:', e);
        if (typeof showToast === 'function') showToast('Could not load node types', 'error');
    }
}
