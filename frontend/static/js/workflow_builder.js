/**
 * n8n-style workflow canvas: palette, zoom/pan, draggable nodes, port wiring, inspector.
 */

let wbNodeTypes = [];
let wbNodeIdCounter = 0;

/** Last execution snapshot for on-canvas + inspector display */
let wbLastRun = {
    success: null,
    outputsByNodeId: {},
    errorsByNodeId: {},
    globalError: null,
    at: 0,
};

/** Selected wire: { kind:'map', source, target, outP, inP } | { kind:'identity', source, target } */
let wbSelectedWire = null;

const wbState = {
    nodes: [],
    /** key "source\ntarget" -> { source, target, portMap: Record<out,in> } */
    edgeMap: new Map(),
    selectedId: null,
    view: { tx: 48, ty: 48, scale: 1 },
};

let wbLinkDrag = null;
let wbNodeDrag = null;
let wbPanDrag = null;
let wbRafEdges = 0;

function wbNextNodeId() {
    wbNodeIdCounter += 1;
    return `n${wbNodeIdCounter}`;
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

function wbMeta(typeId) {
    return wbNodeTypes.find((t) => t.id === typeId) || null;
}

function wbEdgeKey(source, target) {
    return `${source}\n${target}`;
}

function wbMergeEdge(source, target, partial) {
    const k = wbEdgeKey(source, target);
    const cur = wbState.edgeMap.get(k) || { source, target, portMap: {} };
    Object.assign(cur.portMap, partial);
    wbState.edgeMap.set(k, cur);
}

function wbRemoveEdgesForNode(nodeId) {
    for (const k of [...wbState.edgeMap.keys()]) {
        const e = wbState.edgeMap.get(k);
        if (e.source === nodeId || e.target === nodeId) wbState.edgeMap.delete(k);
    }
}

function wbWireSelected(sw, ev) {
    if (ev) {
        ev.stopPropagation();
        ev.preventDefault();
    }
    wbSelectedWire = sw;
    wbState.selectedId = null;
    wbDrawEdges();
    wbRenderNodes();
    wbSyncInspector();
}

function wbRemoveSelectedWire() {
    if (!wbSelectedWire) return;
    const sw = wbSelectedWire;
    const k = wbEdgeKey(sw.source, sw.target);
    const e = wbState.edgeMap.get(k);
    if (!e) {
        wbSelectedWire = null;
        wbDrawEdges();
        return;
    }
    if (sw.kind === 'identity') {
        wbState.edgeMap.delete(k);
    } else if (sw.kind === 'map') {
        delete e.portMap[sw.outP];
        if (Object.keys(e.portMap).length === 0) wbState.edgeMap.delete(k);
    }
    wbSelectedWire = null;
    wbScheduleRedrawEdges();
    wbSyncInspector();
}

function wbRemoveWireBySpec(source, target, outP, inP, isIdentity) {
    const k = wbEdgeKey(source, target);
    const e = wbState.edgeMap.get(k);
    if (!e) return;
    if (isIdentity) wbState.edgeMap.delete(k);
    else {
        delete e.portMap[outP];
        if (Object.keys(e.portMap).length === 0) wbState.edgeMap.delete(k);
    }
    if (
        wbSelectedWire &&
        wbSelectedWire.source === source &&
        wbSelectedWire.target === target &&
        (isIdentity ? wbSelectedWire.kind === 'identity' : wbSelectedWire.kind === 'map' && wbSelectedWire.outP === outP)
    ) {
        wbSelectedWire = null;
    }
    wbScheduleRedrawEdges();
    wbSyncInspector();
}

function wbRenameNodeId(oldId, newId) {
    const collected = [];
    wbState.edgeMap.forEach((e) => {
        collected.push({
            source: e.source === oldId ? newId : e.source,
            target: e.target === oldId ? newId : e.target,
            portMap: { ...e.portMap },
        });
    });
    wbState.edgeMap.clear();
    collected.forEach((e) => {
        const k = wbEdgeKey(e.source, e.target);
        const ex = wbState.edgeMap.get(k);
        if (ex) Object.assign(ex.portMap, e.portMap);
        else wbState.edgeMap.set(k, { source: e.source, target: e.target, portMap: { ...e.portMap } });
    });
}

function wbSharedPorts(sourceType, targetType) {
    const so = wbMeta(sourceType)?.outputs || {};
    const ti = wbMeta(targetType)?.inputs || {};
    const outKeys = Object.keys(so);
    const inKeys = new Set(Object.keys(ti));
    return outKeys.filter((k) => inKeys.has(k));
}

/** API edges: merge port maps; omit empty port_map */
function wbEdgesForApi() {
    const list = [];
    wbState.edgeMap.forEach((e) => {
        const o = { source: e.source, target: e.target };
        const pm = e.portMap || {};
        if (Object.keys(pm).length) o.port_map = { ...pm };
        list.push(o);
    });
    return list;
}

function wbGetState() {
    const nodes = wbState.nodes.map((n) => ({
        id: n.id.trim(),
        type: n.type,
        params: (() => {
            try {
                return typeof n.params === 'object' && n.params !== null ? { ...n.params } : {};
            } catch {
                return {};
            }
        })(),
    }));

    let initial_inputs = {};
    const ta = document.getElementById('wb-initial-inputs');
    if (ta && ta.value.trim()) {
        try {
            initial_inputs = JSON.parse(ta.value);
        } catch {
            initial_inputs = {};
        }
    }
    return { nodes, edges: wbEdgesForApi(), initial_inputs };
}

function wbSetInitialInputs(obj) {
    const ta = document.getElementById('wb-initial-inputs');
    if (ta) ta.value = JSON.stringify(obj, null, 2);
}

function wbApplyWorldTransform() {
    const w = document.getElementById('wb-canvas-world');
    if (!w) return;
    const { tx, ty, scale } = wbState.view;
    w.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    const zl = document.getElementById('wb-zoom-label');
    if (zl) zl.textContent = `${Math.round(scale * 100)}%`;
}

function wbEventToWorld(e) {
    const world = document.getElementById('wb-canvas-world');
    if (!world) return [0, 0];
    const rect = world.getBoundingClientRect();
    const rw = world.offsetWidth;
    const rh = world.offsetHeight;
    const wx = ((e.clientX - rect.left) / rect.width) * rw;
    const wy = ((e.clientY - rect.top) / rect.height) * rh;
    return [wx, wy];
}

function wbPortWorldPos(nodeId, portName, dir) {
    const nodeEl = document.querySelector(`[data-wb-node="${CSS.escape(nodeId)}"]`);
    const portEl = nodeEl?.querySelector(
        `.wb-port[data-port="${CSS.escape(portName)}"][data-dir="${dir}"]`,
    );
    if (!nodeEl || !portEl) return null;
    const nx = nodeEl.offsetLeft + portEl.offsetLeft + portEl.offsetWidth / 2;
    const ny = nodeEl.offsetTop + portEl.offsetTop + portEl.offsetHeight / 2;
    return [nx, ny];
}

function wbBezierPath(x1, y1, x2, y2) {
    const dx = Math.min(140, Math.max(40, Math.abs(x2 - x1) * 0.45));
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function wbScheduleRedrawEdges() {
    if (wbRafEdges) cancelAnimationFrame(wbRafEdges);
    wbRafEdges = requestAnimationFrame(() => {
        wbRafEdges = 0;
        wbDrawEdges();
    });
}

function wbWireIsSelected(source, target, kind, outP, inP) {
    const sw = wbSelectedWire;
    if (!sw) return false;
    if (sw.source !== source || sw.target !== target) return false;
    if (kind === 'identity') return sw.kind === 'identity';
    return sw.kind === 'map' && sw.outP === outP && sw.inP === inP;
}

function wbDrawEdges() {
    const svg = document.getElementById('wb-edges-svg');
    if (!svg) return;
    svg.innerHTML = '';
    const ns = 'http://www.w3.org/2000/svg';

    wbState.edgeMap.forEach((e) => {
        const srcNode = wbState.nodes.find((n) => n.id === e.source);
        const tgtNode = wbState.nodes.find((n) => n.id === e.target);
        if (!srcNode || !tgtNode) return;

        const pm = e.portMap || {};
        const explicit = Object.keys(pm).length > 0;
        const pairs = explicit
            ? Object.entries(pm)
            : wbSharedPorts(srcNode.type, tgtNode.type).map((k) => [k, k]);

        pairs.forEach(([outP, inP]) => {
            const a = wbPortWorldPos(e.source, outP, 'out');
            const b = wbPortWorldPos(e.target, inP, 'in');
            if (!a || !b) return;
            const d = wbBezierPath(a[0], a[1], b[0], b[1]);
            const vis = document.createElementNS(ns, 'path');
            vis.setAttribute('d', d);
            const sel = explicit
                ? wbWireIsSelected(e.source, e.target, 'map', outP, inP)
                : wbWireIsSelected(e.source, e.target, 'identity', null, null);
            vis.setAttribute('class', sel ? 'wb-wire wb-wire-sel' : 'wb-wire');
            svg.appendChild(vis);

            const hit = document.createElementNS(ns, 'path');
            hit.setAttribute('d', d);
            hit.setAttribute('class', 'wb-wire-hit');
            hit.addEventListener('mousedown', (ev) => {
                ev.stopPropagation();
                if (explicit) wbWireSelected({ kind: 'map', source: e.source, target: e.target, outP, inP }, ev);
                else wbWireSelected({ kind: 'identity', source: e.source, target: e.target }, ev);
            });
            hit.addEventListener('click', (ev) => ev.stopPropagation());
            svg.appendChild(hit);
        });
    });

    if (wbLinkDrag) {
        const path = document.createElementNS(ns, 'path');
        path.setAttribute('d', wbBezierPath(wbLinkDrag.x1, wbLinkDrag.y1, wbLinkDrag.x2, wbLinkDrag.y2));
        path.setAttribute('class', 'wb-wire-temp');
        svg.appendChild(path);
    }
}

function wbFormatPreview(v, max = 80) {
    if (v === null || v === undefined) return String(v);
    if (typeof v === 'string') return v.length <= max ? v : `${v.slice(0, max)}…`;
    if (typeof v === 'number' || typeof v === 'boolean') return String(v);
    if (Array.isArray(v)) {
        const n = v.length;
        if (n === 0) return '[]';
        if (n <= 3 && v.every((x) => typeof x === 'number')) return `[${v.map((x) => Number(x.toFixed?.(4) ?? x)).join(', ')}]`;
        return `[${n} items]`;
    }
    if (typeof v === 'object') return `{${Object.keys(v).length} keys}`;
    return String(v);
}

function wbExtractFailedNodeId(msg) {
    const m = String(msg || '').match(/Node '([^']+)'/);
    return m ? m[1] : null;
}

function wbClearRunOutputs() {
    wbLastRun = { success: null, outputsByNodeId: {}, errorsByNodeId: {}, globalError: null, at: 0 };
    wbRenderNodes();
    wbSyncInspector();
}

function wbCatClass(cat) {
    const c = (cat || 'tool').toLowerCase();
    if (c === 'data') return 'wb-node-cat-data';
    if (c === 'model') return 'wb-node-cat-model';
    if (c === 'agent') return 'wb-node-cat-agent';
    return 'wb-node-cat-tool';
}

function wbRenderNodes() {
    const layer = document.getElementById('wb-nodes-layer');
    if (!layer) return;
    layer.innerHTML = '';

    wbState.nodes.forEach((n) => {
        const meta = wbMeta(n.type);
        const title = meta?.name || n.type;
        const cat = meta?.category || 'tool';
        const ins = meta ? Object.keys(meta.inputs || {}) : [];
        const outs = meta ? Object.keys(meta.outputs || {}) : [];
        const rows = Math.max(ins.length, outs.length, 1);

        const el = document.createElement('div');
        el.className = `wb-node${wbState.selectedId === n.id ? ' wb-node-selected' : ''}`;
        el.dataset.wbNode = n.id;
        el.style.left = `${n.x}px`;
        el.style.top = `${n.y}px`;

        const head = document.createElement('div');
        head.className = `wb-node-header ${wbCatClass(cat)}`;
        head.innerHTML = `<span class="material-symbols-outlined text-[16px] opacity-80">account_tree</span><span class="truncate">${title}</span>`;

        const body = document.createElement('div');
        body.className = 'wb-node-body';

        const idRow = document.createElement('div');
        idRow.className = 'wb-node-id-row';
        idRow.textContent = n.id;
        body.appendChild(idRow);

        for (let i = 0; i < rows; i++) {
            const row = document.createElement('div');
            row.className = 'wb-port-row';
            const inName = ins[i];
            const outName = outs[i];

            const left = document.createElement('div');
            left.style.display = 'flex';
            left.style.alignItems = 'center';
            left.style.flex = '1';
            if (inName) {
                const dot = document.createElement('div');
                dot.className = 'wb-port wb-port-in';
                dot.dataset.port = inName;
                dot.dataset.dir = 'in';
                dot.title = `Input: ${inName}`;
                left.appendChild(dot);
                const lab = document.createElement('span');
                lab.className = 'wb-port-label';
                lab.textContent = `${inName} · ${meta.inputs[inName] || ''}`;
                left.appendChild(lab);
            } else left.appendChild(document.createElement('span'));

            const right = document.createElement('div');
            right.style.display = 'flex';
            right.style.alignItems = 'center';
            right.style.flex = '1';
            right.style.justifyContent = 'flex-end';
            if (outName) {
                const lab = document.createElement('span');
                lab.className = 'wb-port-label wb-port-label-out';
                lab.textContent = `${outName} · ${meta.outputs[outName] || ''}`;
                right.appendChild(lab);
                const dot = document.createElement('div');
                dot.className = 'wb-port wb-port-out';
                dot.dataset.port = outName;
                dot.dataset.dir = 'out';
                dot.title = `Output: ${outName}`;
                right.appendChild(dot);
            } else right.appendChild(document.createElement('span'));

            row.appendChild(left);
            row.appendChild(right);
            body.appendChild(row);
        }

        const runBox = document.createElement('div');
        runBox.className = 'wb-node-run';
        const lr = wbLastRun;
        if (lr.at) {
            const outs = lr.outputsByNodeId[n.id];
            const err = lr.errorsByNodeId[n.id];
            if (lr.success && outs && typeof outs === 'object') {
                const chip = document.createElement('div');
                chip.className = 'wb-run-chip wb-run-ok';
                chip.innerHTML =
                    '<span class="material-symbols-outlined text-[12px]">check_circle</span> Output';
                runBox.appendChild(chip);
                Object.keys(outs).forEach((k) => {
                    const row = document.createElement('div');
                    row.className = 'wb-run-kv';
                    row.innerHTML = `<span class="wb-run-k">${escapeHtml(k)}</span>: ${escapeHtml(wbFormatPreview(outs[k]))}`;
                    runBox.appendChild(row);
                });
            } else if (err) {
                const chip = document.createElement('div');
                chip.className = 'wb-run-chip wb-run-err';
                chip.innerHTML = '<span class="material-symbols-outlined text-[12px]">error</span> Failed';
                runBox.appendChild(chip);
                const msg = document.createElement('div');
                msg.className = 'wb-run-kv text-rose-200/90';
                msg.textContent = wbFormatPreview(err, 200);
                runBox.appendChild(msg);
            } else if (lr.success === false && lr.globalError && wbExtractFailedNodeId(lr.globalError) === n.id) {
                const chip = document.createElement('div');
                chip.className = 'wb-run-chip wb-run-err';
                chip.textContent = 'Failed (see message)';
                runBox.appendChild(chip);
                const msg = document.createElement('div');
                msg.className = 'wb-run-kv text-rose-200/90';
                msg.textContent = wbFormatPreview(lr.globalError, 220);
                runBox.appendChild(msg);
            } else if (lr.success === false && lr.globalError) {
                const chip = document.createElement('div');
                chip.className = 'wb-run-chip wb-run-wait';
                chip.textContent = 'Run failed (global)';
                runBox.appendChild(chip);
            }
        }
        if (runBox.childNodes.length) body.appendChild(runBox);

        el.appendChild(head);
        el.appendChild(body);
        layer.appendChild(el);

        head.addEventListener('mousedown', (ev) => {
            if (ev.button !== 0) return;
            ev.preventDefault();
            wbSelectedWire = null;
            wbDrawEdges();
            wbState.selectedId = n.id;
            wbRenderNodes();
            wbSyncInspector();
            wbNodeDrag = {
                id: n.id,
                sx: ev.clientX,
                sy: ev.clientY,
                ox: n.x,
                oy: n.y,
            };
        });

        el.addEventListener('mousedown', (ev) => {
            if (ev.target.closest('.wb-port') || ev.target.closest('.wb-node-header')) return;
            wbSelectedWire = null;
            wbDrawEdges();
            wbState.selectedId = n.id;
            wbRenderNodes();
            wbSyncInspector();
        });

        el.querySelectorAll('.wb-port-out').forEach((dot) => {
            dot.addEventListener('mousedown', (ev) => {
                ev.stopPropagation();
                ev.preventDefault();
                const p = wbPortWorldPos(n.id, dot.dataset.port, 'out');
                if (!p) return;
                wbLinkDrag = {
                    sourceId: n.id,
                    outPort: dot.dataset.port,
                    x1: p[0],
                    y1: p[1],
                    x2: p[0],
                    y2: p[1],
                };
            });
        });

        el.querySelectorAll('.wb-port-in').forEach((dot) => {
            dot.addEventListener('mouseup', (ev) => {
                if (!wbLinkDrag) return;
                ev.stopPropagation();
                ev.preventDefault();
                const tgtId = n.id;
                const inPort = dot.dataset.port;
                if (wbLinkDrag.sourceId === tgtId) {
                    wbLinkDrag = null;
                    wbDrawEdges();
                    return;
                }
                wbMergeEdge(wbLinkDrag.sourceId, tgtId, { [wbLinkDrag.outPort]: inPort });
                wbLinkDrag = null;
                wbScheduleRedrawEdges();
            });
        });
    });

    wbScheduleRedrawEdges();
}

function wbSyncInspector() {
    const title = document.getElementById('wb-inspector-title');
    const hint = document.getElementById('wb-inspector-hint');
    const wireLabel = document.getElementById('wb-inspector-wire-label');
    const wireBtn = document.getElementById('wb-inspector-remove-wire');
    const fields = document.getElementById('wb-inspector-fields');
    const connWrap = document.getElementById('wb-inspector-connections-wrap');
    const connEl = document.getElementById('wb-inspector-connections');
    const lrWrap = document.getElementById('wb-inspector-lastrun-wrap');
    const lrPre = document.getElementById('wb-inspector-lastrun');
    const idInp = document.getElementById('wb-inspector-id');
    const typeEl = document.getElementById('wb-inspector-type');
    const descEl = document.getElementById('wb-inspector-desc');
    const paramsTa = document.getElementById('wb-inspector-params');
    const n = wbState.nodes.find((x) => x.id === wbState.selectedId);

    if (title) title.textContent = 'Inspector';

    if (!n && wbSelectedWire) {
        if (hint) hint.classList.add('hidden');
        if (fields) fields.classList.add('hidden');
        connWrap?.classList.add('hidden');
        lrWrap?.classList.add('hidden');
        if (wireLabel) {
            wireLabel.classList.remove('hidden');
            if (wbSelectedWire.kind === 'identity')
                wireLabel.textContent = `${wbSelectedWire.source} → ${wbSelectedWire.target} (auto-mapped ports)`;
            else
                wireLabel.textContent = `${wbSelectedWire.source}.${wbSelectedWire.outP} → ${wbSelectedWire.target}.${wbSelectedWire.inP}`;
        }
        wireBtn?.classList.remove('hidden');
        if (title) title.textContent = 'Connection';
        return;
    }

    if (wireLabel) wireLabel.classList.add('hidden');
    wireBtn?.classList.add('hidden');

    if (!n) {
        if (hint) {
            hint.classList.remove('hidden');
            hint.textContent = 'Select a node or click a wire to delete it';
        }
        if (fields) fields.classList.add('hidden');
        connWrap?.classList.add('hidden');
        lrWrap?.classList.add('hidden');
        return;
    }

    if (hint) {
        hint.classList.add('hidden');
        hint.textContent = 'Select a node on the canvas';
    }
    if (fields) fields.classList.remove('hidden');
    if (title) title.textContent = 'Node';

    const meta = wbMeta(n.type);
    if (idInp) idInp.value = n.id;
    if (typeEl) typeEl.textContent = n.type;
    if (descEl) descEl.textContent = meta?.description || '';
    if (paramsTa) paramsTa.value = JSON.stringify(n.params || {}, null, 2);

    if (connEl && connWrap) {
        connEl.innerHTML = '';
        const addRow = (label, source, target, outP, inP, identity) => {
            const row = document.createElement('div');
            row.className = 'wb-conn-row';
            const span = document.createElement('span');
            span.className = 'min-w-0 truncate';
            span.textContent = label;
            row.appendChild(span);
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.textContent = '×';
            btn.title = 'Remove connection';
            btn.addEventListener('click', () => wbRemoveWireBySpec(source, target, outP, inP, identity));
            row.appendChild(btn);
            connEl.appendChild(row);
        };

        wbState.edgeMap.forEach((e) => {
            const pm = e.portMap || {};
            const explicit = Object.keys(pm).length > 0;
            const srcNode = wbState.nodes.find((x) => x.id === e.source);
            const tgtNode = wbState.nodes.find((x) => x.id === e.target);
            if (!srcNode || !tgtNode) return;

            if (e.source === n.id) {
                if (explicit) {
                    Object.entries(pm).forEach(([op, ip]) => {
                        addRow(`${op} → ${e.target}.${ip}`, e.source, e.target, op, ip, false);
                    });
                } else {
                    wbSharedPorts(srcNode.type, tgtNode.type).forEach((k) => {
                        addRow(`${k} → ${e.target}.${k}`, e.source, e.target, k, k, true);
                    });
                }
            }
            if (e.target === n.id) {
                if (explicit) {
                    Object.entries(pm).forEach(([op, ip]) => {
                        addRow(`${e.source}.${op} → ${ip}`, e.source, e.target, op, ip, false);
                    });
                } else {
                    wbSharedPorts(srcNode.type, tgtNode.type).forEach((k) => {
                        addRow(`${e.source}.${k} → ${k}`, e.source, e.target, k, k, true);
                    });
                }
            }
        });

        connWrap.classList.toggle('hidden', connEl.childNodes.length === 0);
    }

    if (lrPre && lrWrap) {
        const jo = wbLastRun.outputsByNodeId[n.id];
        const je = wbLastRun.errorsByNodeId[n.id];
        const gf = wbLastRun.globalError;
        const failedHere = gf && wbExtractFailedNodeId(gf) === n.id;
        if (wbLastRun.at && (jo || je || failedHere)) {
            lrWrap.classList.remove('hidden');
            if (jo) lrPre.textContent = JSON.stringify(jo, null, 2);
            else if (je) lrPre.textContent = String(je);
            else lrPre.textContent = String(gf);
        } else {
            lrWrap.classList.add('hidden');
            lrPre.textContent = '';
        }
    }
}

function wbWireInspectorOnce() {
    const idInp = document.getElementById('wb-inspector-id');
    const paramsTa = document.getElementById('wb-inspector-params');
    if (!idInp || idInp.dataset.wbBound === '1') return;
    idInp.dataset.wbBound = '1';
    idInp.addEventListener('change', () => {
        const n = wbState.nodes.find((x) => x.id === wbState.selectedId);
        if (!n) return;
        const nv = idInp.value.trim();
        if (!nv || wbState.nodes.some((o) => o.id === nv && o !== n)) {
            idInp.value = n.id;
            return;
        }
        const old = n.id;
        n.id = nv;
        wbRenameNodeId(old, nv);
        if (wbState.selectedId === old) wbState.selectedId = nv;
        wbRenderNodes();
        wbSyncInspector();
    });
    paramsTa?.addEventListener('change', () => {
        const n = wbState.nodes.find((x) => x.id === wbState.selectedId);
        if (!n || !paramsTa) return;
        try {
            n.params = JSON.parse(paramsTa.value || '{}');
        } catch {
            paramsTa.value = JSON.stringify(n.params || {}, null, 2);
        }
    });
}

function wbDeleteSelectedNode() {
    const id = wbState.selectedId;
    if (!id) return;
    if (wbSelectedWire && (wbSelectedWire.source === id || wbSelectedWire.target === id)) wbSelectedWire = null;
    wbState.nodes = wbState.nodes.filter((n) => n.id !== id);
    wbRemoveEdgesForNode(id);
    wbState.selectedId = null;
    wbRenderNodes();
    wbSyncInspector();
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
    blank: { label: 'Blank canvas', nodes: [], edges: [], initial_inputs: {} },
    uniprot_mw_text: {
        label: 'UniProt → preprocess → MW → text JSON',
        nodes: [
            { id: 'fetch', type: 'uniprot_fasta', params: {}, x: 80, y: 200 },
            { id: 'prep', type: 'fasta_preprocess', params: {}, x: 400, y: 200 },
            { id: 'mw', type: 'molecular_weight', params: {}, x: 720, y: 200 },
            { id: 'out', type: 'export_text_json', params: {}, x: 1040, y: 200 },
        ],
        edges: [{ source: 'fetch', target: 'prep' }, { source: 'prep', target: 'mw' }, { source: 'mw', target: 'out', port_map: { report: 'text' } }],
        initial_inputs: { fetch: { protein_id: 'P04637' } },
    },
    uniprot_record_dict: {
        label: 'UniProt → biochem record → dict JSON',
        nodes: [
            { id: 'fetch', type: 'uniprot_fasta', params: {}, x: 80, y: 200 },
            { id: 'prep', type: 'fasta_preprocess', params: {}, x: 400, y: 200 },
            { id: 'rec', type: 'compact_biochem_record', params: {}, x: 720, y: 200 },
            { id: 'out', type: 'export_dict_json', params: {}, x: 1040, y: 200 },
        ],
        edges: [
            { source: 'fetch', target: 'prep' },
            { source: 'prep', target: 'rec' },
            { source: 'rec', target: 'out', port_map: { record: 'record' } },
        ],
        initial_inputs: { fetch: { protein_id: 'P04637' } },
    },
    pdb_summary: {
        label: 'RCSB PDB summary',
        nodes: [{ id: 'pdb', type: 'rcsb_entry_summary', params: {}, x: 400, y: 240 }],
        edges: [],
        initial_inputs: { pdb: { pdb_id: '1YCR' } },
    },
    dummy_embed_export: {
        label: 'UniProt → dummy embed + export JSON',
        nodes: [
            { id: 'fetch', type: 'uniprot_fasta', params: {}, x: 80, y: 200 },
            { id: 'prep', type: 'fasta_preprocess', params: {}, x: 400, y: 200 },
            { id: 'emb', type: 'dummy_embedder', params: { dim: 8 }, x: 720, y: 120 },
            { id: 'out', type: 'export_json', params: {}, x: 1040, y: 200 },
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

function wbApplyTemplate(key) {
    const t = WB_TEMPLATES[key];
    if (!t) return;
    wbState.nodes = t.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        params: { ...(n.params || {}) },
        x: n.x ?? 100 + wbState.nodes.length * 40,
        y: n.y ?? 160,
    }));
    wbState.edgeMap.clear();
    t.edges.forEach((e) => {
        const pm = e.port_map || {};
        wbMergeEdge(e.source, e.target, pm);
    });
    wbState.selectedId = null;
    wbSelectedWire = null;
    wbClearRunOutputs();
    wbSetInitialInputs(t.initial_inputs || {});
    wbRenderNodes();
    wbSyncInspector();
    if (wbState.nodes.length) wbFitView();
    else {
        wbState.view = { tx: 48, ty: 48, scale: 1 };
        wbApplyWorldTransform();
    }
}

function wbFitView() {
    if (!wbState.nodes.length) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    wbState.nodes.forEach((n) => {
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x + 260);
        maxY = Math.max(maxY, n.y + 200);
    });
    const vp = document.getElementById('wb-canvas-viewport');
    if (!vp) return;
    const pw = vp.clientWidth;
    const ph = vp.clientHeight;
    const bw = maxX - minX + 120;
    const bh = maxY - minY + 120;
    const s = Math.min(pw / bw, ph / bh, 1.2);
    wbState.view.scale = Math.max(0.35, Math.min(1.25, s));
    wbState.view.tx = (pw - bw * wbState.view.scale) / 2 - minX * wbState.view.scale + 40 * wbState.view.scale;
    wbState.view.ty = (ph - bh * wbState.view.scale) / 2 - minY * wbState.view.scale + 20 * wbState.view.scale;
    wbApplyWorldTransform();
    wbScheduleRedrawEdges();
}

function wbAddNodeAt(typeId, wx, wy) {
    const id = wbNextNodeId();
    wbState.nodes.push({
        id,
        type: typeId,
        params: { ...(wbMeta(typeId)?.default_params || {}) },
        x: Math.round(wx - 120),
        y: Math.round(wy - 40),
    });
    wbState.selectedId = id;
    wbRenderNodes();
    wbSyncInspector();
}

function wbRenderPalette() {
    const list = document.getElementById('wb-palette-list');
    const q = (document.getElementById('wb-palette-search')?.value || '').toLowerCase().trim();
    if (!list) return;
    list.innerHTML = '';
    const byCat = {};
    wbNodeTypes.forEach((t) => {
        if (q && !t.name.toLowerCase().includes(q) && !t.id.toLowerCase().includes(q)) return;
        const c = t.category || 'tool';
        if (!byCat[c]) byCat[c] = [];
        byCat[c].push(t);
    });
    const order = ['data', 'tool', 'model', 'agent'];
    order.forEach((cat) => {
        const items = byCat[cat];
        if (!items?.length) return;
        const h = document.createElement('div');
        h.className = 'wb-palette-cat';
        h.textContent = cat;
        list.appendChild(h);
        items.sort((a, b) => a.name.localeCompare(b.name));
        items.forEach((t) => {
            const row = document.createElement('div');
            row.className = 'wb-palette-item';
            row.innerHTML = `<span class="material-symbols-outlined text-primary text-[18px]">add_circle</span><span class="min-w-0"><span class="block truncate font-medium">${t.name}</span><span class="block truncate text-[9px] text-text-subtle font-mono">${t.id}</span></span>`;
            row.addEventListener('click', () => {
                const [wx, wy] = wbEventToWorld({
                    clientX: document.getElementById('wb-canvas-viewport').getBoundingClientRect().left + 400,
                    clientY: document.getElementById('wb-canvas-viewport').getBoundingClientRect().top + 220,
                });
                wbAddNodeAt(t.id, wx, wy);
            });
            list.appendChild(row);
        });
    });
}

function wbRenderResult(data) {
    const statusEl = document.getElementById('wb-status');
    const sinkEl = document.getElementById('wb-sink-json');
    const nodesEl = document.getElementById('wb-builder-nodes-json');

    if (data.success) {
        wbLastRun = {
            success: true,
            outputsByNodeId: { ...(data.node_outputs || {}) },
            errorsByNodeId: {},
            globalError: null,
            at: Date.now(),
        };
    } else {
        const gid = wbExtractFailedNodeId(data.error);
        const errMap = {};
        if (gid) errMap[gid] = data.error || 'Error';
        wbLastRun = {
            success: false,
            outputsByNodeId: { ...(data.node_outputs || {}) },
            errorsByNodeId: errMap,
            globalError: data.error || null,
            at: Date.now(),
        };
    }

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

    wbRenderNodes();
    wbSyncInspector();
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

function wbWireCanvasInteractions() {
    const vp = document.getElementById('wb-canvas-viewport');
    if (!vp || vp.dataset.wbWired === '1') return;
    vp.dataset.wbWired = '1';

    vp.addEventListener('mousedown', (e) => {
        if (e.target.closest('.wb-node') || e.target.closest('button')) return;
        if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
            e.preventDefault();
            wbPanDrag = { sx: e.clientX, sy: e.clientY, tx: wbState.view.tx, ty: wbState.view.ty };
            vp.classList.add('wb-panning');
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (wbNodeDrag) {
            const n = wbState.nodes.find((x) => x.id === wbNodeDrag.id);
            if (n) {
                const dx = (e.clientX - wbNodeDrag.sx) / wbState.view.scale;
                const dy = (e.clientY - wbNodeDrag.sy) / wbState.view.scale;
                n.x = Math.round(wbNodeDrag.ox + dx);
                n.y = Math.round(wbNodeDrag.oy + dy);
                const el = document.querySelector(`[data-wb-node="${CSS.escape(n.id)}"]`);
                if (el) {
                    el.style.left = `${n.x}px`;
                    el.style.top = `${n.y}px`;
                }
                wbScheduleRedrawEdges();
            }
        }
        if (wbPanDrag) {
            wbState.view.tx = wbPanDrag.tx + (e.clientX - wbPanDrag.sx);
            wbState.view.ty = wbPanDrag.ty + (e.clientY - wbPanDrag.sy);
            wbApplyWorldTransform();
            wbScheduleRedrawEdges();
        }
        if (wbLinkDrag) {
            const [x2, y2] = wbEventToWorld(e);
            wbLinkDrag.x2 = x2;
            wbLinkDrag.y2 = y2;
            wbDrawEdges();
        }
    });

    window.addEventListener('mouseup', () => {
        if (wbNodeDrag) {
            wbNodeDrag = null;
            wbScheduleRedrawEdges();
        }
        if (wbPanDrag) {
            wbPanDrag = null;
            vp.classList.remove('wb-panning');
        }
        if (wbLinkDrag) {
            wbLinkDrag = null;
            wbDrawEdges();
        }
    });

    vp.addEventListener(
        'wheel',
        (e) => {
            if (!e.ctrlKey && !e.metaKey) return;
            e.preventDefault();
            const rect = vp.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const old = wbState.view.scale;
            const delta = e.deltaY > 0 ? 0.92 : 1.08;
            const ns = Math.min(1.8, Math.max(0.25, old * delta));
            const k = ns / old;
            wbState.view.tx = mx - (mx - wbState.view.tx) * k;
            wbState.view.ty = my - (my - wbState.view.ty) * k;
            wbState.view.scale = ns;
            wbApplyWorldTransform();
            wbScheduleRedrawEdges();
        },
        { passive: false },
    );

    vp.addEventListener('click', (e) => {
        if (e.target.closest('.wb-node')) return;
        wbState.selectedId = null;
        wbSelectedWire = null;
        wbDrawEdges();
        wbRenderNodes();
        wbSyncInspector();
    });
}

function wbWireZoomButtons() {
    document.getElementById('wb-zoom-in')?.addEventListener('click', () => {
        wbState.view.scale = Math.min(1.8, wbState.view.scale * 1.12);
        wbApplyWorldTransform();
        wbScheduleRedrawEdges();
    });
    document.getElementById('wb-zoom-out')?.addEventListener('click', () => {
        wbState.view.scale = Math.max(0.25, wbState.view.scale / 1.12);
        wbApplyWorldTransform();
        wbScheduleRedrawEdges();
    });
    document.getElementById('wb-zoom-fit')?.addEventListener('click', () => wbFitView());
}

function wbWireBottomTabs() {
    document.querySelectorAll('.wb-bottom-tab').forEach((btn) => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.wbTab;
            document.querySelectorAll('.wb-bottom-tab').forEach((b) => {
                const on = b.dataset.wbTab === tab;
                b.classList.toggle('text-white', on);
                b.classList.toggle('text-text-subtle', !on);
                b.classList.toggle('border-primary', on);
                b.classList.toggle('bg-white/5', on);
                b.classList.toggle('border-transparent', !on);
            });
            document.querySelectorAll('.wb-bottom-panel').forEach((p) => p.classList.add('hidden'));
            if (tab === 'inputs') document.getElementById('wb-initial-inputs')?.classList.remove('hidden');
            if (tab === 'sink') document.getElementById('wb-sink-json')?.classList.remove('hidden');
            if (tab === 'nodes') document.getElementById('wb-builder-nodes-json')?.classList.remove('hidden');
        });
    });
}

async function initWorkflowBuilder() {
    wbWireTemplateSelect();
    wbWireCanvasInteractions();
    wbWireZoomButtons();
    wbWireBottomTabs();
    wbWireInspectorOnce();

    document.getElementById('wb-inspector-delete')?.addEventListener('click', () => wbDeleteSelectedNode());
    document.getElementById('wb-inspector-remove-wire')?.addEventListener('click', () => wbRemoveSelectedWire());
    document.getElementById('wb-clear-run')?.addEventListener('click', () => wbClearRunOutputs());

    document.getElementById('wb-palette-search')?.addEventListener('input', () => wbRenderPalette());

    const runBtn = document.getElementById('wb-run-btn');
    if (runBtn && runBtn.dataset.bound !== '1') {
        runBtn.dataset.bound = '1';
        runBtn.addEventListener('click', () => wbRun());
    }

    window.addEventListener('keydown', (e) => {
        const view = document.getElementById('workflow-builder-view');
        if (!view || view.classList.contains('hidden')) return;
        if (e.key !== 'Delete' && e.key !== 'Backspace') return;
        const t = e.target;
        if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA') return;
        if (wbSelectedWire) {
            e.preventDefault();
            wbRemoveSelectedWire();
            return;
        }
        if (!wbState.selectedId) return;
        e.preventDefault();
        wbDeleteSelectedNode();
    });

    try {
        const res = await fetch('/api/workflows/node-types');
        const data = await res.json();
        wbNodeTypes = data.node_types || [];
        const hint = document.getElementById('wb-node-type-count');
        if (hint) hint.textContent = String(wbNodeTypes.length);

        wbRenderPalette();

        if (!wbState.nodes.length) {
            wbApplyTemplate(document.getElementById('wb-template-select')?.value || 'uniprot_mw_text');
        } else {
            wbApplyWorldTransform();
            wbRenderNodes();
            wbSyncInspector();
        }
    } catch (e) {
        console.error('Failed to load node types:', e);
        if (typeof showToast === 'function') showToast('Could not load node types', 'error');
    }
}
