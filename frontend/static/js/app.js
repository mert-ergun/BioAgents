/**
 * BioAgents UI — Full Implementation
 * Multi-Agent System for Computational Biology
 */

// =====================================================
// CONFIGURATION
// =====================================================

const CONFIG = {
    apiBaseUrl: window.location.origin,
    wsUrl: `ws://${window.location.host}/ws`,
    reconnectDelay: 3000,
    maxReconnectAttempts: 5,
    storageKey: 'bioagents_sessions',
};

// Agent configuration
const AGENTS = {
    supervisor: { label: 'Orchestrator', icon: 'alt_route', color: 'slate', description: 'Routing tasks...' },
    research: { label: 'LitSearcher', icon: 'menu_book', color: 'blue', description: 'Searching literature...' },
    analysis: { label: 'Analyzer', icon: 'analytics', color: 'emerald', description: 'Analyzing data...' },
    coder: { label: 'CodeGen', icon: 'code', color: 'amber', description: 'Generating code...' },
    report: { label: 'Reporter', icon: 'description', color: 'pink', description: 'Writing report...' },
    tool_builder: { label: 'ToolBuilder', icon: 'build', color: 'indigo', description: 'Building tools...' },
    protein_design: { label: 'ProteinFolder', icon: 'polymer', color: 'primary', description: 'Folding protein...' },
    critic: { label: 'Validator', icon: 'fact_check', color: 'rose', description: 'Validating results...' },
    ml: { label: 'ML-Expert', icon: 'psychology', color: 'purple', description: 'Training ML model...', isTraining: true },
    dl: { label: 'DL-Specialist', icon: 'memory', color: 'cyan', description: 'Designing neural network...', isTraining: true },
};

// =====================================================
// STATE MANAGEMENT
// =====================================================

const state = {
    ws: null,
    isConnected: false,
    isProcessing: false,
    reconnectAttempts: 0,
    currentSessionId: null,
    sessions: [],
    messages: [],
    auditLog: [],
    artifacts: [],
    codeSteps: [],
    currentAgent: null,
    agentProgress: {},
    nglStage: null,
    currentPdbContent: null,
    settings: {
        autoScroll: true,
        notifications: true,
        theme: 'dark',
        apiKeys: {
            openai: '',
            gemini: '',
        },
    },
    notifications: [],
    sidebarOpen: true,
    // Reference tracking
    references: new Map(),  // ref_id -> Reference object
    referencesByMessage: new Map(),  // message_id -> [ref_ids]
    referenceCounter: 0,
    displayNumbers: new Map(),  // ref_id -> display number
};

// =====================================================
// REFERENCE MANAGEMENT
// =====================================================

function addReference(reference) {
    // Check if reference already exists
    if (state.references.has(reference.id)) {
        return reference.id;
    }

    // Add reference to map
    state.references.set(reference.id, reference);

    // Assign display number if not already assigned
    if (!state.displayNumbers.has(reference.id)) {
        state.referenceCounter++;
        state.displayNumbers.set(reference.id, state.referenceCounter);
    }

    return reference.id;
}

function getReferenceNumber(refId) {
    return state.displayNumbers.get(refId) || null;
}

function getAllReferences() {
    return Array.from(state.references.values())
        .sort((a, b) => {
            const numA = state.displayNumbers.get(a.id) || 0;
            const numB = state.displayNumbers.get(b.id) || 0;
            return numA - numB;
        });
}

function getReferencesByType(type) {
    return getAllReferences().filter(ref => ref.type === type);
}

function attachReferencesToMessage(messageId, refIds) {
    state.referencesByMessage.set(messageId, refIds);
}

function getMessageReferences(messageId) {
    return state.referencesByMessage.get(messageId) || [];
}

// =====================================================
// CITATION PARSING
// =====================================================

function parseCitationsInContent(content, messageRefs) {
    if (!content || !messageRefs || messageRefs.length === 0) {
        return content;
    }

    // Build a mapping of reference IDs for quick lookup
    const refMap = new Map();
    messageRefs.forEach(ref => {
        refMap.set(ref.id, ref);
    });

    let result = content;

    // STEP 1: Convert inline academic citations to numbered format
    // Pattern: (Source Name, Year) or (Source Name et al., Year)
    const academicCitationPattern = /\(([A-Za-z\s&\-]+(?:\set\sal\.)?),?\s*(\d{4})\)/g;
    const citationsToReplace = [];

    let match;
    while ((match = academicCitationPattern.exec(content)) !== null) {
        const fullMatch = match[0];
        const source = match[1].trim();
        const year = match[2];

        // Try to find matching reference
        const matchingRef = messageRefs.find(ref => {
            if (ref.type === 'paper') {
                // Match by journal name or first author
                if (ref.journal && source.includes(ref.journal)) return true;
                if (ref.authors && ref.authors.length > 0 && source.includes(ref.authors[0])) return true;
            }
            // Match by title containing source keywords
            if (ref.title && ref.title.toLowerCase().includes(source.toLowerCase())) return true;
            if (source.toLowerCase().includes(ref.title.toLowerCase())) return true;

            return false;
        });

        if (matchingRef) {
            const displayNum = state.displayNumbers.get(matchingRef.id);
            if (displayNum) {
                citationsToReplace.push({
                    original: fullMatch,
                    replacement: `<sup class="citation" data-ref-id="${matchingRef.id}" data-ref-num="${displayNum}">${displayNum}</sup>`,
                    position: match.index
                });
            }
        }
    }

    // Apply replacements in reverse order to maintain positions
    for (let i = citationsToReplace.length - 1; i >= 0; i--) {
        const {original, replacement, position} = citationsToReplace[i];
        result = result.substring(0, position) + replacement + result.substring(position + original.length);
    }

    // STEP 2: Handle existing [1], [2] format citations
    const citationPattern = /\[(\d+(?:[-,]\d+)*)\]/g;
    const matches = [...result.matchAll(citationPattern)];

    // Process in reverse to maintain correct positions
    for (let i = matches.length - 1; i >= 0; i--) {
        const match = matches[i];
        const fullMatch = match[0];
        const numbers = match[1];
        const position = match.index;

        // Parse the numbers (handle ranges like 1-3 and lists like 1,2)
        const refNumbers = [];
        const parts = numbers.split(',');
        parts.forEach(part => {
            if (part.includes('-')) {
                const [start, end] = part.split('-').map(Number);
                for (let n = start; n <= end; n++) {
                    refNumbers.push(n);
                }
            } else {
                refNumbers.push(Number(part));
            }
        });

        // Find matching reference IDs
        const matchingRefs = messageRefs.filter(ref => {
            const displayNum = state.displayNumbers.get(ref.id);
            return displayNum && refNumbers.includes(displayNum);
        });

        if (matchingRefs.length > 0) {
            // Create clickable citation spans
            const citationHtml = refNumbers.map(num => {
                const ref = messageRefs.find(r => state.displayNumbers.get(r.id) === num);
                const refId = ref ? ref.id : '';
                return `<sup class="citation" data-ref-id="${refId}" data-ref-num="${num}">${num}</sup>`;
            }).join(',');

            result = result.substring(0, position) + citationHtml + result.substring(position + fullMatch.length);
        }
    }

    return result;
}

function renderReferenceHoverCard(refId, targetElement) {
    const ref = state.references.get(refId);
    if (!ref) return null;

    const displayNum = state.displayNumbers.get(refId);

    // Create hover card HTML based on reference type
    let cardContent = '';

    if (ref.type === 'paper') {
        const authors = ref.authors && ref.authors.length > 0
            ? ref.authors.join(', ')
            : 'Unknown authors';
        const year = ref.year || 'n.d.';
        const journal = ref.journal ? ` <em>${ref.journal}</em>` : '';

        cardContent = `
            <div class="reference-hover-card" id="refCard-${refId}">
                <div class="flex items-start justify-between mb-2">
                    <span class="text-xs font-bold text-primary">[${displayNum}] Paper</span>
                    <button onclick="closeHoverCard('${refId}')" class="text-slate-400 hover:text-white">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
                <div class="text-sm font-medium text-white mb-1">${ref.title}</div>
                <div class="text-xs text-slate-400 mb-2">${authors} (${year})${journal}</div>
                ${ref.doi ? `<div class="text-xs text-primary mb-1">DOI: ${ref.doi}</div>` : ''}
                ${ref.pmid ? `<div class="text-xs text-primary mb-1">PMID: ${ref.pmid}</div>` : ''}
                ${ref.url ? `<a href="${ref.url}" target="_blank" class="text-xs text-primary hover:underline">View paper →</a>` : ''}
            </div>
        `;
    } else if (ref.type === 'database') {
        const ids = ref.identifiers && ref.identifiers.length > 0
            ? ref.identifiers.join(', ')
            : '';

        cardContent = `
            <div class="reference-hover-card" id="refCard-${refId}">
                <div class="flex items-start justify-between mb-2">
                    <span class="text-xs font-bold text-primary">[${displayNum}] Database</span>
                    <button onclick="closeHoverCard('${refId}')" class="text-slate-400 hover:text-white">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
                <div class="text-sm font-medium text-white mb-1">${ref.title}</div>
                <div class="text-xs text-slate-400 mb-2">${ref.database_name}</div>
                ${ids ? `<div class="text-xs text-slate-300 mb-2">IDs: ${ids}</div>` : ''}
                ${ref.url ? `<a href="${ref.url}" target="_blank" class="text-xs text-primary hover:underline">View entry →</a>` : ''}
            </div>
        `;
    } else if (ref.type === 'structure') {
        const source = ref.source || 'Unknown';
        const method = ref.method ? ` (${ref.method})` : '';

        cardContent = `
            <div class="reference-hover-card" id="refCard-${refId}">
                <div class="flex items-start justify-between mb-2">
                    <span class="text-xs font-bold text-primary">[${displayNum}] Structure</span>
                    <button onclick="closeHoverCard('${refId}')" class="text-slate-400 hover:text-white">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
                <div class="text-sm font-medium text-white mb-1">${ref.title}</div>
                <div class="text-xs text-slate-400 mb-2">${source}${method}</div>
                ${ref.pdb_id ? `<div class="text-xs text-slate-300 mb-1">PDB: ${ref.pdb_id}</div>` : ''}
                ${ref.uniprot_id ? `<div class="text-xs text-slate-300 mb-1">UniProt: ${ref.uniprot_id}</div>` : ''}
                ${ref.url ? `<a href="${ref.url}" target="_blank" class="text-xs text-primary hover:underline">View structure →</a>` : ''}
            </div>
        `;
    } else if (ref.type === 'tool') {
        cardContent = `
            <div class="reference-hover-card" id="refCard-${refId}">
                <div class="flex items-start justify-between mb-2">
                    <span class="text-xs font-bold text-primary">[${displayNum}] Tool</span>
                    <button onclick="closeHoverCard('${refId}')" class="text-slate-400 hover:text-white">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
                <div class="text-sm font-medium text-white mb-1">${ref.title}</div>
                ${ref.description ? `<div class="text-xs text-slate-400 mb-2">${ref.description}</div>` : ''}
                ${ref.source_url ? `<a href="${ref.source_url}" target="_blank" class="text-xs text-primary hover:underline">View tool →</a>` : ''}
            </div>
        `;
    } else if (ref.type === 'artifact') {
        const size = ref.size ? ` (${formatFileSize(ref.size)})` : '';

        cardContent = `
            <div class="reference-hover-card" id="refCard-${refId}">
                <div class="flex items-start justify-between mb-2">
                    <span class="text-xs font-bold text-primary">[${displayNum}] Artifact</span>
                    <button onclick="closeHoverCard('${refId}')" class="text-slate-400 hover:text-white">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
                <div class="text-sm font-medium text-white mb-1">${ref.file_name}</div>
                <div class="text-xs text-slate-400 mb-2">${ref.file_type || 'File'}${size}</div>
                ${ref.agent ? `<div class="text-xs text-slate-300 mb-1">Generated by: ${ref.agent}</div>` : ''}
            </div>
        `;
    } else {
        // Generic reference
        cardContent = `
            <div class="reference-hover-card" id="refCard-${refId}">
                <div class="flex items-start justify-between mb-2">
                    <span class="text-xs font-bold text-primary">[${displayNum}] Reference</span>
                    <button onclick="closeHoverCard('${refId}')" class="text-slate-400 hover:text-white">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
                <div class="text-sm font-medium text-white mb-1">${ref.title}</div>
                ${ref.url ? `<a href="${ref.url}" target="_blank" class="text-xs text-primary hover:underline">View →</a>` : ''}
            </div>
        `;
    }

    return cardContent;
}

function showHoverCard(refId, event) {
    // Remove any existing hover cards
    document.querySelectorAll('.reference-hover-card').forEach(card => card.remove());

    const cardHtml = renderReferenceHoverCard(refId, event.target);
    if (!cardHtml) return;

    // Create and position the card
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = cardHtml;
    const card = tempDiv.firstElementChild;

    document.body.appendChild(card);

    // Position near the citation
    const rect = event.target.getBoundingClientRect();
    card.style.position = 'fixed';
    card.style.left = `${rect.left}px`;
    card.style.top = `${rect.bottom + 5}px`;
    card.style.zIndex = '1000';

    // Adjust if off-screen
    const cardRect = card.getBoundingClientRect();
    if (cardRect.right > window.innerWidth) {
        card.style.left = `${window.innerWidth - cardRect.width - 10}px`;
    }
    if (cardRect.bottom > window.innerHeight) {
        card.style.top = `${rect.top - cardRect.height - 5}px`;
    }
}

function closeHoverCard(refId) {
    const card = document.getElementById(`refCard-${refId}`);
    if (card) card.remove();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function updateReferencePanel() {
    const referenceList = document.getElementById('referenceList');
    const referenceCount = document.getElementById('referenceCount');

    if (!referenceList || !referenceCount) return;

    const allRefs = getAllReferences();

    if (allRefs.length === 0) {
        referenceList.innerHTML = `
            <div class="text-center py-6 text-text-subtle text-xs">
                <span class="material-symbols-outlined text-2xl text-border-dark mb-2 block">article</span>
                No references yet
            </div>
        `;
        referenceCount.textContent = '0 refs';
        return;
    }

    referenceCount.textContent = `${allRefs.length} ref${allRefs.length !== 1 ? 's' : ''}`;

    // Group references by type
    const byType = {
        paper: [],
        database: [],
        tool: [],
        structure: [],
        artifact: []
    };

    allRefs.forEach(ref => {
        if (byType[ref.type]) {
            byType[ref.type].push(ref);
        }
    });

    let html = '';

    // Render each type group
    Object.entries(byType).forEach(([type, refs]) => {
        if (refs.length === 0) return;

        const typeLabels = {
            paper: '📄 Papers',
            database: '🗃️ Databases',
            tool: '🔧 Tools',
            structure: '🧬 Structures',
            artifact: '📁 Artifacts'
        };

        html += `<div class="mb-3">`;
        html += `<div class="text-[9px] font-bold text-slate-600 uppercase tracking-wider mb-1.5">${typeLabels[type]}</div>`;

        refs.forEach(ref => {
            const displayNum = state.displayNumbers.get(ref.id);
            const truncTitle = ref.title.length > 50 ? ref.title.substring(0, 50) + '...' : ref.title;

            let subtitle = '';
            if (type === 'paper' && ref.authors && ref.authors.length > 0) {
                subtitle = `${ref.authors[0]} ${ref.year ? `(${ref.year})` : ''}`;
            } else if (type === 'database') {
                subtitle = ref.database_name || '';
            } else if (type === 'structure') {
                subtitle = ref.source || '';
            } else if (type === 'tool') {
                subtitle = ref.tool_name || '';
            } else if (type === 'artifact') {
                subtitle = ref.file_type || '';
            }

            html += `
                <div class="reference-item p-2 rounded bg-surface-dark border border-white/5 hover:border-primary/20 mb-1.5 transition-all" data-ref-id="${ref.id}" data-ref-num="${displayNum}">
                    <div class="flex items-start gap-2">
                        <span class="citation-number text-[10px]">[${displayNum}]</span>
                        <div class="flex-1 min-w-0">
                            <div class="text-[11px] font-medium text-white truncate">${truncTitle}</div>
                            ${subtitle ? `<div class="text-[10px] text-slate-500 truncate">${subtitle}</div>` : ''}
                            ${ref.url ? `<a href="${ref.url}" target="_blank" class="text-[10px] text-primary hover:underline" onclick="event.stopPropagation()">View →</a>` : ''}
                        </div>
                    </div>
                </div>
            `;
        });

        html += `</div>`;
    });

    referenceList.innerHTML = html;

    // Add click handlers
    document.querySelectorAll('.reference-item').forEach(item => {
        item.addEventListener('click', (e) => {
            const refId = e.currentTarget.dataset.refId;
            if (refId) {
                // Show expanded view in a modal or highlight
                showReferenceDetails(refId);
            }
        });
    });
}

function scrollToReference(refNum) {
    // Switch to references tab
    const referencesTab = document.querySelector('[data-tab="references"]');
    if (referencesTab) {
        referencesTab.click();
    }

    // Scroll to the reference item
    setTimeout(() => {
        const refItem = document.querySelector(`[data-ref-num="${refNum}"]`);
        if (refItem) {
            refItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
            refItem.classList.add('ring-2', 'ring-primary');
            setTimeout(() => {
                refItem.classList.remove('ring-2', 'ring-primary');
            }, 2000);
        }
    }, 100);
}

function showReferenceDetails(refId) {
    const ref = state.references.get(refId);
    if (!ref) return;

    // For now, just show the hover card
    const fakeEvent = {
        target: document.querySelector(`[data-ref-id="${refId}"]`),
        clientX: window.innerWidth / 2,
        clientY: window.innerHeight / 2
    };

    if (fakeEvent.target) {
        showHoverCard(refId, fakeEvent);
    }
}

// =====================================================
// DOM ELEMENTS
// =====================================================

const elements = {};

function cacheElements() {
    elements.chatMessages = document.getElementById('chatMessages');
    elements.welcomeState = document.getElementById('welcomeState');
    elements.queryInput = document.getElementById('queryInput');
    elements.submitBtn = document.getElementById('submitBtn');
    elements.newResearchBtn = document.getElementById('newResearchBtn');
    elements.statusDot = document.getElementById('statusDot');
    elements.statusText = document.getElementById('statusText');
    elements.projectTitle = document.getElementById('projectTitle');
    elements.agentList = document.getElementById('agentList');
    elements.structureViewer = document.getElementById('structureViewer');
    elements.viewerPlaceholder = document.getElementById('viewerPlaceholder');
    elements.viewerContainer = document.getElementById('viewerContainer');
    elements.artifactsList = document.getElementById('artifactsList');
    elements.auditLog = document.getElementById('auditLog');
    elements.terminalContent = document.getElementById('terminalContent');
    elements.toastContainer = document.getElementById('toastContainer');
    elements.quickChips = document.getElementById('quickChips');
    elements.recentSessions = document.getElementById('recentSessions');
    elements.sidebar = document.querySelector('aside');
    elements.mobileMenuBtn = document.querySelector('button.lg\\:hidden');
}

// =====================================================
// INITIALIZATION
// =====================================================

document.addEventListener('DOMContentLoaded', () => {
    cacheElements();
    initializeApp();
});

function initializeApp() {
    // Load saved data
    loadFromStorage();

    // Initialize all components
    initInputHandlers();
    initExampleChips();
    initTabSwitcher();
    initSidebarNavigation();
    initMobileMenu();
    initViewerControls();
    initExportSession();
    initNotifications();
    initSettings();
    initFileAttachment();
    initVoiceInput();

    // Render initial state
    renderAgentList();
    renderRecentSessions();
    renderNotificationBadge();

    // Connect WebSocket
    connectWebSocket();

    // Configure markdown
    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true, gfm: true });
    }

    // Auto-resize textarea
    elements.queryInput?.addEventListener('input', autoResizeTextarea);

    // Start new session if none exists
    if (!state.currentSessionId) {
        startNewSession();
    }

    logTerminal('System initialized');
    logTerminal('GPU Acceleration: Ready');
    logTerminal('Awaiting user input...');
}

// =====================================================
// SESSION MANAGEMENT
// =====================================================

async function startNewSession() {
    const sessionId = 'session_' + Date.now();
    const session = {
        id: sessionId,
        title: 'New Research Session',
        createdAt: new Date().toISOString(),
        messages: [],
        artifacts: [],
        auditLog: [],
    };

    state.currentSessionId = sessionId;
    state.messages = [];
    state.artifacts = [];  // Clear artifacts for new session
    state.auditLog = [];
    state.codeSteps = [];
    state.currentAgent = null;
    state.currentPdbContent = null;  // Clear loaded structure

    // Clear the 3D viewer
    if (state.nglStage) {
        state.nglStage.removeAllComponents();
    }

    // Clear artifacts on server
    try {
        await fetch(`${CONFIG.apiBaseUrl}/api/clear-artifacts`, { method: 'POST' });
    } catch (e) {
        console.log('Could not clear server artifacts');
    }

    // Add to sessions list
    state.sessions.unshift(session);
    if (state.sessions.length > 20) {
        state.sessions = state.sessions.slice(0, 20);
    }

    saveToStorage();
    renderRecentSessions();
    renderArtifacts();  // Re-render empty artifacts panel
    resetChatUI();

    elements.projectTitle.textContent = 'New Research Session';
    logTerminal('New session started: ' + sessionId);
}

function loadSession(sessionId) {
    const session = state.sessions.find(s => s.id === sessionId);
    if (!session) return;

    state.currentSessionId = sessionId;
    state.messages = session.messages || [];
    state.artifacts = session.artifacts || [];
    state.auditLog = session.auditLog || [];
    state.codeSteps = session.codeSteps || [];

    elements.projectTitle.textContent = session.title;

    // Re-render everything
    resetChatUI();
    state.messages.forEach(msg => {
        if (msg.role === 'user') {
            renderUserMessage(msg);
        } else {
            renderAssistantMessage(msg);
        }
    });

    renderArtifacts();
    updateAuditLog(state.auditLog);
    renderCodeExecution(state.codeSteps, state.currentAgent);
    renderRecentSessions();

    showToast('Session loaded', 'success');
    logTerminal('Loaded session: ' + session.title);
}

function saveCurrentSession() {
    const session = state.sessions.find(s => s.id === state.currentSessionId);
    if (session) {
        session.messages = state.messages;
        session.artifacts = state.artifacts;
        session.auditLog = state.auditLog;
        session.codeSteps = state.codeSteps;

        // Update title from first user message
        if (state.messages.length > 0) {
            const firstUserMsg = state.messages.find(m => m.role === 'user');
            if (firstUserMsg) {
                session.title = firstUserMsg.content.substring(0, 40) + (firstUserMsg.content.length > 40 ? '...' : '');
            }
        }

        saveToStorage();
        renderRecentSessions();
    }
}

function deleteSession(sessionId) {
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    saveToStorage();
    renderRecentSessions();

    if (sessionId === state.currentSessionId) {
        startNewSession();
    }

    showToast('Session deleted', 'success');
}

// =====================================================
// STORAGE
// =====================================================

function saveToStorage() {
    try {
        const data = {
            sessions: state.sessions,
            currentSessionId: state.currentSessionId,
            settings: state.settings,
        };
        localStorage.setItem(CONFIG.storageKey, JSON.stringify(data));
    } catch (e) {
        console.error('Storage save error:', e);
    }
}

function loadFromStorage() {
    try {
        const data = localStorage.getItem(CONFIG.storageKey);
        if (data) {
            const parsed = JSON.parse(data);
            state.sessions = parsed.sessions || [];
            state.currentSessionId = parsed.currentSessionId;
            state.settings = { ...state.settings, ...parsed.settings };
            if (!state.settings.apiKeys) {
                state.settings.apiKeys = { openai: '', gemini: '' };
            }

            // Load current session data (but NOT artifacts - they should be regenerated)
            const session = state.sessions.find(s => s.id === state.currentSessionId);
            if (session) {
                state.messages = session.messages || [];
                state.artifacts = [];  // Don't load old artifacts
                state.auditLog = session.auditLog || [];
            }
        }
    } catch (e) {
        console.error('Storage load error:', e);
    }
}

// =====================================================
// WEBSOCKET CONNECTION
// =====================================================

function connectWebSocket() {
    try {
        state.ws = new WebSocket(CONFIG.wsUrl);

        state.ws.onopen = () => {
            state.isConnected = true;
            state.reconnectAttempts = 0;
            updateStatus('ready');
            logTerminal('WebSocket connected');
            addNotification('Connected to server', 'success');
        };

        state.ws.onclose = () => {
            state.isConnected = false;
            updateStatus('disconnected');
            logTerminal('WebSocket disconnected');

            if (state.reconnectAttempts < CONFIG.maxReconnectAttempts) {
                state.reconnectAttempts++;
                logTerminal(`Reconnecting... (attempt ${state.reconnectAttempts})`);
                setTimeout(connectWebSocket, CONFIG.reconnectDelay);
            }
        };

        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('error');
        };

        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Parse error:', e);
            }
        };
    } catch (error) {
        console.error('WebSocket error:', error);
        updateStatus('rest');
        logTerminal('Falling back to REST API');
    }
}

function handleMessage(data) {
    switch (data.type) {
        case 'agent_update':
            setActiveAgent(data.agent, data.progress);
            break;
        case 'message':
            // Pass the entire message object to preserve references
            addAssistantMessage(data.content, data.agent, data.references);
            break;
        case 'audit':
            updateAuditLog(data.entries);
            break;
        case 'artifact':
            addArtifact(data.artifact);
            break;
        case 'structure':
            loadStructure(data.pdbContent);
            break;
        case 'complete':
            // Handle all_references if provided
            if (data.all_references) {
                console.log('Received all references:', data.all_references);
            }
            handleComplete();
            break;
        case 'error':
            handleError(data.message);
            break;
        case 'log':
            logTerminal(data.message);
            break;
        case 'metrics':
            updateLiveMetrics(data.metrics);
            break;
        case 'code_execution':
            renderCodeExecution(data.steps, data.agent);
            break;
    }
}

function renderCodeExecution(steps, agentId) {
    state.codeSteps = steps; // Save to state
    const list = document.getElementById('codeExecutionList');
    if (!list) return;

    if (!steps || steps.length === 0) {
        list.innerHTML = `
            <div class="text-center py-6 text-text-subtle text-xs">
                <span class="material-symbols-outlined text-2xl text-border-dark mb-2 block">terminal</span>
                No code executed yet
            </div>
        `;
        return;
    }

    const agent = AGENTS[agentId] || { label: agentId, color: 'slate' };
    const colorClass = getColorClass(agent.color);

    const html = steps.map(step => {
        let contentHtml = '';

        if (step.thought) {
            contentHtml += `
                <div class="mb-3">
                    <p class="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Reasoning</p>
                    <div class="bg-surface-dark/50 rounded-lg p-3 border border-white/5 text-xs text-slate-300 italic leading-relaxed">
                        ${escapeHtml(step.thought)}
                    </div>
                </div>
            `;
        }

        if (step.code) {
            contentHtml += `
                <div class="mb-3">
                    <p class="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Generated Code</p>
                    <div class="bg-black/60 rounded-lg p-3 border border-white/5 font-mono text-[11px] text-amber-200/90 overflow-x-auto">
                        <pre><code>${escapeHtml(step.code)}</code></pre>
                    </div>
                </div>
            `;
        }

        if (step.output) {
            contentHtml += `
                <div class="mb-3">
                    <p class="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Standard Output</p>
                    <div class="bg-slate-900/80 rounded-lg p-3 border border-white/5 font-mono text-[11px] text-emerald-400/90 overflow-x-auto whitespace-pre">
                        <code>${escapeHtml(step.output)}</code>
                    </div>
                </div>
            `;
        }

        if (step.logs && step.logs.length > 50) { // Only show significant logs
            contentHtml += `
                <div>
                    <p class="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Execution Logs</p>
                    <div class="bg-slate-950/50 rounded-lg p-3 border border-white/5 font-mono text-[10px] text-slate-500 overflow-x-auto whitespace-pre">
                        <code>${escapeHtml(step.logs)}</code>
                    </div>
                </div>
            `;
        }

        return `
            <div class="p-4 rounded-xl bg-surface-dark border border-white/5 animate-fadeIn">
                <div class="flex items-center gap-2 mb-3">
                    <div class="h-6 w-6 rounded-full ${colorClass.bg} flex items-center justify-center border ${colorClass.border}">
                        <span class="text-[10px] font-bold ${colorClass.text}">${step.step}</span>
                    </div>
                    <span class="text-[11px] font-bold text-white uppercase tracking-tight">Step ${step.step} — ${agent.label}</span>
                </div>
                ${contentHtml}
            </div>
        `;
    }).join('');

    list.innerHTML = html;

    // Switch to Code tab automatically when code is executed
    const codeTabBtn = document.querySelector('.tab-btn[data-tab="code"]');
    if (codeTabBtn) codeTabBtn.click();
}

function updateLiveMetrics(metrics) {
    if (!metrics) return;

    const lossEl = document.getElementById('live-loss');
    const lossBar = document.getElementById('loss-bar');
    const accEl = document.getElementById('live-accuracy');
    const accBar = document.getElementById('accuracy-bar');

    if (metrics.loss !== undefined && lossEl && lossBar) {
        lossEl.textContent = metrics.loss.toFixed(4);
        // Normalize loss for bar (assume max loss 2.0 for visualization)
        const lossPercent = Math.max(0, Math.min(100, (metrics.loss / 2.0) * 100));
        lossBar.style.width = lossPercent + '%';
    }

    if (metrics.accuracy !== undefined && accEl && accBar) {
        const accVal = metrics.accuracy > 1 ? metrics.accuracy : metrics.accuracy * 100;
        accEl.textContent = accVal.toFixed(2) + '%';
        accBar.style.width = accVal + '%';
    }
}

function handleComplete() {
    setProcessingState(false);
    setActiveAgent(null);
    saveCurrentSession();
    addNotification('Query completed', 'success');
    logTerminal('Processing complete');
}

function handleError(message) {
    showToast(message, 'error');
    setProcessingState(false);
    addNotification('Error: ' + message, 'error');
    logTerminal('ERROR: ' + message);
}

// =====================================================
// INPUT HANDLERS
// =====================================================

function initInputHandlers() {
    elements.submitBtn?.addEventListener('click', handleSubmit);

    elements.queryInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    });

    elements.newResearchBtn?.addEventListener('click', () => {
        if (state.messages.length > 0) {
            if (confirm('Start a new session? Current session will be saved.')) {
                saveCurrentSession();
                startNewSession();
            }
        } else {
            startNewSession();
        }
    });
}

async function handleSubmit() {
    const query = elements.queryInput.value.trim();
    if (!query || state.isProcessing) return;

    elements.queryInput.value = '';
    autoResizeTextarea();

    // Hide welcome state and quick chips
    const welcomeEl = document.getElementById('welcomeState');
    if (welcomeEl) welcomeEl.remove();
    elements.quickChips?.classList.add('hidden');

    addUserMessage(query);
    setProcessingState(true);
    logTerminal(`Query: "${query.substring(0, 50)}${query.length > 50 ? '...' : ''}"`);

    try {
        const apiKeys = getApiKeysPayload();
        if (state.isConnected && state.ws?.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: 'query', content: query, api_keys: apiKeys }));
        } else {
            await sendQueryViaRest(query, apiKeys);
        }
    } catch (error) {
        console.error('Query error:', error);
        handleError('Failed to send query: ' + error.message);
    }
}

function getApiKeysPayload() {
    const keys = state.settings?.apiKeys || {};
    const payload = {};
    if (keys.openai?.trim()) payload.openai = keys.openai.trim();
    if (keys.gemini?.trim()) payload.gemini = keys.gemini.trim();
    return Object.keys(payload).length ? payload : undefined;
}

async function sendQueryViaRest(query, apiKeys) {
    logTerminal('Using REST API fallback');

    const body = { query };
    if (apiKeys && Object.keys(apiKeys).length) body.api_keys = apiKeys;

    const response = await fetch(`${CONFIG.apiBaseUrl}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(l => l.startsWith('data: '));

        for (const line of lines) {
            try {
                handleMessage(JSON.parse(line.slice(6)));
            } catch (e) { /* ignore parse errors */ }
        }
    }

    handleComplete();
}

function autoResizeTextarea() {
    const el = elements.queryInput;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

// =====================================================
// FILE ATTACHMENT
// =====================================================

function initFileAttachment() {
    const attachBtn = document.querySelector('button[title="Attach File"]');
    if (!attachBtn) return;

    // Create hidden file input
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.pdb,.fasta,.fa,.csv,.json,.txt,.pdf,.xlsx,.tsv';
    fileInput.style.display = 'none';
    fileInput.id = 'fileInput';
    document.body.appendChild(fileInput);

    attachBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            showToast(`Uploading: ${file.name}...`, 'info');
            logTerminal(`Uploading file: ${file.name}`);

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${CONFIG.apiBaseUrl}/api/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const data = await response.json();
            const ext = file.name.split('.').pop().toLowerCase();

            let prompt = '';
            if (ext === 'pdb') {
                prompt = `I have uploaded a PDB structure: ${file.name}\nPath: ${data.path}\n\nPlease analyze this structure.`;
                // Also try to load it in the viewer
                const content = await file.text();
                loadStructure(content);
            } else if (ext === 'pdf') {
                prompt = `I have uploaded a PDF document: ${file.name}\nPath: ${data.path}\n\nPlease extract and summarize the information from this PDF.`;
            } else if (ext === 'csv' || ext === 'xlsx' || ext === 'tsv') {
                prompt = `I have uploaded a dataset: ${file.name}\nPath: ${data.path}\n\nPlease load this data and perform analysis/modeling.`;
            } else if (ext === 'fasta' || ext === 'fa') {
                const content = await file.text();
                prompt = `I have uploaded a FASTA sequence: ${file.name}\nPath: ${data.path}\n\nContent:\n\`\`\`\n${content}\n\`\`\``;
            } else {
                prompt = `I have uploaded a file: ${file.name}\nPath: ${data.path}\n\nPlease process this file.`;
            }

            elements.queryInput.value = prompt;
            autoResizeTextarea();
            showToast(`File uploaded successfully: ${file.name}`, 'success');
            logTerminal(`File uploaded to: ${data.path}`);

            // Add to artifacts list so user can see it
            addArtifact({
                name: file.name,
                path: data.path,
                type: ext,
                size: file.size,
                isUpload: true
            });

        } catch (error) {
            console.error('Upload error:', error);
            showToast('Failed to upload file: ' + error.message, 'error');
        }

        fileInput.value = '';
    });
}

// =====================================================
// VOICE INPUT
// =====================================================

function initVoiceInput() {
    const voiceBtn = document.querySelector('button[title="Voice Input"]');
    if (!voiceBtn) return;

    let recognition = null;
    let isListening = false;

    // Check for browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        voiceBtn.addEventListener('click', () => {
            showToast('Voice input not supported in this browser', 'warning');
        });
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        isListening = true;
        voiceBtn.classList.add('text-primary', 'animate-pulse');
        voiceBtn.classList.remove('text-slate-400');
        logTerminal('Voice input active...');
    };

    recognition.onend = () => {
        isListening = false;
        voiceBtn.classList.remove('text-primary', 'animate-pulse');
        voiceBtn.classList.add('text-slate-400');
    };

    recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join('');

        elements.queryInput.value = transcript;
        autoResizeTextarea();
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        showToast('Voice recognition error', 'error');
        isListening = false;
        voiceBtn.classList.remove('text-primary', 'animate-pulse');
        voiceBtn.classList.add('text-slate-400');
    };

    voiceBtn.addEventListener('click', () => {
        if (isListening) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });
}

// =====================================================
// MESSAGES
// =====================================================

function addUserMessage(content) {
    const message = { role: 'user', content, timestamp: new Date().toISOString() };
    state.messages.push(message);
    renderUserMessage(message);
    scrollToBottom();
    saveCurrentSession();
}

function addAssistantMessage(content, agent = null, references = null) {
    const message = {
        role: 'assistant',
        content,
        agent,
        timestamp: new Date().toISOString(),
        references: references || []
    };
    state.messages.push(message);
    renderAssistantMessage(message);
    scrollToBottom();
}

function renderUserMessage(message) {
    const time = formatTime(new Date(message.timestamp));
    const html = `
        <div class="flex flex-row-reverse gap-3 animate-fadeIn">
            <div class="h-8 w-8 rounded-full bg-slate-700 flex-shrink-0 flex items-center justify-center border border-slate-600">
                <span class="material-symbols-outlined text-white text-[16px]">person</span>
            </div>
            <div class="max-w-[80%] flex flex-col items-end gap-1">
                <div class="bg-primary/20 border border-primary/20 text-white px-4 py-3 rounded-2xl rounded-tr-none text-sm leading-relaxed">
                    <p class="whitespace-pre-wrap">${escapeHtml(message.content)}</p>
                </div>
                <span class="text-[10px] text-slate-500 font-mono">${time}</span>
            </div>
        </div>
    `;
    elements.chatMessages.insertAdjacentHTML('beforeend', html);
}

function renderAssistantMessage(message) {
    const agent = AGENTS[message.agent] || AGENTS.supervisor;
    const colorClass = getColorClass(agent.color);
    const time = formatTime(new Date(message.timestamp));

    // Add references from message if provided
    let messageRefs = [];
    if (message.references && Array.isArray(message.references)) {
        console.log('Processing references for message:', message.references);
        message.references.forEach(ref => {
            addReference(ref);
            messageRefs.push(ref);
        });
        updateReferencePanel();
        console.log('Total references now:', state.references.size);
    }

    // Parse markdown
    let contentHtml = message.content;
    if (typeof marked !== 'undefined') {
        contentHtml = marked.parse(message.content);
    }

    // Parse citations in content if there are references
    if (messageRefs.length > 0) {
        contentHtml = parseCitationsInContent(contentHtml, messageRefs);
    }

    const html = `
        <div class="flex gap-3 animate-fadeIn">
            <div class="h-8 w-8 rounded-full ${colorClass.bg} flex-shrink-0 flex items-center justify-center border ${colorClass.border} shadow-lg">
                <span class="material-symbols-outlined ${colorClass.text} text-[16px]">${agent.icon}</span>
            </div>
            <div class="max-w-[85%] flex flex-col items-start gap-1 min-w-0">
                <div class="flex items-center gap-2 mb-1">
                    <span class="text-[10px] font-bold ${colorClass.bg} ${colorClass.text} px-2 py-0.5 rounded border ${colorClass.border}">${agent.label.toUpperCase()}</span>
                </div>
                <div class="bg-surface-dark border border-white/5 text-slate-200 px-4 py-3 rounded-2xl rounded-tl-none text-sm leading-relaxed message-content overflow-x-auto max-w-full">
                    ${contentHtml}
                </div>
                <span class="text-[10px] text-slate-500 font-mono">${time}</span>
            </div>
        </div>
    `;
    elements.chatMessages.insertAdjacentHTML('beforeend', html);

    // Attach hover card event listeners to citations
    setTimeout(() => {
        document.querySelectorAll('.citation').forEach(citation => {
            citation.addEventListener('mouseenter', (e) => {
                const refId = e.target.dataset.refId;
                if (refId) showHoverCard(refId, e);
            });
            citation.addEventListener('mouseleave', (e) => {
                setTimeout(() => {
                    const refId = e.target.dataset.refId;
                    if (refId) closeHoverCard(refId);
                }, 300);
            });
            citation.addEventListener('click', (e) => {
                e.preventDefault();
                const refNum = e.target.dataset.refNum;
                scrollToReference(refNum);
            });
        });
    }, 0);
}

function scrollToBottom() {
    if (state.settings.autoScroll && elements.chatMessages) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

function resetChatUI() {
    if (!elements.chatMessages) return;

    elements.chatMessages.innerHTML = `
        <div class="flex flex-col items-center justify-center flex-1 min-h-[300px] text-center px-4" id="welcomeState">
            <div class="mb-6 p-4 rounded-full bg-primary/5 border border-primary/20 shadow-[0_0_30px_rgba(7,182,213,0.1)]">
                <span class="material-symbols-outlined text-4xl text-primary">genetics</span>
            </div>
            <h1 class="text-2xl md:text-3xl font-bold text-white mb-3 tracking-tight">What research problem<br/>are we solving today?</h1>
            <p class="text-text-subtle text-sm max-w-md font-light">Deploy multi-agent swarms for protein folding, docking simulations, and literature review.</p>
        </div>
    `;

    elements.quickChips?.classList.remove('hidden');
    renderAgentList();
    renderArtifacts();

    // Clear code execution
    const codeList = document.getElementById('codeExecutionList');
    if (codeList) {
        codeList.innerHTML = `
            <div class="text-center py-6 text-text-subtle text-xs">
                <span class="material-symbols-outlined text-2xl text-border-dark mb-2 block">terminal</span>
                No code executed yet
            </div>
        `;
    }
}

// =====================================================
// AGENTS
// =====================================================

function renderAgentList() {
    if (!elements.agentList) return;

    const html = Object.entries(AGENTS).map(([id, agent]) => {
        const isActive = state.currentAgent === id;
        const progress = state.agentProgress[id] || 0;
        const colorClass = getColorClass(agent.color);

        if (isActive) {
            let trainingVisual = '';
            if (agent.isTraining) {
                trainingVisual = `
                    <div class="mt-3 p-3 bg-black/40 rounded-lg border border-white/5 overflow-hidden relative">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Live Monitor</span>
                            <div class="flex gap-1">
                                <span class="w-1 h-3 bg-primary/40 animate-[bounce_1s_infinite]"></span>
                                <span class="w-1 h-3 bg-primary/60 animate-[bounce_1.2s_infinite]"></span>
                                <span class="w-1 h-3 bg-primary/80 animate-[bounce_0.8s_infinite]"></span>
                            </div>
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between text-[10px] font-mono">
                                <span class="text-slate-500">Loss</span>
                                <span class="text-rose-400" id="live-loss">0.0000</span>
                            </div>
                            <div class="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div class="h-full bg-rose-500/50 transition-all duration-1000" id="loss-bar" style="width: 0%"></div>
                            </div>
                            <div class="flex justify-between text-[10px] font-mono">
                                <span class="text-slate-500">Accuracy</span>
                                <span class="text-emerald-400" id="live-accuracy">0.00%</span>
                            </div>
                            <div class="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div class="h-full bg-emerald-500/50 transition-all duration-1000" id="accuracy-bar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="mt-3 flex justify-center">
                            <svg class="w-full h-8 opacity-50" viewBox="0 0 100 20">
                                <path d="M0 10 Q 10 2, 20 10 T 40 10 T 60 10 T 80 10 T 100 10" fill="none" stroke="currentColor" stroke-width="0.5" class="text-primary animate-[pulse_2s_infinite]"></path>
                            </svg>
                        </div>
                    </div>
                `;
            }

            return `
                <div class="relative flex flex-col gap-2 p-4 rounded-xl border border-primary/50 bg-primary/5 shadow-[0_0_20px_rgba(7,182,213,0.1)] transition-all duration-300">
                    <div class="absolute top-4 right-4 h-2 w-2">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary text-[20px]">${agent.icon}</span>
                        <span class="text-sm font-bold text-white">${agent.label}</span>
                    </div>
                    <div class="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full bg-primary transition-all duration-500" style="width: ${progress || 50}%"></div>
                    </div>
                    <p class="text-xs text-primary font-medium">${agent.description}</p>
                    ${trainingVisual}
                    <div class="flex gap-2 mt-1">
                        <span class="text-[10px] font-mono text-slate-400 bg-black/20 px-1.5 py-0.5 rounded border border-white/5">CPU: ${Math.floor(Math.random() * 30 + 60)}%</span>
                    </div>
                </div>
            `;
        } else {
            return `
                <div class="flex flex-col gap-2 p-3 rounded-lg border border-slate-700 bg-surface-dark opacity-60 hover:opacity-80 transition-opacity">
                    <div class="flex justify-between items-start">
                        <div class="flex items-center gap-2">
                            <span class="material-symbols-outlined text-slate-400 text-[18px]">${agent.icon}</span>
                            <span class="text-sm font-semibold text-slate-300">${agent.label}</span>
                        </div>
                        <span class="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded">IDLE</span>
                    </div>
                    <p class="text-xs text-slate-500">Waiting...</p>
                </div>
            `;
        }
    }).join('');

    elements.agentList.innerHTML = html;
}

let metricsInterval = null;

function setActiveAgent(agentId, progress = 50) {
    state.currentAgent = agentId;
    if (agentId) {
        state.agentProgress[agentId] = progress;
    }
    renderAgentList();

    // Clear existing interval
    if (metricsInterval) {
        clearInterval(metricsInterval);
        metricsInterval = null;
    }

    if (agentId) {
        const agent = AGENTS[agentId] || { label: agentId };
        updateStatus('processing');
        logTerminal(`Agent: ${agent.label} activated`);

        // Start simulated metrics if it's a training agent
        if (agent.isTraining) {
            let simLoss = 1.2;
            let simAcc = 45;
            metricsInterval = setInterval(() => {
                simLoss = Math.max(0.1, simLoss - (Math.random() * 0.05));
                simAcc = Math.min(99, simAcc + (Math.random() * 2));
                updateLiveMetrics({ loss: simLoss, accuracy: simAcc });
            }, 2000);
        }
    }
}

// =====================================================
// ARTIFACTS
// =====================================================

function addArtifact(artifact) {
    // Extract protein ID from PDB filename (e.g., "AF-P01308-F1-model_v6.pdb" -> "P01308")
    const extractProteinId = (name) => {
        if (!name) return null;
        // Match UniProt ID pattern in filename
        const match = name.match(/([A-Z][A-Z0-9]{4,5})/i);
        return match ? match[1].toUpperCase() : null;
    };

    const newProteinId = artifact.type === 'pdb' ? extractProteinId(artifact.name) : null;

    // Avoid duplicates by checking path, name, or protein ID for PDB files
    const isDuplicate = state.artifacts.some(a => {
        // Exact matches
        if (a.path === artifact.path || a.name === artifact.name) return true;

        // For PDB files, check if same protein ID
        if (a.type === 'pdb' && artifact.type === 'pdb' && newProteinId) {
            const existingProteinId = extractProteinId(a.name);
            if (existingProteinId === newProteinId) return true;
        }

        return false;
    });

    if (!isDuplicate) {
        state.artifacts.push(artifact);
        renderArtifacts();
        const action = artifact.isUpload ? 'Uploaded' : 'Generated';
        addNotification(`${action}: ${artifact.name}`, 'success');
        logTerminal(`${action}: ${artifact.name}`);
    }
}

function renderArtifacts() {
    if (!elements.artifactsList) return;

    if (state.artifacts.length === 0) {
        elements.artifactsList.innerHTML = `
            <div class="text-center py-8 text-text-subtle text-xs">
                <span class="material-symbols-outlined text-3xl text-border-dark mb-2 block">folder_off</span>
                No artifacts generated yet
            </div>
        `;
        return;
    }

    // Group artifacts by type
    const grouped = {};
    state.artifacts.forEach(a => {
        const type = a.type || 'other';
        if (!grouped[type]) grouped[type] = [];
        grouped[type].push(a);
    });

    // Priority order for display
    const typeOrder = ['pdb', 'md', 'csv', 'json', 'fasta', 'py', 'png', 'jpg', 'txt'];
    const sortedTypes = Object.keys(grouped).sort((a, b) => {
        const aIdx = typeOrder.indexOf(a);
        const bIdx = typeOrder.indexOf(b);
        return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx);
    });

    let html = '';
    sortedTypes.forEach(type => {
        const artifacts = grouped[type];
        const iconInfo = getFileIconInfo(type);

        // Add a subtle category header if multiple types
        if (sortedTypes.length > 1) {
            html += `<p class="text-[9px] font-bold text-slate-600 uppercase tracking-wider mt-2 mb-1 first:mt-0">${iconInfo.label || type.toUpperCase()}</p>`;
        }

        artifacts.forEach((artifact, idx) => {
            const globalIndex = state.artifacts.indexOf(artifact);
            html += `
                <div class="artifact-item flex items-center gap-3 p-2.5 rounded-lg bg-surface-dark/80 border border-white/5 hover:border-primary/30 transition-all group cursor-pointer" data-index="${globalIndex}">
                    <div class="h-9 w-9 rounded-lg ${iconInfo.bg} ${iconInfo.text} flex items-center justify-center border ${iconInfo.border} flex-shrink-0">
                        <span class="material-symbols-outlined text-[18px]">${iconInfo.icon}</span>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-xs font-medium text-white truncate group-hover:text-primary transition-colors">${escapeHtml(artifact.name)}</p>
                        <p class="text-[10px] text-slate-500">${formatBytes(artifact.size)}${artifact.preview ? ' • Click to preview' : ''}</p>
                    </div>
                    <div class="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button class="artifact-view p-1.5 text-slate-500 hover:text-white hover:bg-white/10 rounded transition-colors" title="Preview">
                            <span class="material-symbols-outlined text-[16px]">visibility</span>
                        </button>
                        <button class="artifact-download p-1.5 text-slate-500 hover:text-primary hover:bg-primary/10 rounded transition-colors" title="Download">
                            <span class="material-symbols-outlined text-[16px]">download</span>
                        </button>
                    </div>
                </div>
            `;
        });
    });

    // Add summary at top
    const summary = `
        <div class="flex items-center justify-between mb-2 pb-2 border-b border-white/5">
            <span class="text-[10px] text-slate-400">${state.artifacts.length} file${state.artifacts.length !== 1 ? 's' : ''} generated</span>
            <button class="clear-artifacts text-[10px] text-slate-500 hover:text-red-400 transition-colors">Clear</button>
        </div>
    `;

    elements.artifactsList.innerHTML = summary + html;

    // Add event listeners
    elements.artifactsList.querySelectorAll('.artifact-item').forEach((item) => {
        const index = parseInt(item.dataset.index);
        const artifact = state.artifacts[index];

        if (!artifact) return;

        item.querySelector('.artifact-download')?.addEventListener('click', (e) => {
            e.stopPropagation();
            downloadArtifact(artifact.path, artifact.name);
        });

        item.querySelector('.artifact-view')?.addEventListener('click', (e) => {
            e.stopPropagation();
            previewArtifact(artifact);
        });

        // Click on item to view
        item.addEventListener('click', () => {
            previewArtifact(artifact);
        });
    });

    // Clear artifacts button
    elements.artifactsList.querySelector('.clear-artifacts')?.addEventListener('click', () => {
        state.artifacts = [];
        renderArtifacts();
        showToast('Artifacts cleared', 'info');
    });
}

function previewArtifact(artifact) {
    if (artifact.type === 'pdb') {
        if (artifact.path?.startsWith('http')) {
            // Load from URL
            fetch(artifact.path)
                .then(r => r.text())
                .then(content => loadStructure(content))
                .catch(e => showToast('Failed to load structure', 'error'));
        } else if (state.currentPdbContent) {
            showToast('Structure displayed in viewer', 'info');
        }
    } else {
        // Show preview modal for text-based and image artifacts
        showArtifactPreviewModal(artifact);
    }
}

function showArtifactPreviewModal(artifact) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4';

    // Determine icon and color based on type
    const typeConfig = {
        csv: { icon: 'table_chart', color: 'emerald', label: 'Table Data' },
        json: { icon: 'data_object', color: 'amber', label: 'JSON Data' },
        md: { icon: 'description', color: 'blue', label: 'Report' },
        py: { icon: 'code', color: 'yellow', label: 'Python Code' },
        js: { icon: 'code', color: 'yellow', label: 'JavaScript' },
        fasta: { icon: 'genetics', color: 'primary', label: 'Sequence' },
        txt: { icon: 'article', color: 'slate', label: 'Text' },
        sh: { icon: 'terminal', color: 'green', label: 'Shell Script' },
    };

    const config = typeConfig[artifact.type] || typeConfig.txt;

    // Format preview content based on type
    let previewHtml = '';
    const preview = artifact.preview || '';

    if (artifact.type === 'csv') {
        // Render CSV as table
        const lines = preview.split('\n').filter(l => l.trim());
        if (lines.length > 0) {
            previewHtml = '<div class="overflow-x-auto"><table class="w-full text-xs border-collapse">';
            previewHtml += '<thead class="bg-surface-dark"><tr>';
            const headers = lines[0].split(',');
            headers.forEach(h => {
                previewHtml += `<th class="border border-white/10 px-3 py-2 text-left text-slate-300 font-medium">${escapeHtml(h)}</th>`;
            });
            previewHtml += '</tr></thead><tbody>';

            for (let i = 1; i < Math.min(lines.length, 15); i++) {
                previewHtml += '<tr class="hover:bg-white/5">';
                const cells = lines[i].split(',');
                cells.forEach(c => {
                    previewHtml += `<td class="border border-white/10 px-3 py-2 text-slate-400">${escapeHtml(c)}</td>`;
                });
                previewHtml += '</tr>';
            }
            previewHtml += '</tbody></table></div>';
            if (lines.length > 15) {
                previewHtml += `<p class="text-xs text-slate-500 mt-2">... and ${lines.length - 15} more rows</p>`;
            }
        }
    } else if (artifact.type === 'png' || artifact.type === 'jpg' || artifact.type === 'jpeg') {
        // Render image
        const imgUrl = `${CONFIG.apiBaseUrl}/api/download?path=${encodeURIComponent(artifact.path)}`;
        previewHtml = `
            <div class="flex flex-col items-center justify-center p-4">
                <img src="${imgUrl}" class="max-w-full max-h-[60vh] rounded shadow-lg border border-white/10" alt="${escapeHtml(artifact.name)}" />
                <p class="text-[10px] text-slate-500 mt-4 font-mono">${artifact.path}</p>
            </div>
        `;
    } else if (artifact.type === 'json') {
        // Syntax highlight JSON
        const highlighted = preview
            .replace(/"([^"]+)":/g, '<span class="text-blue-400">"$1"</span>:')
            .replace(/: "([^"]+)"/g, ': <span class="text-amber-300">"$1"</span>')
            .replace(/: (\d+)/g, ': <span class="text-emerald-400">$1</span>')
            .replace(/: (true|false)/g, ': <span class="text-purple-400">$1</span>');
        previewHtml = `<pre class="text-xs font-mono whitespace-pre-wrap overflow-auto max-h-96">${highlighted}</pre>`;
    } else if (artifact.type === 'md') {
        // Render markdown
        if (typeof marked !== 'undefined') {
            previewHtml = `<div class="prose prose-sm prose-invert max-w-none overflow-auto max-h-96">${marked.parse(preview)}</div>`;
        } else {
            previewHtml = `<pre class="text-xs whitespace-pre-wrap overflow-auto max-h-96">${escapeHtml(preview)}</pre>`;
        }
    } else {
        // Code or text
        previewHtml = `<pre class="text-xs font-mono whitespace-pre-wrap overflow-auto max-h-96 text-slate-300">${escapeHtml(preview)}</pre>`;
    }

    modal.innerHTML = `
        <div class="bg-surface-dark border border-white/10 rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden animate-fadeIn flex flex-col">
            <div class="flex items-center justify-between p-4 border-b border-white/5 flex-shrink-0">
                <div class="flex items-center gap-3">
                    <div class="h-10 w-10 rounded-lg bg-${config.color}-500/20 flex items-center justify-center border border-${config.color}-500/30">
                        <span class="material-symbols-outlined text-${config.color}-400">${config.icon}</span>
                    </div>
                    <div>
                        <h3 class="text-base font-bold text-white">${escapeHtml(artifact.name)}</h3>
                        <p class="text-xs text-slate-400">${config.label} • ${formatBytes(artifact.size)}</p>
                    </div>
                </div>
                <div class="flex gap-2">
                    <button class="download-btn flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary/20 hover:bg-primary/30 border border-primary/30 text-xs font-medium text-primary transition-all">
                        <span class="material-symbols-outlined text-[16px]">download</span>
                        Download
                    </button>
                    <button class="modal-close p-2 text-slate-400 hover:text-white hover:bg-white/10 rounded-lg transition-all">
                        <span class="material-symbols-outlined">close</span>
                    </button>
                </div>
            </div>
            <div class="flex-1 overflow-auto p-4 bg-black/20">
                ${previewHtml || '<p class="text-slate-500 text-sm">No preview available</p>'}
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Event listeners
    modal.querySelector('.modal-close')?.addEventListener('click', () => modal.remove());
    modal.querySelector('.download-btn')?.addEventListener('click', () => {
        downloadArtifact(artifact.path, artifact.name);
    });
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
}

function downloadArtifact(path, filename) {
    const link = document.createElement('a');
    link.href = `${CONFIG.apiBaseUrl}/api/download?path=${encodeURIComponent(path)}`;
    link.download = filename || 'download';
    link.click();
    showToast('Downloading: ' + filename, 'success');
}

function getFileIconInfo(type) {
    const icons = {
        'pdb': { icon: 'view_in_ar', bg: 'bg-indigo-500/20', text: 'text-indigo-400', border: 'border-indigo-500/30', label: '3D Structure' },
        'csv': { icon: 'table_chart', bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Table' },
        'png': { icon: 'image', bg: 'bg-pink-500/20', text: 'text-pink-400', border: 'border-pink-500/30', label: 'Image' },
        'jpg': { icon: 'image', bg: 'bg-pink-500/20', text: 'text-pink-400', border: 'border-pink-500/30', label: 'Image' },
        'jpeg': { icon: 'image', bg: 'bg-pink-500/20', text: 'text-pink-400', border: 'border-pink-500/30', label: 'Image' },
        'json': { icon: 'data_object', bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30', label: 'Data' },
        'txt': { icon: 'article', bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500/30', label: 'Text' },
        'fasta': { icon: 'genetics', bg: 'bg-primary/20', text: 'text-primary', border: 'border-primary/30', label: 'Sequence' },
        'md': { icon: 'description', bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30', label: 'Report' },
        'pdf': { icon: 'picture_as_pdf', bg: 'bg-rose-500/20', text: 'text-rose-400', border: 'border-rose-500/30', label: 'PDF Document' },
        'xlsx': { icon: 'table_view', bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Excel' },
        'xls': { icon: 'table_view', bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Excel' },
        'py': { icon: 'code', bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30', label: 'Python' },
        'js': { icon: 'javascript', bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30', label: 'JavaScript' },
        'sh': { icon: 'terminal', bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30', label: 'Script' },
        'html': { icon: 'html', bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30', label: 'HTML' },
        'sql': { icon: 'database', bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/30', label: 'SQL' },
    };
    return icons[type?.toLowerCase()] || icons.txt;
}

// =====================================================
// AUDIT LOG
// =====================================================

function updateAuditLog(entries) {
    state.auditLog = entries;
    if (!elements.auditLog) return;

    const formattedLog = JSON.stringify(entries, null, 2);

    // Syntax highlight JSON
    const highlighted = formattedLog
        .replace(/"([^"]+)":/g, '<span class="text-blue-400">"$1"</span>:')
        .replace(/: "([^"]+)"/g, ': <span class="text-amber-300">"$1"</span>')
        .replace(/: (\d+)/g, ': <span class="text-emerald-400">$1</span>')
        .replace(/: (true|false)/g, ': <span class="text-purple-400">$1</span>')
        .replace(/: (null)/g, ': <span class="text-slate-500">$1</span>');

    elements.auditLog.innerHTML = `<pre class="whitespace-pre-wrap text-[10px] leading-relaxed">${highlighted}</pre>`;
}

// =====================================================
// 3D MOLECULAR VIEWER
// =====================================================

function initViewerControls() {
    const viewerContainer = document.getElementById('viewerContainer');
    if (!viewerContainer) return;

    // Fullscreen button
    const fullscreenBtn = viewerContainer.querySelector('.viewer-fullscreen');
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', () => {
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else {
                viewerContainer.requestFullscreen().catch(err => {
                    showToast('Fullscreen not available', 'warning');
                });
            }
        });
    }

    // Spin toggle
    const spinBtn = viewerContainer.querySelector('.viewer-spin');
    let isSpinning = false;
    if (spinBtn) {
        spinBtn.addEventListener('click', () => {
            if (state.nglStage) {
                isSpinning = !isSpinning;
                state.nglStage.setSpin(isSpinning);
                spinBtn.classList.toggle('text-primary', isSpinning);
                spinBtn.classList.toggle('bg-primary/20', isSpinning);
            } else {
                showToast('No structure loaded', 'info');
            }
        });
    }

    // Reset view
    const resetBtn = viewerContainer.querySelector('.viewer-reset');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            if (state.nglStage) {
                state.nglStage.autoView();
                showToast('View reset', 'info');
            }
        });
    }

    // Zoom in
    const zoomInBtn = viewerContainer.querySelector('.viewer-zoom-in');
    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', () => {
            if (state.nglStage) {
                const zoom = state.nglStage.viewer.camera.zoom;
                state.nglStage.viewer.camera.zoom = zoom * 1.2;
                state.nglStage.viewer.requestRender();
            }
        });
    }

    // Zoom out
    const zoomOutBtn = viewerContainer.querySelector('.viewer-zoom-out');
    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', () => {
            if (state.nglStage) {
                const zoom = state.nglStage.viewer.camera.zoom;
                state.nglStage.viewer.camera.zoom = zoom / 1.2;
                state.nglStage.viewer.requestRender();
            }
        });
    }

    // Drag and drop PDB files
    viewerContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        viewerContainer.classList.add('border-primary', 'border-2');
    });

    viewerContainer.addEventListener('dragleave', () => {
        viewerContainer.classList.remove('border-primary', 'border-2');
    });

    viewerContainer.addEventListener('drop', async (e) => {
        e.preventDefault();
        viewerContainer.classList.remove('border-primary', 'border-2');

        const file = e.dataTransfer.files[0];
        if (file && (file.name.endsWith('.pdb') || file.name.endsWith('.cif'))) {
            try {
                const content = await file.text();
                loadStructure(content);
                showToast(`Loaded: ${file.name}`, 'success');
            } catch (err) {
                showToast('Failed to load file', 'error');
            }
        } else {
            showToast('Please drop a PDB or CIF file', 'warning');
        }
    });
}

function loadStructure(pdbContent) {
    if (!pdbContent) return;

    state.currentPdbContent = pdbContent;

    if (typeof NGL === 'undefined') {
        showToast('NGL.js not loaded', 'error');
        return;
    }

    const viewer = elements.structureViewer;
    if (!viewer) return;

    // Clear and prepare viewer
    viewer.innerHTML = '';
    viewer.style.width = '100%';
    viewer.style.height = '100%';

    try {
        // Dispose existing stage
        if (state.nglStage) {
            state.nglStage.dispose();
        }

        // Create new stage
        state.nglStage = new NGL.Stage(viewer, {
            backgroundColor: '#000000',
            quality: 'high',
        });

        // Load structure
        const blob = new Blob([pdbContent], { type: 'text/plain' });
        state.nglStage.loadFile(blob, { ext: 'pdb' }).then((component) => {
            // Add representations
            component.addRepresentation('cartoon', {
                color: 'chainname',
                smoothSheet: true,
            });
            component.addRepresentation('ball+stick', {
                sele: 'ligand',
                colorScheme: 'element',
            });
            component.addRepresentation('surface', {
                sele: 'polymer',
                opacity: 0.1,
                color: 'electrostatic',
            });

            state.nglStage.autoView();
            state.nglStage.setSpin(true);

            // Stop spin after 3 seconds
            setTimeout(() => {
                if (state.nglStage) state.nglStage.setSpin(false);
            }, 3000);
        });

        // Handle resize
        const resizeObserver = new ResizeObserver(() => {
            if (state.nglStage) state.nglStage.handleResize();
        });
        resizeObserver.observe(viewer);

        showToast('Structure loaded', 'success');
        logTerminal('3D structure rendered');

        // Don't add artifact here - the server sends it separately
        // This prevents duplicate artifacts

    } catch (error) {
        console.error('NGL error:', error);
        showToast('Failed to render structure', 'error');
    }
}

// =====================================================
// SIDEBAR & NAVIGATION
// =====================================================

function initSidebarNavigation() {
    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => {
                l.classList.remove('bg-primary/20', 'border-primary/20', 'text-primary', 'border');
                l.classList.add('text-text-subtle');
            });
            link.classList.add('bg-primary/20', 'border', 'border-primary/20', 'text-primary');
            link.classList.remove('text-text-subtle');

            // Handle navigation
            const text = link.textContent.trim();
            if (text.includes('History')) {
                showHistoryPanel();
            } else if (text.includes('Agent Library')) {
                showAgentLibrary();
            } else if (text.includes('Project Index')) {
                showProjectIndex();
            }
        });
    });
}

function initMobileMenu() {
    const mobileBtn = document.getElementById('mobileMenuBtn');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');

    if (mobileBtn && sidebar) {
        mobileBtn.addEventListener('click', () => {
            sidebar.classList.toggle('hidden');
            sidebar.classList.toggle('lg:flex');
            sidebar.classList.toggle('flex');
            sidebar.classList.toggle('fixed');
            sidebar.classList.toggle('z-50');
            sidebar.classList.toggle('left-0');
            sidebar.classList.toggle('top-0');
            overlay?.classList.toggle('hidden');
        });
    }

    // Close sidebar when clicking overlay
    overlay?.addEventListener('click', () => {
        sidebar?.classList.add('hidden');
        sidebar?.classList.remove('flex', 'fixed', 'z-50', 'left-0', 'top-0');
        sidebar?.classList.add('lg:flex');
        overlay?.classList.add('hidden');
    });
}

function renderRecentSessions() {
    if (!elements.recentSessions) return;

    const recentHtml = state.sessions.slice(0, 5).map(session => {
        const isActive = session.id === state.currentSessionId;
        return `
            <div class="session-item flex items-center gap-2 group ${isActive ? 'bg-white/5' : ''}" data-id="${session.id}">
                <button class="session-load flex-1 text-left px-3 py-2 text-sm ${isActive ? 'text-primary' : 'text-text-subtle'} hover:text-white rounded-lg truncate transition-colors">
                    ${escapeHtml(session.title)}
                </button>
                <button class="session-delete hidden group-hover:block p-1 text-slate-600 hover:text-red-400 rounded transition-colors" title="Delete">
                    <span class="material-symbols-outlined text-[16px]">close</span>
                </button>
            </div>
        `;
    }).join('');

    elements.recentSessions.innerHTML = recentHtml || '<p class="px-3 py-2 text-xs text-slate-600">No recent sessions</p>';

    // Add event listeners
    elements.recentSessions.querySelectorAll('.session-item').forEach(item => {
        const sessionId = item.dataset.id;

        item.querySelector('.session-load')?.addEventListener('click', () => {
            loadSession(sessionId);
        });

        item.querySelector('.session-delete')?.addEventListener('click', (e) => {
            e.stopPropagation();
            if (confirm('Delete this session?')) {
                deleteSession(sessionId);
            }
        });
    });
}

function showHistoryPanel() {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm';

    const sessionsHtml = state.sessions.length > 0
        ? state.sessions.map(s => `
            <div class="session-row flex items-center gap-3 p-3 hover:bg-white/5 rounded-lg cursor-pointer transition-all group" data-id="${s.id}">
                <span class="material-symbols-outlined text-slate-400 text-[18px]">description</span>
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium text-white truncate">${escapeHtml(s.title)}</p>
                    <p class="text-[10px] text-slate-400">${new Date(s.createdAt).toLocaleString()} • ${s.messages?.length || 0} messages</p>
                </div>
                <button class="session-delete opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-red-400 transition-all">
                    <span class="material-symbols-outlined text-[16px]">delete</span>
                </button>
            </div>
        `).join('')
        : '<p class="text-center py-8 text-slate-400">No sessions in history</p>';

    modal.innerHTML = `
        <div class="bg-surface-dark border border-white/10 rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden animate-fadeIn">
            <div class="flex items-center justify-between p-4 border-b border-white/5">
                <h3 class="text-lg font-bold text-white flex items-center gap-2">
                    <span class="material-symbols-outlined text-primary">history</span>
                    Session History
                </h3>
                <button class="modal-close p-1.5 text-slate-400 hover:text-white hover:bg-white/10 rounded-lg transition-all">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="max-h-96 overflow-y-auto custom-scrollbar p-2">
                ${sessionsHtml}
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Close button
    modal.querySelector('.modal-close')?.addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });

    // Session rows
    modal.querySelectorAll('.session-row').forEach(row => {
        row.addEventListener('click', (e) => {
            if (!e.target.closest('.session-delete')) {
                loadSession(row.dataset.id);
                modal.remove();
            }
        });

        row.querySelector('.session-delete')?.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteSession(row.dataset.id);
            row.remove();
        });
    });
}

function showAgentLibrary() {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm';

    const agentsHtml = Object.entries(AGENTS).map(([id, agent]) => {
        const colorClass = getColorClass(agent.color);
        return `
            <div class="flex items-center gap-3 p-3 rounded-lg border border-white/5 bg-surface-card/50 hover:border-primary/30 transition-all">
                <div class="h-10 w-10 rounded-lg ${colorClass.bg} flex items-center justify-center border ${colorClass.border}">
                    <span class="material-symbols-outlined ${colorClass.text}">${agent.icon}</span>
                </div>
                <div class="flex-1">
                    <p class="text-sm font-bold text-white">${agent.label}</p>
                    <p class="text-xs text-slate-400">${agent.description}</p>
                </div>
            </div>
        `;
    }).join('');

    modal.innerHTML = `
        <div class="bg-surface-dark border border-white/10 rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden animate-fadeIn">
            <div class="flex items-center justify-between p-4 border-b border-white/5">
                <h3 class="text-lg font-bold text-white flex items-center gap-2">
                    <span class="material-symbols-outlined text-primary">smart_toy</span>
                    Agent Library
                </h3>
                <button class="modal-close p-1.5 text-slate-400 hover:text-white hover:bg-white/10 rounded-lg transition-all">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="max-h-96 overflow-y-auto custom-scrollbar p-4 grid gap-3">
                ${agentsHtml}
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    modal.querySelector('.modal-close')?.addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
}

function showProjectIndex() {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm';

    const artifactsHtml = state.artifacts.length > 0
        ? state.artifacts.map(a => {
            const iconInfo = getFileIconInfo(a.type);
            return `
                <div class="flex items-center gap-3 p-3 rounded-lg border border-white/5 hover:border-primary/30 transition-all cursor-pointer" onclick="window.BioAgents.downloadArtifact && window.open('${CONFIG.apiBaseUrl}/api/download?path=${encodeURIComponent(a.path)}')">
                    <div class="h-10 w-10 rounded-lg ${iconInfo.bg} flex items-center justify-center border ${iconInfo.border}">
                        <span class="material-symbols-outlined ${iconInfo.text}">${iconInfo.icon}</span>
                    </div>
                    <div class="flex-1">
                        <p class="text-sm font-medium text-white">${escapeHtml(a.name)}</p>
                        <p class="text-xs text-slate-400">${formatBytes(a.size)}</p>
                    </div>
                </div>
            `;
        }).join('')
        : '<p class="text-center py-8 text-slate-400">No artifacts in this project</p>';

    modal.innerHTML = `
        <div class="bg-surface-dark border border-white/10 rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden animate-fadeIn">
            <div class="flex items-center justify-between p-4 border-b border-white/5">
                <h3 class="text-lg font-bold text-white flex items-center gap-2">
                    <span class="material-symbols-outlined text-primary">folder_open</span>
                    Project Files
                </h3>
                <button class="modal-close p-1.5 text-slate-400 hover:text-white hover:bg-white/10 rounded-lg transition-all">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="max-h-96 overflow-y-auto custom-scrollbar p-4 grid gap-2">
                ${artifactsHtml}
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    modal.querySelector('.modal-close')?.addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
}

// =====================================================
// EXPORT SESSION
// =====================================================

function initExportSession() {
    const exportBtn = document.querySelector('.export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportCurrentSession);
    }
}

function exportCurrentSession() {
    const session = {
        id: state.currentSessionId,
        exportedAt: new Date().toISOString(),
        messages: state.messages,
        artifacts: state.artifacts.map(a => ({ name: a.name, type: a.type, size: a.size })),
        auditLog: state.auditLog,
        codeSteps: state.codeSteps,
    };

    const blob = new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bioagents_session_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Session exported', 'success');
    logTerminal('Session exported to JSON');
}

// =====================================================
// NOTIFICATIONS
// =====================================================

function initNotifications() {
    const notifBtn = document.querySelector('.notifications-btn');
    if (notifBtn) {
        notifBtn.addEventListener('click', showNotifications);
    }

    // Help button
    const helpBtn = document.querySelector('.help-btn');
    if (helpBtn) {
        helpBtn.addEventListener('click', showHelp);
    }
}

function showHelp() {
    const helpContent = `
BioAgents Help
==============

Quick Start:
1. Type a research question in the input box
2. Click "Run Agent" or press Enter
3. Watch agents collaborate to solve your problem

Features:
• Protein Analysis - Analyze sequences and properties
• Structure Prediction - Visualize 3D molecular structures
• Literature Search - Search scientific databases
• Code Generation - Generate analysis scripts
• Report Writing - Create comprehensive summaries

Tips:
• Use example chips for common queries
• Drag & drop PDB files into the viewer
• Attach FASTA files for sequence analysis
• Export sessions to save your work

Keyboard Shortcuts:
• Enter - Submit query
• Shift+Enter - New line in input
    `.trim();

    alert(helpContent);
}

function addNotification(message, type = 'info') {
    state.notifications.unshift({
        id: Date.now(),
        message,
        type,
        timestamp: new Date().toISOString(),
        read: false,
    });

    // Keep only last 20 notifications
    if (state.notifications.length > 20) {
        state.notifications = state.notifications.slice(0, 20);
    }

    renderNotificationBadge();
}

function renderNotificationBadge() {
    const unreadCount = state.notifications.filter(n => !n.read).length;
    const badge = document.querySelector('.notifications-badge') ||
                  document.querySelector('button:has(.material-symbols-outlined) span.bg-primary');

    if (badge) {
        badge.style.display = unreadCount > 0 ? 'block' : 'none';
    }
}

function showNotifications() {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 z-50 flex items-start justify-end p-4 pointer-events-none';
    modal.id = 'notificationsModal';

    const notifHtml = state.notifications.length > 0
        ? state.notifications.slice(0, 10).map(n => {
            const typeColors = {
                success: 'text-emerald-400 bg-emerald-500/10',
                error: 'text-red-400 bg-red-500/10',
                warning: 'text-amber-400 bg-amber-500/10',
                info: 'text-primary bg-primary/10',
            };
            const colors = typeColors[n.type] || typeColors.info;
            return `
                <div class="flex items-start gap-3 p-3 ${colors} rounded-lg">
                    <span class="material-symbols-outlined text-[16px] mt-0.5">${n.type === 'success' ? 'check_circle' : n.type === 'error' ? 'error' : 'info'}</span>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm text-white">${escapeHtml(n.message)}</p>
                        <p class="text-[10px] text-slate-400 mt-0.5">${new Date(n.timestamp).toLocaleTimeString()}</p>
                    </div>
                </div>
            `;
        }).join('')
        : '<p class="text-center py-8 text-slate-400 text-sm">No notifications</p>';

    modal.innerHTML = `
        <div class="bg-surface-dark border border-white/10 rounded-xl shadow-2xl w-80 overflow-hidden pointer-events-auto animate-fadeIn mt-12">
            <div class="flex items-center justify-between p-3 border-b border-white/5">
                <h3 class="text-sm font-bold text-white">Notifications</h3>
                <div class="flex gap-1">
                    <button class="notif-clear text-[10px] text-slate-400 hover:text-white px-2 py-1 hover:bg-white/5 rounded transition-all">Clear all</button>
                    <button class="notif-close p-1 text-slate-400 hover:text-white hover:bg-white/10 rounded transition-all">
                        <span class="material-symbols-outlined text-[16px]">close</span>
                    </button>
                </div>
            </div>
            <div class="max-h-80 overflow-y-auto custom-scrollbar p-2 space-y-2">
                ${notifHtml}
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Mark all as read
    state.notifications.forEach(n => n.read = true);
    renderNotificationBadge();

    // Close button
    modal.querySelector('.notif-close')?.addEventListener('click', () => modal.remove());

    // Clear all
    modal.querySelector('.notif-clear')?.addEventListener('click', () => {
        state.notifications = [];
        renderNotificationBadge();
        modal.remove();
        showToast('Notifications cleared', 'info');
    });

    // Click outside to close
    setTimeout(() => {
        document.addEventListener('click', function closeNotif(e) {
            if (!modal.contains(e.target)) {
                modal.remove();
                document.removeEventListener('click', closeNotif);
            }
        });
    }, 100);
}

// =====================================================
// SETTINGS
// =====================================================

function initSettings() {
    const settingsBtn = document.querySelector('.settings-btn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', showSettingsPanel);
    }

    // Terminal clear button
    const terminalClear = document.querySelector('.terminal-clear');
    if (terminalClear) {
        terminalClear.addEventListener('click', () => {
            if (elements.terminalContent) {
                elements.terminalContent.innerHTML = '';
                logTerminal('Log cleared');
            }
        });
    }
}

function showSettingsPanel() {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm';
    modal.id = 'settingsModal';

    modal.innerHTML = `
        <div class="bg-surface-dark border border-white/10 rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-fadeIn">
            <div class="flex items-center justify-between p-4 border-b border-white/5">
                <h3 class="text-lg font-bold text-white flex items-center gap-2">
                    <span class="material-symbols-outlined text-primary">settings</span>
                    Settings
                </h3>
                <button class="settings-close p-1.5 text-slate-400 hover:text-white hover:bg-white/10 rounded-lg transition-all">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="p-4 space-y-4">
                <div class="flex items-center justify-between py-2">
                    <div>
                        <p class="text-sm font-medium text-white">Auto-scroll</p>
                        <p class="text-xs text-slate-400">Scroll to new messages automatically</p>
                    </div>
                    <button class="setting-toggle w-12 h-6 rounded-full ${state.settings.autoScroll ? 'bg-primary' : 'bg-slate-700'} relative transition-colors" data-setting="autoScroll">
                        <span class="absolute top-1 ${state.settings.autoScroll ? 'right-1' : 'left-1'} w-4 h-4 bg-white rounded-full transition-all"></span>
                    </button>
                </div>
                <div class="flex items-center justify-between py-2">
                    <div>
                        <p class="text-sm font-medium text-white">Notifications</p>
                        <p class="text-xs text-slate-400">Show toast notifications</p>
                    </div>
                    <button class="setting-toggle w-12 h-6 rounded-full ${state.settings.notifications ? 'bg-primary' : 'bg-slate-700'} relative transition-colors" data-setting="notifications">
                        <span class="absolute top-1 ${state.settings.notifications ? 'right-1' : 'left-1'} w-4 h-4 bg-white rounded-full transition-all"></span>
                    </button>
                </div>
                <div class="pt-4 border-t border-white/5 space-y-3">
                    <p class="text-sm font-medium text-white flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary text-[18px]">key</span>
                        API Keys (optional)
                    </p>
                    <p class="text-xs text-slate-400">Use your own API keys. Leave blank to use server-configured keys.</p>
                    <div>
                        <label class="block text-xs text-slate-500 mb-1">OpenAI API Key</label>
                        <input type="password" id="apiKeyOpenAI" class="api-key-input w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-sm text-white placeholder-slate-500 focus:border-primary/50 focus:ring-1 focus:ring-primary/30 outline-none" placeholder="sk-..." value="${state.settings.apiKeys?.openai || ''}" autocomplete="off"/>
                    </div>
                    <div>
                        <label class="block text-xs text-slate-500 mb-1">Google Gemini API Key</label>
                        <input type="password" id="apiKeyGemini" class="api-key-input w-full px-3 py-2 rounded-lg bg-black/30 border border-white/10 text-sm text-white placeholder-slate-500 focus:border-primary/50 focus:ring-1 focus:ring-primary/30 outline-none" placeholder="AIza..." value="${state.settings.apiKeys?.gemini || ''}" autocomplete="off"/>
                    </div>
                    <button class="api-keys-save w-full py-2 px-4 text-sm font-medium bg-primary/20 hover:bg-primary/30 border border-primary/30 text-primary rounded-lg transition-all">
                        Save API Keys
                    </button>
                </div>
                <div class="pt-4 border-t border-white/5">
                    <button class="clear-all-data w-full py-2.5 px-4 text-sm font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 border border-red-500/30 rounded-lg transition-all">
                        Clear All Data
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Close button
    modal.querySelector('.settings-close')?.addEventListener('click', () => modal.remove());

    // Click outside to close
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });

    // Toggle buttons
    modal.querySelectorAll('.setting-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const setting = btn.dataset.setting;
            state.settings[setting] = !state.settings[setting];
            saveToStorage();

            // Update toggle visual
            btn.classList.toggle('bg-primary', state.settings[setting]);
            btn.classList.toggle('bg-slate-700', !state.settings[setting]);
            btn.querySelector('span').classList.toggle('right-1', state.settings[setting]);
            btn.querySelector('span').classList.toggle('left-1', !state.settings[setting]);

            showToast(`${setting} ${state.settings[setting] ? 'enabled' : 'disabled'}`, 'success');
        });
    });

    // Save API keys
    modal.querySelector('.api-keys-save')?.addEventListener('click', () => {
        const openaiInput = modal.querySelector('#apiKeyOpenAI');
        const geminiInput = modal.querySelector('#apiKeyGemini');
        if (!state.settings.apiKeys) state.settings.apiKeys = { openai: '', gemini: '' };
        state.settings.apiKeys.openai = openaiInput?.value?.trim() || '';
        state.settings.apiKeys.gemini = geminiInput?.value?.trim() || '';
        saveToStorage();
        showToast('API keys saved', 'success');
    });

    // Clear all data
    modal.querySelector('.clear-all-data')?.addEventListener('click', () => {
        if (confirm('Clear all sessions and data? This cannot be undone.')) {
            localStorage.removeItem(CONFIG.storageKey);
            state.sessions = [];
            state.messages = [];
            state.artifacts = [];
            state.auditLog = [];
            startNewSession();
            modal.remove();
            showToast('All data cleared', 'success');
        }
    });
}

// =====================================================
// TABS
// =====================================================

function initTabSwitcher() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;

            // Update button states
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('text-white', 'border-primary', 'bg-white/5');
                b.classList.add('text-slate-500', 'border-transparent');
            });
            btn.classList.add('text-white', 'border-primary', 'bg-white/5');
            btn.classList.remove('text-slate-500', 'border-transparent');

            // Show/hide content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            document.getElementById(tab + 'Tab')?.classList.remove('hidden');
        });
    });
}

function initExampleChips() {
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            if (query && elements.queryInput) {
                elements.queryInput.value = query;
                autoResizeTextarea();
                elements.queryInput.focus();
            }
        });
    });
}

// =====================================================
// UI HELPERS
// =====================================================

function setProcessingState(isProcessing) {
    state.isProcessing = isProcessing;

    if (elements.submitBtn) {
        elements.submitBtn.disabled = isProcessing;
        elements.submitBtn.innerHTML = isProcessing
            ? '<span class="animate-spin material-symbols-outlined text-[18px]">progress_activity</span><span>Processing...</span>'
            : '<span>Run Agent</span><span class="material-symbols-outlined text-[18px]">send</span>';
    }

    if (elements.queryInput) {
        elements.queryInput.disabled = isProcessing;
    }

    updateStatus(isProcessing ? 'processing' : 'ready');
}

function updateStatus(status) {
    if (!elements.statusDot || !elements.statusText) return;

    elements.statusDot.className = 'w-2 h-2 rounded-full';

    switch (status) {
        case 'ready':
            elements.statusDot.classList.add('bg-green-500', 'animate-pulse');
            elements.statusText.textContent = 'System Ready • Awaiting Input';
            break;
        case 'processing':
            elements.statusDot.classList.add('bg-primary', 'animate-pulse');
            const agentName = state.currentAgent ? AGENTS[state.currentAgent]?.label : 'Agent';
            elements.statusText.textContent = `Processing • ${agentName} Active`;
            break;
        case 'disconnected':
            elements.statusDot.classList.add('bg-yellow-500');
            elements.statusText.textContent = 'Reconnecting...';
            break;
        case 'error':
            elements.statusDot.classList.add('bg-red-500');
            elements.statusText.textContent = 'Connection Error';
            break;
        case 'rest':
            elements.statusDot.classList.add('bg-blue-500', 'animate-pulse');
            elements.statusText.textContent = 'REST Mode • Ready';
            break;
    }
}

function logTerminal(message) {
    if (!elements.terminalContent) return;

    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const line = document.createElement('p');
    line.innerHTML = `<span class="text-slate-500">&gt;</span> <span class="text-slate-600">[${time}]</span> ${escapeHtml(message)}`;
    elements.terminalContent.appendChild(line);

    // Keep only last 15 lines
    while (elements.terminalContent.children.length > 15) {
        elements.terminalContent.firstChild.remove();
    }

    // Scroll terminal
    elements.terminalContent.parentElement?.scrollTo(0, elements.terminalContent.parentElement.scrollHeight);
}

function showToast(message, type = 'info') {
    if (!elements.toastContainer) return;

    const colors = {
        success: 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400',
        error: 'bg-red-500/20 border-red-500/50 text-red-400',
        warning: 'bg-amber-500/20 border-amber-500/50 text-amber-400',
        info: 'bg-primary/20 border-primary/50 text-primary',
    };

    const icons = { success: 'check_circle', error: 'error', warning: 'warning', info: 'info' };

    const toast = document.createElement('div');
    toast.className = `flex items-center gap-3 px-4 py-3 rounded-lg border ${colors[type]} text-sm font-medium shadow-lg backdrop-blur-sm`;
    toast.style.animation = 'slideIn 0.3s ease-out';
    toast.innerHTML = `
        <span class="material-symbols-outlined text-[18px]">${icons[type]}</span>
        <span class="flex-1">${escapeHtml(message)}</span>
        <button class="toast-close p-1 hover:bg-white/10 rounded transition-colors">
            <span class="material-symbols-outlined text-[14px]">close</span>
        </button>
    `;

    toast.querySelector('.toast-close')?.addEventListener('click', () => {
        toast.remove();
    });

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// =====================================================
// UTILITIES
// =====================================================

function getColorClass(color) {
    const colors = {
        primary: { bg: 'bg-primary/10', text: 'text-primary', border: 'border-primary/30' },
        slate: { bg: 'bg-slate-800', text: 'text-slate-400', border: 'border-slate-700' },
        blue: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30' },
        emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30' },
        amber: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
        pink: { bg: 'bg-pink-500/10', text: 'text-pink-400', border: 'border-pink-500/30' },
        indigo: { bg: 'bg-indigo-500/10', text: 'text-indigo-400', border: 'border-indigo-500/30' },
        rose: { bg: 'bg-rose-500/10', text: 'text-rose-400', border: 'border-rose-500/30' },
        purple: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30' },
    };
    return colors[color] || colors.slate;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function formatTime(date) {
    if (!(date instanceof Date)) date = new Date(date);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// =====================================================
// CSS ANIMATIONS
// =====================================================

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(100%); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fadeIn {
        animation: fadeIn 0.3s ease-out;
    }
`;
document.head.appendChild(style);

// =====================================================
// DEMO FUNCTIONS
// =====================================================

async function loadDemoStructure(identifier) {
    /**
     * Load a protein structure from the demo API for 3D viewing.
     * @param {string} identifier - UniProt ID (e.g., "P04637") or PDB ID (e.g., "1YCR")
     */
    showToast(`Loading structure: ${identifier}...`, 'info');
    logTerminal(`Fetching structure: ${identifier}`);

    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/api/demo/structure/${identifier}`);
        if (!response.ok) {
            throw new Error(`Structure not found: ${identifier}`);
        }

        const data = await response.json();
        loadStructure(data.pdbContent);

        // Add as artifact only if not already present
        const artifactName = data.source === 'AlphaFold'
            ? `AF-${identifier}-F1-model_v4.pdb`
            : `${identifier}.pdb`;

        if (!state.artifacts.some(a => a.name.includes(identifier))) {
            addArtifact({
                name: artifactName,
                type: 'pdb',
                size: data.pdbContent.length,
                path: `demo://${artifactName}`,
            });
        }

        showToast(`Loaded ${data.source} structure: ${identifier}`, 'success');
        logTerminal(`Loaded structure from ${data.source}`);

        return data;
    } catch (error) {
        showToast(error.message, 'error');
        logTerminal(`ERROR: ${error.message}`);
        throw error;
    }
}

async function loadDemoArtifacts() {
    /**
     * Load available demo artifacts from the server.
     */
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/api/demo/artifacts`);
        if (!response.ok) return;

        const data = await response.json();

        for (const artifact of data.artifacts) {
            addArtifact(artifact);
        }

        if (data.artifacts.length > 0) {
            showToast(`Found ${data.artifacts.length} artifacts`, 'success');
        }
    } catch (error) {
        console.log('No demo artifacts available');
    }
}

async function runDemo() {
    /**
     * Run a quick demo showcasing the UI features.
     */
    showToast('Starting demo...', 'info');
    logTerminal('=== DEMO MODE ACTIVATED ===');

    // Simulate agent workflow
    const agents = ['supervisor', 'research', 'analysis', 'protein_design', 'report'];

    for (let i = 0; i < agents.length; i++) {
        setActiveAgent(agents[i], (i + 1) * 20);
        await sleep(800);
    }

    // Load a demo structure (p53)
    try {
        await loadDemoStructure('P04637');
    } catch (e) {
        // Try a PDB structure instead
        try {
            await loadDemoStructure('1YCR');
        } catch (e2) {
            logTerminal('Could not load demo structure');
        }
    }

    // Add a demo message
    addAssistantMessage(`## Demo Complete! 🧬

This is a demonstration of the BioAgents UI capabilities:

### Features Showcased:
1. **Agent Workflow** - Multi-agent orchestration visualization
2. **3D Molecular Viewer** - Interactive protein structure display
3. **Artifacts Panel** - Generated files and outputs
4. **Chat Interface** - Natural language interaction

### Try These Queries:
- "Analyze the properties of human p53 protein (P04637)"
- "Fetch the structure of insulin (P01308)"
- "Search literature on CRISPR-Cas9 applications"

The system uses specialized AI agents for research, analysis, code generation, and protein design.`, 'research');

    setActiveAgent(null);
    logTerminal('Demo complete');
    showToast('Demo complete!', 'success');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// =====================================================
// KEYBOARD SHORTCUTS
// =====================================================

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + D for demo
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        runDemo();
    }

    // Escape to close modals
    if (e.key === 'Escape') {
        document.querySelectorAll('#settingsModal, #notificationsModal').forEach(m => m.remove());
    }
});

// =====================================================
// EXPOSE FOR DEBUGGING
// =====================================================

window.BioAgents = {
    state,
    loadStructure,
    loadDemoStructure,
    loadDemoArtifacts,
    runDemo,
    addArtifact,
    addAssistantMessage,
    setActiveAgent,
    showToast,
    logTerminal,
};
