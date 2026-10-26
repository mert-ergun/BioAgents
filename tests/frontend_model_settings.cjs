// Run with: node tests/frontend_model_settings.cjs
async function runModelSettingsChecks(source) {
    const build = new Function('window', 'document', 'localStorage', 'navigator', 'screen',
        'fetch', 'WebSocket', 'TextDecoder', 'console', source + `
        autoResizeTextarea = () => {}; addUserMessage = () => {};
        setProcessingState = () => {}; logTerminal = () => {};
        showToast = () => {}; handleComplete = () => {};
        return {state, elements, loadFromStorage, showSettingsPanel, handleSubmit, sendQueryViaRest};
    `);
    function setup(settings, configuration = {provider: 'openrouter', model: 'server/custom-model'}, status = 200) {
        const saved = new Map();
        if (settings) saved.set('bioagents_sessions', JSON.stringify({settings, sessions: []}));
        const writes = [], configReads = [], widgets = new Map();
        function widget() {
            let html = '';
            return {
                value: '', get innerHTML() { return html; }, set innerHTML(value) { html = value; },
                set textContent(value) { html = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },
                classList: {add() {}, toggle() {}},
                addEventListener(event, fn) { this[event] = fn; }, appendChild() {},
            };
        }
        const modal = {
            ...widget(), querySelector(sel) {
                if (!widgets.has(sel)) widgets.set(sel, widget());
                return widgets.get(sel);
            }, querySelectorAll() { return []; },
        };
        let modalCreated = false;
        const document = {
            addEventListener() {}, createElement(tag) {
                if (tag === 'div' && !modalCreated) { modalCreated = true; return modal; }
                return widget();
            },
            head: {appendChild() {}}, body: {appendChild() {}}, getElementById() { return null; },
        };
        const storage = {
            getItem(k) { return saved.get(k) || null; }, setItem(k, v) { saved.set(k, v); },
        };
        const fetch = async (url, options) => {
            if (url.endsWith('/api/llm/config')) {
                configReads.push(url);
                return {ok: status === 200, status, json: async () => configuration};
            }
            writes.push(JSON.parse(options.body));
            return {ok: true, body: {getReader() { return {read: async () => ({done: true})}; }}};
        };
        const app = build({location: {protocol: 'http:', host: 'localhost', origin: 'http://localhost'}},
            document, storage, {userAgent: 'test'}, {width: 100, height: 100}, fetch,
            {OPEN: 1}, class TextDecoder {}, {error() {}});
        return {app, saved, writes, configReads, widgets, modal};
    }
    function assert(condition, message) {
        if (!condition) throw new Error(message);
    }
    async function sendBothTransports(environment) {
        await environment.app.sendQueryViaRest('test');
        environment.app.elements.queryInput = {value: 'test'};
        environment.app.state.isConnected = true;
        environment.app.state.ws = {readyState: 1, send(data) { environment.writes.push(JSON.parse(data)); }};
        await environment.app.handleSubmit();
        for (const payload of environment.writes) {
            assert(!('provider' in payload) && !('model' in payload),
                'Browser must always defer model selection to server configuration');
        }
    }
    const fresh = setup();
    await sendBothTransports(fresh);
    for (const provider of ['openai', 'gemini', 'ollama', 'openrouter']) {
        for (const oldProvider of ['openai', 'gemini', 'ollama', 'openrouter']) {
            const configuration = {provider, model: `${provider}-custom-deployment`};
            const environment = setup({llmProvider: oldProvider, llmModel: 'old/custom-model',
                llmSelectionExplicit: true, theme: 'light', apiKeys: {gemini: 'test-key'}}, configuration);
            environment.app.loadFromStorage();
            assert(!('llmProvider' in environment.app.state.settings)
                && !('llmModel' in environment.app.state.settings), 'All stale model selections must migrate');
            assert(environment.app.state.settings.theme === 'light'
                && environment.app.state.settings.apiKeys.gemini === 'test-key', 'Other preferences must survive');
            const persisted = JSON.parse(environment.saved.get('bioagents_sessions')).settings;
            assert(!('llmProvider' in persisted) && !('llmModel' in persisted), 'Migration must persist');
            await sendBothTransports(environment);
            await environment.app.showSettingsPanel();
            assert(environment.configReads.length === 1, 'Settings must load current server configuration');
            assert(environment.modal.innerHTML.includes(configuration.model)
                && environment.modal.innerHTML.includes(configuration.provider), 'Settings must show the configured custom model');
            assert(!environment.modal.innerHTML.includes('<select'), 'Model selection belongs to the server');
            environment.modal.querySelector('#apiKeyOpenAI').value = 'new-key';
            environment.widgets.get('.settings-save-all').click();
            assert(environment.app.state.settings.apiKeys.openai === 'new-key', 'API key settings must still save');
            assert(!('llmProvider' in environment.app.state.settings), 'Saving settings must not restore overrides');
        }
    }
    // Even an in-memory stale field must not affect either transport.
    fresh.app.state.settings.llmProvider = 'gemini';
    fresh.app.state.settings.llmModel = 'stale-model';
    await sendBothTransports(fresh);
    const missing = setup(null, null, 503);
    await missing.app.showSettingsPanel();
    assert(missing.modal.innerHTML.includes('Unavailable'), 'A failed config lookup must not invent a model');
    await sendBothTransports(missing);
    const defaults = setup(null, {provider: 'openai', model: null});
    await defaults.app.showSettingsPanel();
    assert(defaults.modal.innerHTML.includes('Agent defaults'), 'Unconfigured models must be shown honestly');
    const escaped = setup(null, {provider: 'openai', model: '<img src=x onerror=bad()>'});
    await escaped.app.showSettingsPanel();
    assert(escaped.modal.innerHTML.includes('&lt;img') && !escaped.modal.innerHTML.includes('<img'),
        'Server model names must be rendered as text');
    return 'Frontend model selection checks passed';
}

if (typeof module !== 'undefined' && require.main === module) {
    const fs = require('node:fs');
    const path = require('node:path');
    runModelSettingsChecks(fs.readFileSync(path.join(__dirname, '../frontend/static/js/app.js'), 'utf8'))
        .then(message => console.log(message))
        .catch(error => { console.error(error); process.exitCode = 1; });
}
