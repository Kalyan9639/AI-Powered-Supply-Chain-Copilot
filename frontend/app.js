import { setLanguage } from './i18n.js';
const CLOUD_RUN_API_BASE = 'https://ai-powered-supply-chain-copilot-473731334420.us-central1.run.app/api';
const API_BASE = ['localhost', '127.0.0.1'].includes(window.location.hostname)
    ? '/api'
    : CLOUD_RUN_API_BASE;
const RUNTIME_CONFIG_STORAGE_KEY = 'agentic_copilot_runtime_config_v1';
// Set default language on load
setLanguage('en');
let currentConfig = null;
let currentAssessments = [];
let runtimeConfigRestorePromise = null;

function loadBrowserRuntimeConfig() {
    try {
        const raw = localStorage.getItem(RUNTIME_CONFIG_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : null;
        return parsed && typeof parsed === 'object' ? parsed : null;
    } catch (error) {
        console.warn('Unable to load browser runtime config:', error);
        return null;
    }
}

function saveBrowserRuntimeConfig(data) {
    try {
        localStorage.setItem(RUNTIME_CONFIG_STORAGE_KEY, JSON.stringify({
            hf_api_key: data.hf_api_key || '',
            hf_model: data.hf_model || '',
            scrape_interval_minutes: data.scrape_interval_minutes ?? 1,
            target_urls: data.target_urls || [],
            log_level: data.log_level || 'INFO',
            debug: Boolean(data.debug)
        }));
    } catch (error) {
        console.warn('Unable to save browser runtime config:', error);
    }
}

async function restoreRuntimeConfigFromBrowser() {
    const cachedConfig = loadBrowserRuntimeConfig();
    if (!cachedConfig || !cachedConfig.hf_api_key) {
        return null;
    }

    const response = await fetch(`${API_BASE}/runtime-config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cachedConfig)
    });

    if (!response.ok) {
        throw new Error(`Runtime config restore failed with status ${response.status}`);
    }

    const restoredConfig = await response.json();
    saveBrowserRuntimeConfig(restoredConfig);
    return restoredConfig;
}

function ensureRuntimeConfigRestored() {
    if (!runtimeConfigRestorePromise) {
        runtimeConfigRestorePromise = restoreRuntimeConfigFromBrowser().catch(error => {
            console.warn('Runtime config restore skipped:', error);
            return null;
        });
    }

    return runtimeConfigRestorePromise;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

async function fetchAssessments() {
    try {
        const response = await fetch(`${API_BASE}/assessments?limit=10`);
        const data = await response.json();
        renderAssessments(data);
    } catch (error) {
        console.error('Error fetching assessments:', error);
        showToast('Failed to load assessments');
    }
}

async function runCycle() {
    const btn = document.querySelector('button[onclick="runCycle()"]');
    const originalText = btn.innerText;
    btn.disabled = true;
    btn.innerText = 'Running...';
    try {
        await ensureRuntimeConfigRestored();
        const response = await fetch(`${API_BASE}/run-cycle`, { method: 'POST' });
        if (!response.ok) {
            const errorPayload = await response.json().catch(() => null);
            throw new Error(errorPayload?.detail || `Run failed with status ${response.status}`);
        }
        const data = await response.json();
        showToast(`Cycle complete! ${data.assessments_count} assessments generated.`);
        renderAssessments(data.assessments || []);
    } catch (error) {
        console.error('Error running cycle:', error);
        showToast('Error running cycle: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerText = originalText;
    }
}

function normalizeAssessments(assessments) {
    return (assessments || []).map(a => ({ ...a, ui_status: a.ui_status || 'pending' }));
}

function reviewAssessment(newsUrl) {
    if (!newsUrl) {
        showToast('No source article link is available for this assessment.');
        return;
    }
    window.open(newsUrl, '_blank', 'noopener,noreferrer');
}

function setAssessmentDecision(newsId, decision) {
    currentAssessments = currentAssessments.map(a => (
        a.news_id === newsId ? { ...a, ui_status: decision } : a
    ));
    renderAssessments(currentAssessments);
}

function renderAssessments(assessments) {
    const container = document.getElementById('assessments-container');
    currentAssessments = normalizeAssessments(assessments || []);
    const sorted = sortAssessments(currentAssessments);
    if (!sorted || sorted.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 40px;">No assessments yet. Run an agent cycle to analyze news.</div>';
        return;
    }
    container.innerHTML = sorted.map(a => `
        <div class="action-item risk-${a.risk_level.toLowerCase()}">
            <div class="news-title">${escapeHtml(a.news_title || 'Risk Assessment')}</div>
            <div class="news-content">${escapeHtml(a.reasoning)}</div>
            ${a.proposed_action ? `<div class="reasoning"><strong>Proposed Action:</strong> ${escapeHtml(a.proposed_action)}</div>` : ''}
            <div class="news-meta">
                <span>Confidence: ${(a.confidence * 100).toFixed(0)}%</span>
                <span class="risk-badge risk-${a.risk_level.toLowerCase()}">${a.risk_level} Risk</span>
            </div>
            <div class="action-buttons">
                ${a.risk_level === 'Low' || a.ui_status !== 'pending' ? '' : `<button class="btn-primary" aria-label="Approve" onclick='setAssessmentDecision(${JSON.stringify(a.news_id)}, "approved")'>Approve</button>`}
                <button class="btn-secondary" aria-label="Review" onclick='reviewAssessment(${JSON.stringify(a.news_url || '')})'>Review</button>
                ${a.risk_level === 'Low' || a.ui_status !== 'pending' ? '' : `<button class="btn-danger" aria-label="Discard" onclick='setAssessmentDecision(${JSON.stringify(a.news_id)}, "discarded")'>Discard</button>`}
            </div>
        </div>`).join('');
    document.getElementById('total-news').innerText = sorted.length;
    const highRiskCount = sorted.filter(x => x.risk_level === 'High').length;
    document.getElementById('high-risk').innerText = highRiskCount;
}

function sortAssessments(assessments) {
    const rank = { High: 0, Medium: 1, Low: 2 };
    return [...assessments].sort((a, b) => {
        const r = (rank[a.risk_level] ?? 3) - (rank[b.risk_level] ?? 3);
        if (r !== 0) return r;
        const c = (b.confidence ?? 0) - (a.confidence ?? 0);
        return c;
    });
}

async function fetchConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();
        currentConfig = config;
        document.getElementById('business-name').value = config.business_name;
        document.getElementById('commodity').value = config.commodity;
        document.getElementById('region').value = config.region;
        document.getElementById('rules-textarea').value = (config.rules || []).join('\n');
        document.getElementById('save-status').innerText = '';
    } catch (error) {
        console.error('Error fetching config:', error);
        document.getElementById('save-status').innerText = 'Unable to load business profile.';
    }
}

async function saveConfig() {
    const rules = document.getElementById('rules-textarea').value
        .split('\n')
        .map(r => r.trim())
        .filter(Boolean);
    const payload = {
        business_name: document.getElementById('business-name').value.trim(),
        commodity: document.getElementById('commodity').value.trim(),
        region: document.getElementById('region').value.trim(),
        rules,
        created_at: currentConfig?.created_at || new Date().toISOString(),
    };
    try {
        const response = await fetch(`${API_BASE}/config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(`Save failed with status ${response.status}`);
        currentConfig = await response.json();
        document.getElementById('save-status').innerText = 'Business profile saved.';
    } catch (error) {
        console.error('Error saving config:', error);
        document.getElementById('save-status').innerText = 'Failed to save business profile.';
    }
}

function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// Initialize
ensureRuntimeConfigRestored();
fetchAssessments();
fetchConfig();
