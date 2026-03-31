// ========== Tab 切换 ==========
function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${name}')"]`).classList.add('active');
    document.getElementById(`tab-${name}`).classList.add('active');

    // 加载数据
    if (name === 'providers' || name === 'models') loadConfig();
    if (name === 'keys') loadKeys();
    if (name === 'usage') loadUsage();
    if (name === 'logs') loadLogs();
}

// ========== API 辅助 ==========
async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    return resp.json();
}

// ========== 全局状态 ==========
let _currentProviders = [];  // 用于 adapter 下拉框
let _adapterRows = [];       // 当前 model form 中的 adapter 列表

// ========== 供应商管理 ==========
async function loadConfig() {
    const config = await api('GET', '/api/config');
    _currentProviders = config.providers || [];
    renderProviders(config.providers || []);
    renderModels(config.models || {});
}

function renderProviders(providers) {
    const el = document.getElementById('providers-list');
    if (!providers.length) { el.innerHTML = '<p style="color:#999">暂无供应商配置</p>'; return; }
    el.innerHTML = providers.map(p => `
        <div class="card" id="provider-card-${esc(p.name)}">
            <div class="card-header">
                <span class="card-title">${esc(p.name)}</span>
                <span class="badge ${p.enabled ? 'badge-green' : 'badge-red'}">${p.enabled ? '启用' : '禁用'}</span>
            </div>
            <div class="card-body">
                <table>
                    <tr><td>协议</td><td><span class="badge badge-blue">${esc(p.provider)}</span></td></tr>
                    <tr><td>Base URL</td><td>${esc(p.base_url)}</td></tr>
                    <tr><td>API Key</td><td>${esc(p.api_key).substring(0, 12)}***</td></tr>
                    ${p.custom_headers && Object.keys(p.custom_headers).length ? `<tr><td>Headers</td><td>${esc(JSON.stringify(p.custom_headers))}</td></tr>` : ''}
                </table>
            </div>
            <div class="card-actions" style="margin-top:8px">
                <button class="btn btn-sm" onclick="editProvider('${esc(p.name)}')">编辑</button>
                <button class="btn btn-sm btn-test" onclick="testProvider('${esc(p.name)}')">测试连通</button>
                <button class="btn btn-sm ${p.enabled ? 'btn-danger' : ''}" onclick="toggleProvider('${esc(p.name)}')">${p.enabled ? '禁用' : '启用'}</button>
                <button class="btn btn-sm btn-danger" onclick="deleteProvider('${esc(p.name)}')">删除</button>
            </div>
            <div id="provider-test-${esc(p.name)}" class="test-result" style="display:none"></div>
        </div>
    `).join('');
}

function showAddProvider() {
    document.getElementById('provider-form-title').textContent = '新建供应商';
    document.getElementById('pf-original-name').value = '';
    document.getElementById('pf-name').value = '';
    document.getElementById('pf-provider').value = 'openai';
    document.getElementById('pf-base-url').value = '';
    document.getElementById('pf-api-key').value = '';
    document.getElementById('pf-headers').value = '{}';
    document.getElementById('provider-form').style.display = 'flex';
}

function editProvider(name) {
    api('GET', '/api/config').then(config => {
        const p = (config.providers || []).find(x => x.name === name);
        if (!p) return;
        document.getElementById('provider-form-title').textContent = '编辑供应商';
        document.getElementById('pf-original-name').value = name;
        document.getElementById('pf-name').value = p.name;
        document.getElementById('pf-provider').value = p.provider;
        document.getElementById('pf-base-url').value = p.base_url;
        document.getElementById('pf-api-key').value = p.api_key;
        document.getElementById('pf-headers').value = JSON.stringify(p.custom_headers || {});
        document.getElementById('provider-form').style.display = 'flex';
    });
}

function hideProviderForm() { document.getElementById('provider-form').style.display = 'none'; }

async function saveProvider() {
    const original = document.getElementById('pf-original-name').value;
    const name = document.getElementById('pf-name').value.trim();
    if (!name) return alert('请填写名称');
    let headers = {};
    try { headers = JSON.parse(document.getElementById('pf-headers').value || '{}'); } catch(e) { return alert('Headers JSON 格式错误'); }

    const body = {
        name, provider: document.getElementById('pf-provider').value,
        base_url: document.getElementById('pf-base-url').value.trim(),
        api_key: document.getElementById('pf-api-key').value.trim(),
        custom_headers: headers, enabled: true,
    };
    if (original) {
        await api('PUT', `/api/config/providers/${encodeURIComponent(original)}`, body);
    } else {
        await api('POST', '/api/config/providers', body);
    }
    hideProviderForm();
    loadConfig();
}

async function toggleProvider(name) { await api('PATCH', `/api/config/providers/${encodeURIComponent(name)}/toggle`); loadConfig(); }
async function deleteProvider(name) { if (confirm('确认删除?')) { await api('DELETE', `/api/config/providers/${encodeURIComponent(name)}`); loadConfig(); } }

// ========== 供应商测试 ==========
async function testProvider(name) {
    const el = document.getElementById(`provider-test-${name}`);
    el.style.display = 'block';
    el.innerHTML = '<span class="test-loading">测试中...</span>';
    const result = await api('POST', `/api/config/providers/${encodeURIComponent(name)}/test`);
    if (result.success) {
        el.innerHTML = `<span class="test-ok">连通成功</span> <span class="test-meta">状态: ${result.status_code || 'OK'} · 延迟: ${result.latency_ms}ms</span>`;
    } else {
        el.innerHTML = `<span class="test-fail">连通失败</span> <span class="test-meta">${esc(result.error || '')}</span>`;
    }
}

// ========== 模型配置 ==========
function renderModels(models) {
    const el = document.getElementById('models-list');
    const entries = Object.entries(models);
    if (!entries.length) { el.innerHTML = '<p style="color:#999">暂无模型配置</p>'; return; }
    el.innerHTML = entries.map(([name, m]) => `
        <div class="card" id="model-card-${esc(name)}">
            <div class="card-header">
                <span class="card-title">${esc(name)}</span>
                <span class="badge badge-blue">${esc(m.mode)}</span>
                <span style="margin-left:8px;color:#999;font-size:12px">${esc(m.description || '')}</span>
            </div>
            <div class="adapter-list">
                ${(m.adapters || []).map(a => `
                    <div class="adapter-item">
                        <span class="adapter-priority">P${a.priority}</span>
                        <span class="adapter-provider">${esc(a.adapter)} / ${esc(a.model_name)}</span>
                        <span style="color:#999">timeout: ${a.timeout}s</span>
                    </div>
                `).join('')}
            </div>
            <div class="card-actions" style="margin-top:8px">
                <button class="btn btn-sm" onclick="editModel('${esc(name)}')">编辑</button>
                <button class="btn btn-sm btn-test" onclick="testModel('${esc(name)}')">测试调用</button>
                <button class="btn btn-sm btn-danger" onclick="deleteModel('${esc(name)}')">删除</button>
            </div>
            <div id="model-test-${esc(name)}" class="test-result" style="display:none"></div>
        </div>
    `).join('');
}

function showAddModel() {
    document.getElementById('model-form-title').textContent = '新建模型';
    document.getElementById('mf-original-name').value = '';
    document.getElementById('mf-name').value = '';
    document.getElementById('mf-mode').value = 'chain';
    document.getElementById('mf-description').value = '';
    _adapterRows = [];
    renderAdapterRows();
    document.getElementById('model-form').style.display = 'flex';
}

function editModel(name) {
    api('GET', '/api/config').then(config => {
        const m = config.models[name];
        if (!m) return;
        document.getElementById('model-form-title').textContent = '编辑模型';
        document.getElementById('mf-original-name').value = name;
        document.getElementById('mf-name').value = name;
        document.getElementById('mf-mode').value = m.mode;
        document.getElementById('mf-description').value = m.description || '';
        _adapterRows = (m.adapters || []).map(a => ({...a}));
        renderAdapterRows();
        document.getElementById('model-form').style.display = 'flex';
    });
}

function hideModelForm() { document.getElementById('model-form').style.display = 'none'; }

// ========== 适配器列表动态编辑 ==========
function addAdapterRow() {
    _adapterRows.push({ adapter: '', model_name: '', priority: _adapterRows.length + 1, timeout: 60 });
    renderAdapterRows();
}

function removeAdapterRow(index) {
    _adapterRows.splice(index, 1);
    // 重新编号 priority
    _adapterRows.forEach((r, i) => r.priority = i + 1);
    renderAdapterRows();
}

function renderAdapterRows() {
    const container = document.getElementById('mf-adapters-list');
    if (!_adapterRows.length) {
        container.innerHTML = '<p style="color:#999;font-size:13px">暂无适配器，点击下方按钮添加</p>';
        return;
    }

    container.innerHTML = _adapterRows.map((row, i) => {
        const options = _currentProviders.map(p => {
            const sel = p.name === row.adapter ? ' selected' : '';
            return `<option value="${esc(p.name)}"${sel}>${esc(p.name)}</option>`;
        }).join('');

        return `
        <div class="adapter-form-card">
            <div class="adapter-form-row1">
                <span class="adapter-form-priority">${row.priority}.</span>
                <select class="adapter-form-select" onchange="_adapterRows[${i}].adapter=this.value">
                    <option value="">-- 选择供应商 --</option>
                    ${options}
                </select>
                <button class="btn btn-sm btn-danger adapter-form-remove" onclick="removeAdapterRow(${i})">✕</button>
            </div>
            <div class="adapter-form-row2">
                <input class="adapter-form-model" placeholder="上游模型名，如 glm-5、qwen-max" value="${esc(row.model_name)}"
                       onchange="_adapterRows[${i}].model_name=this.value">
            </div>
            <div class="adapter-form-row2" style="margin-top:4px">
                <label style="font-size:12px;color:#888;flex-shrink:0">超时</label>
                <input class="adapter-form-timeout" type="number" value="${row.timeout}" min="1"
                       onchange="_adapterRows[${i}].timeout=parseInt(this.value)||60">
                <span style="font-size:12px;color:#888">秒</span>
            </div>
        </div>
        `;
    }).join('');
}

async function saveModel() {
    const original = document.getElementById('mf-original-name').value;
    const name = document.getElementById('mf-name').value.trim();
    if (!name) return alert('请填写模型名称');

    // 校验 adapter 数据
    const validAdapters = _adapterRows.filter(r => r.adapter && r.model_name);
    if (!validAdapters.length) return alert('请至少添加一个有效的适配器（供应商和模型名必填）');

    const body = {
        mode: document.getElementById('mf-mode').value,
        description: document.getElementById('mf-description').value.trim(),
        adapters: validAdapters,
    };

    if (original) {
        await api('PUT', `/api/config/models/${encodeURIComponent(original)}`, body);
    } else {
        await api('POST', `/api/config/models?name=${encodeURIComponent(name)}`, body);
    }
    hideModelForm();
    loadConfig();
}

async function deleteModel(name) { if (confirm('确认删除?')) { await api('DELETE', `/api/config/models/${encodeURIComponent(name)}`); loadConfig(); } }

// ========== 模型测试 ==========
async function testModel(name) {
    const el = document.getElementById(`model-test-${name}`);
    el.style.display = 'block';
    el.innerHTML = '<span class="test-loading">测试中...</span>';
    const result = await api('POST', `/api/config/models/${encodeURIComponent(name)}/test`);
    if (result.success) {
        el.innerHTML = `
            <span class="test-ok">调用成功</span>
            <span class="test-meta">适配器: ${esc(result.adapter_used)} · 模型: ${esc(result.model_name || '')} · 延迟: ${result.latency_ms}ms</span>
            ${result.preview ? `<div class="test-preview">"${esc(result.preview)}"</div>` : ''}
            ${result.usage ? `<span class="test-meta">Token: ${result.usage.prompt_tokens || 0} → ${result.usage.completion_tokens || 0}</span>` : ''}
        `;
    } else {
        el.innerHTML = `<span class="test-fail">调用失败</span> <span class="test-meta">${esc(result.error || '')} · ${result.latency_ms}ms</span>`;
    }
}

// ========== API Key 管理 ==========
async function loadKeys() {
    const keys = await api('GET', '/api/keys');
    renderKeys(keys);
}

function renderKeys(keys) {
    const el = document.getElementById('keys-list');
    if (!keys.length) { el.innerHTML = '<p style="color:#999">暂无 API Key</p>'; return; }
    el.innerHTML = keys.map(k => `
        <div class="card">
            <div class="card-header">
                <span class="card-title">${esc(k.name || 'unnamed')}</span>
                <span class="badge ${k.enabled ? 'badge-green' : 'badge-red'}">${k.enabled ? '启用' : '禁用'}</span>
            </div>
            <div class="card-body">
                <table>
                    <tr><td>Key</td><td><code>${esc(k.key)}</code></td></tr>
                    <tr><td>每分钟限流</td><td>${k.rate_limit || '不限'}</td></tr>
                    <tr><td>每日限流</td><td>${k.daily_limit || '不限'}</td></tr>
                    ${k.allowed_models && k.allowed_models.length ? `<tr><td>可用模型</td><td>${k.allowed_models.map(m => `<span class="badge badge-blue">${esc(m)}</span>`).join(' ')}</td></tr>` : ''}
                    ${k.description ? `<tr><td>描述</td><td>${esc(k.description)}</td></tr>` : ''}
                </table>
            </div>
            <div class="card-actions" style="margin-top:8px">
                <button class="btn btn-sm" onclick="toggleKey('${esc(k.key_raw || k.key)}')">${k.enabled ? '禁用' : '启用'}</button>
                <button class="btn btn-sm btn-danger" onclick="deleteKey('${esc(k.key_raw || k.key)}')">删除</button>
            </div>
        </div>
    `).join('');
}

function showAddKey() { document.getElementById('key-form').style.display = 'flex'; }
function hideKeyForm() { document.getElementById('key-form').style.display = 'none'; }

async function createKey() {
    const name = document.getElementById('kf-name').value.trim();
    const desc = document.getElementById('kf-description').value.trim();
    const rate = parseInt(document.getElementById('kf-rate-limit').value) || 0;
    const daily = parseInt(document.getElementById('kf-daily-limit').value) || 0;
    const models = document.getElementById('kf-allowed-models').value.split(',').map(s => s.trim()).filter(Boolean);

    const result = await api('POST', '/api/keys', { name, description: desc, rate_limit: rate, daily_limit: daily, allowed_models: models });
    hideKeyForm();
    document.getElementById('new-key-value').value = result.key;
    document.getElementById('new-key-result').style.display = 'flex';
    loadKeys();
}

function copyKey() {
    navigator.clipboard.writeText(document.getElementById('new-key-value').value);
}

async function toggleKey(key) { await api('PATCH', `/api/keys/${encodeURIComponent(key)}/toggle`); loadKeys(); }
async function deleteKey(key) { if (confirm('确认删除?')) { await api('DELETE', `/api/keys/${encodeURIComponent(key)}`); loadKeys(); } }

// ========== 用量统计 ==========
let usageState = { groupBy: 'provider', dateFrom: null, dateTo: null };

async function loadUsage() {
    const groupBy = document.getElementById('usage-group-by').value;
    const dateRange = document.getElementById('usage-date').value;
    const today = new Date().toISOString().split('T')[0];

    if (dateRange === 'today') { usageState.dateFrom = today; usageState.dateTo = today; }
    else if (dateRange === 'week') {
        const d = new Date(); d.setDate(d.getDate() - 7);
        usageState.dateFrom = d.toISOString().split('T')[0]; usageState.dateTo = today;
    } else if (dateRange === 'month') {
        const d = new Date(); d.setDate(d.getDate() - 30);
        usageState.dateFrom = d.toISOString().split('T')[0]; usageState.dateTo = today;
    }
    usageState.groupBy = groupBy;

    const params = new URLSearchParams({ group_by: groupBy, date_from: usageState.dateFrom, date_to: usageState.dateTo });
    const data = await api('GET', `/api/usage?${params}`);

    const el = document.getElementById('usage-summary');
    const groupLabel = { provider: '服务商', model: '模型', api_key: 'API Key' }[groupBy];
    el.innerHTML = `
        <div class="usage-summary">共 <strong>${data.total}</strong> 次请求，按<strong>${groupLabel}</strong>分组</div>
        <table class="usage-table">
            <tr><th>${groupLabel}</th><th>调用量</th><th>成功率</th><th>输入Token</th><th>输出Token</th><th>平均延迟</th><th></th></tr>
            ${(data.groups || []).map(g => `
                <tr>
                    <td><strong>${esc(g.name)}</strong></td>
                    <td>${g.total_requests}</td>
                    <td>${g.success_rate}%</td>
                    <td>${formatNum(g.tokens_in)}</td>
                    <td>${formatNum(g.tokens_out)}</td>
                    <td>${g.avg_latency_ms}ms</td>
                    <td><button class="btn btn-sm drill-btn" onclick="drillUsage('${esc(g.name)}')">下钻</button></td>
                </tr>
            `).join('')}
        </table>
    `;
    document.getElementById('usage-detail').style.display = 'none';
}

async function drillUsage(itemName) {
    const dimensions = ['provider', 'model', 'api_key'].filter(d => d !== usageState.groupBy);
    if (!dimensions.length) return;
    const subGroup = dimensions[0];

    const params = new URLSearchParams({
        group_by: usageState.groupBy, sub_group: subGroup,
        date_from: usageState.dateFrom, date_to: usageState.dateTo,
    });
    const data = await api('GET', `/api/usage/${encodeURIComponent(itemName)}/detail?${params}`);

    const el = document.getElementById('usage-detail');
    const subLabel = { provider: '服务商', model: '模型', api_key: 'API Key' }[subGroup];

    el.innerHTML = `
        <div class="detail-header">
            <h4>${esc(itemName)} - 按${subLabel}下钻</h4>
            <button class="btn btn-sm" onclick="this.parentElement.parentElement.style.display='none'">关闭</button>
        </div>
        <table class="usage-table">
            <tr><th>${subLabel}</th><th>调用量</th><th>成功</th><th>失败</th><th>输入Token</th><th>输出Token</th><th>平均延迟</th></tr>
            ${data.map(g => `
                <tr>
                    <td>${esc(g.name)}</td>
                    <td>${g.total_requests}</td>
                    <td>${g.success_count}</td>
                    <td>${g.fail_count}</td>
                    <td>${formatNum(g.tokens_in)}</td>
                    <td>${formatNum(g.tokens_out)}</td>
                    <td>${g.avg_latency_ms}ms</td>
                </tr>
            `).join('')}
        </table>
    `;
    el.style.display = 'block';
}

// ========== 调试日志 ==========
let logRefreshTimer = null;

async function loadLogs() {
    const level = document.getElementById('log-level').value;
    const requestId = document.getElementById('log-request-id').value.trim();
    const apiKey = document.getElementById('log-api-key').value.trim();

    const params = new URLSearchParams({ tail: 200 });
    if (level) params.set('level', level);
    if (requestId) params.set('request_id', requestId);
    if (apiKey) params.set('api_key', apiKey);

    const data = await api('GET', `/api/logs?${params}`);
    renderLogs(data.logs || []);

    // 自动刷新
    const autoRefresh = document.getElementById('log-auto-refresh').checked;
    if (autoRefresh && !logRefreshTimer) {
        logRefreshTimer = setInterval(loadLogs, 3000);
    } else if (!autoRefresh && logRefreshTimer) {
        clearInterval(logRefreshTimer);
        logRefreshTimer = null;
    }
}

function renderLogs(logs) {
    const el = document.getElementById('log-container');
    if (!logs.length) { el.innerHTML = '<div class="log-line log-INFO">暂无日志</div>'; return; }
    el.innerHTML = logs.map(l => {
        const ts = l.timestamp ? l.timestamp.split('T')[1].split('.')[0] : '';
        const level = l.level || 'INFO';
        const rid = l.request_id ? `[${l.request_id}]` : '';
        return `<div class="log-line log-${level}"><span class="log-timestamp">${ts}</span> <span class="log-${level}">[${level}]</span> ${rid} ${esc(l.message)}</div>`;
    }).join('');
    el.scrollTop = el.scrollHeight;
}

// ========== 工具函数 ==========
function esc(s) { if (s == null) return ''; const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }
function formatNum(n) { if (n == null) return '0'; return n >= 1000 ? (n / 1000).toFixed(1) + 'K' : String(n); }

// 页面加载
document.addEventListener('DOMContentLoaded', () => loadConfig());
