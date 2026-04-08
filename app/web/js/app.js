// ========== State Management ==========
const AppState = {
  providers: [],
  adapterRows: [],
  usage: { groupBy: "provider", dateFrom: null, dateTo: null },
  conversations: { page: 0, limit: 50, selectedLine: null },
  truncatedTexts: {},
  timers: { log: null, queue: null },
  auth: { token: null, isLoggedIn: false },
};

// ========== Toast Notifications ==========
function showToast(message, type = "info") {
  const existing = document.querySelector(".toast-notification");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = `toast-notification toast-${type}`;
  toast.textContent = message;
  toast.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 2000;
        padding: 12px 20px; border-radius: 6px; font-size: 14px;
        background: ${type === "error" ? "#e74c3c" : type === "success" ? "#2e7d32" : "#1565c0"};
        color: #fff; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: toastFadeIn 0.3s ease;
    `;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "toastFadeOut 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ========== API Helper with Error Handling ==========
async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };

  // Add auth header if logged in
  if (AppState.auth.token) {
    opts.headers["Authorization"] = `Bearer ${AppState.auth.token}`;
  }

  if (body) opts.body = JSON.stringify(body);

  try {
    const resp = await fetch(path, opts);

    // Handle auth errors
    if (resp.status === 401 || resp.status === 403) {
      // Clear invalid token
      AppState.auth.token = null;
      AppState.auth.isLoggedIn = false;
      localStorage.removeItem("admin_token");

      // Show login modal
      showLoginModal();

      throw new Error(t("auth.required"));
    }

    if (!resp.ok) {
      let errorData;
      try {
        errorData = await resp.json();
      } catch {
        errorData = { error: resp.statusText || `HTTP ${resp.status}` };
      }
      throw new Error(
        errorData.error?.message ||
          errorData.error ||
          errorData.message ||
          `HTTP ${resp.status}`,
      );
    }

    const text = await resp.text();
    try {
      return JSON.parse(text);
    } catch {
      console.warn("Response is not JSON:", path);
      return { raw: text };
    }
  } catch (e) {
    if (e.name === "TypeError" && e.message.includes("fetch")) {
      showToast(t("error.network"), "error");
    } else {
      showToast(t("error.api") + ": " + e.message, "error");
    }
    console.error(`API ${method} ${path} failed:`, e);
    throw e;
  }
}

// ========== Timer Cleanup ==========
function clearAllTimers() {
  if (AppState.timers.log) {
    clearInterval(AppState.timers.log);
    AppState.timers.log = null;
  }
  if (AppState.timers.queue) {
    clearInterval(AppState.timers.queue);
    AppState.timers.queue = null;
  }
}

// ========== Authentication ==========

function checkAuthOnLoad() {
  const savedToken = localStorage.getItem("admin_token");
  if (savedToken) {
    AppState.auth.token = savedToken;
    AppState.auth.isLoggedIn = true;
    // Verify token by making a test API call
    api("GET", "/api/config")
      .then(() => {
        updateAuthUI();
      })
      .catch(() => {
        // Token invalid, clear it
        AppState.auth.token = null;
        AppState.auth.isLoggedIn = false;
        localStorage.removeItem("admin_token");
        showLoginModal();
      });
  } else {
    showLoginModal();
  }
}

function showLoginModal() {
  const modal = document.getElementById("login-modal");
  if (modal) {
    modal.classList.add("active");
    modal.style.display = "flex";
    // Focus the input
    const input = modal.querySelector("input");
    if (input) input.focus();
  }
}

function hideLoginModal() {
  const modal = document.getElementById("login-modal");
  if (modal) {
    modal.classList.remove("active");
    modal.style.display = "none";
  }
}

function handleLogin() {
  const input = document.getElementById("login-api-key");
  const token = input?.value?.trim();

  if (!token) {
    showToast(t("auth.invalidKey"), "error");
    return;
  }

  // Validate by making an API call
  fetch("/api/config", {
    headers: { Authorization: `Bearer ${token}` },
  })
    .then((resp) => {
      if (resp.ok) {
        AppState.auth.token = token;
        AppState.auth.isLoggedIn = true;
        localStorage.setItem("admin_token", token);
        hideLoginModal();
        updateAuthUI();
        showToast(t("common.success"), "success");
        // Reload current tab
        switchTab(localStorage.getItem("lastTab") || "providers");
      } else {
        showToast(t("auth.invalidKey"), "error");
      }
    })
    .catch(() => {
      showToast(t("auth.invalidKey"), "error");
    });
}

function handleLogout() {
  AppState.auth.token = null;
  AppState.auth.isLoggedIn = false;
  localStorage.removeItem("admin_token");
  updateAuthUI();
  showLoginModal();
}

function updateAuthUI() {
  const loginBtn = document.getElementById("auth-btn");
  const userInfo = document.getElementById("auth-user");

  if (AppState.auth.isLoggedIn) {
    if (loginBtn) loginBtn.textContent = t("auth.logout");
    if (loginBtn) loginBtn.onclick = handleLogout;
    if (userInfo) userInfo.textContent = "✓";
  } else {
    if (loginBtn) loginBtn.textContent = t("auth.login");
    if (loginBtn) loginBtn.onclick = showLoginModal;
    if (userInfo) userInfo.textContent = "";
  }
}

// ========== Modal Handlers ==========
function setupModalHandlers() {
  // Escape key to close modals (but NOT the login modal)
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const visibleModals = document.querySelectorAll('.modal[style*="flex"]');
      visibleModals.forEach((m) => {
        if (!m.id || m.id !== "login-modal") m.style.display = "none";
      });
    }
    // Enter key to submit login form
    if (e.key === "Enter") {
      const loginModal = document.getElementById("login-modal");
      if (loginModal && loginModal.style.display === "flex") {
        handleLogin();
      }
    }
  });

  // Click outside modal to close (but NOT the login modal)
  document.addEventListener("click", (e) => {
    if (
      e.target.classList.contains("modal") &&
      e.target.id &&
      e.target.id !== "login-modal"
    ) {
      e.target.style.display = "none";
    }
  });
}

// ========== Tab Switching ==========
function switchTab(name) {
  clearAllTimers();
  localStorage.setItem("lastTab", name);

  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.remove("active");
    t.setAttribute("aria-selected", "false");
  });
  document
    .querySelectorAll(".tab-content")
    .forEach((t) => t.classList.remove("active"));

  const activeTab = document.querySelector(`[onclick="switchTab('${name}')"]`);
  if (activeTab) {
    activeTab.classList.add("active");
    activeTab.setAttribute("aria-selected", "true");
  }

  const activeContent = document.getElementById(`tab-${name}`);
  if (activeContent) activeContent.classList.add("active");

  if (name === "providers" || name === "models") loadConfig();
  if (name === "keys") loadKeys();
  if (name === "queue") loadQueueStats();
  if (name === "usage") loadUsage();
  if (name === "logs") loadLogs();
  if (name === "conversations") loadConversations();
}

// ========== Providers ==========
async function loadConfig() {
  const config = await api("GET", "/api/config");
  AppState.providers = config.providers || [];
  renderProviders(config.providers || []);
  renderModels(config.models || {});
}

function renderProviders(providers) {
  const el = document.getElementById("providers-list");
  if (!providers.length) {
    el.innerHTML = `<p style="color:#999">${t("provider.empty")}</p>`;
    return;
  }

  el.innerHTML = providers
    .map(
      (p) => `
        <div class="card" id="provider-card-${esc(p.name)}">
            <div class="card-header">
                <span class="card-title">${esc(p.name)}</span>
                <span class="badge ${p.enabled ? "badge-green" : "badge-red"}">${p.enabled ? t("common.enabled") : t("common.disabled")}</span>
                ${p.max_concurrent > 0 ? `<span class="badge badge-orange" title="${t("provider.queue.enabled")}">Q</span>` : ""}
            </div>
            <div class="card-body">
                <table>
                    <tr><td>${t("provider.td.protocol")}</td><td><span class="badge badge-blue">${esc(p.provider)}</span></td></tr>
                    <tr><td>Base URL</td><td>${esc(p.base_url)}</td></tr>
                    <tr><td>API Key</td><td>${esc(p.api_key).substring(0, 12)}***</td></tr>
                    ${p.custom_headers && Object.keys(p.custom_headers).length ? `<tr><td>Headers</td><td>${esc(JSON.stringify(p.custom_headers))}</td></tr>` : ""}
                    ${
                      p.max_concurrent > 0
                        ? `
                    <tr><td>${t("provider.td.queue")}</td><td>
                        <span class="badge badge-orange">${t("provider.queue.enabled")}</span>
                        <span class="queue-info">${t("provider.queue.info", p.max_concurrent, p.max_queue_size, p.queue_timeout)}</span>
                    </td></tr>
                    `
                        : ""
                    }
                </table>
            </div>
            <div class="card-actions" style="margin-top:8px">
                <button class="btn btn-sm" onclick="editProvider('${esc(p.name)}')">${t("common.edit")}</button>
                <button class="btn btn-sm btn-test" onclick="testProvider('${esc(p.name)}')">${t("provider.btn.test")}</button>
                <button class="btn btn-sm ${p.enabled ? "btn-danger" : ""}" onclick="toggleProvider('${esc(p.name)}')">${p.enabled ? t("common.disable") : t("common.enable")}</button>
                <button class="btn btn-sm btn-danger" onclick="deleteProvider('${esc(p.name)}')">${t("common.delete")}</button>
            </div>
            <div id="provider-test-${esc(p.name)}" class="test-result" style="display:none"></div>
        </div>
    `,
    )
    .join("");
}

function showAddProvider() {
  document.getElementById("provider-form-title").textContent = t(
    "provider.formTitle.add",
  );
  document.getElementById("pf-original-name").value = "";
  document.getElementById("pf-name").value = "";
  document.getElementById("pf-provider").value = "openai";
  document.getElementById("pf-base-url").value = "";
  document.getElementById("pf-api-key").value = "";
  document.getElementById("pf-headers").value = "{}";
  document.getElementById("pf-max-concurrent").value = "0";
  document.getElementById("pf-max-queue-size").value = "100";
  document.getElementById("pf-queue-timeout").value = "300";
  document.getElementById("provider-form").style.display = "flex";
}

async function editProvider(name) {
  const config = await api("GET", "/api/config");
  const p = (config.providers || []).find((x) => x.name === name);
  if (!p) return;

  document.getElementById("provider-form-title").textContent = t(
    "provider.formTitle.edit",
  );
  document.getElementById("pf-original-name").value = name;
  document.getElementById("pf-name").value = p.name;
  document.getElementById("pf-provider").value = p.provider;
  document.getElementById("pf-base-url").value = p.base_url;
  document.getElementById("pf-api-key").value = p.api_key;
  document.getElementById("pf-headers").value = JSON.stringify(
    p.custom_headers || {},
  );
  document.getElementById("pf-max-concurrent").value = p.max_concurrent || 0;
  document.getElementById("pf-max-queue-size").value = p.max_queue_size || 100;
  document.getElementById("pf-queue-timeout").value = p.queue_timeout || 300;
  document.getElementById("provider-form").style.display = "flex";
}

function hideProviderForm() {
  document.getElementById("provider-form").style.display = "none";
}

async function saveProvider() {
  const original = document.getElementById("pf-original-name").value;
  const name = document.getElementById("pf-name").value.trim();
  if (!name) {
    showToast(t("provider.validation.nameRequired"), "error");
    return;
  }

  let headers = {};
  try {
    headers = JSON.parse(document.getElementById("pf-headers").value || "{}");
  } catch (e) {
    showToast(t("provider.validation.headersJson"), "error");
    return;
  }

  const body = {
    name,
    provider: document.getElementById("pf-provider").value,
    base_url: document.getElementById("pf-base-url").value.trim(),
    api_key: document.getElementById("pf-api-key").value.trim(),
    custom_headers: headers,
    enabled: true,
    max_concurrent:
      parseInt(document.getElementById("pf-max-concurrent").value) || 0,
    max_queue_size:
      parseInt(document.getElementById("pf-max-queue-size").value) || 100,
    queue_timeout:
      parseFloat(document.getElementById("pf-queue-timeout").value) || 300,
  };

  try {
    if (original) {
      await api(
        "PUT",
        `/api/config/providers/${encodeURIComponent(original)}`,
        body,
      );
    } else {
      await api("POST", "/api/config/providers", body);
    }
    hideProviderForm();
    showToast(t("common.saved"), "success");
    loadConfig();
  } catch (e) {
    // Error already shown by api()
  }
}

async function toggleProvider(name) {
  try {
    await api(
      "PATCH",
      `/api/config/providers/${encodeURIComponent(name)}/toggle`,
    );
    showToast(t("common.updated"), "success");
    loadConfig();
  } catch (e) {}
}

async function deleteProvider(name) {
  if (!confirm(t("common.confirmDelete"))) return;
  try {
    await api("DELETE", `/api/config/providers/${encodeURIComponent(name)}`);
    showToast(t("common.deleted"), "success");
    loadConfig();
  } catch (e) {}
}

// ========== Provider Test ==========
async function testProvider(name) {
  const el = document.getElementById(`provider-test-${name}`);
  el.style.display = "block";
  el.innerHTML = `<span class="test-loading">${t("provider.test.testing")}</span>`;

  try {
    const result = await api(
      "POST",
      `/api/config/providers/${encodeURIComponent(name)}/test`,
    );
    if (result.success) {
      el.innerHTML = `<span class="test-ok">${t("provider.test.success")}</span> <span class="test-meta">${t("provider.test.status")}: ${result.status_code || "OK"} · ${t("provider.test.latency")}: ${result.latency_ms}ms</span>`;
    } else {
      el.innerHTML = `<span class="test-fail">${t("provider.test.fail")}</span> <span class="test-meta">${esc(result.error || "")}</span>`;
    }
  } catch (e) {
    el.innerHTML = `<span class="test-fail">${t("provider.test.fail")}</span>`;
  }
}

// ========== Models ==========
function renderModels(models) {
  const el = document.getElementById("models-list");
  const entries = Object.entries(models);
  if (!entries.length) {
    el.innerHTML = `<p style="color:#999">${t("model.empty")}</p>`;
    return;
  }

  el.innerHTML = entries
    .map(
      ([name, m]) => `
        <div class="card" id="model-card-${esc(name)}">
            <div class="card-header">
                <span class="card-title">${esc(name)}</span>
                <span class="badge badge-blue">${esc(m.mode)}</span>
                <span style="margin-left:8px;color:#999;font-size:12px">${esc(m.description || "")}</span>
            </div>
            <div class="adapter-list">
                ${(m.adapters || [])
                  .map(
                    (a) => `
                    <div class="adapter-item">
                        <span class="adapter-priority">${t("model.adapter.priority", a.priority)}</span>
                        <span class="adapter-provider">${esc(a.adapter)} / ${esc(a.model_name)}</span>
                        <span style="color:#999">${t("adapter.timeout.label", a.timeout)}</span>
                    </div>
                `,
                  )
                  .join("")}
            </div>
            <div class="card-actions" style="margin-top:8px">
                <button class="btn btn-sm" onclick="editModel('${esc(name)}')">${t("common.edit")}</button>
                <button class="btn btn-sm btn-test" onclick="testModel('${esc(name)}')">${t("model.btn.test")}</button>
                <button class="btn btn-sm btn-danger" onclick="deleteModel('${esc(name)}')">${t("common.delete")}</button>
            </div>
            <div id="model-test-${esc(name)}" class="test-result" style="display:none"></div>
        </div>
    `,
    )
    .join("");
}

function showAddModel() {
  document.getElementById("model-form-title").textContent = t(
    "model.formTitle.add",
  );
  document.getElementById("mf-original-name").value = "";
  document.getElementById("mf-name").value = "";
  document.getElementById("mf-mode").value = "chain";
  document.getElementById("mf-description").value = "";
  AppState.adapterRows = [];
  renderAdapterRows();
  document.getElementById("model-form").style.display = "flex";
}

async function editModel(name) {
  const config = await api("GET", "/api/config");
  const m = config.models[name];
  if (!m) return;

  document.getElementById("model-form-title").textContent = t(
    "model.formTitle.edit",
  );
  document.getElementById("mf-original-name").value = name;
  document.getElementById("mf-name").value = name;
  document.getElementById("mf-mode").value = m.mode;
  document.getElementById("mf-description").value = m.description || "";
  AppState.adapterRows = (m.adapters || []).map((a) => ({ ...a }));
  renderAdapterRows();
  document.getElementById("model-form").style.display = "flex";
}

function hideModelForm() {
  document.getElementById("model-form").style.display = "none";
}

// ========== Adapter Rows ==========
function addAdapterRow() {
  AppState.adapterRows.push({
    adapter: "",
    model_name: "",
    priority: AppState.adapterRows.length + 1,
    timeout: 60,
  });
  renderAdapterRows();
}

function removeAdapterRow(index) {
  AppState.adapterRows.splice(index, 1);
  AppState.adapterRows.forEach((r, i) => (r.priority = i + 1));
  renderAdapterRows();
}

function moveAdapterRow(index, direction) {
  const target = index + direction;
  if (target < 0 || target >= AppState.adapterRows.length) return;

  const temp = AppState.adapterRows[index];
  AppState.adapterRows[index] = AppState.adapterRows[target];
  AppState.adapterRows[target] = temp;
  AppState.adapterRows.forEach((r, i) => (r.priority = i + 1));
  renderAdapterRows();
}

function renderAdapterRows() {
  const container = document.getElementById("mf-adapters-list");
  if (!AppState.adapterRows.length) {
    container.innerHTML = `<p style="color:#999;font-size:13px">${t("adapter.empty")}</p>`;
    return;
  }

  container.innerHTML = AppState.adapterRows
    .map((row, i) => {
      const options = AppState.providers
        .map((p) => {
          const sel = p.name === row.adapter ? " selected" : "";
          return `<option value="${esc(p.name)}"${sel}>${esc(p.name)}</option>`;
        })
        .join("");

      return `
        <div class="adapter-form-card">
            <div class="adapter-form-row1">
                <div class="adapter-form-arrows">
                    <button class="btn-arrow${i === 0 ? " btn-arrow-disabled" : ""}"
                        onclick="moveAdapterRow(${i}, -1)" ${i === 0 ? "disabled" : ""}>▲</button>
                    <button class="btn-arrow${i === AppState.adapterRows.length - 1 ? " btn-arrow-disabled" : ""}"
                        onclick="moveAdapterRow(${i}, 1)" ${i === AppState.adapterRows.length - 1 ? "disabled" : ""}>▼</button>
                </div>
                <span class="adapter-form-priority">${row.priority}.</span>
                <select class="adapter-form-select" onchange="AppState.adapterRows[${i}].adapter=this.value">
                    <option value="">${t("adapter.selectProvider")}</option>
                    ${options}
                </select>
                <button class="btn btn-sm btn-danger adapter-form-remove" onclick="removeAdapterRow(${i})">✕</button>
            </div>
            <div class="adapter-form-row2">
                <input class="adapter-form-model" placeholder="${t("adapter.ph.modelName")}" value="${esc(row.model_name)}"
                       onchange="AppState.adapterRows[${i}].model_name=this.value">
            </div>
            <div class="adapter-form-row2" style="margin-top:4px">
                <label style="font-size:12px;color:#888;flex-shrink:0">${t("adapter.label.timeout")}</label>
                <input class="adapter-form-timeout" type="number" value="${row.timeout}" min="1"
                       onchange="AppState.adapterRows[${i}].timeout=parseInt(this.value)||60">
                <span style="font-size:12px;color:#888">${t("adapter.unit.seconds")}</span>
            </div>
        </div>
        `;
    })
    .join("");
}

async function saveModel() {
  const original = document.getElementById("mf-original-name").value;
  const name = document.getElementById("mf-name").value.trim();
  if (!name) {
    showToast(t("model.validation.nameRequired"), "error");
    return;
  }

  const validAdapters = AppState.adapterRows.filter(
    (r) => r.adapter && r.model_name,
  );
  if (!validAdapters.length) {
    showToast(t("model.validation.adapterRequired"), "error");
    return;
  }

  const body = {
    mode: document.getElementById("mf-mode").value,
    description: document.getElementById("mf-description").value.trim(),
    adapters: validAdapters,
  };

  try {
    if (original) {
      await api(
        "PUT",
        `/api/config/models/${encodeURIComponent(original)}`,
        body,
      );
    } else {
      await api(
        "POST",
        `/api/config/models?name=${encodeURIComponent(name)}`,
        body,
      );
    }
    hideModelForm();
    showToast(t("common.saved"), "success");
    loadConfig();
  } catch (e) {}
}

async function deleteModel(name) {
  if (!confirm(t("common.confirmDelete"))) return;
  try {
    await api("DELETE", `/api/config/models/${encodeURIComponent(name)}`);
    showToast(t("common.deleted"), "success");
    loadConfig();
  } catch (e) {}
}

// ========== Model Test ==========
async function testModel(name) {
  const el = document.getElementById(`model-test-${name}`);
  el.style.display = "block";
  el.innerHTML = `<span class="test-loading">${t("model.test.chainProbe")}</span>`;

  try {
    const result = await api(
      "POST",
      `/api/config/models/${encodeURIComponent(name)}/test`,
    );
    renderModelTestResult(el, result);
  } catch (e) {
    el.innerHTML = `<span class="test-fail">${t("model.test.allFailed")}</span>`;
  }
}

function renderModelTestResult(el, result) {
  let html = "";

  if (result.chain && result.chain.length > 0) {
    html += `<table class="usage-table chain-test-table">
            <tr><th>${t("chain.th.priority")}</th><th>${t("chain.th.provider")}</th><th>${t("chain.th.model")}</th><th>${t("chain.th.status")}</th><th>${t("chain.th.latency")}</th><th>Token</th></tr>`;

    result.chain.forEach((c) => {
      const isHit =
        result.success && c.adapter === result.adapter_used && c.success;
      const statusBadge = c.skipped
        ? `<span class="badge badge-gray">${t("chain.status.skipped")}</span>`
        : c.success
          ? `<span class="badge badge-green">${t("common.success")}${isHit ? " ✓" : ""}</span>`
          : `<span class="badge badge-red">${t("common.failed")}</span>`;
      const tokenStr = c.usage
        ? `${c.usage.prompt_tokens || 0}→${c.usage.completion_tokens || 0}`
        : "-";
      const errorStr = c.error
        ? `<div class="test-chain-error">${esc(c.error)}</div>`
        : "";

      html += `<tr class="${isHit ? "chain-hit-row" : ""}">
                <td>${t("model.adapter.priority", c.priority)}</td>
                <td>${esc(c.adapter)}</td>
                <td>${esc(c.model_name)}</td>
                <td>${statusBadge}${errorStr}</td>
                <td>${c.latency_ms ? c.latency_ms + "ms" : "-"}</td>
                <td>${tokenStr}</td>
            </tr>`;
    });
    html += "</table>";
  }

  if (result.success) {
    html =
      `<span class="test-ok">${t("model.test.success")}</span> <span class="test-meta">${t("model.test.hit")}: ${esc(result.adapter_used)} · ${t("provider.test.latency")}: ${result.latency_ms}ms</span>
            ${result.preview ? `<div class="test-preview">"${esc(result.preview)}"</div>` : ""}` +
      html;
  } else {
    html =
      `<span class="test-fail">${t("model.test.allFailed")}</span> <span class="test-meta">${esc(result.error || "")}</span>` +
      html;
  }

  el.innerHTML = html;
}

// ========== API Keys ==========
async function loadKeys() {
  const keys = await api("GET", "/api/keys");
  renderKeys(keys);
}

function renderKeys(keys) {
  const el = document.getElementById("keys-list");
  if (!keys.length) {
    el.innerHTML = `<p style="color:#999">${t("key.empty")}</p>`;
    return;
  }

  el.innerHTML = keys
    .map(
      (k) => `
        <div class="card">
            <div class="card-header">
                <span class="card-title">${esc(k.name || "unnamed")}</span>
                <span class="badge ${k.enabled ? "badge-green" : "badge-red"}">${k.enabled ? t("common.enabled") : t("common.disabled")}</span>
            </div>
            <div class="card-body">
                <table>
                    <tr><td>Key</td><td><code>${esc(k.key)}</code></td></tr>
                    <tr><td>${t("key.td.rateLimit")}</td><td>${k.rate_limit || t("common.unlimited")}</td></tr>
                    <tr><td>${t("key.td.dailyLimit")}</td><td>${k.daily_limit || t("common.unlimited")}</td></tr>
                    ${k.allowed_models && k.allowed_models.length ? `<tr><td>${t("key.td.allowedModels")}</td><td>${k.allowed_models.map((m) => `<span class="badge badge-blue">${esc(m)}</span>`).join(" ")}</td></tr>` : ""}
                    ${k.description ? `<tr><td>${t("common.description")}</td><td>${esc(k.description)}</td></tr>` : ""}
                </table>
            </div>
            <div class="card-actions" style="margin-top:8px">
                <button class="btn btn-sm" onclick="toggleKey('${esc(k.key_raw || k.key)}')">${k.enabled ? t("common.disable") : t("common.enable")}</button>
                <button class="btn btn-sm btn-danger" onclick="deleteKey('${esc(k.key_raw || k.key)}')">${t("common.delete")}</button>
            </div>
        </div>
    `,
    )
    .join("");
}

function showAddKey() {
  document.getElementById("key-form").style.display = "flex";
}
function hideKeyForm() {
  document.getElementById("key-form").style.display = "none";
}

async function createKey() {
  const name = document.getElementById("kf-name").value.trim();
  const desc = document.getElementById("kf-description").value.trim();
  const rate = parseInt(document.getElementById("kf-rate-limit").value) || 0;
  const daily = parseInt(document.getElementById("kf-daily-limit").value) || 0;
  const models = document
    .getElementById("kf-allowed-models")
    .value.split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  try {
    const result = await api("POST", "/api/keys", {
      name,
      description: desc,
      rate_limit: rate,
      daily_limit: daily,
      allowed_models: models,
    });

    hideKeyForm();
    document.getElementById("new-key-value").value = result.key;
    document.getElementById("new-key-result").style.display = "flex";
    showToast(t("key.created.success"), "success");
    loadKeys();
  } catch (e) {}
}

function copyKey() {
  const key = document.getElementById("new-key-value").value;
  navigator.clipboard
    .writeText(key)
    .then(() => {
      showToast(t("common.copied"), "success");
    })
    .catch(() => {
      showToast(t("error.copy"), "error");
    });
}

async function toggleKey(key) {
  try {
    await api("PATCH", `/api/keys/${encodeURIComponent(key)}/toggle`);
    showToast(t("common.updated"), "success");
    loadKeys();
  } catch (e) {}
}

async function deleteKey(key) {
  if (!confirm(t("common.confirmDelete"))) return;
  try {
    await api("DELETE", `/api/keys/${encodeURIComponent(key)}`);
    showToast(t("common.deleted"), "success");
    loadKeys();
  } catch (e) {}
}

// ========== Usage ==========
async function loadUsage() {
  const groupBy = document.getElementById("usage-group-by").value;
  const dateRange = document.getElementById("usage-date").value;
  const today = new Date().toISOString().split("T")[0];

  if (dateRange === "today") {
    AppState.usage.dateFrom = today;
    AppState.usage.dateTo = today;
  } else if (dateRange === "week") {
    const d = new Date();
    d.setDate(d.getDate() - 7);
    AppState.usage.dateFrom = d.toISOString().split("T")[0];
    AppState.usage.dateTo = today;
  } else if (dateRange === "month") {
    const d = new Date();
    d.setDate(d.getDate() - 30);
    AppState.usage.dateFrom = d.toISOString().split("T")[0];
    AppState.usage.dateTo = today;
  }
  AppState.usage.groupBy = groupBy;

  const params = new URLSearchParams({
    group_by: groupBy,
    date_from: AppState.usage.dateFrom,
    date_to: AppState.usage.dateTo,
  });

  try {
    const data = await api("GET", `/api/usage?${params}`);
    renderUsage(data, groupBy);
  } catch (e) {}
}

function renderUsage(data, groupBy) {
  const el = document.getElementById("usage-summary");
  const _dimLabel = {
    provider: t("usage.dim.provider"),
    model: t("usage.dim.model"),
    api_key: t("usage.dim.apiKey"),
  };
  const groupLabel = _dimLabel[groupBy];

  el.innerHTML = `
        <div class="usage-summary">${t("usage.summary", data.total, groupLabel)}</div>
        <table class="usage-table">
            <tr><th>${groupLabel}</th><th>${t("usage.th.requests")}</th><th>${t("usage.th.successRate")}</th><th>${t("usage.th.tokensIn")}</th><th>${t("usage.th.tokensOut")}</th><th>${t("usage.th.avgLatency")}</th><th></th></tr>
            ${(data.groups || [])
              .map(
                (g) => `
                <tr>
                    <td><strong>${esc(g.name)}</strong></td>
                    <td>${g.total_requests}</td>
                    <td>${g.success_rate}%</td>
                    <td>${formatNum(g.tokens_in)}</td>
                    <td>${formatNum(g.tokens_out)}</td>
                    <td>${g.avg_latency_ms}ms</td>
                    <td>
                        <button class="btn btn-sm drill-btn"
                            data-top-name="${esc(g.name)}"
                            onclick="drillUsageFromBtn(this)">${t("usage.btn.drill")}</button>
                    </td>
                </tr>
            `,
              )
              .join("")}
        </table>
    `;
  document.getElementById("usage-detail").style.display = "none";
}

async function drillUsageFromBtn(btn) {
  const topItemName = btn.dataset.topName;
  await drillUsage(topItemName, {});
}

async function drillUsage(topItemName, filters) {
  filters = filters || {};

  const allDims = ["provider", "model", "api_key"];
  const _dimLabel = {
    provider: t("usage.dim.provider"),
    model: t("usage.dim.model"),
    api_key: t("usage.dim.apiKey"),
  };
  const locked = new Set([AppState.usage.groupBy, ...Object.keys(filters)]);
  const remaining = allDims.filter((d) => !locked.has(d));
  if (!remaining.length) return;
  const subGroup = remaining[0];
  const subLabel = _dimLabel[subGroup];

  const params = new URLSearchParams({
    group_by: AppState.usage.groupBy,
    sub_group: subGroup,
    date_from: AppState.usage.dateFrom,
    date_to: AppState.usage.dateTo,
  });
  for (const [dim, val] of Object.entries(filters)) {
    params.set(`filter_${dim}`, val);
  }

  try {
    const data = await api(
      "GET",
      `/api/usage/${encodeURIComponent(topItemName)}/detail?${params}`,
    );
    renderDrillUsage(
      data,
      topItemName,
      filters,
      subGroup,
      subLabel,
      _dimLabel,
      remaining,
    );
  } catch (e) {}
}

function renderDrillUsage(
  data,
  topItemName,
  filters,
  subGroup,
  subLabel,
  _dimLabel,
  remaining,
) {
  const crumbs = [`${_dimLabel[AppState.usage.groupBy]}: ${topItemName}`];
  for (const [dim, val] of Object.entries(filters)) {
    crumbs.push(`${_dimLabel[dim]}: ${val}`);
  }

  const canDrillMore = remaining.length > 1;
  const el = document.getElementById("usage-detail");

  el.innerHTML = `
        <div class="detail-header">
            <h4>${crumbs.join(" → ")} → ${t("usage.drill.by", subLabel)}</h4>
            <button class="btn btn-sm" onclick="document.getElementById('usage-detail').style.display='none'">${t("common.close")}</button>
        </div>
        <table class="usage-table">
            <tr><th>${subLabel}</th><th>${t("usage.th.requests")}</th><th>${t("usage.th.success")}</th><th>${t("usage.th.failed")}</th><th>${t("usage.th.tokensIn")}</th><th>${t("usage.th.tokensOut")}</th><th>${t("usage.th.avgLatency")}</th>${canDrillMore ? "<th></th>" : ""}</tr>
            ${data
              .map((g) => {
                const nextFiltersJson = JSON.stringify({
                  ...filters,
                  [subGroup]: g.name,
                });
                return `
                <tr>
                    <td>${esc(g.name)}</td>
                    <td>${g.total_requests}</td>
                    <td>${g.success_count}</td>
                    <td>${g.fail_count}</td>
                    <td>${formatNum(g.tokens_in)}</td>
                    <td>${formatNum(g.tokens_out)}</td>
                    <td>${g.avg_latency_ms}ms</td>
                    ${
                      canDrillMore
                        ? `<td>
                        <button class="btn btn-sm drill-btn"
                            data-top-name="${esc(topItemName)}"
                            data-filters="${esc(nextFiltersJson)}"
                            onclick="drillUsageFromDataBtn(this)">${t("usage.btn.drill")}</button>
                    </td>`
                        : ""
                    }
                </tr>`;
              })
              .join("")}
        </table>
    `;
  el.style.display = "block";
}

async function drillUsageFromDataBtn(btn) {
  const topItemName = btn.dataset.topName;
  const filters = JSON.parse(btn.dataset.filters);
  await drillUsage(topItemName, filters);
}

// ========== Logs ==========
async function loadLogs() {
  const level = document.getElementById("log-level").value;
  const requestId = document.getElementById("log-request-id").value.trim();
  const apiKey = document.getElementById("log-api-key").value.trim();

  const params = new URLSearchParams({ tail: 200 });
  if (level) params.set("level", level);
  if (requestId) params.set("request_id", requestId);
  if (apiKey) params.set("api_key", apiKey);

  try {
    const data = await api("GET", `/api/logs?${params}`);
    renderLogs(data.logs || []);
  } catch (e) {}

  const autoRefresh = document.getElementById("log-auto-refresh").checked;
  if (autoRefresh && !AppState.timers.log) {
    AppState.timers.log = setInterval(loadLogs, 3000);
  } else if (!autoRefresh && AppState.timers.log) {
    clearInterval(AppState.timers.log);
    AppState.timers.log = null;
  }
}

function renderLogs(logs) {
  const el = document.getElementById("log-container");
  if (!logs.length) {
    el.innerHTML = `<div class="log-line log-INFO">${t("log.empty")}</div>`;
    return;
  }

  el.innerHTML = logs
    .map((l) => {
      const ts = l.timestamp ? l.timestamp.split("T")[1].split(".")[0] : "";
      const level = l.level || "INFO";
      const rid = l.request_id ? `[${l.request_id}]` : "";
      return `<div class="log-line log-${level}"><span class="log-timestamp">${ts}</span> <span class="log-${level}">[${level}]</span> ${rid} ${esc(l.message)}</div>`;
    })
    .join("");
  el.scrollTop = el.scrollHeight;
}

// ========== Conversations ==========
async function loadConversations() {
  // Clear truncated texts to prevent memory leak
  AppState.truncatedTexts = {};

  const apiKey = document.getElementById("conv-api-key").value;
  const model = document.getElementById("conv-model").value;
  const success = document.getElementById("conv-success").value;

  const params = new URLSearchParams({
    limit: AppState.conversations.limit,
    offset: AppState.conversations.page * AppState.conversations.limit,
  });
  if (apiKey) params.set("api_key", apiKey);
  if (model) params.set("model", model);
  if (success) params.set("success", success);

  try {
    const data = await api("GET", `/api/conversations?${params}`);
    _fillSelect("conv-api-key", data.api_keys || [], apiKey);
    _fillSelect("conv-model", data.models || [], model);
    renderConvList(data);
    renderConvPagination(data);
  } catch (e) {}
}

function _fillSelect(id, options, currentVal) {
  const sel = document.getElementById(id);
  sel.innerHTML =
    `<option value="">${t("common.all")}</option>` +
    options
      .map(
        (o) =>
          `<option value="${esc(o)}"${o === currentVal ? " selected" : ""}>${esc(o)}</option>`,
      )
      .join("");
}

function renderConvList(data) {
  const el = document.getElementById("conv-list");
  if (!data.items || !data.items.length) {
    el.innerHTML = `<p style="color:#999;padding:16px;text-align:center">${t("conv.empty")}</p>`;
    return;
  }

  el.innerHTML = data.items
    .map((item) => {
      const ts = item.timestamp
        ? item.timestamp.split("T")[1]?.split(".")[0] || ""
        : "";
      const date = item.timestamp ? item.timestamp.split("T")[0] : "";
      const active =
        item.id === AppState.conversations.selectedLine ? " active" : "";
      const statusBadge = item.success
        ? `<span class="badge badge-green">${t("common.success")}</span>`
        : `<span class="badge badge-red">${t("common.failed")}</span>`;
      const toolBadge = item.has_tool_use
        ? ' <span class="badge badge-blue">Tool</span>'
        : "";

      return `
        <div class="conv-item${active}" onclick="loadConvDetail(${item.id})">
            <div class="conv-item-header">
                <span class="badge badge-blue">${esc(item.model)}</span>
                ${statusBadge}${toolBadge}
                <span class="conv-item-time">${date} ${ts}</span>
            </div>
            <div class="conv-item-meta">${esc(item.adapter)} · ${item.latency_ms}ms · ${formatNum(item.tokens_in)}→${formatNum(item.tokens_out)} · key:${esc(item.api_key)}</div>
            <div class="conv-item-preview">${esc(item.output_preview)}</div>
        </div>`;
    })
    .join("");
}

function renderConvPagination(data) {
  const el = document.getElementById("conv-pagination");
  const totalPages = Math.max(1, Math.ceil(data.total / data.limit));
  const curPage = AppState.conversations.page + 1;

  el.innerHTML = `
        <button class="btn btn-sm" onclick="convPrev()" ${AppState.conversations.page <= 0 ? "disabled" : ""}>${t("conv.pagination.prev")}</button>
        <span>${curPage} / ${totalPages}</span>
        <span style="color:#999;font-size:12px">(${t("conv.pagination.count", data.total)})</span>
        <button class="btn btn-sm" onclick="convNext(${totalPages})" ${curPage >= totalPages ? "disabled" : ""}>${t("conv.pagination.next")}</button>
    `;
}

function convPrev() {
  if (AppState.conversations.page > 0) {
    AppState.conversations.page--;
    loadConversations();
  }
}

function convNext(totalPages) {
  if (AppState.conversations.page + 1 < totalPages) {
    AppState.conversations.page++;
    loadConversations();
  }
}

async function loadConvDetail(line) {
  AppState.conversations.selectedLine = line;

  document.querySelectorAll(".conv-item").forEach((el) => {
    const match = el.getAttribute("onclick");
    if (match && match.includes(`loadConvDetail(${line})`)) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });

  const detail = document.getElementById("conv-detail");
  detail.innerHTML = `<div class="conv-detail-empty">${t("common.loading")}</div>`;

  try {
    const rec = await api("GET", `/api/conversations/${line}`);
    if (rec.error) {
      detail.innerHTML = `<div class="conv-detail-empty">${t("conv.detail.loadError", rec.error)}</div>`;
      return;
    }
    renderConvDetail(rec);
  } catch (e) {
    detail.innerHTML = `<div class="conv-detail-empty">${t("conv.detail.loadError", t("error.network"))}</div>`;
  }
}

function renderConvDetail(rec) {
  const el = document.getElementById("conv-detail");
  const ts = rec.timestamp || "";

  let html = `<div class="conv-detail-header"><table>
        <tr><td><strong>${t("conv.detail.time")}</strong></td><td>${esc(ts)}</td><td><strong>${t("conv.detail.model")}</strong></td><td><span class="badge badge-blue">${esc(rec.model)}</span></td></tr>
        <tr><td><strong>${t("conv.detail.adapter")}</strong></td><td>${esc(rec.adapter)}</td><td><strong>${t("conv.detail.apiKey")}</strong></td><td>${esc(rec.api_key)}</td></tr>
        <tr><td><strong>${t("conv.detail.latency")}</strong></td><td>${rec.latency_ms}ms</td><td><strong>${t("conv.detail.token")}</strong></td><td>${formatNum(rec.tokens_in)} → ${formatNum(rec.tokens_out)}</td></tr>
        ${rec.request_id ? `<tr><td><strong>Request ID</strong></td><td colspan="3">${esc(rec.request_id)}</td></tr>` : ""}
    </table></div>`;

  html += `<div class="conv-detail-section-title">${t("conv.detail.inputMessages")}</div>`;
  if (rec.messages && rec.messages.length) {
    html += renderMessagesGrouped(rec.messages);
  } else {
    html += `<p style="color:#999">${t("common.noMessages")}</p>`;
  }

  if (rec.output && rec.output.length) {
    html += `<div class="conv-detail-section"><div class="conv-detail-section-title">${t("conv.detail.output")}</div>`;
    html += rec.output
      .map((block) => {
        if (block.type === "text") {
          return renderMessageBubble(
            "assistant",
            t("role.assistant"),
            block.text,
          );
        } else if (block.type === "tool_use") {
          const aStr = block.arguments || "";
          const aLen =
            typeof aStr === "string"
              ? aStr.length
              : JSON.stringify(aStr).length;
          return `<div class="conv-message conv-msg-assistant">
                    <div class="conv-message-role">TOOL CALL <span class="conv-tool-name">${esc(block.name)}</span> ${_roleMeta(aLen)}</div>
                    <div class="conv-tool-block">${formatArgs(block.arguments)}</div>
                </div>`;
        }
        return "";
      })
      .join("");
    html += "</div>";
  }

  if (rec.error) {
    html += `<div class="conv-error-block"><strong>${t("conv.detail.error")}</strong>${esc(rec.error)}</div>`;
  }

  el.innerHTML = html;
}

function renderMessagesGrouped(messages) {
  const toolResultMap = {};
  messages.forEach((m) => {
    if (m.role === "tool" && m.tool_call_id) {
      toolResultMap[m.tool_call_id] = m;
    }
  });

  const rendered = new Set();
  let html = "";

  messages.forEach((msg, idx) => {
    if (rendered.has(idx)) return;

    if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length) {
      if (msg.content) {
        html += renderMessageBubble(
          "assistant",
          t("role.assistant"),
          typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content),
        );
      }
      msg.tool_calls.forEach((tc) => {
        const func = tc.function || {};
        const tcId = tc.id || "";
        const result = tcId ? toolResultMap[tcId] : null;
        if (result) {
          messages.forEach((m, i) => {
            if (m === result) rendered.add(i);
          });
        }
        const argsStr = func.arguments || "";
        const argsLen =
          typeof argsStr === "string"
            ? argsStr.length
            : JSON.stringify(argsStr).length;

        html += `<div class="conv-tool-group">`;
        html += `<div class="conv-message conv-msg-assistant conv-tool-call">
                    <div class="conv-message-role">TOOL CALL <span class="conv-tool-name">${esc(func.name || "")}</span>
                    ${tcId ? `<span class="conv-tool-group-id">${esc(tcId)}</span>` : ""} ${_roleMeta(argsLen)}</div>
                    <div class="conv-tool-block">${formatArgs(func.arguments)}</div>
                </div>`;
        if (result) {
          const rLen = _contentLen(result.content);
          html += `<div class="conv-message conv-msg-tool">
                        <div class="conv-message-role">${t("conv.toolResult")} ${_roleMeta(rLen)}</div>
                        <div class="conv-message-content">${renderContent(result.content)}</div>
                    </div>`;
        }
        html += `</div>`;
      });
      return;
    }

    if (msg.role === "tool") {
      const toolId = msg.tool_call_id
        ? `<span class="conv-tool-name">${esc(msg.tool_call_id)}</span> `
        : "";
      const rLen = _contentLen(msg.content);
      html += `<div class="conv-message conv-msg-tool">
                <div class="conv-message-role">${t("conv.toolResult")} ${toolId}${_roleMeta(rLen)}</div>
                <div class="conv-message-content">${renderContent(msg.content)}</div>
            </div>`;
      return;
    }

    html += renderMessage(msg);
  });

  return html;
}

function renderMessage(msg) {
  const role = msg.role || "user";
  const content = msg.content;

  if (role === "tool") {
    const toolId = msg.tool_call_id
      ? `<span class="conv-tool-name">${esc(msg.tool_call_id)}</span> `
      : "";
    const rLen = _contentLen(content);
    return `<div class="conv-message conv-msg-tool">
            <div class="conv-message-role">${t("conv.toolResult")} ${toolId}${_roleMeta(rLen)}</div>
            <div class="conv-message-content">${renderContent(content)}</div>
        </div>`;
  }

  if (typeof content === "string") {
    return renderMessageBubble(role, _roleLabel(role), content);
  }

  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (!block || typeof block !== "object") return "";
        if (block.type === "text") {
          return renderMessageBubble(role, _roleLabel(role), block.text || "");
        } else if (block.type === "thinking") {
          const thinkingText = block.thinking || "";
          if (!thinkingText) return "";
          return `<div class="conv-message conv-msg-thinking">
                    <div class="conv-message-role">${t("role.thinking")} ${_roleMeta(thinkingText.length)}</div>
                    <div class="conv-thinking-content">${esc(thinkingText)}</div>
                </div>`;
        } else if (block.type === "tool_use") {
          const inputStr = block.input || "";
          const inputLen =
            typeof inputStr === "string"
              ? inputStr.length
              : JSON.stringify(inputStr).length;
          return `<div class="conv-message conv-msg-${_roleClass(role)}">
                    <div class="conv-message-role">${_roleLabel(role)} - TOOL CALL <span class="conv-tool-name">${esc(block.name)}</span> ${_roleMeta(inputLen)}</div>
                    <div class="conv-tool-block">${formatArgs(block.input)}</div>
                </div>`;
        } else if (block.type === "tool_result") {
          const rLen = _contentLen(block.content);
          return `<div class="conv-message conv-msg-tool">
                    <div class="conv-message-role">${t("conv.toolResult")} ${_roleMeta(rLen)}</div>
                    <div class="conv-message-content">${renderContent(block.content)}</div>
                </div>`;
        } else if (block.type === "image" || block.type === "image_url") {
          return `<div class="conv-message conv-msg-${_roleClass(role)}">
                    <div class="conv-message-role">${_roleLabel(role)}</div>
                    <div class="conv-message-content" style="color:#999">${t("common.image")}</div>
                </div>`;
        } else if (block.type === "redacted_thinking") {
          return `<div class="conv-message conv-msg-thinking">
                    <div class="conv-message-role">${t("role.thinking")} ${_roleMeta(0)}</div>
                    <div class="conv-thinking-content" style="color:#999;font-style:italic">${t("conv.redacted")}</div>
                </div>`;
        }
        return "";
      })
      .join("");
  }

  let parts = "";
  if (msg.tool_calls && msg.tool_calls.length) {
    parts = msg.tool_calls
      .map((tc) => {
        const func = tc.function || {};
        const aStr = func.arguments || "";
        const aLen =
          typeof aStr === "string" ? aStr.length : JSON.stringify(aStr).length;
        return `<div class="conv-message conv-msg-assistant">
                <div class="conv-message-role">${t("role.assistant")} - TOOL CALL <span class="conv-tool-name">${esc(func.name || "")}</span>
                ${tc.id ? `<span class="conv-tool-group-id">${esc(tc.id)}</span>` : ""} ${_roleMeta(aLen)}</div>
                <div class="conv-tool-block">${formatArgs(func.arguments)}</div>
            </div>`;
      })
      .join("");
  }
  return (
    parts ||
    `<div class="conv-message conv-msg-${_roleClass(role)}">
        <div class="conv-message-role">${_roleLabel(role)} ${_roleMeta(0)}</div>
        <div class="conv-message-content" style="color:#999">${t("common.empty")}</div>
    </div>`
  );
}

function renderMessageBubble(role, label, text) {
  const cls = _roleClass(role);
  const needTruncate = text && text.length > 10000;
  const uid = "mc_" + Math.random().toString(36).slice(2, 8);
  const displayText = needTruncate ? text.substring(0, 5000) : text;
  const charsBadge = text
    ? `<span class="conv-chars-badge">${text.length} chars</span>`
    : "";
  const foldLabel = `<span class="conv-fold-label">${t("conv.fold")}</span>`;

  if (needTruncate) {
    AppState.truncatedTexts[uid] = text;
  }

  return `<div class="conv-message conv-msg-${cls}">
        <div class="conv-message-role">${esc(label)} ${charsBadge}${foldLabel}</div>
        <div class="conv-message-content" id="${uid}">${esc(displayText)}</div>
    </div>`;
}

function convExpandText(uid) {
  const full = AppState.truncatedTexts[uid];
  if (!full) return;
  const el = document.getElementById(uid);
  if (el) el.textContent = full;
  delete AppState.truncatedTexts[uid];
}

function formatArgs(args) {
  if (!args) return "";
  try {
    const obj = typeof args === "string" ? JSON.parse(args) : args;
    return esc(JSON.stringify(obj, null, 2));
  } catch (e) {
    return esc(String(args));
  }
}

function _contentLen(content) {
  if (typeof content === "string") return content.length;
  if (Array.isArray(content)) {
    return content.reduce((s, b) => {
      return (
        s +
        (typeof b === "string"
          ? b.length
          : b && b.text
            ? b.text.length
            : JSON.stringify(b).length)
      );
    }, 0);
  }
  if (content == null) return 0;
  return JSON.stringify(content).length;
}

function _roleMeta(len) {
  const badge = len ? `<span class="conv-chars-badge">${len} chars</span>` : "";
  return `${badge}<span class="conv-fold-label">${t("conv.fold")}</span>`;
}

function renderContent(content) {
  if (typeof content === "string") return esc(content);
  if (Array.isArray(content)) {
    return content
      .map((b) => {
        if (typeof b === "string") return esc(b);
        if (b && b.type === "text") return esc(b.text || "");
        return esc(JSON.stringify(b));
      })
      .join("");
  }
  if (content == null)
    return `<span style="color:#999">${t("common.empty")}</span>`;
  return esc(JSON.stringify(content));
}

function _roleLabel(role) {
  return (
    {
      user: t("role.user"),
      assistant: t("role.assistant"),
      system: t("role.system"),
      tool: t("role.tool"),
    }[role] || role
  );
}

function _roleClass(role) {
  return (
    { user: "user", assistant: "assistant", system: "system", tool: "tool" }[
      role
    ] || "user"
  );
}

// ========== Queue Stats ==========
async function loadQueueStats() {
  try {
    const data = await api("GET", "/api/config/queue-stats");
    renderQueueStats(data);
  } catch (e) {}
}

function renderQueueStats(data) {
  const summaryEl = document.getElementById("queue-summary");
  const listEl = document.getElementById("queue-list");

  if (!data.queues || Object.keys(data.queues).length === 0) {
    summaryEl.innerHTML = `<p style="color:#999">${t("queue.empty")}</p>`;
    listEl.innerHTML = "";
    return;
  }

  const activeQueues = data.active_queues || 0;
  const totalProviders = data.total_providers || 0;

  summaryEl.innerHTML = `
        <div class="queue-summary-stats">
            <div class="stat-item">
                <span class="stat-value">${activeQueues}</span>
                <span class="stat-label">${t("queue.stat.activeQueues")}</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${totalProviders}</span>
                <span class="stat-label">${t("queue.stat.totalProviders")}</span>
            </div>
        </div>
    `;

  const queues = data.queues || {};
  listEl.innerHTML = Object.entries(queues)
    .map(([name, q]) => {
      const isActive = q.max_concurrent > 0;
      const queuePercent = Math.min(
        100,
        (q.current_queue_size / q.max_queue_size) * 100,
      );
      const queueBarColor =
        queuePercent > 80
          ? "var(--color-danger)"
          : queuePercent > 50
            ? "var(--color-warning)"
            : "var(--color-success)";

      return `
        <div class="card queue-card ${isActive ? "queue-active" : "queue-inactive"}">
            <div class="card-header">
                <span class="card-title">${esc(name)}</span>
                ${
                  isActive
                    ? `<span class="badge badge-orange">${t("provider.queue.enabled")}</span>`
                    : `<span class="badge badge-gray">${t("provider.queue.disabled")}</span>`
                }
            </div>
            <div class="card-body">
                <div class="queue-metrics">
                    <div class="metric-row">
                        <span class="metric-label">${t("queue.metric.maxConcurrent")}:</span>
                        <span class="metric-value">${q.max_concurrent}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">${t("queue.metric.activeRequests")}:</span>
                        <span class="metric-value">${q.active_requests || 0}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">${t("queue.metric.queueSize")}:</span>
                        <span class="metric-value">${q.current_queue_size} / ${q.max_queue_size}</span>
                    </div>
                    ${
                      isActive
                        ? `
                    <div class="queue-progress-bar">
                        <div class="queue-progress-fill" style="width: ${queuePercent}%; background: ${queueBarColor}"></div>
                    </div>
                    `
                        : ""
                    }
                    <div class="metric-row">
                        <span class="metric-label">${t("queue.metric.totalRequests")}:</span>
                        <span class="metric-value">${q.total_requests || 0}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">${t("queue.metric.rejectedRequests")}:</span>
                        <span class="metric-value ${(q.rejected_requests || 0) > 0 ? "text-red" : ""}">${q.rejected_requests || 0}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">${t("queue.metric.avgWaitTime")}:</span>
                        <span class="metric-value">${(q.avg_wait_time || 0).toFixed(2)}s</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">${t("queue.metric.queueTimeout")}:</span>
                        <span class="metric-value">${q.queue_timeout || 300}s</span>
                    </div>
                </div>
            </div>
        </div>
        `;
    })
    .join("");
}

function toggleQueueAutoRefresh() {
  const autoRefresh = document.getElementById("queue-auto-refresh").checked;
  if (autoRefresh && !AppState.timers.queue) {
    AppState.timers.queue = setInterval(loadQueueStats, 5000);
  } else if (!autoRefresh && AppState.timers.queue) {
    clearInterval(AppState.timers.queue);
    AppState.timers.queue = null;
  }
}

// ========== Utilities ==========
function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

function formatNum(n) {
  if (n == null) return "0";
  return n >= 1000 ? (n / 1000).toFixed(1) + "K" : String(n);
}

// ========== Page Initialization ==========
document.addEventListener("DOMContentLoaded", () => {
  checkAuthOnLoad();
  setupModalHandlers();
  loadConfig();

  // Conversation detail: click to toggle expand/fold
  document
    .getElementById("conv-detail")
    .addEventListener("click", function (e) {
      const msg = e.target.closest(".conv-message");
      if (!msg) return;

      msg.classList.toggle("expanded");
      const lbl = msg.querySelector(".conv-fold-label");
      if (lbl) {
        lbl.textContent = msg.classList.contains("expanded")
          ? t("conv.expanded")
          : t("conv.fold");
      }

      if (msg.classList.contains("expanded")) {
        const content = msg.querySelector(".conv-message-content");
        if (content && content.id && AppState.truncatedTexts[content.id]) {
          convExpandText(content.id);
        }
      }
    });

  // Add toast animation styles
  const style = document.createElement("style");
  style.textContent = `
        @keyframes toastFadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes toastFadeOut { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(-10px); } }
    `;
  document.head.appendChild(style);
});
