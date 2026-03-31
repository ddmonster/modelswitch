// ========== i18n ==========
window.I18N = {
    current: localStorage.getItem('lang') || 'zh',
    locales: {
        zh: {
            // Tabs
            "tab.providers": "供应商管理",
            "tab.models": "模型配置",
            "tab.keys": "API Key",
            "tab.usage": "用量统计",
            "tab.logs": "调试日志",
            "tab.conversations": "对话记录",

            // Common
            "common.save": "保存",
            "common.cancel": "取消",
            "common.delete": "删除",
            "common.refresh": "刷新",
            "common.edit": "编辑",
            "common.close": "关闭",
            "common.copy": "复制",
            "common.create": "创建",
            "common.enabled": "启用",
            "common.disabled": "禁用",
            "common.enable": "启用",
            "common.disable": "禁用",
            "common.confirmDelete": "确认删除?",
            "common.all": "全部",
            "common.success": "成功",
            "common.failed": "失败",
            "common.unlimited": "不限",
            "common.filter": "过滤",
            "common.description": "描述",
            "common.image": "[图片]",
            "common.empty": "[空]",
            "common.loading": "加载中...",
            "common.noMessages": "无消息",

            // Provider
            "provider.add": "+ 新建供应商",
            "provider.formTitle.add": "新建供应商",
            "provider.formTitle.edit": "编辑供应商",
            "provider.empty": "暂无供应商配置",
            "provider.label.name": "名称",
            "provider.label.protocol": "协议类型",
            "provider.label.baseUrl": "Base URL",
            "provider.label.apiKey": "API Key",
            "provider.label.headers": "自定义 Headers (JSON)",
            "provider.option.openai": "OpenAI 兼容",
            "provider.option.anthropic": "Anthropic",
            "provider.ph.name": "如 dashscope",
            "provider.ph.apiKey": "${ENV_VAR} 或直接填入",
            "provider.td.protocol": "协议",
            "provider.btn.test": "测试连通",
            "provider.test.testing": "测试中...",
            "provider.test.success": "连通成功",
            "provider.test.fail": "连通失败",
            "provider.test.status": "状态",
            "provider.test.latency": "延迟",
            "provider.validation.nameRequired": "请填写名称",
            "provider.validation.headersJson": "Headers JSON 格式错误",

            // Model
            "model.add": "+ 新建模型",
            "model.formTitle.add": "新建模型",
            "model.formTitle.edit": "编辑模型",
            "model.empty": "暂无模型配置",
            "model.label.name": "模型名称",
            "model.label.mode": "调用模式",
            "model.label.description": "描述",
            "model.label.adapters": "适配器列表",
            "model.option.chain": "Chain (fallback 链)",
            "model.option.adapter": "Adapter (直接调用)",
            "model.ph.name": "如 glm5",
            "model.ph.description": "模型描述",
            "model.btn.test": "测试调用",
            "model.test.chainProbe": "测试中（逐个探测 chain 中的适配器）...",
            "model.test.success": "调用成功",
            "model.test.allFailed": "全部失败",
            "model.test.hit": "命中",
            "model.validation.nameRequired": "请填写模型名称",
            "model.validation.adapterRequired": "请至少添加一个有效的适配器（供应商和模型名必填）",

            // Adapter
            "adapter.add": "+ 添加适配器",
            "adapter.empty": "暂无适配器，点击下方按钮添加",
            "adapter.selectProvider": "-- 选择供应商 --",
            "adapter.ph.modelName": "上游模型名，如 glm-5、qwen-max",
            "adapter.label.timeout": "超时",
            "adapter.unit.seconds": "秒",

            // API Key
            "key.add": "+ 新建 Key",
            "key.formTitle": "新建 API Key",
            "key.empty": "暂无 API Key",
            "key.label.name": "名称",
            "key.label.description": "描述",
            "key.label.rateLimit": "每分钟限流",
            "key.label.dailyLimit": "每日限流 (0=不限)",
            "key.label.allowedModels": "允许的模型 (逗号分隔，空=全部)",
            "key.ph.name": "如 张三",
            "key.ph.description": "用途说明",
            "key.created.title": "API Key 创建成功",
            "key.created.warning": "请立即保存此 Key，关闭后无法再次查看",
            "key.td.rateLimit": "每分钟限流",
            "key.td.dailyLimit": "每日限流",
            "key.td.allowedModels": "可用模型",

            // Usage
            "usage.label.dimension": "主维度：",
            "usage.label.date": "日期：",
            "usage.dim.provider": "服务商",
            "usage.dim.model": "模型",
            "usage.dim.apiKey": "API Key",
            "usage.date.today": "今日",
            "usage.date.week": "本周",
            "usage.date.month": "本月",
            "usage.summary": "共 {0} 次请求，按{1}分组",
            "usage.th.requests": "调用量",
            "usage.th.successRate": "成功率",
            "usage.th.tokensIn": "输入Token",
            "usage.th.tokensOut": "输出Token",
            "usage.th.avgLatency": "平均延迟",
            "usage.th.success": "成功",
            "usage.th.failed": "失败",
            "usage.btn.drill": "下钻",
            "usage.drill.by": "按{0}下钻",

            // Logs
            "log.label.level": "级别：",
            "log.label.requestId": "Request ID：",
            "log.label.apiKey": "API Key：",
            "log.label.autoRefresh": "自动刷新",
            "log.empty": "暂无日志",

            // Conversations
            "conv.label.apiKey": "API Key：",
            "conv.label.model": "模型：",
            "conv.label.status": "状态：",
            "conv.empty": "暂无对话记录",
            "conv.detail.empty": "点击左侧对话记录查看详情",
            "conv.detail.loadError": "加载失败：{0}",
            "conv.detail.time": "时间",
            "conv.detail.model": "模型",
            "conv.detail.adapter": "适配器",
            "conv.detail.apiKey": "API Key",
            "conv.detail.latency": "延迟",
            "conv.detail.token": "Token",
            "conv.detail.inputMessages": "输入消息",
            "conv.detail.output": "输出",
            "conv.detail.error": "错误：",
            "conv.pagination.prev": "上一页",
            "conv.pagination.next": "下一页",
            "conv.pagination.count": "{0} 条",
            "conv.fold": "fold",
            "conv.expanded": "expanded",
            "conv.toolResult": "工具结果",

            // Role labels
            "role.user": "用户",
            "role.assistant": "助手",
            "role.system": "系统",
            "role.tool": "工具结果",

            // Chain test table
            "chain.th.priority": "优先级",
            "chain.th.provider": "供应商",
            "chain.th.model": "模型",
            "chain.th.status": "状态",
            "chain.th.latency": "延迟",
            "chain.status.skipped": "跳过",
        },
        en: {
            // Tabs
            "tab.providers": "Providers",
            "tab.models": "Models",
            "tab.keys": "API Keys",
            "tab.usage": "Usage",
            "tab.logs": "Logs",
            "tab.conversations": "Conversations",

            // Common
            "common.save": "Save",
            "common.cancel": "Cancel",
            "common.delete": "Delete",
            "common.refresh": "Refresh",
            "common.edit": "Edit",
            "common.close": "Close",
            "common.copy": "Copy",
            "common.create": "Create",
            "common.enabled": "Enabled",
            "common.disabled": "Disabled",
            "common.enable": "Enable",
            "common.disable": "Disable",
            "common.confirmDelete": "Confirm delete?",
            "common.all": "All",
            "common.success": "Success",
            "common.failed": "Failed",
            "common.unlimited": "Unlimited",
            "common.filter": "Filter",
            "common.description": "Description",
            "common.image": "[Image]",
            "common.empty": "[Empty]",
            "common.loading": "Loading...",
            "common.noMessages": "No messages",

            // Provider
            "provider.add": "+ New Provider",
            "provider.formTitle.add": "New Provider",
            "provider.formTitle.edit": "Edit Provider",
            "provider.empty": "No providers configured",
            "provider.label.name": "Name",
            "provider.label.protocol": "Protocol",
            "provider.label.baseUrl": "Base URL",
            "provider.label.apiKey": "API Key",
            "provider.label.headers": "Custom Headers (JSON)",
            "provider.option.openai": "OpenAI Compatible",
            "provider.option.anthropic": "Anthropic",
            "provider.ph.name": "e.g. dashscope",
            "provider.ph.apiKey": "${ENV_VAR} or direct value",
            "provider.td.protocol": "Protocol",
            "provider.btn.test": "Test",
            "provider.test.testing": "Testing...",
            "provider.test.success": "Connected",
            "provider.test.fail": "Connection failed",
            "provider.test.status": "Status",
            "provider.test.latency": "Latency",
            "provider.validation.nameRequired": "Please enter a name",
            "provider.validation.headersJson": "Invalid Headers JSON format",

            // Model
            "model.add": "+ New Model",
            "model.formTitle.add": "New Model",
            "model.formTitle.edit": "Edit Model",
            "model.empty": "No models configured",
            "model.label.name": "Model Name",
            "model.label.mode": "Call Mode",
            "model.label.description": "Description",
            "model.label.adapters": "Adapter List",
            "model.option.chain": "Chain (fallback)",
            "model.option.adapter": "Adapter (direct)",
            "model.ph.name": "e.g. glm5",
            "model.ph.description": "Model description",
            "model.btn.test": "Test",
            "model.test.chainProbe": "Testing (probing each adapter in chain)...",
            "model.test.success": "Call succeeded",
            "model.test.allFailed": "All failed",
            "model.test.hit": "Hit",
            "model.validation.nameRequired": "Please enter a model name",
            "model.validation.adapterRequired": "Please add at least one valid adapter (provider and model name required)",

            // Adapter
            "adapter.add": "+ Add Adapter",
            "adapter.empty": "No adapters. Click button below to add",
            "adapter.selectProvider": "-- Select Provider --",
            "adapter.ph.modelName": "Upstream model, e.g. glm-5, qwen-max",
            "adapter.label.timeout": "Timeout",
            "adapter.unit.seconds": "s",

            // API Key
            "key.add": "+ New Key",
            "key.formTitle": "New API Key",
            "key.empty": "No API Keys",
            "key.label.name": "Name",
            "key.label.description": "Description",
            "key.label.rateLimit": "Rate Limit (per min)",
            "key.label.dailyLimit": "Daily Limit (0=unlimited)",
            "key.label.allowedModels": "Allowed Models (comma-separated, empty=all)",
            "key.ph.name": "e.g. John",
            "key.ph.description": "Purpose",
            "key.created.title": "API Key Created",
            "key.created.warning": "Save this key now. It cannot be viewed again after closing",
            "key.td.rateLimit": "Rate Limit/min",
            "key.td.dailyLimit": "Daily Limit",
            "key.td.allowedModels": "Allowed Models",

            // Usage
            "usage.label.dimension": "Dimension:",
            "usage.label.date": "Date:",
            "usage.dim.provider": "Provider",
            "usage.dim.model": "Model",
            "usage.dim.apiKey": "API Key",
            "usage.date.today": "Today",
            "usage.date.week": "This Week",
            "usage.date.month": "This Month",
            "usage.summary": "{0} requests, grouped by {1}",
            "usage.th.requests": "Requests",
            "usage.th.successRate": "Success Rate",
            "usage.th.tokensIn": "Tokens In",
            "usage.th.tokensOut": "Tokens Out",
            "usage.th.avgLatency": "Avg Latency",
            "usage.th.success": "Success",
            "usage.th.failed": "Failed",
            "usage.btn.drill": "Drill",
            "usage.drill.by": "Drill by {0}",

            // Logs
            "log.label.level": "Level:",
            "log.label.requestId": "Request ID:",
            "log.label.apiKey": "API Key:",
            "log.label.autoRefresh": "Auto Refresh",
            "log.empty": "No logs",

            // Conversations
            "conv.label.apiKey": "API Key:",
            "conv.label.model": "Model:",
            "conv.label.status": "Status:",
            "conv.empty": "No conversations",
            "conv.detail.empty": "Click a conversation on the left to view details",
            "conv.detail.loadError": "Load failed: {0}",
            "conv.detail.time": "Time",
            "conv.detail.model": "Model",
            "conv.detail.adapter": "Adapter",
            "conv.detail.apiKey": "API Key",
            "conv.detail.latency": "Latency",
            "conv.detail.token": "Token",
            "conv.detail.inputMessages": "Input Messages",
            "conv.detail.output": "Output",
            "conv.detail.error": "Error: ",
            "conv.pagination.prev": "Prev",
            "conv.pagination.next": "Next",
            "conv.pagination.count": "{0} items",
            "conv.fold": "fold",
            "conv.expanded": "expanded",
            "conv.toolResult": "Tool Result",

            // Role labels
            "role.user": "User",
            "role.assistant": "Assistant",
            "role.system": "System",
            "role.tool": "Tool Result",

            // Chain test table
            "chain.th.priority": "Priority",
            "chain.th.provider": "Provider",
            "chain.th.model": "Model",
            "chain.th.status": "Status",
            "chain.th.latency": "Latency",
            "chain.status.skipped": "Skipped",
        }
    }
};

function t(key) {
    let s = I18N.locales[I18N.current]?.[key] ?? I18N.locales['zh']?.[key] ?? key;
    if (arguments.length > 1) {
        for (let i = 1; i < arguments.length; i++) {
            s = s.replace('{' + (i - 1) + '}', arguments[i]);
        }
    }
    return s;
}

function applyI18n() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    // Update lang switcher active state
    document.querySelectorAll('.lang-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.lang === I18N.current);
    });
}

function setLocale(lang) {
    I18N.current = lang;
    localStorage.setItem('lang', lang);
    applyI18n();
    // Re-render active tab content
    const activeTab = document.querySelector('.tab.active');
    if (activeTab) activeTab.click();
}

// Apply on load
document.addEventListener('DOMContentLoaded', () => applyI18n());
