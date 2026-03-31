# ModelSwitch - LLM Gateway Proxy

ModelSwitch 是一个 LLM 网关代理，对外暴露 OpenAI 兼容和 Anthropic 兼容 API，后端支持多提供商自动切换。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 健康检查
curl http://localhost:8000/api/config/health
```

## API Key 使用指南

### 认证方式

网关支持三种认证方式（任选其一）：

```bash
# 方式 1: Authorization Bearer（推荐）
-H "Authorization: Bearer sk-your-api-key"

# 方式 2: x-api-key 头
-H "x-api-key: sk-your-api-key"

# 方式 3: 直接使用 sk- 前缀的值
-H "Authorization: sk-your-api-key"
```

### 可用端点

| 端点 | 协议 | 说明 |
|------|------|------|
| `POST /openai/chat/completions` | OpenAI | 聊天补全 |
| `POST /v1/chat/completions` | OpenAI | 聊天补全（向后兼容） |
| `GET /openai/models` | OpenAI | 模型列表 |
| `POST /anthropic/messages` | Anthropic | Messages API |
| `POST /v1/messages` | Anthropic | Messages API（向后兼容） |

### 使用示例

#### OpenAI 协议调用

```bash
curl -s http://localhost:8000/openai/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'
```

#### Anthropic 协议调用

```bash
curl -s http://localhost:8000/anthropic/messages \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'
```

#### 流式调用

```bash
curl -s http://localhost:8000/openai/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [{"role": "user", "content": "讲个故事"}],
    "max_tokens": 500,
    "stream": true
  }'
```

#### Tool Use（函数调用）

```bash
curl -s http://localhost:8000/anthropic/messages \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [{"role": "user", "content": "北京天气怎么样？"}],
    "max_tokens": 500,
    "tools": [{
      "name": "get_weather",
      "description": "获取城市天气",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
      }
    }]
  }'
```

### Python SDK 示例

#### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000/openai"
)

response = client.chat.completions.create(
    model="glm-5",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

#### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000"
)

response = client.messages.create(
    model="glm-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "你好"}]
)
print(response.content[0].text)
```

### 配置 Claude Code

在 `~/.claude/settings.json` 中配置：

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "YOUR_API_KEY",
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_MODEL": "glm-5"
  }
}
```

### Web 管理界面

访问 `http://localhost:8000/` 进入 Web 管理界面，支持：

- **Providers**: 查看和管理上游提供商
- **Models**: 配置模型及其适配器链
- **API Keys**: 创建、查看、删除 API Key
- **Usage Stats**: 查看用量统计
- **Debug Logs**: 查看请求日志

## API Key 管理

### 通过 Web 界面

1. 访问 `http://localhost:8000/`
2. 切换到 "API Keys" 标签
3. 点击 "新建 Key" 创建新的 API Key
4. 可设置：名称、描述、速率限制、日限额、允许的模型、过期时间

### 通过 API

```bash
# 列出所有 API Key
curl -s http://localhost:8000/api/keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"

# 创建新 Key
curl -s -X POST http://localhost:8000/api/keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-app",
    "description": "我的应用",
    "rate_limit": 60,
    "daily_limit": 1000,
    "allowed_models": ["glm-5"]
  }'

# 删除 Key
curl -s -X DELETE http://localhost:8000/api/keys/my-app \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

## 可用模型

模型配置在 `config.yaml` 中定义，可根据上游提供商支持的模型自定义。

## 故障排查

### 日志查看

```bash
# 请求日志
tail -f logs/gateway.log

# 会话日志（完整请求/响应）
tail -f logs/conversations.jsonl
```

### 常见问题

1. **401 Unauthorized**: 检查 API Key 是否正确
2. **403 Forbidden**: 检查 API Key 是否启用的，或模型是否在 allowed_models 列表中
3. **404 Model not found**: 检查模型名称是否正确（区分大小写，建议用小写）
4. **502/503 Upstream error**: 上游提供商不可用，检查 provider 配置

## License

MIT