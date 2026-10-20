# LangChain Day16：MCP、外部工具服务与标准化集成

## 1. 今天学到了什么

Day16 的核心不是再写一个普通 Python 函数，而是把外部能力包装成一个遵守统一协议的独立服务。

今天完成了以下内容：

1. 理解 JSON-RPC 请求、响应和错误对象；
2. 理解 stdio 和 Streamable HTTP 两种传输方式；
3. 使用正式 MCP SDK 创建 Server 和 Client；
4. 发布和调用 Tool；
5. 发布和读取 Resource；
6. 发布和获取 Prompt；
7. 把 MCP Tool 适配为 LangChain Tool；
8. 让 LangChain Agent 调用 MCP 服务；
9. 连接多个 MCP Server；
10. 处理工具错误和协议错误；
11. 使用 ToolAnnotations 描述工具风险；
12. 在高风险工具执行前进行人工审批；
13. 使用 Bearer Token 进行身份认证；
14. 使用 scope 进行细粒度授权；
15. 编写 MCP 契约测试；
16. 整合完整 MCP Server、Client 和 SQLite 审计日志。

---

## 2. MCP 是什么

MCP 是 Model Context Protocol，即模型上下文协议。

它解决的核心问题是：让大模型应用使用统一方式连接外部工具、资料和提示词，不必为每种服务重新设计一套私有接口。
  
```text
大模型应用
    ↓ MCP Client
统一 MCP 协议
    ↓
MCP Server
    ├── Tools
    ├── Resources
    └── Prompts
```

MCP 不等于大模型，也不负责模型推理。它主要定义客户端和能力服务之间如何发现和调用能力。

---

## 3. MCP 中的主要角色

### 3.1 Host

Host 是承载整个大模型应用的程序，例如桌面助手、IDE 或 LangChain Agent 应用。

### 3.2 Client

Client 位于 Host 内部，负责：

- 连接 MCP Server；
- 协商协议版本；
- 发现服务端能力；
- 发送工具、资源和 Prompt 请求；
- 接收结构化返回值。

### 3.3 Server

Server 负责公开实际能力，例如：

- 计算工具；
- 数据库查询；
- 企业资料；
- 文件操作；
- 可复用提示词模板。

一个 Host 可以同时连接多个 MCP Server。

---

## 4. JSON-RPC 是 MCP 消息的基础

一个简化请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "add",
    "arguments": {"a": 20, "b": 4}
  }
}
```

对应的简化响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{"type": "text", "text": "24"}]
  }
}
```

关键点：

- `jsonrpc` 表示协议版本；
- `id` 用来匹配请求和响应；
- `method` 表示调用的方法；
- `params` 表示请求参数；
- 成功时返回 `result`；
- 失败时返回 `error`。

MCP SDK 会替我们完成 JSON 序列化、消息匹配、Schema 校验和错误转换。

---

## 5. stdio 与 Streamable HTTP

### 5.1 stdio

客户端启动一个本地 MCP Server 子进程，通过标准输入和标准输出传输协议消息。

```text
本地客户端
    ↓ stdin
本地 MCP Server 子进程
    ↑ stdout
```

适合：

- 本机工具；
- IDE 插件；
- 命令行服务；
- 不需要开放网络端口的场景。

服务端日志应该写到 `stderr`，避免污染用于协议消息的 `stdout`。

### 5.2 Streamable HTTP

客户端通过 HTTP 连接独立运行的远程服务。

```text
MCP Client
    ↓ HTTP POST /mcp
网络
    ↓
MCP Server
```

适合：

- 跨机器访问；
- 多客户端共享；
- 集中认证和授权；
- 云服务或企业内部服务。

---

## 6. Tool、Resource 和 Prompt 的区别

| 能力 | 主要用途 | 是否执行动作 | 调用方式 |
|---|---|---:|---|
| Tool | 计算、查询、修改外部数据 | 是 | `tools/call` |
| Resource | 读取由 URI 标识的资料 | 通常只读 | `resources/read` |
| Prompt | 根据参数生成消息模板 | 否 | `prompts/get` |

### 6.1 Tool

```python
@mcp_server.tool()
async def add(a: float, b: float) -> float:
    return a + b
```

SDK 会根据函数名、文档字符串和类型注解生成工具名称、说明和 JSON Schema。

### 6.2 Resource

```python
@mcp_server.resource("course://notes/mcp")
async def read_mcp_note() -> str:
    return "MCP 学习资料"
```

Resource 使用稳定 URI。客户端不必知道资料实际来自文件、数据库还是远程 API。

### 6.3 Prompt

```python
@mcp_server.prompt(name="teach-topic")
async def teach_topic(topic: str) -> UserMessage:
    return UserMessage(f"请讲解 {topic}")
```

Prompt 只生成消息，不自动调用大模型。

---

## 7. MCP Tool 如何进入 LangChain Agent

MCP SDK 的 Tool 描述与 LangChain 的 `BaseTool` 不是同一种 Python 对象，因此需要适配器转换。

```text
MCP Server
    ↓ tools/list
MCP Tool 描述
    ↓ MCPAdapter
LangChain BaseTool
    ↓ create_agent(tools=...)
Agent 可调用工具
```

当模型第二次请求另一个工具时，仍然会经过相同路径：

```text
AIMessage.tool_calls
    ↓ ToolNode
适配后的 BaseTool
    ↓ MCPAdapter
MCP Client
    ↓ tools/call
MCP Server 中的真实工具函数
    ↓
ToolMessage
```

适配通常在初始化阶段完成，不是每一次调用都重新安装或重新注册工具。但每一次真正的工具执行仍然要通过 MCP Client 向 Server 发送 `tools/call`。

---

## 8. 多 MCP Server

不同能力可以由不同服务负责：

```text
LangChain Agent
    ├── calculator MCP Server
    │      ├── add
    │      └── multiply
    │
    └── knowledge MCP Server
           └── explain_topic
```

为了避免不同服务中出现同名工具，可以使用带服务名前缀的工具名，例如：

```text
calculator_add
calculator_multiply
knowledge_explain_topic
```

这样 Agent 能清楚地选择工具来源。

---

## 9. 错误分层

### 9.1 业务工具错误

例如余额不足、金额非法和除数为零。

服务端可以抛出 `ToolError`，客户端通常会收到：

```text
CallToolResult.is_error = True
```

连接一般仍然可以继续使用。

### 9.2 协议或请求错误

例如资源 URI 不存在、请求格式错误或参数不符合协议。

这类错误可能以 `MCPError` 抛出。

### 9.3 连接错误

例如服务端未启动、端口错误、网络中断或认证失败。

客户端应在传输层捕获异常并给出明确提示。

---

## 10. ToolAnnotations

安全注解用于描述工具特征：

```python
types.ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=False,
)
```

含义：

- `readOnlyHint`：是否只读取数据；
- `destructiveHint`：是否可能修改或破坏数据；
- `idempotentHint`：相同调用重复执行是否仍产生相同效果；
- `openWorldHint`：是否与开放外部世界交互。

必须记住：ToolAnnotations 是提示，不是强制安全边界。

客户端不能盲目信任服务端声明，还应维护自己的高风险工具规则。

---

## 11. 人工审批

客户端审批的正确位置是在发送高风险 `tools/call` 之前：

```text
发现 transfer_money
    ↓
检查工具名称和 annotations
    ↓
展示完整参数
    ↓
用户输入 yes / no
    ├── no  → 不发送 tools/call
    └── yes → 发送 tools/call
```

拒绝审批不是工具执行失败，而是客户端根本没有发送执行请求，因此不会产生副作用。

---

## 12. 认证与授权的区别

### 12.1 认证 Authentication

认证回答：你是谁？

```text
Bearer Token
    ↓
TokenVerifier.verify_token()
    ↓
AccessToken
    ├── subject
    ├── client_id
    ├── scopes
    └── resource
```

### 12.2 授权 Authorization

授权回答：你可以做什么？

```python
if "account:write" not in access_token.scopes:
    raise ToolError("缺少权限")
```

Reader：

```text
course:read
account:read
```

Operator：

```text
course:read
account:read
account:write
```

角色和权限解耦后，权限粒度更细。角色只是权限集合，服务端真正判断的是具体 scope。

---

## 13. 为什么要有客户端和服务端双重保护

完整转账保护包含：

```text
客户端检查 ToolAnnotations
    ↓
客户端检查当前 scopes
    ↓
人工审批
    ↓
发送 tools/call
    ↓
服务端再次检查 account:write
    ↓
验证业务参数
    ↓
执行操作并记录审计日志
```

客户端检查改善用户体验并避免无效请求；服务端检查才是真正不能绕过的安全边界。

即使恶意客户端跳过人工审批，服务端仍会执行身份、scope 和业务规则检查。

---

## 14. 审计日志

最终服务端把重要操作写入 SQLite：

```text
created_at
subject
action
arguments_json
result
```

数据流：

```text
MCP 工具收到请求
    ↓
取得 AccessToken.subject
    ↓
执行业务验证或操作
    ↓
write_audit_log()
    ↓
INSERT INTO audit_logs
```

审计日志用于回答：

- 谁执行了操作；
- 什么时间执行；
- 调用了什么能力；
- 参数是什么；
- 最终成功还是被拒绝。

审计程序使用 `?` 占位符执行参数化 SQL，而不是直接拼接用户输入，可降低 SQL 注入风险。

---

## 15. 契约测试

契约测试从客户端可观察视角验证服务是否遵守约定：

- Server 名称和协议版本；
- 工具是否存在；
- 参数 JSON Schema；
- ToolAnnotations；
- 成功返回格式；
- 业务错误格式；
- Resource URI 和正文；
- Prompt 名称、参数和渲染结果。

进程内 Client 测试不需要监听端口，也不需要调用大模型，因此速度快而且结果稳定。

```bash
python day16/python/day16_mcp_contract_tests.py
```

---

## 16. 最终项目结构

```text
day16/
├── database/
│   └── day16_final_mcp.db
├── notes/
│   └── LangChain-Day16-学习笔记.md
├── python/
│   ├── day16_final_mcp_server.py
│   ├── day16_final_mcp_client.py
│   ├── day16_final_audit_viewer.py
│   └── 其他分步练习文件
└── requirements.txt
```

---

## 17. 最终项目运行方法

终端 1：

```bash
python day16/python/day16_final_mcp_server.py
```

终端 2，运行 Reader：

```bash
python day16/python/day16_final_mcp_client.py reader
```

终端 2，运行 Operator：

```bash
python day16/python/day16_final_mcp_client.py operator
```

查看所有审计记录：

```bash
python day16/python/day16_final_audit_viewer.py all
```

查看转账记录：

```bash
python day16/python/day16_final_audit_viewer.py action transfer_money
```

查看统计：

```bash
python day16/python/day16_final_audit_viewer.py stats
```

---

## 18. 最终系统完整数据流

```text
用户选择 reader / operator
    ↓
客户端选择对应 Token
    ↓
Authorization: Bearer <token>
    ↓
Streamable HTTP POST /mcp
    ↓
TokenVerifier
    ↓
AccessToken(subject, client_id, scopes, resource)
    ↓
MCP Client 发现服务端能力
    ├── list_tools
    ├── list_resources
    └── list_prompts
    ↓
调用具体能力
    ├── Tool：执行动作
    ├── Resource：读取资料
    └── Prompt：渲染消息模板
    ↓
客户端检查高风险 ToolAnnotations
    ↓
人工审批
    ↓
服务端检查 scope 和业务参数
    ↓
执行操作
    ↓
返回 CallToolResult
    ↓
写入 SQLite 审计日志
```

---

## 19. 今天最重要的结论

1. MCP 是客户端与外部能力服务之间的标准协议，不是大模型本身。
2. Tool 执行动作，Resource 提供 URI 资料，Prompt 提供消息模板。
3. MCP Tool 需要适配为 LangChain BaseTool 后才能交给 Agent。
4. 工具发现通常在初始化阶段进行，但每次执行仍要发送 `tools/call`。
5. ToolAnnotations 只提供风险提示，不能代替真正的权限检查。
6. 认证解决“你是谁”，scope 授权解决“你能做什么”。
7. 客户端审批改善安全交互，服务端授权才是不可绕过的安全边界。
8. 高风险操作需要保存身份、参数、结果和时间等审计信息。
9. 契约测试能防止服务升级后悄悄破坏客户端依赖的接口。
10. 最终系统是 Client、协议、Server、业务逻辑、安全控制和审计的组合。

---

## 20. 自测题

1. MCP Host、Client 和 Server 分别负责什么？
2. stdio 和 Streamable HTTP 分别适合什么场景？
3. Tool、Resource 和 Prompt 有什么区别？
4. 为什么 MCP Tool 不能直接当作 LangChain BaseTool 使用？
5. 第二次工具调用是否仍然会经过 MCP Server？
6. `destructiveHint=True` 会不会自动阻止工具执行？
7. Reader 为什么不能执行 `transfer_money`？
8. 为什么客户端检查过 scope 后，服务端还要再次检查？
9. 拒绝人工审批后，为什么不会产生工具副作用？
10. 契约测试和普通业务单元测试分别关注什么？

