# LangChain Day19：浏览器聊天界面与 WebSocket 前后端集成

## 1. 今天完成了什么

Day19 把 Day18 的终端 WebSocket 客户端升级成真正的浏览器聊天页面。

今天固定完成了七步：

1. 浏览器与 Python WebSocket 基础双向通信。
2. 使用 `start / token / done` 更新同一个 DOM 元素。
3. 将真实的 `llm.astream()` 模型流发送到浏览器。
4. 使用后台 Task 支持取消、状态查询和按钮状态切换。
5. 检测断线、应用层心跳与指数退避重连。
6. 整合协议校验、模型限流、会话记忆和静态资源。
7. 整理最终笔记和完整代码数据流。

最终代码：

```text
day19/
├── python/day19_final_browser_chat_server.py
├── html/day19_final_browser_chat.html
├── javascript/day19_final_browser_chat.js
├── css/day19_final_browser_chat.css
└── notes/LangChain-Day19-学习笔记.md
```

---

## 2. 最终系统的四层结构

```text
HTML
├── 标题
├── 连接状态
├── 消息区域
├── 输入框
└── 发送、取消、状态按钮

CSS
├── 页面布局
├── 用户与助手消息样式
├── 按钮状态
└── 移动端适配

JavaScript
├── WebSocket 连接
├── 协议消息构造
├── DOM 流式更新
├── 自动重连
├── 心跳
└── 用户交互状态

Python
├── 静态文件路由
├── WebSocket 服务
├── Pydantic 协议校验
├── DeepSeek 流式调用
├── Task 取消
├── Semaphore 限流
└── 会话记忆
```

---

## 3. HTTP 与 WebSocket 的职责

最终程序包含四条路由：

```python
app.router.add_get("/", index_handler)
app.router.add_get("/app.js", javascript_handler)
app.router.add_get("/app.css", css_handler)
app.router.add_get("/ws", websocket_handler)
```

前三条是普通 HTTP 请求：

```text
GET /
→ 返回 HTML

GET /app.js
→ 返回 JavaScript

GET /app.css
→ 返回 CSS
```

最后一条会把 HTTP 连接升级为 WebSocket：

```text
GET /ws?client_id=browser-xxx
→ HTTP Upgrade
→ 双向 WebSocket 长连接
```

---

## 4. 浏览器加载顺序

浏览器访问首页后：

```text
GET /
→ 解析 HTML
→ 发现 /app.css
→ 请求 CSS
→ 发现 defer /app.js
→ 请求 JavaScript
→ HTML 解析完成
→ 执行 JavaScript
→ connect()
→ 创建 WebSocket
```

HTML 使用：

```html
<script src="/app.js" defer></script>
```

`defer` 表示下载脚本时不阻塞 HTML 解析，并在 HTML 解析完成后执行。

因此 JavaScript 执行时，页面中的按钮和输入框已经存在。

---

## 5. DOM 是什么

DOM 是浏览器把 HTML 转换成的对象树。

例如：

```html
<p id="connection-status">正在连接</p>
```

JavaScript 可以取得这个对象：

```javascript
const connectionStatus = document.querySelector("#connection-status");
```

再修改页面内容：

```javascript
connectionStatus.textContent = "连接状态：已连接";
```

这不是重新加载整个网页，而是只修改页面中的一个节点。

---

## 6. 为什么流式回答只创建一个 DOM 元素

收到 `start` 时创建助手消息框：

```javascript
activeAnswerElement = addMessage("助手", "", "assistant");
```

收到每个 `token` 时：

```javascript
activeAnswerElement.textContent += payload.text;
```

数据变化：

```text
""
→ "WebSocket"
→ "WebSocket 是"
→ "WebSocket 是一种"
→ "WebSocket 是一种双向通信协议"
```

如果每个 token 都创建新元素，页面会出现几十个甚至几百个消息框。

正确做法是：一轮回答只创建一个元素，所有 token 持续追加到它。

---

## 7. 前后端统一协议

客户端命令格式：

```json
{
  "version": "1.0",
  "type": "ask",
  "request_id": "req-123",
  "payload": {
    "question": "什么是 WebSocket？"
  }
}
```

字段职责：

| 字段 | 类型 | 作用 |
|---|---|---|
| `version` | `str` | 协议版本 |
| `type` | `str` | 命令或事件类型 |
| `request_id` | `str \| null` | 标识一次请求 |
| `payload` | `object` | 业务数据 |

浏览器支持发送：

```text
ask
cancel
status
ping
```

服务端支持返回：

```text
connected
queued
start
token
done
cancel_requested
cancelled
status_result
pong
busy
error
cancel_not_found
```

---

## 8. Pydantic 协议校验

服务端通过 `ClientCommand` 描述客户端消息：

```python
class ClientCommand(BaseModel):
    version: Literal["1.0"]
    type: Literal["ask", "cancel", "status", "ping"]
    request_id: str | None
    payload: CommandPayload
```

原始 WebSocket 文本通过：

```python
ClientCommand.model_validate_json(raw_text)
```

完成两项工作：

```text
JSON 字符串解析
+
字段类型和规则校验
```

校验成功：

```text
raw_text: str
→ ClientCommand
```

校验失败：

```text
ValidationError
→ error 事件
→ 浏览器显示错误
```

`extra="forbid"` 表示不允许未定义字段，避免协议悄悄接受拼错的参数。

---

## 9. connection_id、client_id 和 request_id

三种编号不能混为一谈。

### connection_id

标识一条具体 WebSocket 连接。

```text
刷新网页前：conn-111
刷新网页后：conn-222
```

每次重连都会变化。

### client_id

标识一个浏览器会话。

JavaScript 把它保存到：

```javascript
localStorage
```

刷新页面后仍然不变，用于寻找服务端的 `SessionState`。

### request_id

标识一次具体命令或模型问题：

```text
req-xxx
status-xxx
ping-xxx
```

总结：

```text
client_id
└── 一个浏览器会话
    ├── connection_id 1
    ├── connection_id 2（重连）
    └── 多个 request_id
```

---

## 10. localStorage 的作用

JavaScript 首次运行时：

```javascript
const created = `browser-${crypto.randomUUID()}`;
localStorage.setItem(CLIENT_ID_KEY, created);
```

下次刷新时：

```javascript
localStorage.getItem(CLIENT_ID_KEY)
```

`localStorage` 保存在浏览器本地，不会随着刷新消失。

它只保存 `client_id`，没有保存完整聊天内容。

聊天历史仍由服务端 `SessionState` 保存，浏览器连接成功后从服务端取回。

---

## 11. SessionState 与 ConnectionState

服务端把状态拆成两层。

### SessionState

```python
history: list[BaseMessage]
completed_requests: int
cancelled_requests: int
```

它由 `client_id` 标识，可以跨 WebSocket 重连复用。

### ConnectionState

```python
connection_id: str
client_id: str
session: SessionState
active_request_id: str | None
active_task: asyncio.Task[None] | None
send_lock: asyncio.Lock
```

它只服务于当前 WebSocket，连接断开后销毁。

关系：

```text
sessions: dict[str, SessionState]
    ↓ client_id
SessionState
    ↑
ConnectionState
    ↑
当前 WebSocket
```

---

## 12. 当前记忆的持久化范围

当前服务端使用：

```python
sessions: dict[str, SessionState] = {}
```

因此：

```text
刷新网页或 WebSocket 重连
→ 历史仍存在

停止并重新运行 Python 服务
→ 历史消失
```

它是“进程内会话记忆”，不是数据库持久化。

生产环境通常会换成：

```text
SQLite
PostgreSQL
Redis
LangGraph Checkpointer
```

---

## 13. 为什么模型调用必须放进 Task

错误结构：

```python
await run_model(...)
```

此时 WebSocket 主循环必须等模型结束，期间无法接收取消指令。

正确结构：

```python
state.active_task = asyncio.create_task(run_model(...))
```

于是同一事件循环中存在两条并发路线：

```text
WebSocket 接收协程
├── ask
├── cancel
├── status
└── ping

模型 Task
├── 等待 Semaphore
├── llm.astream()
├── 发送 token
└── 发送 done
```

---

## 14. 模型流到 DOM 的完整转换

```text
model_messages: list[BaseMessage]
    ↓
llm.astream(model_messages)
    ↓
AsyncIterator[AIMessageChunk]
    ↓
chunk.content: str
    ↓
Python dict
    ↓
websocket.send_json()
    ↓
WebSocket 文本帧
    ↓
JavaScript message 事件
    ↓
JSON.parse(event.data)
    ↓
普通 JavaScript object
    ↓
payload.text
    ↓
activeAnswerElement.textContent += text
    ↓
用户看到逐渐增长的回答
```

---

## 15. 全局 Semaphore

```python
model_semaphore = asyncio.Semaphore(2)
```

表示整个服务最多同时运行两个模型流。

```text
浏览器 A：获得许可
浏览器 B：获得许可
浏览器 C：等待
```

请求首先收到 `queued`。

取得许可后才收到 `start`。

成功、失败或取消都必须在 `finally` 中归还许可：

```python
if acquired:
    model_semaphore.release()
```

---

## 16. send_lock

同一连接上可能有多个协程发送消息：

```text
模型 Task → token
主接收协程 → status_result
取消分支 → cancel_requested
心跳分支 → pong
```

通过：

```python
async with state.send_lock:
```

保证这些写操作按顺序完成。

`send_lock` 管的是单条连接的写入顺序。

`model_semaphore` 管的是整个服务的模型并发量。

二者职责完全不同。

---

## 17. 事务式记忆提交

模型流生成期间：

```python
text_parts.append(text)
```

但还没有修改正式历史。

只有完整结束后才提交：

```python
session.history.extend(
    [HumanMessage(question), AIMessage(complete_answer)]
)
```

如果取消或断线：

```text
partial_answer 可以显示
但不会写入 history
```

这样下一轮模型不会读到残缺对话。

当前历史限制：

```python
session.history = session.history[-12:]
```

最多保留最近12条消息，约6轮完整问答。

---

## 18. 取消流程

```text
用户点击取消
→ JavaScript 发送 cancel + request_id
→ 服务端检查 active_request_id
→ 发送 cancel_requested
→ active_task.cancel()
→ Task 在 await 处抛出 CancelledError
→ run_model 捕获异常
→ 发送 cancelled
→ 不提交历史
→ finally 释放 Semaphore
→ 清空 active_task
→ 浏览器重新启用输入框
```

取消当前 Task 不会关闭 WebSocket，也不会停止整个服务端。

---

## 19. 自动重连

连接关闭时浏览器安排重连：

```text
第1次等待 1 秒
第2次等待 2 秒
第3次等待 4 秒
第4次等待 8 秒
以后最多等待 8 秒
```

代码原理：

```javascript
Math.min(2 ** reconnectAttempt, 8)
```

成功连接后：

```javascript
reconnectAttempt = 0;
```

必须创建新的 `WebSocket` 对象，因为已经关闭的 WebSocket 不能重新打开。

---

## 20. 心跳

最终版本有两层心跳概念。

### 协议层心跳

```python
web.WebSocketResponse(heartbeat=20)
```

由 aiohttp 维护 WebSocket ping/pong。

### 应用层心跳

浏览器定期发送：

```json
{"type": "ping", "request_id": "ping-xxx", "payload": {}}
```

服务端返回 `pong`。

应用层心跳以后可以携带认证续期、客户端状态或延迟统计。

---

## 21. 前端控制状态

JavaScript 主要依赖：

```javascript
websocket
activeRequestId
activeAnswerElement
reconnectAttempt
reconnectTimer
heartbeatTimer
```

`updateControls()` 根据两个核心条件控制按钮：

```text
连接是否在线
当前是否有活动请求
```

| 状态 | 输入/发送 | 取消 | 状态查询 |
|---|---:|---:|---:|
| 未连接 | 禁用 | 禁用 | 禁用 |
| 已连接、空闲 | 启用 | 禁用 | 启用 |
| 已连接、回答中 | 禁用 | 启用 | 启用 |

---

## 22. 安全地写入页面

最终页面通过：

```javascript
content.textContent = text;
```

而不是：

```javascript
content.innerHTML = text;
```

`textContent` 会把模型输出当作普通文本，降低模型输出或用户内容被解释为恶意 HTML/JavaScript 的风险。

如果以后需要 Markdown，应当使用带安全过滤的 Markdown 渲染方案。

---

## 23. 最终服务端调用链

```text
main()
→ create_app()
→ web.run_app()
→ 浏览器 GET /
→ index_handler()
→ 返回 HTML
→ 浏览器加载 CSS 和 JavaScript
→ 浏览器连接 /ws?client_id=...
→ websocket_handler()
→ 创建 ConnectionState
→ 查找 SessionState
→ connected + history
→ 收到 TEXT
→ handle_command()
→ Pydantic 校验
→ ask 分支
→ create_task(run_model())
→ queued
→ Semaphore.acquire()
→ start
→ llm.astream()
→ token × N
→ 提交历史
→ done
→ release()
```

---

## 24. 最终客户端调用链

```text
浏览器执行 app.js
→ getOrCreateClientId()
→ connect()
→ new WebSocket()
→ open 事件
→ startHeartbeat()
→ connected 事件
→ renderHistory()
→ 用户提交 form
→ 生成 request_id
→ sendCommand("ask")
→ queued/start/token/done
→ 更新状态和 DOM
```

---

## 25. 最终完整数据流

```text
用户输入 question: str
    ↓
JavaScript command object
    ↓
JSON.stringify()
    ↓
WebSocket TEXT
    ↓
Python raw_text: str
    ↓
ClientCommand.model_validate_json()
    ↓
ClientCommand
    ↓
HumanMessage
    ↓
list[BaseMessage]
    ↓
llm.astream()
    ↓
AIMessageChunk
    ↓
chunk.content: str
    ↓
token JSON
    ↓
浏览器 JavaScript object
    ↓
DOM Text 节点
    ↓
完整成功后 HumanMessage + AIMessage
    ↓
SessionState.history
```

---

## 26. 运行最终程序

在项目根目录执行：

```bash
python day19/python/day19_final_browser_chat_server.py
```

浏览器打开：

```text
http://127.0.0.1:8021
```

测试顺序：

1. 告诉模型自己的名字。
2. 追问自己的名字，验证记忆。
3. 刷新页面，再次追问，验证 `client_id` 会话恢复。
4. 提交长问题后点击取消。
5. 查询状态。
6. 打开第二个浏览器或无痕窗口，验证会话隔离。

---

## 27. 今天全部代码

```text
第1步
├── python/day19_browser_basic_server.py
└── html/day19_browser_basic.html

第2步
├── python/day19_dom_stream_server.py
└── html/day19_dom_stream.html

第3步
├── python/day19_browser_llm_server.py
└── html/day19_browser_llm.html

第4步
├── python/day19_browser_control_server.py
└── html/day19_browser_control.html

第5步
├── python/day19_browser_reconnect_server.py
└── html/day19_browser_reconnect.html

第6步最终版
├── python/day19_final_browser_chat_server.py
├── html/day19_final_browser_chat.html
├── javascript/day19_final_browser_chat.js
└── css/day19_final_browser_chat.css
```

---

## 28. 最重要的结论

1. HTML 负责结构，CSS 负责样式，JavaScript 负责浏览器行为，Python 负责后端业务。
2. `llm.astream()` 产生模型流，WebSocket 把它继续传到浏览器。
3. 一轮回答只创建一个 DOM 元素，每个 token 更新同一元素。
4. 模型必须放进后台 Task，主接收循环才能处理取消和状态命令。
5. `request_id` 标识请求，`connection_id` 标识连接，`client_id` 标识可恢复会话。
6. `send_lock` 保护单连接写入顺序，`Semaphore` 限制全局模型并发。
7. 取消和断线产生的残缺回答不应写入正式历史。
8. 自动重连必须创建新的 WebSocket 对象，并使用退避策略。
9. `localStorage` 只保存客户端标识，真正会话状态由服务端管理。
10. 当前记忆只能跨重连，不能跨服务端重启；数据库持久化是后续升级方向。

---

## 29. 自测题

1. 为什么不能对关闭的 WebSocket 对象直接重新调用 `open()`？
2. `client_id` 和 `connection_id` 分别解决什么问题？
3. 为什么模型流必须放进 `asyncio.Task`？
4. 为什么取消的部分回答不能直接写进正式历史？
5. `send_lock` 和 `model_semaphore` 有什么区别？
6. 为什么收到每个 token 时不能创建一个新的助手消息框？
7. 浏览器刷新后，服务端如何找到原来的会话？
8. 为什么使用 `textContent` 比直接使用 `innerHTML` 更安全？
9. 当前代码为什么不能在 Python 服务重启后恢复历史？
10. `queued` 和 `start` 两个事件有什么区别？
