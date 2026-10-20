# LangChain Day18：WebSocket 双向实时对话与可取消模型流

## 1. 今天完成了什么

Day18 固定完成 7 个步骤：

1. WebSocket 基础双向通信；
2. 通过 WebSocket 传输模型流；
3. `version/type/request_id/payload` 结构化协议；
4. 多客户端并发与连接状态隔离；
5. 在模型输出期间发送取消指令；
6. 最终可靠实时对话程序；
7. 学习笔记与最终代码解读。

核心成果：

```text
一条持久 WebSocket
├── 客户端 → ask / cancel / status
└── 服务端 → queued / start / token / done / cancelled / error
```

---

## 2. WebSocket 与 SSE 的区别

| 对比项 | SSE | WebSocket |
|---|---|---|
| 方向 | 主要是服务端到客户端 | 真正双向 |
| 建立方式 | 普通持续 HTTP 响应 | HTTP Upgrade 后切换协议 |
| 数据形式 | UTF-8 文本事件 | Text/Binary Frame |
| 客户端控制 | 通常另发 HTTP 请求 | 同一连接直接发送 |
| 自动重连 | 浏览器 EventSource 支持 | 应用自行实现 |
| 适用场景 | 文本推送、通知、模型输出 | 聊天、游戏、协作、实时控制 |

SSE 更简单；WebSocket 更适合模型输出期间随时取消、暂停或发送其他控制消息。

---

## 3. WebSocket 握手

```text
客户端发送 HTTP GET /ws
+ Upgrade: websocket
    ↓
服务端接受升级
    ↓
HTTP 101 Switching Protocols
    ↓
建立持久 WebSocket
```

aiohttp 服务端：

```python
websocket = web.WebSocketResponse(heartbeat=20)
await websocket.prepare(request)
```

客户端：

```python
websocket = await session.ws_connect(
    "ws://127.0.0.1:8015/ws"
)
```

---

## 4. 基础收发

服务端持续接收：

```python
async for message in websocket:
    ...
```

发送 JSON：

```python
await websocket.send_json(data)
```

客户端和服务端都能调用 `send_json()` 和 `receive()`，因此双方地位比 SSE 更对称。

WebSocket 连接可以依次传输很多业务消息，不必每次提问都重新进行 HTTP 握手。

---

## 5. 模型流转换为 WebSocket 消息

```text
HumanMessage
→ llm.astream()
→ AIMessageChunk
→ chunk.content
→ {type: token, payload: {text: ...}}
→ websocket.send_json()
→ 客户端立即打印
```

完成时再发送：

```json
{
  "type": "done",
  "payload": {
    "chunk_count": 50,
    "answer_length": 120
  }
}
```

---

## 6. 统一消息协议

最终协议外壳：

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

字段作用：

| 字段 | 类型 | 作用 |
|---|---|---|
| `version` | `str` | 协议兼容性 |
| `type` | `str` | 消息路由 |
| `request_id` | `str | None` | 请求归属与精准取消 |
| `payload` | `dict` | 业务数据 |

统一协议让客户端不必根据每种消息猜测字段位置。

---

## 7. connection_id 与 request_id

```text
connection_id
└── 一条 WebSocket 的整个生命周期
    ├── request_id=req-001
    ├── request_id=req-002
    └── request_id=req-003
```

`connection_id` 用于连接级日志和会话隔离。

`request_id` 用于：

- 归属流式 token；
- 区分多个请求；
- 查找后台模型 Task；
- 精准取消；
- 关联错误和完成消息。

---

## 8. 多客户端隔离

aiohttp 为每条连接调用一次：

```python
websocket_handler(request)
```

定义在处理函数内部的状态属于该连接：

```python
connection_id
client_name
request_count
history
active_task
```

因此不同客户端不会共享局部状态：

```text
连接 A → handler A → history A
连接 B → handler B → history B
连接 C → handler C → history C
```

只有确实需要全局共享的数据，例如模型 Semaphore，才放在模块级别。

---

## 9. 为什么模型流必须放入 Task

错误结构：

```python
async for message in websocket:
    await stream_answer(...)
```

`stream_answer()` 没结束时，接收循环无法读取 `cancel`。

正确结构：

```python
task = asyncio.create_task(run_model(...))
```

主循环立刻恢复：

```text
接收循环 → 持续读取 ask/cancel/status
模型 Task → 独立读取模型流并发送 token
```

这两个协程在同一事件循环中并发推进。

---

## 10. 精准取消

客户端：

```json
{
  "type": "cancel",
  "request_id": "req-123",
  "payload": {}
}
```

服务端：

```python
if request_id == state.active_request_id:
    state.active_task.cancel()
```

模型 Task：

```python
except asyncio.CancelledError:
    await send(..., "cancelled", ...)
```

两阶段反馈：

```text
cancel_requested → 服务端收到指令
cancelled        → 模型 Task 已真正停止
```

---

## 11. send_lock

最终服务端中，主接收协程和模型 Task 都可能发送消息。

```python
send_lock: asyncio.Lock
```

发送时：

```python
async with state.send_lock:
    await websocket.send_json(...)
```

它把同一连接的写操作串行化，避免多个协程同时写 WebSocket。

Lock 只保护短暂的发送操作，不能包住整个模型生成，否则 cancel 回复仍会被阻塞。

---

## 12. 全局 Semaphore

```python
model_semaphore = asyncio.Semaphore(2)
```

它由所有连接共享，限制整个服务最多同时运行两个模型请求。

```text
WebSocket 连接可以很多
    ↓
ask 可以先进入 queued
    ↓
只有获得 Semaphore 的请求调用模型
```

必须在 `finally` 中释放：

```python
finally:
    if acquired:
        model_semaphore.release()
```

这样正常完成、模型异常、客户端取消和断线都不会泄漏许可。

---

## 13. 每连接对话记忆

最终状态类：

```python
@dataclass
class ConnectionState:
    connection_id: str
    history: list[BaseMessage]
    active_request_id: str | None
    active_task: asyncio.Task[None] | None
    completed_requests: int
    send_lock: asyncio.Lock
```

`history` 在同一 WebSocket 中持续存在，关闭连接后消失。

模型输入：

```python
model_messages = [
    SystemMessage(...),
    *state.history,
    HumanMessage(content=question),
]
```

---

## 14. 事务式记忆提交

只有模型完整成功后才执行：

```python
state.history.extend([
    HumanMessage(content=question),
    AIMessage(content=complete_answer),
])
```

如果取消：

```text
partial_answer 返回客户端
saved_to_history=False
```

这样下一轮不会读取一个残缺回答。

```text
开始生成 ≠ 已经形成有效历史
完整成功 → 提交历史
取消/失败 → 回滚本轮
```

---

## 15. Pydantic 协议校验

```python
class ClientMessage(BaseModel):
    version: Literal["1.0"]
    type: Literal["ask", "cancel", "status"]
    request_id: str | None
    payload: Payload
```

解析：

```python
message = ClientMessage.model_validate_json(raw_text)
```

它会同时完成：

- JSON 解码；
- 字段存在性检查；
- 字段类型检查；
- 协议版本检查；
- `type` 枚举检查；
- 多余字段拒绝。

---

## 16. 最终服务端函数调用关系

```text
main()
→ create_app()
→ web.run_app()
→ GET /ws
→ websocket_handler()
→ ConnectionState()
→ receive client JSON
→ handle_client_message()
    ├── ask → create_task(run_model())
    ├── cancel → active_task.cancel()
    └── status → send(status_result)
→ run_model()
→ Semaphore.acquire()
→ llm.astream()
→ send(token)
→ 成功后提交 history
→ send(done)
→ finally release + clear active state
```

---

## 17. 最终客户端的两个 Task

```python
receiver = asyncio.create_task(receive_loop(...))
keyboard = asyncio.create_task(input_loop(...))
```

键盘任务：

```text
普通文字 → ask
/cancel  → cancel
/status  → status
/exit    → close
```

接收任务：

```text
queued/start/token/done
cancel_requested/cancelled
status_result/error/busy
```

两个 Task 并发，用户不必等模型输出完才能输入 `/cancel`。

---

## 18. asyncio.to_thread(input)

普通 `input()` 会阻塞事件循环。

最终客户端使用：

```python
user_input = await asyncio.to_thread(input, "\n你：")
```

终端阻塞读取被放到工作线程，事件循环仍能继续接收模型 token。

这就是为什么模型输出和用户命令可以同时工作。

---

## 19. 客户端状态

```python
@dataclass
class ClientState:
    active_request_id: str | None
    text_parts: list[str]
    running: bool
```

| 字段 | 作用 |
|---|---|
| `active_request_id` | 当前请求编号，供 `/cancel` 使用 |
| `text_parts` | 当前回答的文本片段 |
| `running` | 输入和接收循环是否继续 |

收到 `done` 或 `cancelled` 后清空活动请求，用户才能发送下一问。

---

## 20. 最终完整数据流

```text
用户终端输入
→ asyncio.to_thread(input)
→ envelope(ask, request_id, question)
→ WebSocket Text Frame
→ Pydantic ClientMessage
→ create_task(run_model)
→ queued
→ Semaphore.acquire()
→ start
→ state.history + 新 HumanMessage
→ llm.astream()
→ AIMessageChunk
→ send_lock
→ token JSON
→ WebSocket Client
→ receive_loop()
→ 立即打印
→ 模型成功完成
→ HumanMessage + AIMessage 提交 history
→ done
→ release Semaphore
→ 清空 active_request_id/active_task
```

取消支路：

```text
用户输入 /cancel
→ cancel(request_id)
→ active_task.cancel()
→ CancelledError
→ cancelled(partial_answer, saved_to_history=False)
→ release Semaphore
→ 不修改 history
```

---

## 21. 数据类型变化

```text
终端输入 str
→ dict[str, Any]
→ JSON WebSocket Text Frame
→ ClientMessage(Pydantic)
→ HumanMessage
→ list[BaseMessage]
→ AIMessageChunk
→ text: str
→ dict payload
→ JSON WebSocket Text Frame
→ 客户端 dict
→ text_parts: list[str]
→ complete_answer: str
→ AIMessage
→ history: list[BaseMessage]
```

---

## 22. 最终代码运行

终端一：

```bash
python day18/python/day18_final_websocket_server.py
```

终端二：

```bash
python day18/python/day18_final_websocket_client.py
```

建议测试：

```text
你：我叫小林，正在学习 WebSocket。
你：你还记得我叫什么吗？
你：请详细讲解十点异步编程。
你：/cancel
你：/status
你：/exit
```

---

## 23. 今天所有代码

```text
day18/python/
├── day18_websocket_basic_server.py
├── day18_websocket_basic_client.py
├── day18_websocket_llm_server.py
├── day18_websocket_llm_client.py
├── day18_websocket_protocol_server.py
├── day18_websocket_protocol_client.py
├── day18_websocket_multi_server.py
├── day18_websocket_multi_client.py
├── day18_websocket_cancel_server.py
├── day18_websocket_cancel_client.py
├── day18_final_websocket_server.py
└── day18_final_websocket_client.py
```

---

## 24. 最重要的结论

1. WebSocket 在一次 HTTP Upgrade 后提供持久双向通道。
2. `request_id` 把 ask、token、cancel 和 done 关联起来。
3. 多条连接通过 handler 局部状态实现隔离。
4. 模型生成必须放入后台 Task，主循环才能继续接收 cancel。
5. 同一连接上的并发写入应使用 Lock 串行化。
6. 全局 Semaphore 限制所有连接的模型并发总量。
7. 取消的残缺回答不应写入正式对话历史。
8. `asyncio.to_thread(input)` 避免终端输入阻塞网络接收。
9. 连接关闭时必须取消并回收所有后台 Task。
10. SSE 适合简单推送，WebSocket 适合需要实时双向控制的对话。

---

## 25. 自测题

1. WebSocket 握手为什么最初仍使用 HTTP？
2. WebSocket 与 SSE 的核心方向差异是什么？
3. `connection_id` 和 `request_id` 有何区别？
4. 为什么直接 `await stream_answer()` 无法及时处理 cancel？
5. 为什么模型流需要 `create_task()`？
6. `send_lock` 应保护单次发送还是整个生成过程？
7. Semaphore 为什么是所有连接共享的？
8. 取消后的部分回答为什么不写入 history？
9. `input()` 为什么会影响异步客户端？
10. WebSocket 断开时需要清理哪些资源？

