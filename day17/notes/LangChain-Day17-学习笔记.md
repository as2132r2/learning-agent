# LangChain Day17：流式输出、SSE 与实时对话服务

## 1. 今天的学习目标

Day17 把模型的流式输出从本地终端扩展到 HTTP 客户端，并逐步加入真实服务需要的并发、安全清理和连接可靠性机制。

今天的主线可以压缩成六个模块：

1. LangChain 同步与异步流式输出；
2. SSE 服务端与客户端协议解析；
3. 多客户端异步并发；
4. Semaphore 模型并发限制；
5. 客户端取消、服务端清理、心跳和背压；
6. `Last-Event-ID` 断线续传。

---

## 2. 为什么需要流式输出

普通调用：

```python
response = llm.invoke(messages)
```

模型生成完整回答后才返回一个 `AIMessage`：

```text
发送问题
→ 等待完整回答
→ 一次性显示
```

流式调用：

```python
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

模型边生成，程序边接收：

```text
发送问题
→ 收到 AIMessageChunk
→ 立即显示
→ 收到下一个 AIMessageChunk
→ 最终拼接完整回答
```

流式输出主要降低用户的感知等待时间，不一定缩短完整生成的总时间。

---

## 3. AIMessage 与 AIMessageChunk

`invoke()` 返回：

```text
AIMessage
```

`stream()` 或 `astream()` 逐次返回：

```text
AIMessageChunk
```

一个 chunk 不保证等于一个汉字、一个单词或一个完整 Token。拆分大小由模型服务、网络缓冲和 SDK 决定。

第一个或最后一个 chunk 的 `content` 可能是空字符串，这通常用于流开始、结束或元数据，不代表错误。

完整回答需要按原顺序拼接：

```python
text_parts: list[str] = []

for chunk in llm.stream(messages):
    text_parts.append(chunk.content)

complete_answer = "".join(text_parts)
```

---

## 4. 同步流与异步流

同步流：

```python
for chunk in llm.stream(messages):
    ...
```

异步流：

```python
async for chunk in llm.astream(messages):
    ...
```

对应关系：

| 同步 | 异步 |
|---|---|
| `stream()` | `astream()` |
| `for` | `async for` |
| 普通函数 | `async def` |
| 直接调用 | `await` |
| `time.sleep()` 阻塞 | `asyncio.sleep()` 不阻塞 |

异步的关键不是自动使用多个 CPU，而是在等待网络 I/O 时把执行权交还给事件循环。

```text
模型请求等待下一个 chunk
    ↓ 让出执行权
事件循环执行其他 Task
    ↓
新 chunk 到达
    ↓
恢复模型流协程
```

---

## 5. Task 与协程清理

创建后台 Task：

```python
task = asyncio.create_task(
    heartbeat(stop_event),
    name="heartbeat-task",
)
```

发出停止信号后，还应该等待 Task 真正结束：

```python
stop_event.set()
result = await task
```

只调用 `set()` 或 `cancel()`，不代表后台 Task 已经完成。通常还需要 `await task`，从而：

- 确认任务已回收；
- 获得返回值；
- 观察任务异常；
- 避免遗留 Task。

---

## 6. SSE 是什么

SSE 是 Server-Sent Events，即服务端发送事件。

它基于普通 HTTP 长响应，让服务端不断向客户端推送文本事件。

```text
HTTP Client
→ GET /stream
→ HTTP 200
→ Content-Type: text/event-stream
→ 响应保持打开
← start
← token
← token
← done
```

SSE 通常是单向的：服务端持续向客户端发送数据。客户端如果要再次提交内容，一般使用新的 HTTP 请求。

---

## 7. SSE 响应头

典型响应头：

```python
headers = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
```

作用：

- `text/event-stream`：声明 SSE 响应；
- `no-cache`：避免缓存实时数据；
- `keep-alive`：维持连接；
- `X-Accel-Buffering: no`：提示部分代理不要缓冲整个响应。

---

## 8. SSE 事件格式

```text
id: req-001:5
event: token
data: {"text": "异步"}

```

最后的空行表示一个事件结束。

主要字段：

| 字段 | 作用 |
|---|---|
| `id` | 事件编号，支持恢复和去重 |
| `event` | 事件名称 |
| `data` | 业务内容，可以出现多行 |
| `retry` | 建议客户端重连等待时间 |
| `: comment` | SSE 注释，可作为心跳 |

---

## 9. 模型流如何转换为 SSE

服务端核心流程：

```python
async for chunk in llm.astream(messages):
    await response.write(
        encode_sse(
            event="token",
            data={"text": chunk.content},
        )
    )
```

完整数据流：

```text
HumanMessage
→ llm.astream()
→ AIMessageChunk
→ JSON data
→ SSE 文本编码
→ UTF-8 bytes
→ response.write()
→ HTTP ClientResponse.content
→ SSEEvent
→ 立即打印
```

---

## 10. SSE 客户端解析

客户端异步读取字节流：

```python
async for raw_line in response.content:
    line = raw_line.decode("utf-8")
```

解析规则：

```text
id: ...    → event_id
event: ... → event name
data: ...  → 收集 data 行
空行       → 当前事件完成
: ...      → 注释或心跳
```

只有 `token` 的文本需要加入完整回答：

```python
if event.event == "token":
    text_parts.append(event.data["text"])
```

`start`、`done`、`error` 和心跳都是控制信息，不应该混入正文。

---

## 11. 两层异步

第一层：单个模型回答内部异步接收很多 chunk。

```python
async for chunk in llm.astream(messages):
    ...
```

第二层：多个 SSE 请求协程异步并发。

```text
客户端1 → SSE连接1 → 模型流1
客户端2 → SSE连接2 → 模型流2
客户端3 → SSE连接3 → 模型流3
```

每个流在等待网络数据时，事件循环可以推进其他流。

因此：

```text
单次回答的多个 chunk 是异步接收
多个对话连接也是异步并发处理
```

---

## 12. ClientSession 与独立连接

多个任务可以共享一个 `ClientSession`：

```python
async with ClientSession() as session:
    task1 = create_task(session.get(...))
    task2 = create_task(session.get(...))
```

共享 `ClientSession` 表示共享连接池、超时和公共配置，不代表共享同一个 SSE 响应。

每次 `session.get()` 都有自己的：

- HTTP 请求；
- HTTP 响应；
- SSE 字节流；
- `request_id`；
- 模型调用上下文。

---

## 13. gather 的顺序

实际完成顺序可能是：

```text
任务3 → 任务2 → 任务1
```

但：

```python
results = await asyncio.gather(task1, task2, task3)
```

返回顺序仍然是：

```text
task1 的结果、task2 的结果、task3 的结果
```

`gather()` 按传入参数顺序整理结果，而不是按完成先后排列。

---

## 14. 为什么需要 Semaphore

异步服务器可以维持大量连接，但不能无限调用昂贵的上游模型。

```python
model_semaphore = asyncio.Semaphore(2)
```

请求调用模型前：

```python
await model_semaphore.acquire()
```

结束时：

```python
model_semaphore.release()
```

四个连接、两个许可：

```text
请求1 → 运行
请求2 → 运行
请求3 → 等待
请求4 → 等待
```

Semaphore 限制的是实际模型调用数量，不一定限制 SSE 连接数量。

---

## 15. 为什么 release 必须在 finally 中

以下情况都必须归还许可：

- 模型正常完成；
- 上游连接失败；
- 客户端断开；
- 任务被取消；
- 业务代码抛出异常。

正确结构：

```python
await semaphore.acquire()

try:
    await call_model()
finally:
    semaphore.release()
```

否则异常可能泄漏许可，最终让所有新请求永久等待。

今天实际验证了 `OpenAIConnectionError` 后：

```text
active_model_streams = 0
waiting_model_streams = 0
```

说明异常清理逻辑有效。

---

## 16. 客户端取消

按事件数量取消：

```python
if token_count >= 12:
    break
```

缺点：如果模型一直没有返回首块，取消条件永远不会触发。

按时间取消：

```python
async with asyncio.timeout(1.0):
    await consume_stream()
```

期限到达时，内部协程收到 `CancelledError`，外部看到 `TimeoutError`。

```text
期限到达
→ 消费协程取消
→ ClientResponse 上下文退出
→ HTTP 连接关闭
```

---

## 17. 服务端主动断线检测

如果服务端只在 `response.write()` 时发现断线，而模型迟迟没有首块，就可能无法及时清理。

改进方法是同时运行：

```text
model_task
disconnect_task
```

然后竞争等待：

```python
done, pending = await asyncio.wait(
    {model_task, disconnect_task},
    return_when=asyncio.FIRST_COMPLETED,
)
```

客户端先断开：

```text
disconnect_task 完成
→ model_task.cancel()
→ await model_task
→ 释放 Semaphore
```

模型先完成：

```text
model_task 完成
→ 发送 done
→ 取消 disconnect_task
→ 释放 Semaphore
```

---

## 18. SSE 心跳

当模型长时间没有产生 chunk 时，服务端可以发送 SSE 注释：

```text
: heartbeat 1790501084.785

```

心跳作用：

- 表示连接仍然存活；
- 减少代理因空闲而关闭长连接的概率；
- 不成为业务事件；
- 不混入最终回答。

客户端通常直接忽略冒号行，也可以单独统计。

---

## 19. Queue 与生产者—消费者

把模型读取和 HTTP 写入解耦：

```text
模型生产者
→ asyncio.Queue
→ HTTP/SSE 消费者
```

生产者：

```python
await queue.put(QueueItem(...))
```

消费者：

```python
item = await queue.get()
await response.write(...)
queue.task_done()
```

这样模型侧和网络侧拥有清晰边界，并且消费者等待 Queue 时可以定期发送心跳。

---

## 20. 背压

无限 Queue 可能在客户端或网络很慢时积累大量模型 chunk。

有限 Queue：

```python
queue = asyncio.Queue(maxsize=20)
```

当 Queue 已满：

```text
生产者 await queue.put()
→ 暂停读取更多模型数据
→ 消费者继续发送已有数据
→ Queue 腾出空间
→ 生产者恢复
```

下游处理能力反向限制上游生产速度，这就是背压。

---

## 21. SSE 断线重连

服务端事件：

```text
id: 3
event: number
retry: 500
data: {"number": 3}

```

客户端保存：

```text
last_event_id = 3
retry_milliseconds = 500
```

断线后重新请求：

```http
Last-Event-ID: 3
```

服务端从下一条开始：

```text
next_event_id = 4
```

---

## 22. 真实模型流续传的前提

`Last-Event-ID` 只告诉服务端客户端收到了哪里，并不会自动保存历史事件。

真实模型流要支持续传，服务端必须保存：

- `stream_id`；
- 每个 chunk 的事件 ID；
- 已生成文本；
- 流是否完成；
- 事件过期时间。

可以存放在：

- 当前进程内存；
- Redis；
- 数据库；
- 专门的消息流系统。

如果服务端没有事件缓存，客户端发送 `Last-Event-ID` 也无法恢复原来的模型流。

---

## 23. 模型回答与传输正确性

SSE、异步和限流都正确，不代表模型内容一定正确。

今天模型曾用 JavaScript 术语回答 Python 问题：

```text
for await...of
Promise
Symbol.asyncIterator
```

Python 语境应该是：

```text
async for
Awaitable
__aiter__()
__anext__()
```

因此要区分：

```text
传输可靠性 ≠ 回答正确性
```

可以通过更明确的 SystemMessage、RAG、结构化验证和评测减少内容偏离。

---

## 24. 今天的文件

```text
day17/python/
├── day17_stream_basic.py
├── day17_async_stream.py
├── day17_sse_server.py
├── day17_sse_client.py
├── day17_sse_concurrent_clients.py
├── day17_sse_limited_server.py
├── day17_sse_limit_test.py
├── day17_sse_cancel_client.py
├── day17_sse_disconnect_server.py
├── day17_sse_timeout_cancel.py
├── day17_sse_heartbeat_server.py
├── day17_sse_heartbeat_client.py
├── day17_sse_resume_server.py
└── day17_sse_resume_client.py
```

---

## 25. 核心系统数据流

```text
用户问题
→ HTTP GET /stream
→ SSE handler
→ Semaphore.acquire()
→ 创建 model Task
→ llm.astream(messages)
→ AIMessageChunk
→ 有限 asyncio.Queue
→ SSE token 事件
→ HTTP 长响应
→ 客户端异步解析
→ 立即显示并拼接正文
→ done / error / cancel
→ 清理 Task
→ Semaphore.release()
```

辅助可靠性路径：

```text
空闲期间 → heartbeat
客户端断开 → disconnect_task → cancel model_task
服务错误 → error 事件 + finally 清理
连接中断 → Last-Event-ID → 重连续传
```

---

## 26. 今天最重要的结论

1. `stream()` 返回同步迭代器，`astream()` 返回异步迭代器。
2. chunk 大小不固定，需要按顺序拼接。
3. SSE 用一个持续 HTTP 响应单向推送文本事件。
4. 单次模型流和多个 SSE 对话都可以异步处理。
5. 异步连接可以很多，但上游模型调用必须使用 Semaphore 限流。
6. 所有许可和后台 Task 都必须在异常、取消和断线时清理。
7. 按时间取消可以覆盖“模型没有首块”的情况。
8. 心跳维持安静连接，但不应该进入业务正文。
9. 有限 Queue 可以形成背压，防止内存无限堆积。
10. `Last-Event-ID` 需要配合服务端事件缓存才能实现真实续传。

---

## 27. 自测题

1. `invoke()`、`stream()` 和 `astream()` 分别返回什么？
2. 为什么 chunk 不一定等于一个 Token？
3. SSE 事件为什么必须以空行结束？
4. `text/event-stream` 响应与普通 JSON 响应有什么区别？
5. 为什么共享 `ClientSession` 不等于共享同一个 SSE 响应？
6. `gather()` 的结果顺序由什么决定？
7. Semaphore 限制的是连接数还是模型调用数？
8. 为什么必须在 `finally` 中释放 Semaphore？
9. 按 token 数取消有什么缺点？
10. 服务端怎样主动发现客户端断开？
11. SSE 心跳为什么不会混入回答？
12. 有限 Queue 如何产生背压？
13. `retry` 和 `Last-Event-ID` 分别有什么作用？
14. 为什么真实模型续传需要服务端保存历史事件？

