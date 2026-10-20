# Day17 最终代码详细解读

## 1. 最终代码由哪些文件组成

Day17 没有把所有生产特性强行塞进一个超大文件，而是保留几份职责清晰的完整程序：

| 文件 | 核心职责 |
|---|---|
| `day17_sse_heartbeat_server.py` | 模型流、SSE、Semaphore、Queue、心跳和背压 |
| `day17_sse_heartbeat_client.py` | 解析业务事件和注释心跳，拼接回答 |
| `day17_sse_disconnect_server.py` | 主动检测客户端断线并取消模型 Task |
| `day17_sse_timeout_cancel.py` | 使用 `asyncio.timeout()` 定时取消客户端 |
| `day17_sse_resume_server.py` | 使用事件 ID 提供可恢复的数据 |
| `day17_sse_resume_client.py` | 使用 `retry` 和 `Last-Event-ID` 自动重连 |

其中最能代表今天主线的是：

```text
day17_sse_heartbeat_server.py
day17_sse_heartbeat_client.py
```

它们构成一个完整的实时模型问答链路。

---

## 2. 核心架构

```text
SSE Client
    ↓ GET /stream?question=...
aiohttp stream_handler
    ↓ Semaphore.acquire()
模型生产者 Task
    ↓ llm.astream(messages)
AIMessageChunk
    ↓ QueueItem
asyncio.Queue(maxsize=20)
    ↓ Queue 消费者
SSE token / heartbeat / done / error
    ↓ HTTP StreamResponse
客户端逐行解析
    ↓
立即显示 + 拼接完整回答
```

这条链路中有三个不同层次的数据流：

1. 模型消息流：`AIMessageChunk`；
2. 服务端内部流：`QueueItem`；
3. 网络协议流：SSE UTF-8 字节。

---

## 3. 服务端常量和对象

核心文件：`day17_sse_heartbeat_server.py`。

```python
HOST = "127.0.0.1"
PORT = 8008
MAX_MODEL_CONCURRENCY = 2
HEARTBEAT_INTERVAL_SECONDS = 0.5
QUEUE_MAX_SIZE = 20
```

类型和作用：

| 变量 | 类型 | 作用 |
|---|---|---|
| `HOST` | `str` | 服务监听地址 |
| `PORT` | `int` | HTTP 端口 |
| `MAX_MODEL_CONCURRENCY` | `int` | 同时运行的模型流上限 |
| `HEARTBEAT_INTERVAL_SECONDS` | `float` | 没有模型数据时的心跳间隔 |
| `QUEUE_MAX_SIZE` | `int` | 单条请求内部最多缓存多少条消息 |

模型对象：

```python
llm = ChatOpenAI(...)
```

类型：

```text
langchain_openai.ChatOpenAI
```

职责：

- 保存模型名称、API Key 和 Base URL；
- 把 LangChain 消息转换为模型 API 请求；
- 通过 `astream()` 返回异步消息块。

并发控制对象：

```python
model_semaphore = asyncio.Semaphore(2)
```

类型：

```text
asyncio.Semaphore
```

它内部初始有两个许可。只有获得许可的请求才可以开始调用模型。

---

## 4. QueueItem 数据类

```python
@dataclass
class QueueItem:
    kind: Literal["token", "done", "error"]
    data: dict[str, Any]
```

字段：

| 字段 | 类型 | 作用 |
|---|---|---|
| `kind` | `Literal[...]` | 内部消息种类 |
| `data` | `dict[str, Any]` | 该消息携带的数据 |

示例 token：

```python
QueueItem(
    kind="token",
    data={
        "request_id": "req-001",
        "index": 1,
        "text": "异步",
    },
)
```

它不是 LangChain 消息，也不是 SSE 文本，而是服务端两个协程之间传递数据的内部对象。

```text
AIMessageChunk
→ QueueItem
→ SSE bytes
```

---

## 5. encode_sse()

函数定义：

```python
def encode_sse(
    *,
    event: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> bytes:
```

参数：

| 参数 | 类型 | 作用 |
|---|---|---|
| `event` | `str` | SSE 事件名称 |
| `data` | `dict[str, Any]` | 要编码的 JSON 数据 |
| `event_id` | `str | None` | 可选事件编号 |

开头的 `*` 表示后面的参数必须使用关键字传递：

```python
encode_sse(event="token", data={...})
```

不能写成难以辨认的：

```python
encode_sse("token", {...})
```

返回值：

```text
bytes
```

因为 `response.write()` 写入的是网络字节，而不是 Python 字符串。

内部转换：

```text
dict
→ json.dumps()
→ str
→ encode("utf-8")
→ bytes
```

结果格式：

```text
id: req-001:1
event: token
data: {"text": "异步"}

```

---

## 6. encode_heartbeat()

```python
def encode_heartbeat() -> bytes:
    timestamp = round(time.time(), 3)
    return f": heartbeat {timestamp}\n\n".encode("utf-8")
```

返回示例：

```text
: heartbeat 1790501084.785

```

冒号开头表示 SSE 注释。它会产生网络流量以维持连接，但不会成为业务事件。

---

## 7. produce_chunks() 模型生产者

定义：

```python
async def produce_chunks(
    *,
    question: str,
    request_id: str,
    queue: asyncio.Queue[QueueItem],
) -> None:
```

参数：

| 参数 | 类型 | 作用 |
|---|---|---|
| `question` | `str` | 用户问题 |
| `request_id` | `str` | 当前请求编号 |
| `queue` | `asyncio.Queue[QueueItem]` | 向消费者传递内部消息 |

返回值是 `None`。它不直接返回完整回答，也不直接写 HTTP。

模型调用：

```python
async for chunk in llm.astream(messages):
```

此处 `chunk` 的运行时类型通常是：

```text
AIMessageChunk
```

正文提取：

```python
if isinstance(chunk.content, str):
    text = chunk.content
else:
    text = str(chunk.content)
```

过滤空块：

```python
if not text:
    continue
```

放入 Queue：

```python
await queue.put(QueueItem(...))
```

这里的 `await` 很关键。如果 Queue 已满，生产者会暂停，不会继续无限读取和缓存模型 chunk。

正常结束放入：

```python
QueueItem(kind="done", data={...})
```

模型异常放入：

```python
QueueItem(kind="error", data={...})
```

因此异常被转换为数据，通过统一 Queue 交给网络消费者处理。

---

## 8. health_handler()

```python
async def health_handler(
    request: web.Request,
) -> web.Response:
```

虽然当前函数不读取 `request`，aiohttp 路由处理函数仍必须接收它。

返回值：

```text
web.Response
```

返回普通 JSON，而不是 SSE：

```python
return web.json_response({...})
```

它用于监控配置和服务是否存活。

---

## 9. stream_handler() 是整个服务端入口

定义：

```python
async def stream_handler(
    request: web.Request,
) -> web.StreamResponse:
```

输入是 aiohttp 解析后的 HTTP 请求，输出是可持续写入的流响应。

### 9.1 读取问题

```python
question = request.query.get("question", "").strip()
```

数据变化：

```text
URL 查询参数
→ str
→ strip()
→ question
```

### 9.2 创建请求编号

```python
request_id = "req-" + uuid.uuid4().hex[:12]
```

不同请求使用不同编号，方便日志、事件和错误关联。

### 9.3 创建 StreamResponse

```python
response = web.StreamResponse(
    status=200,
    headers={...},
)
await response.prepare(request)
```

`prepare()` 发送 HTTP 状态和响应头。调用后不能再改成普通 JSON 响应，只能继续写流或关闭连接。

### 9.4 获取 Semaphore

```python
await model_semaphore.acquire()
acquired = True
```

如果许可用完，当前协程异步挂起，但不会阻塞整个服务端。

`acquired` 用来记录当前请求是否真的持有许可，避免错误地重复释放。

### 9.5 创建有限 Queue

```python
queue: asyncio.Queue[QueueItem] = asyncio.Queue(
    maxsize=QUEUE_MAX_SIZE
)
```

每一条 SSE 请求都有自己的 Queue，因此不同用户的 chunk 不会混在一起。

### 9.6 创建生产者 Task

```python
producer_task = asyncio.create_task(
    produce_chunks(...),
    name=f"producer-{request_id}",
)
```

此时同一个请求内有两个并发角色：

```text
producer_task：读取模型
stream_handler：发送 HTTP
```

### 9.7 消费 Queue

```python
item = await asyncio.wait_for(
    queue.get(),
    timeout=HEARTBEAT_INTERVAL_SECONDS,
)
```

两种结果：

```text
期限内有 QueueItem
→ 处理 token/done/error

期限内没有 QueueItem
→ TimeoutError
→ 发送 heartbeat
```

### 9.8 处理 token

```python
await response.write(
    encode_sse(event="token", data=item.data)
)
```

如果客户端网络慢，`response.write()` 可能等待，Queue 会逐渐填充。Queue 满后生产者也等待，这就是背压链路。

### 9.9 处理 done

发送最后的统计事件，然后跳出循环：

```python
break
```

### 9.10 处理 error

模型异常不会让客户端只看到连接神秘关闭，而是收到结构化错误：

```text
event: error
data: {"error_type": "...", "message": "..."}
```

---

## 10. 服务端 finally 清理

```python
finally:
    if producer_task is not None and not producer_task.done():
        producer_task.cancel()
        await producer_task

    if acquired:
        model_semaphore.release()

    await response.write_eof()
```

清理顺序：

1. 停止未结束的模型生产者；
2. 等待 Task 真正退出；
3. 归还模型许可；
4. 结束 HTTP 响应。

这段代码必须覆盖：

- 正常完成；
- 客户端断开；
- 模型连接异常；
- 处理协程取消。

---

## 11. 客户端 SSEFrame

核心文件：`day17_sse_heartbeat_client.py`。

```python
@dataclass
class SSEFrame:
    kind: Literal["event", "comment"]
    event: str | None = None
    data: dict[str, Any] | None = None
    event_id: str | None = None
    comment: str | None = None
```

为什么使用一个统一对象？

因为 HTTP 流中会同时出现：

```text
业务事件
SSE 注释心跳
```

`kind` 先区分大类，再读取对应字段。

---

## 12. iter_sse_frames()

定义：

```python
async def iter_sse_frames(
    response: Any,
) -> AsyncIterator[SSEFrame]:
```

这是异步生成器：

- 使用 `async def`；
- 内部使用 `yield`；
- 调用后得到 `AsyncIterator[SSEFrame]`；
- 使用 `async for` 消费。

逐行读取：

```python
async for raw_line in response.content:
```

`raw_line` 是 `bytes`，需要：

```python
line = raw_line.decode("utf-8")
```

注释路线：

```python
if line.startswith(":"):
    yield SSEFrame(kind="comment", ...)
```

业务事件路线：

```text
读取 event/id/data
→ 遇到空行
→ json.loads(data)
→ yield SSEFrame(kind="event")
```

---

## 13. 客户端 main() 数据状态

```python
heartbeat_count = 0
token_event_count = 0
business_event_count = 0
text_parts: list[str] = []
```

| 变量 | 类型 | 作用 |
|---|---|---|
| `heartbeat_count` | `int` | 收到的注释心跳数量 |
| `token_event_count` | `int` | 收到的正文事件数量 |
| `business_event_count` | `int` | 所有业务事件数量 |
| `text_parts` | `list[str]` | 等待拼接的正文片段 |

心跳处理：

```python
if frame.kind == "comment":
    heartbeat_count += 1
    continue
```

`continue` 确保心跳不会进入下面的业务事件处理。

正文处理：

```python
elif event_name == "token":
    text_parts.append(text)
```

最后：

```python
complete_answer = "".join(text_parts)
```

---

## 14. 主动断线检测代码

核心文件：`day17_sse_disconnect_server.py`。

模型任务：

```python
model_task = asyncio.create_task(
    produce_model_stream(...)
)
```

断线监测任务：

```python
disconnect_task = asyncio.create_task(
    wait_for_disconnect(request)
)
```

竞争：

```python
done, pending = await asyncio.wait(
    {model_task, disconnect_task},
    return_when=asyncio.FIRST_COMPLETED,
)
```

返回类型：

```text
done: set[Task]
pending: set[Task]
```

如果断线先发生：

```python
model_task.cancel()
await model_task
```

如果模型先完成：

```python
result = model_task.result()
```

`result()` 会取得 Task 返回值；如果 Task 内部异常，也会在这里重新抛出。

---

## 15. asyncio.timeout() 定时取消

核心文件：`day17_sse_timeout_cancel.py`。

```python
async with asyncio.timeout(1.0):
    await consume_until_cancelled(session, state)
```

内部协程收到：

```text
asyncio.CancelledError
```

上下文外捕获：

```text
TimeoutError
```

不要在内部吞掉 `CancelledError`：

```python
except asyncio.CancelledError:
    print("正在清理")
    raise
```

重新抛出后，`async with session.get(...)` 才能沿正确取消路径退出。

---

## 16. 自动重连客户端状态

核心文件：`day17_sse_resume_client.py`。

```python
last_event_id: str | None = None
retry_milliseconds = 1000
received_numbers: list[int] = []
completed = False
connection_attempts = 0
```

重连请求头：

```python
headers = {"Accept": "text/event-stream"}

if last_event_id is not None:
    headers["Last-Event-ID"] = last_event_id
```

收到事件后更新：

```python
last_event_id = event.event_id
retry_milliseconds = event.retry_milliseconds
```

连接结束后：

```python
await asyncio.sleep(retry_milliseconds / 1000)
```

然后重新发送请求。

---

## 17. 为什么真实模型续传更复杂

编号演示服务能根据数字重新生成事件：

```text
Last-Event-ID: 3
→ 从数字 4 继续
```

真实模型生成具有上下文和随机性，不能简单重新请求并假设会产生完全相同的 chunk。

生产级续传需要：

```text
stream_id
→ 服务端缓存的事件列表
→ 每个事件的 sequence/id
→ 当前生成状态
→ 完成或失败状态
→ TTL 清理时间
```

客户端重连：

```text
stream_id + Last-Event-ID
→ 重放缺失事件
→ 再订阅新事件
```

---

## 18. 完整函数调用关系

### 服务端

```text
main()
→ create_app()
→ web.run_app()
→ 收到 GET /stream
→ logging_middleware()
→ stream_handler(request)
→ Semaphore.acquire()
→ asyncio.Queue(maxsize=20)
→ create_task(produce_chunks())
→ llm.astream(messages)
→ queue.put(QueueItem)
→ queue.get()
→ encode_sse()
→ response.write()
→ finally 清理 Task、许可和响应
```

### 客户端

```text
asyncio.run(main())
→ ClientSession()
→ session.get(/stream)
→ ClientResponse.content
→ iter_sse_frames(response)
→ SSEFrame
→ comment：统计心跳
→ token：打印并保存正文
→ done：显示统计
→ join(text_parts)
```

---

## 19. 变量从请求到回答的变化

```text
question: str
    ↓ URL 编码
HTTP query parameter
    ↓ request.query.get()
question: str
    ↓ HumanMessage(content=question)
messages: list[BaseMessage]
    ↓ llm.astream()
chunk: AIMessageChunk
    ↓ chunk.content
text: str
    ↓ QueueItem.data
dict[str, Any]
    ↓ json.dumps()
JSON str
    ↓ encode("utf-8")
bytes
    ↓ HTTP response.content
bytes
    ↓ decode + json.loads
dict[str, Any]
    ↓ data["text"]
text: str
    ↓ list.append + join
complete_answer: str
```

---

## 20. 今天代码中的安全边界

必须始终满足：

1. 用户问题不能为空；
2. 模型并发不能无限增长；
3. Queue 不能无限增长；
4. 客户端断开要停止无意义工作；
5. Task 取消后必须等待回收；
6. Semaphore 必须在 `finally` 释放；
7. 模型异常要成为明确的 `error` 事件；
8. 心跳不能混入模型正文；
9. 重连次数必须有上限；
10. 真实续传数据必须设置过期清理机制。

---

## 21. 最终记忆口诀

```text
模型边生成，astream 边读取；
服务边发送，SSE 边解析；
连接可很多，模型要限流；
下游若变慢，Queue 做背压；
连接若安静，心跳保活性；
用户若离开，Task 要取消；
无论成与败，finally 做清理；
断线想恢复，事件必须能重放。
```

