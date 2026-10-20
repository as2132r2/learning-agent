# Day28 最终代码详细解读

本文解释 `day28_final_observability_dashboard.py` 如何复用第 5 步的 Trace 数据，并通过 aiohttp 和浏览器形成可观测 RAG 控制台。

## 一、最终文件结构

```text
day28/
├── python/
│   ├── day28_structured_logging.py
│   ├── day28_trace_spans.py
│   ├── day28_async_context.py
│   ├── day28_http_tracing_server.py
│   ├── day28_http_tracing_client.py
│   ├── day28_rag_stage_tracing.py
│   └── day28_final_observability_dashboard.py
├── html/day28_final.html
├── css/day28_final.css
├── javascript/day28_final.js
├── logs/*.jsonl
├── database/day28_observability.db
└── notes/
```

最终控制台不复制 Trace 实现，而是导入第 5 步的能力：

```python
from day28_rag_stage_tracing import (
    create_schema,
    load_trace,
    open_database,
    run_rag_pipeline,
    stage_diagnostics,
)
```

组合关系：

```text
最终Web控制台
└── RAG阶段追踪与SQLite
    └── Span和ContextVar
        └── 结构化JSON日志
```

---

## 二、服务常量和静态文件

```python
HOST = "127.0.0.1"
PORT = 8042
```

只监听本机，适合学习。静态文件通过当前 Python 文件计算绝对路径：

```python
DAY28_DIR = Path(__file__).resolve().parents[1]
```

因此从项目根目录运行时仍能找到 HTML、CSS 和 JavaScript。

`VALID_TRACE_ID` 限制详情接口参数：

```python
r"^trace-[a-f0-9]{16}$"
```

数据库查询虽然已经使用 SQL 参数绑定，但入口继续校验 ID，可以尽早拒绝无效请求。

---

## 三、为什么需要 Semaphore

```python
pipeline_semaphore = asyncio.Semaphore(2)
```

浏览器可能连续点击，多个用户也可能同时访问。控制台最多允许两个流水线同时执行。

它保护：

- 本地线程资源；
- SQLite 写事务；
- 未来替换成真实模型后的 API 配额；
- 整体响应稳定性。

Semaphore 控制并发数量，不负责让同步代码异步化。

---

## 四、`read_overview()`：聚合总览指标

这个同步函数执行三类 SQL。

### 1. Trace 总体统计

```sql
COUNT(*)
SUM(status = 'success')
SUM(status = 'failed')
AVG(duration_ms)
```

Python 继续计算：

```text
success_rate = success / total × 100%
```

### 2. 平均最慢阶段

```sql
SELECT name, AVG(duration_ms), COUNT(*)
FROM spans
WHERE parent_span_id IS NOT NULL
GROUP BY name
ORDER BY average_ms DESC
LIMIT 1
```

排除根 Span，因为根 Span 天然包含整条流水线，通常总是最慢；我们真正关心的是内部业务阶段。

### 3. 错误分组

失败 Span 按阶段名和错误类型聚合，可以回答：

```text
哪个阶段最常失败？
主要是哪种错误？
```

返回值是普通字典，可以直接 JSON 序列化。

---

## 五、`read_recent_traces()`：Trace 列表

该函数只取页面列表需要的字段：

```text
trace_id
request_id
question
status
started_at
duration_ms
error_type
error_message
```

按 `started_at DESC` 排序并限制 30 条，避免页面首次加载读取整个数据库。

生产系统应增加游标分页，而不是永远固定最近 30 条。

---

## 六、`read_trace_detail()`：单条调用链详情

```python
trace, spans = load_trace(trace_id)
```

再组合：

- Trace 概览；
- 全部 Span；
- 文本调用树；
- 最慢阶段和失败阶段诊断。

虽然当前前端主要使用 `spans` 绘制耗时条，但 `tree_lines` 也被 API 返回，便于命令行或其他客户端直接显示父子树。

---

## 七、错误中间件

```python
@web.middleware
async def error_middleware(request, handler):
```

处理两类异常：

```text
web.HTTPException → 保留400、404等HTTP状态
其他Exception    → 服务端记录类型，客户端返回通用500
```

客户端不应看到完整堆栈、磁盘路径或数据库内部错误。

---

## 八、静态文件和健康检查

`static_handler()` 只能从固定映射返回：

```text
/        → HTML
/app.css → CSS
/app.js  → JavaScript
```

客户端不能通过 URL 参数读取任意服务器文件。

`/health` 只报告服务是否运行以及数据库位置和并发配置，不执行 RAG 流水线。

---

## 九、只读 API

### `GET /api/overview`

通过 `asyncio.to_thread(read_overview)` 读取 SQLite 聚合指标。

### `GET /api/traces`

返回最近 Trace，用于页面左栏。

### `GET /api/traces/{trace_id}`

执行流程：

```text
校验格式
→ 在线程中查询数据库
→ 不存在则404
→ 返回Trace、Spans、调用树和诊断
```

只读数据库同样放入线程，避免同步 SQLite 操作阻塞 aiohttp 事件循环。

---

## 十、`POST /api/ask`：创建新 Trace

### 1. 请求验证

请求正文：

```json
{
  "question": "什么是 RAG？",
  "simulate_error": false
}
```

规则：

- 必须是 JSON；
- question 必须是非空字符串；
- simulate_error 必须是布尔值。

### 2. 并发许可

```python
async with pipeline_semaphore:
```

超过两个请求时，后续请求挂起等待许可，但不会阻塞事件循环。

### 3. 线程执行

```python
result = await asyncio.to_thread(
    run_rag_pipeline,
    question,
    simulate_reranker_error=simulate_error,
)
```

为什么不能直接调用：

- 流水线是同步函数；
- 包含 SQLite；
- 教学模拟使用 `time.sleep()`；
- 直接调用会暂停整个 HTTP 事件循环。

### 4. HTTP 状态

```text
流水线成功 → 200
流水线失败 → 502
```

502 表示当前服务因为上游阶段失败而无法完成请求。即使返回 502，失败 Trace 也已经保存，便于排查。

---

## 十一、路由表

```text
GET  /                       页面
GET  /app.css                样式
GET  /app.js                 前端逻辑
GET  /health                 健康检查
GET  /api/overview           总体指标
GET  /api/traces             最近Trace
GET  /api/traces/{trace_id}  Trace详情
POST /api/ask                运行RAG并产生Trace
```

有副作用的 RAG 执行使用 POST，而不是 GET。

---

## 十二、前端状态与安全渲染

```javascript
const state = { selectedTraceId: null };
```

前端只保存当前选中的 Trace ID，其余事实每次从服务端读取。

`escapeHtml()` 转义来自数据库和 API 的文本，防止问题、错误消息或属性被浏览器当成 HTML 执行。

这在展示日志和用户输入时尤其重要。

---

## 十三、页面初始化和刷新

初始化：

```text
refreshAll(true)
├→ GET /api/overview
└→ GET /api/traces
→ 渲染指标和列表
→ 自动选择最新Trace
→ GET /api/traces/{id}
```

两个总览请求通过 `Promise.all()` 并发发出，减少页面等待时间。

页面每 5 秒自动刷新：

```javascript
setInterval(..., 5000)
```

刷新只是读取状态，不会自动产生新的 RAG 请求。

---

## 十四、提交成功和失败请求

表单将问题和错误模拟开关发送到 `/api/ask`。

这里没有使用统一的 `fetchJson()`，因为 502 仍然包含有价值的 JSON：

```json
{
  "trace_id": "trace-...",
  "status": "failed",
  "error_type": "TimeoutError"
}
```

前端即使看到 502，仍读取正文、保存 trace_id，并刷新到失败详情。

这是可观测性控制台与普通业务页面的不同：失败本身就是需要展示的数据。

---

## 十五、Span 耗时条

宽度计算：

```javascript
span.duration_ms / trace.duration_ms * 100
```

根 Span 接近 100%，内部阶段按相对总耗时显示。

这不是严格的时间轴起点图，只是阶段耗时占比条。真正瀑布图还应考虑：

- 每个 Span 相对 Trace 的开始偏移；
- 并行 Span 的重叠；
- 跨服务时钟偏差。

---

## 十六、完整端到端数据流

```text
浏览器输入问题
        ↓ POST /api/ask
aiohttp ask_handler
        ↓ Semaphore
asyncio.to_thread(run_rag_pipeline)
        ↓
创建request_id、trace_id和Trace记录
        ↓
PersistentSpan(rag_pipeline)
├→ hybrid_retrieval
├→ llm_reranker
├→ structured_answer_generation
└→ answer_quality_evaluation
        ↓
每个Span写入SQLite
        ↓
Trace更新success或failed
        ↓
HTTP返回trace_id
        ↓
浏览器并发读取overview和traces
        ↓
GET /api/traces/{trace_id}
        ↓
Trace概览 + Span列表 + 诊断
        ↓
指标卡、Trace列表、耗时条和错误详情
```

---

## 十七、关键数据类型

| 变量 | 类型 | 作用 |
|---|---|---|
| `pipeline_semaphore` | `asyncio.Semaphore` | 限制流水线并发 |
| `trace_id` | `str` | 调用链唯一编号 |
| `trace` | `dict[str, Any]` | 请求级状态 |
| `spans` | `list[dict[str, Any]]` | 阶段级记录 |
| `diagnostics` | `dict[str, Any]` | 最慢和失败阶段 |
| `simulate_error` | `bool` | 教学故障注入开关 |
| `selectedTraceId` | JavaScript `string/null` | 当前页面选择 |

---

## 十八、当前教学版本的边界

1. RAG 阶段是确定性模拟，没有调用真实模型。
2. 使用 SQLite，没有专门的 Trace 后端。
3. 使用自制 Span，而不是 OpenTelemetry SDK。
4. 没有采样，所有 Trace 都保存。
5. 没有数据保留期限和自动清理。
6. 没有用户认证，不能直接暴露公网。
7. 耗时条不是完整时间轴瀑布图。
8. 多进程时 Semaphore 不能跨进程统一限流。

这些是有意设置的教学边界。当前目标是理解可观测性的基本数据模型和数据流。

---

## 十九、运行与验证

```bash
python day28/python/day28_final_observability_dashboard.py
```

打开：

```text
http://127.0.0.1:8042
```

建议依次验证：

1. 正常提交“什么是 RAG？”；
2. 查看 5 个成功 Span；
3. 勾选 Reranker 超时；
4. 查看 502 返回但 Trace 仍保存；
5. 确认失败停在 Reranker；
6. 观察总览成功率和失败数变化；
7. 刷新页面后确认历史 Trace 仍在。

最终应记住：

```text
可观测性不是多打印几行日志，
而是让一次请求从入口到每个内部阶段都能被关联、计时、持久化和解释。
```
