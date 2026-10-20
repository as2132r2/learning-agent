# Day25 最终代码详细解读

本文重点解释 `day25_final_feedback_rag.py` 及其前端如何连接 Day24 RAG 与 Day25 反馈系统。

## 一、最终项目文件

```text
day25/
├── python/
│   ├── day25_feedback_store.py
│   ├── day25_feedback_api.py
│   ├── day25_feedback_api_client.py
│   ├── day25_feedback_linkage.py
│   ├── day25_feedback_to_evalset.py
│   ├── day25_offline_replay.py
│   └── day25_final_feedback_rag.py
├── html/day25_final.html
├── javascript/day25_final.js
├── css/day25_final.css
├── database/day25_feedback.db
├── datasets/day25_feedback_eval_candidates.jsonl
└── reports/day25_version_comparison.json
```

---

## 二、为什么复用 Day24 而不是复制代码

最终服务导入：

```python
import day24_final_optimized_rag as day24
```

Day24 已经实现：

- 混合检索；
- RRF；
- LLM Reranker；
- 风险分级；
- 模型缓存；
- 调用与 Token 预算；
- 引用审计；
- Groundedness；
- 回答质量评估；
- 一次受控修复；
- 并发限制和取消。

Day25 的职责是增加反馈闭环。如果复制整条流水线，会形成两套相似代码，之后修复 Day24 问题时很容易漏改 Day25。

所以采用组合和代理：

```text
Day25 WebSocket处理器
→ 调用 day24.run_pipeline()
→ RecordingWebSocket观察输出事件
→ 不修改Day24算法
```

---

## 三、路径与端口

```python
HOST, PORT = "127.0.0.1", 8034
HTML = DAY25_DIR / "html" / "day25_final.html"
JS = DAY25_DIR / "javascript" / "day25_final.js"
CSS = DAY25_DIR / "css" / "day25_final.css"
BASE_CSS = PROJECT_DIR / "day24" / "css" / "day24_final.css"
```

Day25 复用 Day24 基础样式，再加载自己的反馈面板样式：

```html
<link rel="stylesheet" href="/base.css">
<link rel="stylesheet" href="/app.css">
```

---

## 四、请求数据模型

### 1. `AskPayload`

```python
class AskPayload(day24.AskPayload):
    session_id: str
    user_id: str
```

继承 Day24 参数：

- `question`
- `risk`
- `max_calls`
- `max_tokens`
- `allow_downgrade`

新增用户和会话信息，让完成的回答能够归属到正确用户。

### 2. `ClientCommand`

```python
class ClientCommand(BaseModel):
    version: Literal["1.0"]
    type: Literal["ask", "cancel"]
    request_id: str
    payload: AskPayload
```

它是 WebSocket 命令信封。

### 3. `FeedbackPayload`

```python
class FeedbackPayload(BaseModel):
    request_id: str
    user_id: str
    rating: Literal["up", "down", "neutral"]
    reason: str | None
```

这里故意没有回答、引用、Token 和模型版本，因为这些字段应由服务端记录。

---

## 五、`save_completed_answer()`

参数：

```python
request_id: str
payload: AskPayload
plan: dict[str, Any]
reranked: dict[str, Any]
done: dict[str, Any]
```

它把三个阶段事件合并成一条 `AnswerRun`：

```text
plan
└── executed_risk

reranked
├── accepted_count
└── results → SourceTrace

done
├── answer
├── wall_seconds
├── cache_hits
├── model_calls
├── total_tokens
└── stages
```

来源转换：

```python
accepted_results = reranked["results"][:accepted_count]
sources = [SourceTrace(...) for item in accepted_results]
```

只有真正通过阈值并交给回答模型的来源才保存为证据。

当前 Day24 事件只暴露总 Token，没有分别暴露输入和输出 Token。教学版暂把聚合 Token 放入一个字段以保持总量，生产版应扩展事件协议，分别保存两个数字。

数据库写入放入：

```python
await asyncio.to_thread(save_completed_answer, ...)
```

原因是 `sqlite3` 为同步接口，不应长时间阻塞事件循环。

---

## 六、`PipelineRecorder`

```python
@dataclass(slots=True)
class PipelineRecorder:
    request_id: str
    payload: AskPayload
    plan: dict[str, Any]
    reranked: dict[str, Any]
    saved: bool
```

它观察三类事件：

```python
if kind == "plan":
    保存计划
elif kind == "reranked":
    保存精排结果
elif kind == "done":
    保存完整回答
```

`saved` 防止异常情况下重复处理 `done`。

---

## 七、`RecordingWebSocket`

这是一个代理对象：

```python
class RecordingWebSocket:
    real_ws: web.WebSocketResponse
    recorder: PipelineRecorder
```

Day24 只需要 WebSocket 提供：

- `.closed`
- `await .send_json(data)`

代理提供相同接口，因此可以传给 Day24 流水线：

```python
await day24.run_pipeline(proxy_ws, state, request_id, core_payload)
```

发送顺序：

```python
async def send_json(self, data):
    await self.recorder.observe(data)
    await self.real_ws.send_json(data)
```

注意必须先 `observe`，再真实发送。

若顺序相反：

```text
真实发送done
→ 浏览器显示反馈按钮
→ 用户快速点击
→ 数据库可能尚无AnswerRun
```

当前顺序保证浏览器看到回答时，回答追踪一定已经落库。

---

## 八、`run_tracked_pipeline()`

Day25 的 `AskPayload` 比 Day24 多出 `session_id/user_id`，而 Day24 的 Pydantic 模型禁止额外字段。

因此先显式转换：

```python
core_payload = day24.AskPayload(
    question=command.payload.question,
    risk=command.payload.risk,
    max_calls=command.payload.max_calls,
    max_tokens=command.payload.max_tokens,
    allow_downgrade=command.payload.allow_downgrade,
)
```

用户字段由 `PipelineRecorder` 自己保留，不传进 Day24 核心。

---

## 九、WebSocket 处理器

每个连接拥有自己的：

```python
state = day24.ConnectionState()
```

状态包含：

- 当前活动 `request_id`；
- 当前 `asyncio.Task`；
- WebSocket 发送锁。

收到 `ask`：

```text
Pydantic验证
→ 检查问题
→ 检查当前连接是否繁忙
→ create_task(run_tracked_pipeline)
```

收到 `cancel`：

```text
request_id与活动任务一致
→ task.cancel()
→ Day24捕获CancelledError
→ 发出cancelled事件
→ finally释放Semaphore
```

连接关闭也会取消尚未结束的任务，并通过 `gather(..., return_exceptions=True)` 回收。

---

## 十、反馈接口

路由：

```text
POST /feedback
```

处理流程：

```text
request.json()
→ FeedbackPayload.model_validate()
→ 查询answer_runs
→ 回答不存在：404
→ user_id不一致：403
→ save_linked_feedback()
→ 新增返回201，更新返回200
```

权限检查：

```python
if answer["user_id"] != payload.user_id:
    raise PermissionError("不能评价其他用户的回答。")
```

教学版的 `user_id` 仍来自页面输入。生产环境必须从经过验证的访问令牌获取，不应相信请求正文中的用户身份。

---

## 十一、反馈统计接口

路由：

```text
GET /feedback/statistics
```

SQL 使用：

```sql
answer_runs
LEFT JOIN linked_feedback
GROUP BY answer_version
```

`LEFT JOIN` 的原因：即使某版本暂时没有反馈，也应保留回答版本统计。

输出包括：

- 反馈总数；
- 点赞、点踩和中立数量；
- 平均耗时；
- 平均模型调用次数。

---

## 十二、前端请求流程

提交问题时生成：

```javascript
active = `req-${crypto.randomUUID()}`;
```

WebSocket 消息：

```javascript
{
  version: "1.0",
  type: "ask",
  request_id: active,
  payload: {
    question,
    session_id,
    user_id,
    risk,
    max_calls,
    max_tokens,
    allow_downgrade
  }
}
```

收到 `done` 后：

```javascript
prepareFeedback(message.request_id);
```

它把这次回答的 `request_id` 和当时的 `user_id` 固定下来。即使用户随后修改输入框，也不会把评价错误关联给另一个用户。

---

## 十三、前端反馈流程

用户点击反馈按钮后调用：

```javascript
fetch("/feedback", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    request_id,
    user_id,
    rating,
    reason
  })
})
```

按钮在请求期间被禁用，避免快速重复提交。保存完成后显示：

- `评价已保存`：首次 INSERT；
- `评价已更新`：同一用户修改评价。

---

## 十四、HTTP 与 WebSocket 的分工

WebSocket 适合：

- 长时间运行的 RAG；
- 多阶段进度事件；
- 取消任务；
- 实时显示检索和审计过程。

普通 HTTP 适合：

- 一次短小的反馈提交；
- 统计查询；
- 文件上传。

最终项目不是所有功能都强行使用同一种协议。

---

## 十五、完整时序图

```text
浏览器                 Day25服务            Day24流水线            SQLite
  │ ask(request_id)        │                      │                    │
  ├───────────────────────>│ run_pipeline()       │                    │
  │                        ├─────────────────────>│                    │
  │ plan/retrieval/events  │<─────────────────────┤                    │
  │<───────────────────────┤                      │                    │
  │                        │<──── done ───────────┤                    │
  │                        │ RecordingWebSocket                        │
  │                        ├──保存回答、来源、指标────────────────────>│
  │<──────── done ─────────┤                      │                    │
  │ 显示评价按钮            │                      │                    │
  │ POST /feedback         │                      │                    │
  ├───────────────────────>│──保存linked_feedback────────────────────>│
  │<──── created/updated ──┤                      │                    │
```

---

## 十六、运行与验证

启动：

```bash
source .venv/bin/activate
python day25/python/day25_final_feedback_rag.py
```

浏览器：

```text
http://127.0.0.1:8034
```

统计：

```text
http://127.0.0.1:8034/feedback/statistics
```

生成候选评测集：

```bash
python day25/python/day25_feedback_to_evalset.py
```

离线回放：

```bash
python day25/python/day25_offline_replay.py
```

---

## 十七、最终理解

Day25 的核心不是三个评价按钮，而是把线上事实变成可复现的改进依据：

```text
回答事实由服务端记录
→ 用户只提供主观反馈
→ request_id连接两者
→ 弱标签经过复核
→ 相同证据离线回放
→ 新旧版本结构化比较
→ 达标后才进入上线决策
```

这才是一条完整、可追踪、可审计的 RAG 持续改进闭环。
