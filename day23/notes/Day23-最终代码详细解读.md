# Day23 最终代码详细解读

解读对象：`day23/python/day23_final_audited_rag.py`及其调用的Day23模块。

## 1. 最终程序不是单文件孤岛

最终服务端负责编排，具体能力来自前面五步：

```text
day22_final_hybrid_rag
├── 上传与双索引同步
├── RRF候选检索
└── JSON转换函数

day22_llm_reranker
└── 候选精排

day23_structured_citations
├── StructuredRAGAnswer
└── 证据筛选

day23_citation_audit
└── 结构化回答生成

day23_answer_repair
├── assess_answer
└── repair_answer
```

最终文件重点是控制调用顺序、并发、事件和质量门控。

---

## 2. 路径与服务配置

```python
PROJECT_DIR = Path(__file__).resolve().parents[2]
HOST, PORT = "127.0.0.1", 8030
```

`Path`用于定位Day22模块和Day23静态文件。`127.0.0.1`只允许本机访问，不直接暴露到局域网。

```python
HTML = DAY23_DIR / "html" / "day23_final.html"
JS = DAY23_DIR / "javascript" / "day23_final.js"
CSS = DAY23_DIR / "css" / "day23_final.css"
```

三者都是`Path`对象。

---

## 3. 整条模型流水线的Semaphore

```python
model_pipeline_semaphore = asyncio.Semaphore(2)
```

一条请求获得许可后，覆盖：

```text
Reranker
→ 初稿生成
→ Groundedness
→ 完整性评估
→ 可能的修复
→ 修复稿重新评估
```

这样限制的是“高成本流水线数量”，而不是某一个短暂API调用。

---

## 4. 浏览器命令模型

```python
class AskPayload(BaseModel):
    question: str | None

class ClientCommand(BaseModel):
    version: Literal["1.0"]
    type: Literal["ask", "cancel"]
    request_id: str
    payload: AskPayload
```

`model_validate_json(raw.data)`同时完成JSON解析与字段校验。

`request_id`贯穿所有事件，让浏览器知道事件属于哪次提问。

---

## 5. ConnectionState

```python
@dataclass
class ConnectionState:
    active_request_id: str | None
    active_task: asyncio.Task[None] | None
    send_lock: asyncio.Lock
```

```text
active_request_id：当前业务请求ID
active_task：当前后台协程Task
send_lock：保证同一WebSocket发送顺序
```

每条浏览器连接拥有自己的状态对象。

---

## 6. send函数

```python
async def send(..., kind, request_id, payload):
```

统一发送：

```json
{
  "type": "version_assessed",
  "request_id": "req-...",
  "payload": {}
}
```

`kind`决定前端分支，`payload`只承载该事件业务数据。

---

## 7. assessment_to_dict

输入：

```python
assessment: AnswerAssessment
version_number: int
label: str
```

`AnswerAssessment`内部包含Pydantic对象和dataclass，不能直接依赖WebSocket自动序列化。

函数显式转换成JSON兼容类型：

```text
str
int
float
bool
None
list
dict
```

输出分为：

```text
回答和claims
citation
groundedness
quality
final
```

如果前一层引用结构失败，后续报告为`None`，前端用可选链显示`—`。

---

## 8. run_audited_rag：第一阶段

```python
await send(..., "retrieving", ...)
```

通知前端开始本地检索。

```python
async with index_lock:
    rrf_results, diagnostics = await asyncio.to_thread(
        retrieve_rrf_candidates,
        question,
    )
```

参数和返回值：

```text
question: str
→ tuple[list[RRFResult], dict[str, Any]]
```

`index_lock`防止上传同步索引时同时检索；`to_thread`防止同步Embedding和SQLite操作阻塞事件循环。

---

## 9. RRF结果事件

```python
rrf_to_dict(item, rank)
```

转换后发送：

```text
vector_rank
vector_distance
keyword_rank
bm25_score
rrf_score
```

浏览器可以观察候选在两条召回路线中的来源。

---

## 10. 获得模型流水线许可

```python
await model_pipeline_semaphore.acquire()
acquired = True
```

`acquired`记录是否真的获得许可。`finally`只在它为True时调用`release()`，避免错误增加Semaphore计数。

---

## 11. Reranker与证据守门

```python
reranked_results, _ = await asyncio.to_thread(
    rerank_candidates,
    question,
    rrf_results,
)
```

返回：

```text
list[RerankedResult]
RerankResponse
```

随后：

```python
evidence = select_evidence(reranked_results)
```

只选择：

```text
relevance_score >= 95
```

无合格资料则发送`retrieval_rejected`，不会生成回答。

---

## 12. 结构化初稿

```python
initial_answer = await asyncio.to_thread(
    generate_structured_answer,
    question,
    evidence,
)
```

输入：

```text
question: str
evidence: list[RerankedResult]
```

输出：

```text
StructuredRAGAnswer
├── can_answer: bool
├── answer: str
└── claims: list[CitedClaim]
```

---

## 13. assess_answer内部调用

```python
current_assessment = await asyncio.to_thread(
    assess_answer,
    question,
    initial_answer,
    evidence,
)
```

内部顺序：

```text
build_readable_answer
→ audit_citations（确定性，不调用模型）
→ evaluate_claim_support（模型调用）
→ build_groundedness_report
→ evaluate_answer_quality（模型调用）
→ build_final_quality_report
```

如果引用审计失败，立即返回，避免把结构非法的回答送入后续评估。

---

## 14. AnswerAssessment

```python
@dataclass
class AnswerAssessment:
    answer: StructuredRAGAnswer
    rendered_answer: str
    citation_audit: CitationAudit
    groundedness_report: GroundednessReport | None
    quality_response: AnswerQualityResponse | None
    final_report: FinalQualityReport | None
```

`passed`要求：

```text
引用审计通过
并且存在最终报告
并且最终报告通过
```

任何一层缺失或失败都不能交付。

---

## 15. 一次受控修复

```python
if not current_assessment.passed:
```

只执行一次：

```python
repaired_answer = await asyncio.to_thread(
    repair_answer,
    question,
    evidence,
    current_assessment,
)
```

修复输入包含当前完整`AnswerAssessment`，`assessment_feedback()`提取可操作问题。

修复后再次调用`assess_answer`，不是复用旧报告。

---

## 16. 两道最终门

检索门：

```text
没有资料达到95分
→ retrieval_rejected
```

答案质量门：

```text
初稿失败
→ 修复一次
→ 仍失败
→ quality_rejected
```

两者不同：前者说明没有足够资料；后者说明有资料，但模型没能产生合格答案。

---

## 17. done事件

只有`current_assessment.passed`才发送：

```python
{
    "answer": current_assessment.rendered_answer,
    "repair_attempts": repair_attempt,
    "sources": [...],
}
```

这保证浏览器“助手最终回答”区域只出现通过审计的版本。

---

## 18. 取消和finally

```python
except asyncio.CancelledError:
    await send(..., "cancelled", ...)
```

```python
finally:
    if acquired:
        model_pipeline_semaphore.release()
```

无论正常、拒绝、异常或取消，都会释放许可并清空连接状态。

限制：`asyncio.to_thread()`中的同步HTTP请求不能保证随着外层Task立即停止。生产版本应使用异步模型客户端、超时和可传播取消的调用链。

---

## 19. WebSocket处理器

```python
async for raw in ws:
```

`ask`创建后台Task：

```python
asyncio.create_task(run_audited_rag(...))
```

因此处理器仍可以接收`cancel`。

同一连接已有任务时发送`busy`，避免两个审计流水线共享同一页面状态。

---

## 20. HTTP路由

```text
GET  /         HTML
GET  /app.js   JavaScript
GET  /app.css  CSS
POST /upload   复用Day22安全上传和双索引同步
GET  /ws       WebSocket
```

GET资源还有自动HEAD路由，因此总数显示9。

---

## 21. JavaScript状态

```javascript
let socket = null;
let activeRequestId = null;
let reconnectAttempt = 0;
```

前端不保存模型对象，只保存连接和当前请求状态。

`version_assessed`到达时，`renderVersion()`创建独立版本卡片，不覆盖之前版本。

---

## 22. 质量指标展示

前端显示五张指标卡：

```text
引用覆盖率
证据支持率
问题相关性
回答完整性
综合最短板
```

合格显示绿色，不合格显示红色。

模型内容使用`textContent`或`pre.textContent`插入页面，避免把文档或模型输出当作HTML执行。

---

## 23. 一次成功请求的类型流动

```text
question: str
  ↓
list[RRFResult]
  ↓
list[RerankedResult]
  ↓
evidence: list[RerankedResult]
  ↓
StructuredRAGAnswer
  ↓
CitationAudit
  ↓
GroundednessResponse
  ↓
GroundednessReport
  ↓
AnswerQualityResponse
  ↓
FinalQualityReport
  ↓
AnswerAssessment
  ↓ assessment_to_dict()
dict[str, Any]
  ↓ WebSocket JSON
浏览器版本卡片
```

---

## 24. 模型调用时间线

初稿通过：

```text
调用1 Reranker
调用2 Structured Answer
调用3 Groundedness Judge
调用4 Quality Judge
```

修复一次：

```text
前4次
调用5 Repair Model
调用6 Repaired Groundedness Judge
调用7 Repaired Quality Judge
```

确定性引用审计不消耗模型调用。

---

## 25. 启动与测试

```bash
python day23/python/day23_final_audited_rag.py
```

打开：

```text
http://127.0.0.1:8030
```

知识库内问题：

```text
什么是 RAG？它的工作流程是什么？
```

知识库外问题：

```text
法国的首都是哪里？
```

前者应生成并审计回答；后者应在Reranker守门阶段拒绝。
