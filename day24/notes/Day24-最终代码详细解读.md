# Day24 最终代码详细解读

解读对象：`day24/python/day24_final_optimized_rag.py`及配套Web文件。

## 1. 最终版复用了哪些能力

```text
Day22
├── 上传与双索引同步
├── 混合检索与RRF
└── Reranker

Day23
├── 结构化回答
├── CitationAudit
├── Groundedness
├── AnswerQuality
└── RepairAnswer

Day24
├── SQLite模型缓存
├── 模型调用指标
├── 风险策略
└── BudgetController
```

最终服务端是编排层，不重新复制底层算法。

---

## 2. 服务配置

```python
HOST, PORT = "127.0.0.1", 8031
PIPELINE_LIMIT = 2
REPAIR_TOKEN_ESTIMATE = 900
```

服务只监听本机。最多同时运行两条高成本模型流水线。

---

## 3. 全局缓存对象

```python
cache = ModelCache(CACHE_DATABASE_PATH)
cache_lock = asyncio.Lock()
```

`ModelCache`内部持有一个SQLite连接。

`cache_lock`只保护短时间的`get/put`，不能在等待外部模型时持锁，否则一条MISS会阻塞所有缓存访问。

应用关闭时：

```python
app.on_cleanup.append(close_resources)
```

关闭SQLite连接。

---

## 4. AskPayload

```python
class AskPayload(BaseModel):
    question: str | None
    risk: Literal["auto", "low", "medium", "high"]
    max_calls: int
    max_tokens: int
    allow_downgrade: bool
```

默认：

```text
risk=auto
max_calls=7
max_tokens=8500
allow_downgrade=False
```

Pydantic限制调用预算0～20、Token预算0～100000。

---

## 5. ExecutionTrace

```python
@dataclass
class ExecutionTrace:
    stages: list[dict[str, Any]]
```

计算属性：

```text
cache_hits
model_calls
total_tokens
```

每阶段记录：

```text
name
cache_hit
elapsed_seconds
model_calls
total_tokens
cache_key前12位
```

---

## 6. async_cached_model_call

参数：

```text
trace：运行轨迹
controller：预算控制器
stage_name：阶段名
cache_input：完整缓存键输入
function：真实模型函数
function_args：位置参数元组
serializer/deserializer：类型转换函数
estimated_tokens：预留Token
```

命中流程：

```text
make_key
→ cache.get
→ deserializer
→ trace追加cache_hit
→ 返回
```

命中不调用`controller.reserve()`，因为没有模型成本。

MISS流程：

```text
controller.reserve
→ to_thread(measured_model_call)
→ controller.complete
→ serializer
→ cache.put
→ trace追加真实指标
```

---

## 7. 为什么缓存输入必须包含正文

Reranker键包含问题和RRF候选；回答键包含问题和证据；评估键还包含结构化回答。

文档正文或上游输出变化后，下游自动MISS，避免返回与当前资料不一致的旧评估。

---

## 8. assessment_payload

将`AnswerAssessment`转换成浏览器JSON：

```text
version
label
passed
answer
citation_coverage
groundedness
relevance
completeness
overall
errors
```

低/中风险未执行的指标使用`None`，序列化成JSON的`null`。

---

## 9. policy_passed

Day23的`AnswerAssessment.passed`要求存在完整`FinalQualityReport`，只适合完整审计。

Day24有低/中/高三种策略，因此单独定义：

```text
low：CitationAudit通过
medium：CitationAudit + Groundedness通过
high：FinalQualityReport通过
```

不能把“未执行完整评估”误判为系统异常，也不能把未执行伪装成满分。

---

## 10. audit_with_policy

共同第一步：

```python
rendered = build_readable_answer(answer)
citation = audit_citations(...)
```

引用失败立即返回，避免继续消耗模型预算。

高风险分支同时创建：

```text
optimized-groundedness
optimized-answer-quality
```

两个Task内部各自先查缓存，MISS才消耗预算并调用模型。

---

## 11. run_pipeline第一阶段：计划

```python
decision = classify_risk(question, payload.risk)
policy, downgrade = choose_executable_policy(...)
```

先发送`plan`事件，让前端显示：

```text
请求风险
实际策略
分类原因
降级原因
调用和Token预算
```

无可执行策略则在检索和模型调用前发送`budget_rejected`。

---

## 12. 本地检索

```python
async with index_lock:
    rrf_results, diagnostics = await asyncio.to_thread(
        retrieve_rrf_candidates,
        question,
    )
```

`index_lock`与上传双索引同步共享，避免检索两套不同版本的索引。

本地检索不占模型预算。

---

## 13. 模型流水线Semaphore

```python
await model_pipeline_semaphore.acquire()
```

许可覆盖从Reranker到最终审计/修复，不在每个阶段之间释放。

`finally`保证任何退出路径都释放许可。

---

## 14. Reranker缓存

缓存输入包含：

```text
question
RRF候选的chunk_id、位置、正文、两路排名和RRF分数
```

输出恢复为：

```text
tuple[list[RerankedResult], RerankResponse]
```

随后`select_evidence()`只保留达到95分的资料。

---

## 15. 回答缓存

缓存输入：

```text
question
通过守门的evidence完整数据
```

输出恢复为`StructuredRAGAnswer`。

CitationAudit始终重新运行，因为它很便宜，也能验证缓存恢复后的对象仍满足当前结构规则。

---

## 16. 并行评估与缓存组合

高风险可能出现四种情况：

```text
Groundedness HIT + Quality HIT：0次调用
Groundedness HIT + Quality MISS：1次调用
Groundedness MISS + Quality HIT：1次调用
Groundedness MISS + Quality MISS：并行2次调用
```

这是缓存和并行的组合，而不是二选一。

---

## 17. 修复预算

初稿失败后：

```python
repair_answer(...)
```

也走缓存和预算控制，键包含：

```text
问题
证据
当前回答
审计反馈
```

修复后重新执行当前风险策略的审计。

如果预算不够，`BudgetExceeded`终止，不会无预算继续修复。

---

## 18. 最终summary

```text
status
answer
repair_count
wall_seconds
cache_hits
model_calls
total_tokens
stages
reserved_calls/tokens
actual_calls/tokens
```

前端使用这些字段生成运行指标卡和逐阶段JSON详情。

---

## 19. 缓存与预算的关系

命中缓存：

```text
不reserve
actual_calls不增加
actual_tokens不增加
```

MISS：

```text
先reserve估算
再调用
最后记录actual
```

但预检采用全部MISS的最坏估算，防止高风险请求只能依赖缓存才执行得起。

---

## 20. 主要事件流

```text
connected
→ plan
→ retrieving
→ rrf_results
→ reranked
→ version_assessed
├─ done
└─ repairing
   → version_assessed
   ├─ done
   └─ quality_rejected
```

拒绝和异常事件：

```text
budget_rejected
budget_exceeded
retrieval_rejected
quality_rejected
cancelled
error
busy
cancel_not_found
```

---

## 21. 前端参数流动

HTML输入：

```text
risk
max-calls
max-tokens
downgrade checkbox
question
```

JavaScript转换：

```javascript
{
  question: q,
  risk: risk.value,
  max_calls: Number(maxCalls.value),
  max_tokens: Number(maxTokens.value),
  allow_downgrade: downgrade.checked
}
```

服务端Pydantic再次校验，不能只依赖浏览器HTML限制。

---

## 22. 一次请求的核心类型流

```text
AskPayload
→ RiskDecision
→ AuditPolicy
→ BudgetController
→ list[RRFResult]
→ list[RerankedResult]
→ StructuredRAGAnswer
→ CitationAudit
→ GroundednessReport | None
→ AnswerQualityResponse | None
→ FinalQualityReport | None
→ AnswerAssessment
→ dict[str, Any]
→ WebSocket JSON
```

---

## 23. 运行方式

```bash
python day24/python/day24_final_optimized_rag.py
```

打开：

```text
http://127.0.0.1:8031
```

相同问题提交两次，观察第二次的缓存命中、真实调用和Token变化。

---

## 24. 当前仍可改进

```text
使用新版UsageMetadataCallbackHandler消除弃用依赖
缓存TTL与知识库版本
用户/租户权限进入缓存键
相同key的single-flight防缓存击穿
异步可取消模型客户端
精确输入Tokenizer和输出max_tokens
分布式缓存与预算服务
指标写入Prometheus/OpenTelemetry
```

最终版已经展示完整控制思想，但不是无需改造即可直接处理敏感生产数据的成品。
