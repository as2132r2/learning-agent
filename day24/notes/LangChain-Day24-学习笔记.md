# LangChain Day24：可审计RAG的性能与成本优化

## 1. 今天解决的问题

Day23的可审计RAG初稿通过需要4次模型调用，修复一次最多7次。Day24解决：

```text
每个阶段到底花多少时间和Token？
相同输入能否避免重复调用？
独立评估器能否并行？
所有问题是否都需要最高级审计？
如何防止调用和Token失控？
```

固定完成七步：

1. 记录阶段耗时、调用次数和Token。
2. SQLite持久化模型缓存。
3. 并行运行两个独立评估器。
4. 风险分级与自适应审计。
5. 调用次数和Token预算控制。
6. 整合最终优化版Web RAG。
7. 整理笔记和最终代码解读。

---

## 2. 性能基线

实测初稿通过：

```text
墙钟耗时：5.557秒
模型调用：4次
输入Token：3470
输出Token：824
Token总量：4294
```

阶段数据：

```text
混合检索与RRF        0.043秒       0 Token
Reranker             1.746秒    1562 Token
结构化回答           1.015秒     796 Token
Groundedness         1.017秒     934 Token
相关性与完整性       1.736秒    1002 Token
```

本地检索和确定性审计很便宜，主要时间和成本来自网络模型调用。

---

## 3. StageMetric

```python
@dataclass
class StageMetric:
    name: str
    kind: str
    elapsed_seconds: float
    model_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    status: str
    detail: str
```

本地阶段和模型阶段使用相同指标结构，便于汇总和比较。

`perf_counter()`测量持续时间，不用于记录业务时间戳。

---

## 4. PipelineMetrics

```python
@dataclass
class PipelineMetrics:
    question: str
    stages: list[StageMetric]
```

计算属性汇总：

```text
阶段耗时之和
模型调用总数
输入Token
输出Token
总Token
```

优化前必须先建立基线，否则无法证明改动是否有效。

---

## 5. 缓存实验结果

清空缓存后连续运行两次：

```text
第一次：4次调用，4269 Token，5.590秒
第二次：0次调用，0 Token，0.073秒
```

结果：

```text
耗时降低约98.7%
约76倍加速
节省4269 Token
减少4次网络请求
```

缓存是减少调用、费用和延迟最明显的方法，但前提是结果可以安全复用。

---

## 6. 缓存键

缓存键输入：

```python
{
    "stage_name": stage_name,
    "model_name": model_name,
    "prompt_version": prompt_version,
    "input": input_data,
}
```

规范化JSON：

```python
json.dumps(..., sort_keys=True, separators=(",", ":"))
```

再计算SHA-256。

问题、候选、正文、回答、模型或提示词版本变化，都会产生新键。

---

## 7. 为什么不能只用问题作为缓存键

同一个问题可能因为以下变化得到不同结果：

```text
知识库文档变化
RRF候选变化
模型版本变化
系统提示词变化
权限范围变化
上游回答变化
```

只使用问题会错误复用旧答案或其他权限范围的数据。

---

## 8. Cache-Aside模式

```text
计算key
→ 查询SQLite
├─ HIT：反序列化并返回
└─ MISS：调用模型 → 序列化 → 写缓存 → 返回
```

缓存命中必须恢复原始类型：

```text
RerankedResult
RerankResponse
StructuredRAGAnswer
GroundednessResponse
AnswerQualityResponse
```

否则后续流水线会收到普通dict，破坏类型契约。

---

## 9. 缓存失效

当前通过内容哈希和`PROMPT_VERSION`实现自然失效：

```python
PROMPT_VERSION = "day24-v1"
```

修改提示词后升级版本即可产生新键。

生产缓存还需要：

```text
TTL/expires_at
知识库版本
租户或用户ID
权限范围
数据敏感级别
最大容量和清理策略
```

---

## 10. 并行实验结果

两个独立评估器：

```text
串行墙钟：2.238秒
并行墙钟：1.638秒
节省：0.600秒
下降：26.8%
加速：1.37倍
```

模型调用均为2次，Token分别1906和1912，基本不变。

结论：

```text
缓存减少调用、Token和延迟
并行主要减少等待时间，不减少计算量
```

---

## 11. 哪些阶段可以并行

依赖关系：

```text
Reranker → 结构化回答
                    ├→ Groundedness
                    └→ AnswerQuality
```

前两个存在数据依赖，不能并行。

两个评估器都只依赖问题、回答和证据，互不依赖，可以并行。

---

## 12. asyncio并行方式

同步模型函数先放入线程：

```python
await asyncio.to_thread(...)
```

再创建Task：

```python
groundedness_task = asyncio.create_task(...)
quality_task = asyncio.create_task(...)
```

使用：

```python
await asyncio.gather(task1, task2)
```

`gather()`结果顺序与传入顺序一致，不按完成顺序排列。

---

## 13. 风险分级

```text
low：定义、简单事实
medium：流程、比较、分析、步骤
high：医疗、法律、金融、密钥、危险操作等
```

分类器使用确定性规则，不额外消耗模型调用，并输出命中的关键词。

用户可以通过`--risk`或Web参数手动提高审计等级。

---

## 14. 三种审计策略

```text
low：Reranker + 回答 + CitationAudit
     2次模型调用

medium：low + Groundedness
        3次模型调用

high：medium + AnswerQuality
      4次模型调用
```

高风险两个评估器并行。

未执行的评估必须表示为`None/未执行`，不能伪造成满分。

---

## 15. 自适应审计的风险

减少审计不是免费优化：

```text
调用减少
→ 成本下降
→ 延迟下降
→ 残余错误风险上升
```

低风险策略只证明引用结构合法，不能证明语义支持或回答完整。

生产环境必须限制谁有权降低风险等级。

---

## 16. 预算估算

根据实测设置保守估算：

```text
Reranker          1700
StructuredAnswer   900
Groundedness      1000
AnswerQuality     1100
```

策略总预算：

```text
low：2次 / 2600 Token
medium：3次 / 3600 Token
high：4次 / 4700 Token
```

修复需要额外的生成和重新评估预算。

---

## 17. 硬预算与软预算

调用次数可以在调用前精确知道，是硬预算。

Token调用前只能估算，模型返回后才知道真实总数，是软预算。

```text
reserved_calls / reserved_tokens
actual_calls / actual_tokens
```

调用前`reserve()`，完成后`complete()`。

真实Token超过软预算时，已经发生的调用无法撤销，但后续阶段停止。

---

## 18. 降级规则

```text
中风险预算不足
├─ 未显式允许 → 拒绝
└─ allow_downgrade → 可以执行low并记录原因

高风险预算不足
└─ 永远拒绝，不允许静默降级
```

不能为了节省成本把医疗、法律或金融问题偷偷降低审计标准。

---

## 19. 最终系统组合

```text
风险判断
→ 最坏情况预算预检
→ 本地混合检索
→ 每个模型阶段先查缓存
→ MISS才预留预算并调用
→ 高风险独立评估器并行
→ 质量失败且预算允许时修复一次
→ 展示缓存、调用、Token和耗时
```

缓存命中不占模型预算，因为没有外部调用。

---

## 20. 最终WebSocket事件

```text
connected
plan
├─ budget_rejected
└─ retrieving
    → rrf_results
    → reranked
    ├─ retrieval_rejected
    └─ version_assessed
       ├─ done
       └─ repairing
          → version_assessed
          ├─ done
          └─ quality_rejected
```

运行中还可能出现：

```text
budget_exceeded
cancelled
error
busy
cancel_not_found
```

---

## 21. 保守预算预检

最终Web版在预检时假设缓存可能全部MISS。

即使当前缓存存在，过低预算仍可能被拒绝。原因是缓存可能：

```text
过期
失效
键不匹配
被清理
因权限隔离不可用
```

高风险流程不能依赖“希望缓存命中”才能支付得起。

---

## 22. 并发缓存的注意点

当前使用`cache_lock`保护单个SQLite连接的短事务，但不会在外部模型调用期间持锁。

两个完全相同的并发MISS可能同时调用模型，再分别写入同一键。这叫cache stampede。

生产环境可使用：

```text
single-flight
按cache_key加锁
分布式锁
请求合并
```

避免重复回源。

---

## 23. 回调弃用提示

当前阶段统计使用`get_openai_callback`，运行时会看到`langchain-community`停止维护提示。

它不影响本次实验。后续生产重构可让模型调用保留原始`AIMessage.usage_metadata`，或使用新版`UsageMetadataCallbackHandler`并显式传入Runnable配置。

---

## 24. 文件导航

```text
day24/python/day24_pipeline_metrics.py
day24/python/day24_pipeline_cache.py
day24/python/day24_parallel_evaluators.py
day24/python/day24_adaptive_audit.py
day24/python/day24_budget_controller.py
day24/python/day24_final_optimized_rag.py
day24/html/day24_final.html
day24/javascript/day24_final.js
day24/css/day24_final.css
day24/database/day24_model_cache.db
```

---

## 25. 一句话总结

Day24把可靠但昂贵的可审计RAG升级为可测量、可缓存、可并行、按风险分级且受预算约束的生产型流水线。
