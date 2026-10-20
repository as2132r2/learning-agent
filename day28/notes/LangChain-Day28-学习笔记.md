# LangChain Day28：结构化日志、Trace 与调用链可观测性

## 一、今天解决的问题

Day27 已经能通过 SLO、告警、事故和回滚判断系统是否健康。Day28 继续解决：当一次请求变慢或失败时，怎样找到具体原因？

```text
用户报告一次请求失败
→ 根据request_id定位请求
→ 根据trace_id找到整条调用链
→ 根据span_id查看每个阶段
→ 比较阶段耗时
→ 找到错误发生位置
→ 关联结构化日志和业务属性
```

Day27 更偏向全局健康监控，Day28 更偏向单次请求根因定位。

---

## 二、固定七步回顾

### 第 1 步：结构化 JSON 日志与关联 ID

文件：`day28/python/day28_structured_logging.py`

传统文本日志只适合阅读：

```text
开始检索知识库
```

结构化日志适合机器检索和统计：

```json
{
  "level": "INFO",
  "event": "retrieval_started",
  "request_id": "req-...",
  "trace_id": "trace-...",
  "details": {"top_k": 2}
}
```

关键组件：

- `logging.LogRecord`：Python 日志事件对象；
- `CorrelationFilter`：自动补充关联 ID；
- `JsonFormatter`：把日志转换为单行 JSON；
- `ContextVar`：保存当前请求上下文；
- JSONL：每行一个 JSON 对象，便于流式写入和逐行解析。

必须在 `finally` 中恢复 ContextVar，防止下一条请求沿用旧 ID。

### 第 2 步：Trace、Span 与父子关系

文件：`day28/python/day28_trace_spans.py`

概念：

```text
Trace：一次完整调用链
Span：调用链中的一个操作
parent_span_id：连接父子操作
```

示例：

```text
POST /ask
├── retrieve_documents
│   └── create_query_embedding
└── generate_answer
```

每个 Span 保存：

- `trace_id`；
- `span_id`；
- `parent_span_id`；
- name 和 kind；
- 开始、结束和耗时；
- status；
- attributes；
- error type 和 message。

`Span` 使用上下文管理器实现，进入时绑定自己的 ID，退出时恢复父 Span。

### 第 3 步：异步上下文传播

文件：`day28/python/day28_async_context.py`

重要规律：

```text
asyncio.create_task()
会复制创建Task那一刻的ContextVar上下文。
```

因此在请求根 Span 中创建检索和策略 Task 时，二者都会继承同一个 trace_id 和根 span_id，但随后各自修改自己的 Context 副本，不会串线。

日志即使交错输出，也能按 trace_id 重新分组。

并行阶段的总等待时间接近最大单任务耗时，而不是所有并行任务耗时之和。

### 第 4 步：HTTP 请求链路传播

文件：

- `day28_http_tracing_server.py`
- `day28_http_tracing_client.py`

服务端中间件统一执行：

```text
读取请求头
→ 校验或生成request_id、trace_id
→ 绑定ContextVar
→ 创建HTTP服务端Span
→ 调用路由
→ 在响应头返回关联ID
→ 清理上下文
```

教学请求头：

```text
X-Request-ID
X-Trace-ID
X-Span-ID
```

生产环境通常使用 W3C `traceparent` 标准，它还能表达上游 parent span 和采样标志。

外部 ID 必须限制字符和长度，不能直接信任客户端输入。

### 第 5 步：RAG 阶段追踪与持久化

文件：`day28/python/day28_rag_stage_tracing.py`

调用链：

```text
rag_pipeline
├── hybrid_retrieval
├── llm_reranker
├── structured_answer_generation
└── answer_quality_evaluation
```

`PersistentSpan` 在 Span 结束后将记录写入 SQLite。

两张核心表：

```text
traces：请求级概览
spans：阶段级细节
```

成功链路可以寻找最慢阶段；失败链路可以确定哪些阶段成功、错误发生在哪一步、后续哪些阶段没有执行。

本步使用确定性函数模拟 RAG 阶段，目的是隔离模型和网络波动，专注学习追踪机制。

### 第 6 步：最终可观测 RAG Web 控制台

文件：

- `day28_final_observability_dashboard.py`
- `day28_final.html`
- `day28_final.css`
- `day28_final.js`

页面展示：

- Trace 总数和成功率；
- 失败请求数；
- 平均总耗时；
- 平均最慢阶段；
- 最近 Trace；
- Span 耗时条；
- Span attributes；
- 失败类型和准确位置。

同步 RAG 流水线通过 `asyncio.to_thread()` 运行，避免阻塞 aiohttp 事件循环；`asyncio.Semaphore(2)` 限制流水线并发。

### 第 7 步：笔记和代码解读

当前文档用于概念复习；`Day28-最终代码详细解读.md` 解释最终系统实现。

---

## 三、日志、指标与追踪的区别

| 类型 | 主要回答 | 示例 |
|---|---|---|
| 日志 Logs | 发生了什么 | Reranker 请求超时 |
| 指标 Metrics | 整体表现如何 | 失败率 2%、P95 1.8 秒 |
| 追踪 Traces | 一次请求经过哪里 | 检索成功，Reranker 失败 |

三者不是替代关系：

```text
指标发现异常
→ Trace定位异常请求和阶段
→ 日志查看具体上下文与错误内容
```

Day27 主要建立指标与告警，Day28 建立日志与追踪。

---

## 四、四类 ID 的职责

| ID | 范围 | 用途 |
|---|---|---|
| user/session ID | 多次请求 | 用户或会话体验 |
| request_id | 一次业务请求 | 客服和接口排查 |
| trace_id | 一条跨组件调用链 | 聚合所有 Span |
| span_id | 一个具体操作 | 阶段耗时和错误定位 |

在简单单体服务中，request_id 和 trace_id 常常一一对应；在异步消息、重试、扇出调用和微服务环境中，两者不一定相同。

---

## 五、什么应该放进 Span attributes

适合记录：

- 模型名、Prompt 版本；
- top_k、候选数量、证据数量；
- Token 数；
- 缓存是否命中；
- HTTP 状态码；
- 数据库操作类型；
- 质量评估结果。

不应直接记录：

- 密码和令牌；
- API Key；
- 完整身份证或银行卡号；
- 未脱敏的私人文档；
- 过大的 Prompt 和响应全文。

可观测性不能成为敏感数据泄漏渠道。

---

## 六、耗时分析需要注意什么

串行阶段：

```text
总耗时 ≈ 阶段1 + 阶段2 + 阶段3 + 系统开销
```

并行阶段：

```text
总等待 ≈ max(并行阶段耗时) + 调度开销
```

因此不能简单把所有 Span 的 duration 相加当作请求总耗时。

Trace 的测量边界也可能比根 Span 更大，例如包含数据库持久化和追踪系统自身开销。

---

## 七、今天最重要的设计原则

1. 每条请求必须有稳定的关联 ID。
2. 业务代码记录事件，Filter 自动注入上下文。
3. 父子关系必须在进入和退出 Span 时严格恢复。
4. 异步 Task 在正确上下文中创建，才能继承正确 Trace。
5. 失败 Span 不应吞掉业务异常。
6. Trace 和 Span 应持久化，不能只显示在终端。
7. 追踪系统本身也有延迟、存储和隐私成本。
8. 不能把用户敏感内容无条件写入日志或 attributes。

---

## 八、复习检查题

1. request_id、trace_id 和 span_id 各自表示什么？
2. `parent_span_id` 如何还原调用树？
3. 为什么 Span 退出时必须恢复父 Span？
4. `create_task()` 在什么时候复制 ContextVar？
5. 为什么同步流水线要通过 `asyncio.to_thread()` 调用？
6. Semaphore 与 `to_thread()` 分别解决什么问题？
7. 为什么并行 Span 的耗时不能直接相加？
8. 日志、指标和追踪应该如何配合？

能够独立回答这些问题，就掌握了 Day28 的核心。

---

## 九、Day29 预告

下一天建议进入“标准化遥测与 OpenTelemetry 思想”：

- W3C `traceparent`；
- 标准 Span kind 和语义属性；
- 采样策略；
- 批量导出与背压；
- Trace 数据保留和脱敏；
- 从自制追踪器迁移到标准遥测接口。

