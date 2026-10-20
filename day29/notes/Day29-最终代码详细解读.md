# Day29 最终代码详细解读

本文解释 `day29_final_telemetry_service.py` 如何整合 W3C 上下文、语义 Span、采样、治理、批量导出和 Web 查询。

## 一、模块组合

```text
最终Web服务
├── traceparent_basic：上下文解析与生成
├── span_semantics：Kind、Status和语义属性
├── sampling_strategies：Parent-Based采样
├── telemetry_governance：脱敏与限制
└── batch_exporter：队列、组批和关闭
```

最终文件是编排层，不重新复制前五步规则。

## 二、全局组件

```python
sampler = ParentBasedSampler(TraceIdRatioBasedSampler(0.50))
pipeline_semaphore = asyncio.Semaphore(2)
```

采样器决定完整 Trace 是否导出；Semaphore 限制同时执行的 RAG 数量。二者职责不同。

## 三、SQLiteJsonlSink

Sink 同时批量写 SQLite 和 JSONL。SQLite 用于页面查询，JSONL 用于观察导出原文。

同步文件和数据库操作通过 `asyncio.to_thread()` 执行。`INSERT OR REPLACE` 以 span_id 幂等写入，重复导出不会产生相同 ID 的多行。

## 四、治理发生在导出之前

`semantic_to_exportable()`：

```text
SemanticSpan → RawSpan → govern_span()
→ GovernedSpan → ExportableSpan → queue
```

事件和治理统计保存为 `telemetry.*` 属性。必须先治理再入队，否则秘密已经进入内存缓冲和其他导出组件。

## 五、远程父上下文与采样

`resolve_parent()` 处理：

```text
无Header → 创建新根上下文
合法Header → 使用远程父上下文
非法Header → 丢弃并创建新上下文
```

Parent-Based Sampler 在有远程父时继承 flags，没有远程父时执行 50% 根采样。`dataclasses.replace()` 把最终决定写入 context 的 trace_flags。

## 六、telemetry_middleware

中间件是最终服务的核心：

```text
提取traceparent
→ 采样决定
→ 创建SERVER Span
→ 把root_context放入request
→ 执行路由
→ 映射HTTP状态
→ 响应返回traceparent和sampled
→ 结束根Span
→ sampled时治理并入队
```

`request` 保存 `root_context`、`semantic_spans` 和 `sampled`，子函数不需要依赖可变全局上下文。

## 七、根 Span 的生命周期

根 Span 必须覆盖完整的 `await handler(request)`，并在得到最终 HTTP 状态后结束。因此教学代码显式调用 builder 的 enter/exit。

正式 OpenTelemetry SDK 通常由 aiohttp 自动埋点或异步上下文管理器处理这一过程。

## 八、RAG 子 Span

成功请求：

```text
SERVER POST /api/ask
├── INTERNAL rag.retrieve
├── CLIENT model HTTP call
└── INTERNAL rag.answer.compose
```

检索和回答组织是进程内逻辑，所以为 INTERNAL；远程模型调用是同步等待的网络请求，所以为 CLIENT。

模型 503 时，CLIENT 和 SERVER 都为 ERROR，回答组织不再执行。

## 九、故意放入的敏感测试数据

教学代码故意构造邮箱、Bearer 和带 API Key 的 URL，用于验证治理层。数据库检查确认原文不存在，只留下 `[EMAIL]` 和 `[REDACTED]`。

这不是推荐记录敏感字段。生产原则仍是源头尽量不采集，脱敏只是第二道防线。

## 十、sampled 与 unsampled

中间件始终建立上下文并执行业务，但只有：

```python
if sampling.sampled:
    exporter.emit(...)
```

才进入导出队列。因此未采样请求可以返回 200，但数据库详情返回 404。这是预期行为。

## 十一、Exporter 生命周期

应用启动：

```text
create_schema → Sink → BatchSpanExporter → start()
```

应用关闭：

```text
aiohttp cleanup → shutdown() → 冲刷队列 → Worker结束
```

Exporter 保存在 `app["exporter"]`，与 Web 应用生命周期绑定，避免遗留后台 Task。

## 十二、查询 API

```text
POST /api/ask                 执行RAG
GET  /api/overview            Trace、Span与Exporter指标
GET  /api/traces              最近已导出Trace
GET  /api/traces/{trace_id}   单条Span详情
```

概览包含 offered、accepted、dropped、exported 和 failures，因为遥测管道本身也必须可观测。

## 十三、浏览器生成 traceparent

JavaScript 使用 `crypto.getRandomValues()` 生成 16 字节 trace_id 和 8 字节 parent_id：

```text
00-{trace_id}-{parent_id}-{01或00}
```

表单允许选择 flags，便于稳定演示 Parent-Based 行为。真实应用通常由前端遥测 SDK 自动注入，而不是用户选择。

## 十四、失败响应仍然可追踪

模型失败返回 HTTP 502，但响应仍携带：

- traceparent；
- X-Telemetry-Sampled；
- trace_id；
- 错误正文。

若 sampled=1，用户提供 trace_id 后即可定位 CLIENT 503 Span。

## 十五、错误 Trace 与错误 Span

一次模型失败会产生两个 ERROR Span：模型 CLIENT Span 和整体 SERVER Span。因此不能把错误 Span 数直接当成失败请求数。

```text
错误Trace数量：失败请求数量
错误Span数量：调用链中失败操作数量
```

## 十六、控制台自观测噪声

中间件覆盖所有路由，页面轮询、静态资源和 Trace 查询也可能被采样。生产配置通常排除健康检查、静态文件、高频内部轮询和遥测导出请求，否则会增加成本甚至递归观测。

## 十七、完整数据流

```text
浏览器traceparent
→ middleware解析与采样
→ SERVER Span
→ ask_handler + Semaphore
→ INTERNAL retrieve
→ CLIENT model
→ INTERNAL compose
→ HTTP响应
→ Span结束
→ govern_span
→ BatchSpanExporter
→ SQLiteJsonlSink
├→ SQLite
└→ JSONL
→ Web API查询并渲染
```

## 十八、关键数据类型

| 类型 | 作用 |
|---|---|
| `TraceContext` | 标准传播上下文 |
| `SamplingResult` | 采样决定和原因 |
| `SemanticSpan` | Kind、Status、attributes 和 events |
| `GovernedSpan` | 已脱敏、已限制的 Span |
| `ExportableSpan` | 批量导出队列元素 |
| `ExporterMetrics` | 导出器自身指标 |

## 十九、当前教学边界

1. RAG 和模型调用是确定性模拟。
2. 没有使用正式 OpenTelemetry SDK 和 OTLP Collector。
3. SQLite 不适合大规模遥测。
4. 只有 Head Sampling，没有 Tail Sampling。
5. 没有跨进程队列、重试和磁盘缓冲。
6. 无认证，不能直接公开部署。
7. 尚未排除控制台自身流量。

## 二十、运行与验证

```bash
python day29/python/day29_final_telemetry_service.py
```

访问 `http://127.0.0.1:8043`，依次验证：

1. sampled=1 正常请求产生 4 个 Span；
2. sampled=0 请求成功但不进入数据库；
3. sampled=1 模型 503 产生错误 Trace；
4. 数据库没有明文邮箱、Bearer 和 API Key；
5. 概览能看到导出与丢弃指标；
6. Ctrl+C 后剩余队列完成导出。

最终要记住：

```text
标准化遥测不是单独一个Trace库，
而是上下文传播、语义、采样、治理、导出和查询组成的完整系统。
```
