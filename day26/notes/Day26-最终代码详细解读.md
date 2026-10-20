# Day26 最终代码详细解读

本文详细解释 `day26_final_canary_rag.py` 如何把 Day24、Day25 和 Day26 串成完整灰度发布系统。

## 一、最终文件结构

```text
day26/
├── python/
│   ├── day26_stable_assignment.py
│   ├── day26_variant_config.py
│   ├── day26_exposure_events.py
│   ├── day26_ab_metrics.py
│   ├── day26_guardrails.py
│   └── day26_final_canary_rag.py
├── html/day26_final.html
├── javascript/day26_final.js
├── css/day26_final.css
├── database/day26_experiments.db
├── reports/day26_ab_metrics.json
└── notes/
```

---

## 二、复用的三层能力

### Day24

提供：混合检索、Reranker、结构化引用回答、风险审计、预算、缓存、并发、修复和 WebSocket 事件。

### Day25

提供：回答与引用追踪、用户反馈保存，以及 `request_id` 关联。

### Day26

增加：稳定分流、配置快照、曝光、实验 Outcome、A/B 指标和护栏。

组合关系：

```text
Day26实验层
└── Day25反馈与追踪层
    └── Day24可审计RAG核心
```

---

## 三、请求级版本上下文

```python
current_variant_config: ContextVar[VariantConfig]
current_config_hash: ContextVar[str]
```

为什么需要两个值：

- `current_variant_config` 决定模型、temperature 和 system prompt；
- `current_config_hash` 进入缓存键，隔离版本缓存。

请求开始时：

```python
token_config = current_variant_config.set(config)
token_hash = current_config_hash.set(digest)
```

请求结束时必须恢复：

```python
finally:
    current_variant_config.reset(token_config)
    current_config_hash.reset(token_hash)
```

`set()` 返回的 Token 不是模型 Token，而是 ContextVar 的恢复令牌。

若不在 `finally` 恢复，复用任务上下文时可能残留旧配置。

---

## 四、`generate_variant_answer()`

函数签名保持与 Day23 原函数一致：

```python
def generate_variant_answer(question, evidence) -> StructuredRAGAnswer
```

它从当前上下文读取配置：

```python
config = current_variant_config.get()
```

再创建模型：

```python
ChatOpenAI(
    model=config.model_name,
    temperature=config.temperature,
)
```

最终系统提示由两部分组成：

```text
冻结版本自己的system_prompt
+
所有版本共用的结构化claims和citation_numbers规则
```

共用结构规则保证后续 Day23 引用审计仍然可以处理两个版本。

---

## 五、运行时适配 Day24

```python
day24.generate_structured_answer = generate_variant_answer
day24.async_cached_model_call = experiment_cached_model_call
```

这是教学项目中的适配方式：保持 Day24 主流程不复制，只替换两个明确扩展点。

大型生产项目更适合把生成函数和缓存命名空间设计为构造参数或依赖注入，而不是运行时替换模块函数。

---

## 六、缓存隔离适配器

原始 Day24 缓存输入包含问题和证据。Day26 包装：

```python
if stage_name in {"StructuredAnswer", "RepairAnswer"}:
    cache_input = {
        "experiment_config_hash": current_config_hash.get(),
        "original_input": cache_input,
    }
```

为什么只处理版本相关阶段：

- Reranker 的输入和逻辑在本实验中相同，可以共享；
- Groundedness 和质量评估的缓存输入已包含具体回答；
- StructuredAnswer 直接受 Prompt 影响，必须隔离；
- RepairAnswer 也可能受版本回答影响，显式隔离更安全。

---

## 七、数据库初始化

`initialize_databases()` 初始化两个数据库：

```text
Day26实验数据库
├── experiments
├── experiment_assignments
├── experiment_variant_configs
├── experiment_exposures
├── experiment_outcomes
├── experiment_feedback_events
└── experiment_guardrail_decisions

Day25反馈数据库
├── answer_runs
├── answer_sources
└── linked_feedback
```

教学中保留两个数据库，用来展示跨系统组合。生产环境若要求反馈和实验事件严格原子一致，可使用同一个事务数据库或可靠事件总线。

---

## 八、实验状态解析

`resolve_experiment(user_id)` 返回：

```python
(variant, config, config_hash, counted_as_exposure)
```

运行中：

```text
读取Assignment
→ 加载对应冻结配置
→ counted_as_exposure=True
```

停止后：

```text
强制control
→ counted_as_exposure=False
```

停止后不继续调用 `assign_user()`，也不写新 Exposure。

---

## 九、Exposure 的写入时机

```python
variant, config, digest, exposed = resolve_experiment(user_id)
if exposed:
    record_exposure(request_id, user_id)
```

它发生在真正调用 RAG 前。

即使之后请求错误或取消，这次 Exposure 仍然真实存在，并会配套写入失败 Outcome。

---

## 十、`OutcomeRecorder`

它观察 WebSocket 终止事件：

```text
done               → success
quality_rejected   → rejected
retrieval_rejected → rejected
budget_rejected    → rejected
budget_exceeded    → rejected
error              → error
cancelled          → cancelled
```

生成：

```python
OutcomeInput(
    request_id,
    status,
    quality_passed,
    latency_ms,
    model_calls,
    total_tokens,
    cache_hits,
    error_type,
)
```

`recorded` 避免同一请求重复保存多个终止结果。

---

## 十一、双层 WebSocket 代理

调用关系：

```text
Day24 run_pipeline
→ Day25 RecordingWebSocket
   ├── 保存AnswerRun和SourceTrace
   └── Day26 OutcomeWebSocket
       ├── 保存Outcome
       └── 真实浏览器WebSocket
```

`done` 的顺序：

```text
先保存Day25回答追踪
→ 再保存Day26成功Outcome
→ 最后发送done给浏览器
```

因此用户看到回答时，两层服务端事实都已落库。

---

## 十二、`run_experiment_pipeline()`

完整步骤：

```text
1. 根据user_id解析实验版本
2. 运行中则记录Exposure
3. 向浏览器发送experiment_assigned
4. 设置ContextVar
5. 创建OutcomeRecorder
6. 调用Day25 tracked pipeline
7. finally恢复ContextVar
```

`experiment_assigned` 包含：

- experiment_id；
- allocation_version；
- variant；
- prompt_version；
- config_hash 短值；
- 是否计入曝光；
- 实验状态。

---

## 十三、WebSocket 连接状态

每个浏览器连接拥有 `day24.ConnectionState`：

```text
active_request_id
active_task
send_lock
```

作用：

- 同一连接一次只跑一个任务；
- cancel 只取消匹配 request_id 的任务；
- 并发事件通过 send_lock 顺序发送；
- 连接关闭时回收活动任务。

---

## 十四、反馈双写

`POST /feedback` 先检查 Day25 的 AnswerRun：

```text
回答不存在 → 404
用户不一致 → 403
```

然后：

```text
Day25 linked_feedback
+
若有实验Exposure，则写Day26 experiment_feedback_events
```

如果实验已经停止后的 control 回退请求没有 Exposure：

- 仍然保存普通产品反馈；
- 不写实验反馈；
- 返回 `experiment_feedback_recorded=false`。

这避免停止后的普通流量污染已结束实验。

---

## 十五、前端实验展示

收到 `experiment_assigned` 后显示：

```text
variant徽章
prompt_version
config_hash短值
是否计入实验曝光
```

用户更换 `user_id` 再提问，可观察稳定版本变化。同一 `user_id` 重复请求应保持相同版本。

生产环境不能允许用户随意输入 `user_id`，应从访问令牌读取。这里保留输入框是为了教学观察。

---

## 十六、HTTP 路由

```text
GET  /                     页面
GET  /app.js               Day26前端逻辑
GET  /app.css              Day26样式
GET  /day24.css            基础RAG样式
GET  /day25.css            反馈面板样式
POST /upload               复用文档上传
GET  /ws                   实验RAG WebSocket
POST /feedback             产品反馈与实验反馈
GET  /experiment/status    实验状态和曝光数
```

---

## 十七、完整时序

```text
浏览器       Day26实验层       Day25追踪层       Day24 RAG       数据库
  │ ask          │                 │                │               │
  ├─────────────>│ Assignment      │                │               │
  │              ├──读取配置──────────────────────────────────────>│
  │              ├──Exposure─────────────────────────────────────>│
  │<─variant─────┤                 │                │               │
  │              ├────────────────────────────────>│               │
  │              │                 │<────events─────┤               │
  │              │                 ├──AnswerRun───────────────────>│
  │              ├──Outcome──────────────────────────────────────>│
  │<────done─────┤                 │                │               │
  │ feedback     │                 │                │               │
  ├─────────────>│──linked_feedback + experiment_feedback────────>│
```

---

## 十八、运行方法

```bash
source .venv/bin/activate
python day26/python/day26_final_canary_rag.py
```

浏览器：

```text
http://127.0.0.1:8035
```

实验状态：

```text
http://127.0.0.1:8035/experiment/status
```

重新生成指标：

```bash
python day26/python/day26_ab_metrics.py
```

运行护栏：

```bash
python day26/python/day26_guardrails.py
```

---

## 十九、最终理解

一个可信的线上实验不能只有两个 Prompt 和一个随机数。它至少需要：

```text
稳定Assignment
+ 不可变配置快照
+ 真实Exposure
+ 完整Outcome
+ 可选Feedback
+ 正确分母
+ 不确定性判断
+ 安全护栏
+ 停止与control回退
+ 全链路审计记录
```

Day26 把 Day25 的“根据反馈持续改进”，推进到了“以可归因、可停止、可复现的方式验证改进”。
