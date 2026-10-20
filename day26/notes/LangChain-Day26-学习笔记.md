# LangChain Day26：提示词 A/B 实验、灰度发布与安全护栏

## 一、今天解决的问题

Day25 已经能够收集反馈、生成候选评测集并离线比较版本，但离线结果不能直接证明新版本适合全部真实用户。

Day26 建立安全的线上实验流程：

```text
候选版本离线验证
→ 给少量真实用户稳定分流
→ 冻结版本配置
→ 记录真实曝光和结果
→ 比较收益指标
→ 持续检查安全护栏
→ 达标后逐步扩大流量
→ 有害时自动停止并回退control
```

---

## 二、固定七步回顾

### 第 1 步：稳定分流与粘性实验分组

文件：`day26/python/day26_stable_assignment.py`

稳定桶：

```text
SHA-256(experiment_id | user_id)
→ 取前8字节转整数
→ 对10000取模
→ 0～9999 bucket
```

50/50 分流：

```text
0000～4999 → control
5000～9999 → candidate
```

分组写入：

```sql
PRIMARY KEY (experiment_id, user_id)
```

后续请求优先读取数据库中的旧分组。因此同一个用户不会刷新页面后换版本。

少量样本不会正好等于配置比例；用户量增加后，哈希桶分布才逐渐接近目标权重。

### 第 2 步：版本配置、提示词快照与实验冻结

文件：`day26/python/day26_variant_config.py`

`VariantConfig` 保存：

- variant；
- prompt_version；
- system_prompt；
- model_name；
- temperature；
- max_evidence_count；
- reranker_threshold。

配置先转为规范 JSON，再计算 SHA-256：

```text
VariantConfig
→ json.dumps(sort_keys=True, separators=(",", ":"))
→ config_hash
```

唯一身份：

```text
experiment_id + allocation_version + variant
```

同一身份的配置冻结后不能修改。若要修改提示词，应创建 `allocation-v2`，不能覆盖 `allocation-v1`。

否则同一个 candidate 名称下会混合多个真实版本，历史指标无法解释和复现。

### 第 3 步：曝光、回答结果与反馈事件

文件：`day26/python/day26_exposure_events.py`

三个事件：

```text
Assignment：用户理论上属于哪组
Exposure：该版本真正开始为一次请求服务
Outcome：请求最终成功、拒绝、错误或取消
FeedbackEvent：用户主动给出的主观评价
```

分组不等于曝光。用户被分配后没有发起问题，不能进入实验请求分母。

事件幂等规则：

```text
Exposure：request_id只能关联一个用户和版本
Outcome：request_id只有一个最终结果
Feedback：同一用户可以修改评价
```

反馈不是每次都有，因此分析使用 `LEFT JOIN`，保留没有反馈的曝光。

### 第 4 步：A/B 指标与基础显著性

文件：`day26/python/day26_ab_metrics.py`

客观指标：

- 成功率；
- 质量通过率；
- 平均延迟；
- 平均 Token；
- 平均模型调用次数。

主观指标：

- 反馈响应率；
- 正向反馈率。

分母：

```text
成功率 = 成功数 / 曝光数
质量通过率 = 质量通过数 / 曝光数
反馈响应率 = 反馈数 / 曝光数
正向反馈率 = 点赞数 / 反馈数
```

统计工具：

- Wilson 95% 置信区间；
- 双侧双比例 z 检验；
- `alpha=0.05`；
- 每组最低有效样本门槛。

当前每组只有三个样本，即使 candidate 正向反馈率表面高 16.67 个百分点，也不能下结论。

反馈存在自选择偏差：主动点击的人不一定代表所有曝光用户。

### 第 5 步：质量、错误率、延迟与成本护栏

文件：`day26/python/day26_guardrails.py`

收益检验与安全护栏的目标不同：

```text
收益检验：candidate是否显著更好？
护栏：candidate是否已经明显有害？
```

两层护栏：

```text
灾难性护栏
├── candidate至少3次曝光
└── 错误率达到50%立即停止

标准护栏
├── 每组至少20次曝光
├── 质量通过率下降不超过5个百分点
├── 拒绝率增加不超过10个百分点
├── 延迟不超过control的1.25倍
└── Token不超过control的1.30倍
```

真实小样本结果为 `continue_collecting`。假想退化场景触发 `stop`。

停止操作在同一事务中保存决策并更新：

```sql
experiments.status = 'stopped'
```

### 第 6 步：最终灰度发布 Web RAG

文件：

- `day26/python/day26_final_canary_rag.py`
- `day26/html/day26_final.html`
- `day26/javascript/day26_final.js`
- `day26/css/day26_final.css`

最终系统复用 Day24 的优化 RAG 和 Day25 的回答追踪与反馈闭环。

运行中：

```text
user_id
→ 稳定Assignment
→ 冻结VariantConfig
→ Exposure
→ 对应Prompt的RAG
→ AnswerRun + Outcome
→ FeedbackEvent
```

停止后：

```text
experiments.status=stopped
→ 全部请求安全回退control
→ 不再计入实验曝光
```

### 第 7 步：笔记和最终代码解读

当前文档用于概念复习；`Day26-最终代码详细解读.md` 解释最终系统实现。

---

## 三、为什么不能每次随机分组

错误方式：

```python
variant = random.choice(["control", "candidate"])
```

问题：

- 同一用户体验来回变化；
- 多轮对话可能跨版本；
- 用户反馈归因不清；
- 重复访问不能稳定复现。

稳定哈希加持久化 Assignment 能解决这些问题。

---

## 四、为什么需要 Exposure

假设一千个用户被分组，但只有一百人真正发送问题。

如果用分组人数作为成功率和点赞率分母，九百名没有体验产品的用户会污染指标。

因此：

```text
Assignment用于路由
Exposure用于实验分析
```

---

## 五、为什么需要 Outcome 的所有状态

如果只记录成功请求：

```text
candidate发生大量错误
→ 错误请求没有进入数据库
→ 数据库里只剩成功请求
→ candidate看起来质量很好
```

这是幸存者偏差。

所以必须记录：

```text
success / rejected / error / cancelled
```

---

## 六、为什么 Prompt 和缓存都要隔离

control 和 candidate 问题、证据相同时，如果缓存键只有：

```text
question + evidence
```

candidate 可能命中 control 已生成的回答，导致实验名义上分组、实际上输出相同。

版本相关缓存键必须包含：

```text
question + evidence + config_hash
```

---

## 七、并发请求为什么使用 ContextVar

不能使用：

```python
current_config = candidate_config
```

这是普通全局变量。两个并发请求会互相覆盖。

Day26 使用：

```python
ContextVar[VariantConfig]
```

每个异步任务有自己的上下文，`asyncio.to_thread()` 也会复制当前上下文，因此在线程内生成回答时仍能读取正确版本。

---

## 八、实验的完整数据流

```text
用户请求
→ 检查实验状态
├─ stopped → control回退，不计实验曝光
└─ running
   → Assignment
   → VariantConfig + config_hash
   → Exposure
   → ContextVar注入Prompt
   → 版本隔离缓存
   → Day24可审计RAG
   → Day25回答追踪
   → Outcome
   → 浏览器回答
   → FeedbackEvent
   → A/B聚合
   → 收益检验 + 安全护栏
```

---

## 九、生产环境还需要什么

- `user_id` 必须来自 Day15 的已验证访问令牌；
- 用真实实验平台或 Feature Flag 服务管理流量；
- 使用 PostgreSQL 支持多实例事务；
- 用消息队列异步发送埋点，避免阻塞回答；
- 处理机器人、员工和重复用户流量；
- 按用户而非仅按请求估计方差；
- 提前完成样本量与检验周期设计；
- 处理多重指标和多重检验；
- 对时间趋势、渠道和用户群分层；
- 使用灰度阶段：1% → 5% → 20% → 50%；
- 停止后保留配置、事件和决策审计记录。

---

## 十、Day26 核心结论

1. 分组必须稳定，Assignment 不等于 Exposure。
2. 版本名称不够，必须保存不可变配置快照和哈希。
3. 成功、拒绝、错误和取消都要记录，避免幸存者偏差。
4. 表面提升不等于统计显著，小样本不能下结论。
5. 收益检验追求证明更好，护栏追求尽快发现有害。
6. Prompt 不同意味着缓存键也必须不同。
7. 并发版本配置应使用请求级上下文，而不是普通全局变量。
8. 实验停止后应安全回退 control，并停止新增实验曝光。
