# LangChain Day25：用户反馈、离线回放与持续改进闭环

## 一、今天解决的问题

Day24 已经能测量和控制 RAG 的质量、成本与耗时，但系统仍然不知道真实用户是否满意。

Day25 建立以下闭环：

```text
线上回答
→ 保存回答、证据、版本和指标
→ 用户点赞、点踩或填写原因
→ 生成候选评测样本
→ 人工复核弱标签
→ 离线回放候选版本
→ 对比新旧版本
→ 验证通过后再考虑上线
```

这叫持续改进闭环，而不是简单收集点赞数。

---

## 二、固定七步回顾

### 第 1 步：反馈数据模型与 SQLite 持久化

文件：`day25/python/day25_feedback_store.py`

定义：

```python
@dataclass(slots=True)
class FeedbackInput:
    request_id: str
    session_id: str
    user_id: str
    question: str
    answer: str
    rating: Rating
    reason: str | None
    answer_version: str
    source_ids: list[str] | None
    pipeline_metrics: dict[str, object] | None
```

第一步先理解一条反馈需要关联哪些信息。

唯一约束：

```sql
UNIQUE (request_id, user_id)
```

同一用户对同一回答只能保留一条当前评价。再次提交执行 UPSERT，而不是新增重复记录。

### 第 2 步：反馈 HTTP API

文件：

- `day25/python/day25_feedback_api.py`
- `day25/python/day25_feedback_api_client.py`

接口：

```text
GET  /health
POST /feedback
GET  /feedback
GET  /feedback?rating=down
GET  /feedback/statistics
```

HTTP JSON 属于不可信输入，必须先检查：

- 必填字符串是否为空；
- `rating` 是否为 `up/down/neutral`；
- `source_ids` 是否是字符串列表；
- `pipeline_metrics` 是否是对象。

### 第 3 步：回答、来源、指标与反馈关联

文件：`day25/python/day25_feedback_linkage.py`

第一步为了学习方便，把很多数据放在一张表中。第三步改成规范化关系：

```text
answer_runs
├── 1:N answer_sources
└── 1:N linked_feedback
```

三张表通过 `request_id` 关联。

`answer_runs` 保存一次回答和执行指标；`answer_sources` 保存该回答使用的证据；`linked_feedback` 只保存用户评价本身。

外键：

```sql
FOREIGN KEY (request_id)
REFERENCES answer_runs(request_id)
ON DELETE CASCADE
```

作用：

- 没有回答不能提交关联反馈；
- 删除回答时自动删除关联来源和反馈；
- 避免孤儿数据。

事务：

```python
with connection:
    保存回答
    删除旧来源
    保存本次完整来源集合
```

回答和来源必须一起成功或一起失败。

### 第 4 步：反馈转候选评测集

文件：`day25/python/day25_feedback_to_evalset.py`

输出：`day25/datasets/day25_feedback_eval_candidates.jsonl`

核心原则：

> 用户反馈是弱标签，不是经过验证的标准答案。

点踩可能代表事实错误，也可能只是回答太慢、太长、风格不合适或用户误操作。

因此分类为：

```text
up           → candidate_pass
down/neutral → needs_review
```

稳定样本编号：

```text
SHA-256(dataset_version | request_id | user_id)
```

可复现导出还依赖：

- 固定 SQL 排序；
- 固定数据集版本；
- JSON `sort_keys=True`；
- 整个文件的 SHA-256 指纹。

### 第 5 步：离线回放与版本对比

文件：`day25/python/day25_offline_replay.py`

默认只回放 `needs_review` 样本：

```bash
python day25/python/day25_offline_replay.py
```

只检查数据流：

```bash
python day25/python/day25_offline_replay.py --dry-run
```

回放过程：

```text
旧线上回答作为 baseline
相同问题 + 相同证据 + 反馈改进要求
→ 候选 v2 重新生成回答
→ PairwiseEvaluation 成对比较
→ baseline / candidate / tie
```

成对评估检查：

- 正确性；
- 相关性；
- 完整性；
- 引用可核验性；
- 历史反馈是否解决。

单个 `needs_review` 样本包含两次模型调用：候选生成一次、成对评估一次。

模型评估依然可能有偏差。正式评测应扩大金标集、随机交换回答顺序、重复评估并进行人工抽检。

### 第 6 步：在线反馈闭环 Web RAG

文件：

- `day25/python/day25_final_feedback_rag.py`
- `day25/html/day25_final.html`
- `day25/javascript/day25_final.js`
- `day25/css/day25_final.css`

最终服务复用 Day24 已验证的 RAG 流水线，通过 `RecordingWebSocket` 观察事件，不复制核心算法。

最重要的时序：

```text
Day24流水线准备发送done
→ RecordingWebSocket拦截done
→ 先保存AnswerRun和SourceTrace
→ 再把done发送给浏览器
→ 浏览器才显示反馈按钮
```

这样避免：浏览器已经收到答案并立即评价，但服务端还没保存回答。

浏览器提交的反馈只有：

```json
{
  "request_id": "req-...",
  "user_id": "user-xiaolin",
  "rating": "down",
  "reason": "没有解释清楚"
}
```

回答、来源、版本、Token 和耗时由服务端根据 `request_id` 获取，不能相信浏览器回传这些审计数据。

### 第 7 步：笔记与代码解读

当前笔记总结概念；`Day25-最终代码详细解读.md` 逐模块解释最终项目。

---

## 三、三种 ID 的区别

### `request_id`

标识一次回答，是回答、来源和反馈之间最关键的关联键。

### `session_id`

标识一段会话。同一个会话可以包含多个 `request_id`。

### `user_id`

标识用户。它与 `request_id` 共同限制同一用户对一条回答只有一条当前评价。

关系：

```text
user_id
└── session_id
    ├── request_id-1
    ├── request_id-2
    └── request_id-3
```

---

## 四、为什么客户端不能提交全部追踪数据

浏览器数据可以被用户修改。如果客户端提交：

```json
{
  "model_calls": 0,
  "total_tokens": 0,
  "quality_passed": true
}
```

服务端不能把这些值当作事实。

安全边界应该是：

```text
客户端可信范围：用户自己的评价和原因
服务端可信范围：回答、引用、模型版本、审计结果、成本和耗时
```

---

## 五、反馈不是训练数据

原始反馈进入正式训练或评测前至少需要：

1. 清理隐私和敏感信息；
2. 去除重复与恶意反馈；
3. 区分内容质量与产品体验问题；
4. 人工确认错误类型；
5. 编写明确的期望行为或金标答案；
6. 固定数据集版本；
7. 防止评测集泄漏进提示词或训练集。

---

## 六、今天建立的完整数据流

```text
用户问题
→ WebSocket ask
→ Day24优化RAG
→ RRF + Reranker + 结构化回答 + 审计
→ RecordingWebSocket
→ answer_runs + answer_sources
→ 浏览器显示回答和评价按钮
→ POST /feedback
→ linked_feedback
→ JSONL候选评测集
→ 人工复核
→ 候选版本离线回放
→ PairwiseEvaluation
→ 版本报告
```

---

## 七、生产环境仍需补充的内容

- 使用 Day15 的认证令牌获取 `user_id`，不能由输入框自报身份；
- 管理员与普通用户分离反馈查询权限；
- 对反馈接口限流，防止批量刷评价；
- 记录反馈修改历史，而不仅保存当前状态；
- 对评价原因做隐私清理；
- 用 PostgreSQL 等数据库支持多实例；
- 建立数据保留和删除策略；
- 对离线实验做版本审批与灰度发布；
- 使用真实输入、输出 Token 分项指标，而不是只保存聚合值。

---

## 八、Day25 核心结论

1. 可观测不等于知道用户满意度，反馈必须与具体回答关联。
2. `request_id` 是连接回答、证据、指标和反馈的主线。
3. 客户端只能提交用户意见，审计数据必须以服务端记录为准。
4. 用户反馈是弱标签，需要复核后才能成为评测金标。
5. 离线回放应保持问题和证据一致，比较才有意义。
6. 模型评估可以辅助筛选，不能完全代替人工验收。
7. 最终目标不是收集更多反馈，而是形成可验证的持续改进闭环。
