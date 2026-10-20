# LangChain Day23：RAG引用审计、质量评估与受控修复

## 1. 今天解决的问题

Day22解决了“检索到的资料是否相关”，Day23继续解决：

```text
模型生成的回答是否真的忠于资料？
每条事实是否有明确引用？
引用的资料是否真的支持该事实？
回答是否切题、完整？
不合格回答能否安全修复？
```

固定完成七步：

1. 生成结构化的结论—引用回答。
2. 解析引用并计算覆盖率。
3. 检查结论是否真的被引用正文支持。
4. 综合评估忠实度、相关性与完整性。
5. 不合格回答受控修复一次。
6. 整合最终可审计Web RAG。
7. 整理笔记与最终代码调用链。

---

## 2. 最终系统结构

```text
用户问题
  ↓
混合检索 + RRF + Reranker
  ↓
达到95分的证据
  ↓
结构化回答
  ├── answer
  └── claims[]
       ├── claim
       └── citation_numbers[]
  ↓
CitationAudit
  ↓
Groundedness Evaluator
  ↓
Relevance + Completeness Evaluator
  ↓
质量门控
  ├── 通过 → 返回
  └── 失败 → 修复一次 → 全量复审
                         ├── 通过 → 返回
                         └── 失败 → 拒绝
```

---

## 3. 为什么自然语言引用不够

普通回答：

```text
RAG是检索增强生成。[资料1]
```

人能阅读，但程序不容易可靠回答：

```text
有几条独立事实？
每条事实是否有引用？
哪些引用属于哪条事实？
引用编号是否越界？
显示时是否丢失引用？
```

所以Day23使用结构化回答。

---

## 4. StructuredRAGAnswer

```python
class CitedClaim(BaseModel):
    claim: str
    citation_numbers: list[int]

class StructuredRAGAnswer(BaseModel):
    can_answer: bool
    answer: str
    claims: list[CitedClaim]
```

示例：

```python
StructuredRAGAnswer(
    can_answer=True,
    answer="RAG是检索增强生成……",
    claims=[
        CitedClaim(
            claim="RAG的中文名称是检索增强生成。",
            citation_numbers=[1],
        )
    ],
)
```

`answer`面向用户，`claims`面向审计系统。后续质量检查应以`claims`为事实单元。

---

## 5. 稳定证据编号

通过Reranker的资料按顺序编号：

```text
[资料1]
chunk_id: ...
来源: ...
正文: ...
```

`citation_numbers=[1]`引用的是当前证据列表中的第一项，而不是数据库永久ID。

引用编号便于模型生成和用户阅读；`chunk_id`用于数据库追踪和审计记录。两者职责不同。

---

## 6. CitationAudit

确定性检查不调用模型，包括：

```text
can_answer与claims是否一致
结论是否为空
每条结论是否有引用
引用是否重复
引用是否越界
结构化引用与渲染文本是否一致
哪些资料被使用
哪些资料未被使用
```

渲染文本引用通过正则解析：

```python
re.compile(r"\[资料(\d+)\]")
```

覆盖率：

```text
带引用结论数 ÷ 结论总数
```

引用编号存在只证明关系结构合法，不证明正文支持结论。

---

## 7. 结构化数据与渲染文本对照

结构化层：

```python
claim.citation_numbers == [1, 2]
```

最终文本：

```text
结论……[资料1][资料2]
```

程序比较两边的编号和出现次数。如果页面格式化时漏掉一个引用，就不能通过审计。

---

## 8. Groundedness

Groundedness回答：

> 已经说出来的每条结论，是否能从它实际引用的正文中得到？

三种状态：

```text
fully_supported：结论全部被直接支持
partially_supported：只支持一部分，或结论扩大范围
unsupported：无关、矛盾或完全没有支持
```

严格通过要求：

```python
verdict == "fully_supported"
and support_score >= 80
```

严格支持率：

```text
通过严格标准的结论数 ÷ 结论总数
```

---

## 9. 为什么只给评估器看被引用的资料

假设资料2支持结论，但模型错误引用资料1。

如果评估器能看到全部资料，它可能用资料2替模型兜底，错误判定“有支持”。

Day23按每条claim的`citation_numbers`取正文：

```python
evidence[citation_number - 1]
```

所以评估的是“引用是否支持结论”，不是“知识库某处是否存在支持”。

---

## 10. 忠实度与完整性的区别

忠实度：

```text
说出来的内容有没有证据？
```

完整性：

```text
资料能回答且问题要求的重要内容，有没有漏掉？
```

例如问题要求定义和流程，回答只给定义：

```text
定义有证据 → 忠实度可以100%
流程被遗漏 → 完整性不足
```

高忠实度不等于高质量完整回答。

---

## 11. AnswerQualityResponse

```python
class AnswerQualityResponse(BaseModel):
    relevance_score: int
    completeness_score: int
    addressed_points: list[str]
    missing_points: list[str]
    irrelevant_points: list[str]
    explanation: str
```

相关性判断是否直接回应问题；完整性只要求覆盖“资料支持且问题要求”的要点，不能强迫模型补充资料外知识。

---

## 12. 最短板综合分

```python
overall_score = min(
    citation_coverage_rate * 100,
    groundedness_rate * 100,
    relevance_score,
    completeness_score,
)
```

不使用平均值，因为：

```text
100、100、100、60
平均值=90，看似优秀
最短板=60，暴露回答严重遗漏
```

高分指标不能抵消一项致命缺陷。

---

## 13. 受控修复

失败时不是重复同一个请求，而是发送：

```text
用户问题
原始合格证据
当前回答
引用错误
不支持结论
遗漏要点
偏题内容
```

修复模型只能根据原证据修正，不能引入新外部知识。

修复后必须重新运行：

```text
CitationAudit
Groundedness
Relevance
Completeness
```

不能因为它叫“修复稿”就自动信任。

---

## 14. 为什么只修复一次

```python
MAX_REPAIR_ATTEMPTS = 1
```

限制可以防止：

```text
无限循环
费用失控
延迟不可预测
多次改写逐渐偏离证据
```

一次修复后仍失败，系统拒绝返回。

---

## 15. AnswerVersion

每一版都保留：

```text
版本号
版本标签
结构化回答
引用审计
Groundedness报告
质量评估
最终报告
```

初稿不会被修复稿覆盖，因此能够比较质量变化，也方便生产审计和问题复现。

---

## 16. 最终WebSocket事件

```text
connected
retrieving
rrf_results
pipeline_queued
reranked
├── retrieval_rejected
└── drafting
      ↓
   auditing(version=1)
      ↓
   version_assessed
      ├── done
      └── repairing
            ↓
         auditing(version=2)
            ↓
         version_assessed
            ├── done
            └── quality_rejected
```

异常和取消：

```text
cancelled
error
busy
cancel_not_found
```

---

## 17. 为什么最终答案不逐Token展示

结构化回答必须完整解析后才能运行引用和质量审计。如果边生成边把Token当最终答案展示，用户可能先看到随后被判定不合格的内容。

Day23改为流式展示“处理阶段”，最终只展示通过门控的完整回答。

这是一种产品取舍：

```text
更低的感知延迟 ↔ 展示前质量保证
```

---

## 18. 模型调用成本

初稿直接通过：

```text
Reranker                 1次
结构化初稿               1次
Groundedness             1次
相关性与完整性           1次
合计                     4次
```

执行一次修复：

```text
前面4次
修复                     1次
修复稿Groundedness       1次
修复稿相关性与完整性     1次
合计                     7次
```

质量提升不是免费的，需要综合考虑成本、延迟和风险等级。

---

## 19. LLM-as-a-Judge的局限

Groundedness和完整性由模型评估，仍可能误判，并不构成数学证明。

生产系统还需要：

```text
人工标注评测集
不同模型交叉评估
评估提示词版本管理
分数校准
高风险领域人工复核
日志与抽样检查
```

确定性检查适合格式、编号、范围与状态一致性；语义检查需要模型，但必须持续评测。

---

## 20. 隐私边界

本地执行：

```text
文档保存与切分
Embedding
SQLiteVec
FTS5
RRF
引用编号和覆盖率检查
```

外部DeepSeek API：

```text
Reranker
结构化回答
Groundedness评估
完整性评估
可能的回答修复
```

候选资料会多次发送给外部模型。敏感系统应考虑本地模型、脱敏、权限过滤、审计和数据保留策略。

---

## 21. 文件导航

```text
day23/python/day23_structured_citations.py
day23/python/day23_citation_audit.py
day23/python/day23_claim_groundedness.py
day23/python/day23_answer_quality.py
day23/python/day23_answer_repair.py
day23/python/day23_final_audited_rag.py
day23/html/day23_final.html
day23/javascript/day23_final.js
day23/css/day23_final.css
```

---

## 22. 复习问题

1. 为什么自然语言中的`[资料1]`不够？
2. `answer`与`claims`分别面向谁？
3. 引用覆盖率与Groundedness有什么区别？
4. 为什么评估结论时只能看它实际引用的资料？
5. 高忠实度为什么不代表回答完整？
6. 为什么综合分采用最短板而不是平均值？
7. 修复与普通重试有什么区别？
8. 修复稿为什么必须重新执行全部审计？
9. 为什么限制修复次数？
10. LLM-as-a-Judge有哪些局限？

---

## 23. 一句话总结

Day23把“带引用的回答”升级为可以检查引用结构、验证证据支持、评估完整性、受控修复并在失败时拒绝交付的可审计RAG流水线。
