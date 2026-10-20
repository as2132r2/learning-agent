# LangChain Day22：混合检索、RRF与Reranker

## 1. 今天解决的问题

Day21只使用向量检索。问题“什么是 RAG？”的向量Top-1却是无关上传资料，正确资料只排第2名。

Day22固定完成七步：

1. SQLite FTS5/BM25关键词检索。
2. 并排比较向量与关键词结果。
3. 使用RRF融合两路排名。
4. 使用LLM Reranker二次排序。
5. 建立评测集并选择守门阈值。
6. 整合最终混合检索Web RAG。
7. 整理笔记与最终代码调用链。

最终效果：

```text
纯向量Top-1来源准确率：75%
RRF Top-1来源准确率：100%
Reranker Top-1来源准确率：100%
当前6题守门准确率：100%
```

---

## 2. 最终系统结构

```text
浏览器问题
    ↓ WebSocket
向量语义检索 ─────┐
                   ├→ RRF融合 → 前5个候选
FTS5关键词检索 ───┘                 ↓
                              LLM Reranker
                                    ↓
                              相关性95分守门
                              ├─ 不通过：拒绝
                              └─ 通过：生成回答
                                    ↓
                              WebSocket流式输出
```

上传链路：

```text
浏览器上传
→ 文件安全校验
→ SHA-256内容寻址保存
→ index_lock
→ 同步SQLiteVec向量索引
→ 重建FTS5关键词索引
```

---

## 3. FTS5与BM25

FTS5是SQLite的全文检索模块。Day22建立虚拟表：

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    source UNINDEXED,
    location UNINDEXED,
    tokenize = 'trigram'
)
```

只有`content`参与正文检索。`UNINDEXED`字段可以随结果返回，但不会建立全文索引。

`trigram`把文本拆成连续三字符片段，更适合没有空格分词的中文基础检索。

查询：

```sql
SELECT ..., bm25(chunks_fts) AS bm25_score
FROM chunks_fts
WHERE chunks_fts MATCH ?
ORDER BY bm25_score ASC
LIMIT ?
```

当前SQLite FTS5实现中，BM25越小排名越靠前，并且可能是负数。它只适合在同一次关键词检索内部排序。

---

## 4. 两种检索的互补关系

向量检索：

```text
优点：理解同义表达和语义相似关系
缺点：Embedding能力有限时可能把“领域相近”误判为“能回答问题”
```

关键词检索：

```text
优点：专有名词、缩写、精确短语命中稳定
缺点：用户换一种表达但没有共同词时容易漏检
```

本次“什么是 RAG？”结果：

```text
向量第1：无关上传文档
向量第2：正确RAG资料
关键词第1：正确RAG资料
```

因此不能简单只选其中一种。

---

## 5. 为什么不能直接相加原始分数

向量距离与BM25的含义和范围完全不同：

```text
向量距离：越小越相关，当前结果约为0.9～1.2
BM25分数：越小越相关，当前可能为负数
```

错误示例：

```python
score = vector_distance + bm25_score
```

这相当于把不同单位的数字直接相加，没有统一统计意义。

---

## 6. RRF倒数排名融合

RRF全称`Reciprocal Rank Fusion`，只使用每一路的排名：

```text
RRF分数 = Σ 1 / (k + rank)
```

本次使用：

```python
RRF_K = 60
```

正确RAG资料：

```text
向量第2：1/(60+2) = 0.016129
关键词第1：1/(60+1) = 0.016393
总分：0.032522
```

错误资料只在向量中排第1：

```text
1/(60+1) + 0 = 0.016393
```

因此正确资料升到融合第1名。

`k`是平滑常数，不是Top-K数量。它让前几名仍有顺序差异，但不会让第1名压倒其他路线共同认可的候选。

---

## 7. LLM Reranker

RRF仍然没有阅读正文来回答“资料是否真正回答问题”。Reranker接收：

```text
用户问题
+ RRF前5个候选的chunk_id、位置和正文
```

一次模型调用批量返回：

```python
class RerankItem(BaseModel):
    chunk_id: str
    relevance_score: int  # 0～100
    reason: str
```

使用结构化输出：

```python
llm.with_structured_output(
    RerankResponse,
    method="function_calling",
)
```

程序继续校验：

```text
模型有没有虚构chunk_id
有没有重复chunk_id
有没有遗漏候选
分数是否在0～100之间
```

结构化输出降低格式错误，但不等于模型数据天然可信。

---

## 8. RRF与Reranker的职责

```text
RRF：召回阶段
目标：利用不同检索方法的互补性，尽量别漏掉正确资料

Reranker：精排阶段
目标：阅读问题和正文，把真正能回答问题的资料排到前面
```

如果直接让Reranker读取整个知识库，会增加Token、延迟和费用。先召回少量候选再精排更加合理。

---

## 9. 评测集

评测题包含：

```text
4道知识库内问题：应该命中day21_sample.txt
2道知识库外问题：应该拒绝回答
```

两个指标：

```text
Top-1来源准确率：第一名是否来自人工标注的正确文件
守门准确率：该回答的问题是否允许，该拒绝的问题是否拒绝
```

实测：

```text
纯向量：3/4 = 75%
RRF：4/4 = 100%
Reranker：4/4 = 100%
```

说明RRF已经修复了本次向量误排，Reranker进一步给出了可解释的相关性分数。

---

## 10. 阈值选择

评测程序测试50～95：

```text
最高分 >= 阈值：允许回答
最高分 < 阈值：拒绝回答
```

当前分布：

```text
相关问题：95、95、100、95
无关问题：0、0
```

所有候选阈值均达到100%，并列时程序选择更高、更保守的95。

95只适用于当前小型评测集，不是永久真理。资料、模型、提示词或真实用户表达变化后，都要重新评测。

---

## 11. 最终WebSocket事件

```text
connected
retrieving
rrf_results
rerank_queued
reranked
├─ rejected
└─ start → token... → done
```

取消时返回：

```text
cancelled
```

异常事件：

```text
error
busy
cancel_not_found
```

前端根据事件分别展示初排、精排、阈值、流式回答和错误。

---

## 12. 并发与一致性

`index_lock`保护：

```text
上传后同步两套索引
检索期间读取索引
```

避免向量库已经更新而FTS库仍是旧数据。

`model_semaphore`限制：

```text
Reranker调用
+ 同请求后续答案生成
```

当前最多两条模型链路并发，防止大量请求同时压向模型服务。

---

## 13. 隐私边界

Embedding在本地执行，但当前Reranker与答案生成使用DeepSeek API：

```text
本地：文档保存、切分、Embedding、SQLiteVec、FTS5、RRF
外部API：候选精排、最终答案生成
```

因此候选正文会发送给外部模型。敏感资料应使用本地Reranker和本地生成模型，或者先做权限检查、脱敏与审计。

---

## 14. 文件导航

```text
day22/python/day22_fts_basic.py
day22/python/day22_retrieval_compare.py
day22/python/day22_rrf_fusion.py
day22/python/day22_llm_reranker.py
day22/python/day22_retrieval_evaluation.py
day22/python/day22_final_hybrid_rag.py
day22/html/day22_final.html
day22/javascript/day22_final.js
day22/css/day22_final.css
day22/database/day22_fts.db
```

---

## 15. 复习问题

1. 为什么向量距离和BM25不能直接相加？
2. RRF为什么能融合不同量纲的检索结果？
3. RRF中的`k=60`是什么，而不是什么？
4. RRF和Reranker分别负责什么？
5. 为什么Reranker只读取召回候选而不是整个知识库？
6. 结构化输出之后为什么仍要验证chunk_id？
7. 为什么阈值必须通过评测集选择？
8. 上传后为什么必须同步更新两套索引？
9. 哪些数据留在本地，哪些数据会发送给外部模型？

---

## 16. 一句话总结

Day22把单路向量检索升级为“关键词与语义双路召回、RRF融合、LLM精排、评测阈值守门”的可观察Web RAG检索链路。
