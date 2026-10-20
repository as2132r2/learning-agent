# LangChain / LangGraph Day 5 学习笔记

## 1. 今天的主题：RAG

RAG 是 Retrieval-Augmented Generation 的缩写，中文通常称为“检索增强生成”。

核心思想：

```text
大模型回答前
先从外部知识库找到相关资料
再让模型根据资料回答
```

完整流程：

```text
原始资料
→ Document
→ 文本切分
→ Embedding
→ 向量数据库
→ Retriever
→ 用户问题
→ 检索相关资料
→ context + question
→ LLM
→ 基于资料的回答
```

RAG 的两个组成部分：

```text
Retrieval：检索资料
Generation：生成答案
```

---

## 2. Document：RAG 的基本数据单位

练习文件：`day5_document_basic.py`

创建一个 Document：

```python
document = Document(
    page_content="文档正文",
    metadata={
        "source": "资料来源",
        "title": "标题",
        "page": 1
    },
    id="document-001"
)
```

三个主要字段：

```text
page_content: str
→ 文档正文
→ 用于切分、生成向量和提供给模型

metadata: dict
→ 来源、标题、页码、分类等附加信息
→ 用于来源展示、过滤和追踪

id: str | None
→ 文档的可选唯一编号
```

多个文档通常使用：

```python
documents: list[Document]
```

重要结论：正文负责承载知识，metadata 负责描述和追踪知识。

---

## 3. 文本切分

练习文件：`day5_text_splitter.py`

长文档通常不能直接作为一个检索单元，需要切成多个 chunk。

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=15,
    add_start_index=True,
    separators=["\n\n", "\n", "。", "；", "，", ""]
)
```

主要参数：

```text
chunk_size
→ 每个文本块期望的最大长度

chunk_overlap
→ 相邻文本块尝试保留的重复内容
→ 不保证每次严格重复指定数量

add_start_index
→ 在 metadata 中加入原文起始位置

separators
→ 优先按照段落、换行、句号、分号等边界切分
```

调用：

```python
chunks = text_splitter.split_documents(
    [original_document]
)
```

数据类型：

```text
输入：list[Document]
输出：list[Document]
```

切分后的 Document 会复制原文档的 metadata，并可增加 `start_index`。

文本块太大：

```text
可能包含太多不相关信息
检索精度下降
占用更多模型上下文
```

文本块太小：

```text
可能割裂完整语义
缺少回答所需上下文
文本块数量增加
```

---

## 4. Embedding：文本变成向量

练习文件：`day5_embedding_basic.py`

Embedding 的作用：

```text
文本 str
→ Embedding 模型
→ list[float]
```

文档批量向量化：

```python
embed_documents(
    texts: list[str]
) -> list[list[float]]
```

问题向量化：

```python
embed_query(
    text: str
) -> list[float]
```

`DeterministicFakeEmbedding`：

```text
优点：不联网、不收费、相同文本结果稳定
缺点：不理解语义，只适合测试数据流程
```

它可能把“什么是 RAG？”错误地匹配到 LangChain，因为数字只是稳定的测试向量。

---

## 5. 余弦相似度

练习文件：`day5_similarity_basic.py`

余弦相似度比较两个向量的方向：

```text
接近 1：方向相似
接近 0：方向关系较弱
接近 -1：方向相反
```

基本公式：

```text
点积 /（向量 A 的长度 × 向量 B 的长度）
```

计算过程：

```python
dot_product = np.dot(array_a, array_b)
norm_a = np.linalg.norm(array_a)
norm_b = np.linalg.norm(array_b)
similarity = dot_product / (norm_a * norm_b)
```

手动检索：

```text
问题向量
→ 和每个文档向量计算相似度
→ 保存分数
→ 从高到低排序
→ 取前 k 个文档
```

重要区别：相似度算法只负责比较向量；Embedding 模型决定向量是否真正表达语义。

---

## 6. InMemoryVectorStore

练习文件：`day5_vector_store.py`

向量数据库负责：

```text
保存向量
保存正文和 metadata
为问题生成向量
计算相似度
排序并返回前 k 个结果
```

建立数据库：

```python
vector_store = InMemoryVectorStore(
    embedding=embedding_model
)

stored_ids = vector_store.add_documents(
    documents=chunks
)
```

带分数检索：

```python
results = vector_store.similarity_search_with_score(
    query=question,
    k=3
)
```

返回类型：

```python
list[tuple[Document, float]]
```

只返回 Document：

```python
documents = vector_store.similarity_search(
    query=question,
    k=2
)
```

返回：

```python
list[Document]
```

限制：程序退出后，内存向量库中的数据消失。

---

## 7. Retriever：统一检索接口

练习文件：`day5_retriever_basic.py`

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)
```

调用：

```python
retrieved_documents = retriever.invoke(question)
```

数据类型：

```text
输入：str
输出：list[Document]
```

Retriever 是 Runnable，因此也支持：

```python
retriever.batch(list_of_questions)
```

VectorStore 与 Retriever 的区别：

```text
VectorStore
→ 保存向量并实现具体搜索

Retriever
→ 对外提供统一的“问题 → Document”接口
```

RAG 链只依赖 Retriever，不需要关心底层具体使用 SQLite、Chroma、FAISS 还是其他数据库。

---

## 8. 真实中文 Embedding

练习文件：`day5_real_embedding.py`

使用模型：

```text
BAAI/bge-small-zh-v1.5
```

特点：

```text
本机 CPU 运行
理解中文语义关系
首次下载，后续读取缓存
输出 512 维向量
不消耗 DeepSeek API Token
```

模型缓存：

```text
.fastembed_cache/
```

FastEmbed 与 LangChain 使用不同接口，因此编写适配器：

```python
class LocalChineseEmbeddings(Embeddings):
    def embed_documents(self, texts): ...
    def embed_query(self, text): ...
```

对象关系：

```text
LocalChineseEmbeddings
└── self.model
    └── FastEmbed TextEmbedding
        └── BGE 中文模型
```

真实检索结果：

```text
检索增强生成 → RAG
开发操作系统 → C语言
图结构工作流 → LangGraph
```

---

## 9. 第一条完整 RAG 链

练习文件：`day5_rag_basic.py`

核心链：

```python
rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

两条并行数据流：

```text
问题
├─→ Retriever → Documents → format_documents → context
└─→ RunnablePassthrough → 原问题
```

合并：

```text
{
    "context": "格式化资料",
    "question": "用户问题"
}
→ ChatPromptTemplate
→ DeepSeek
→ StrOutputParser
→ 最终文本
```

`RunnablePassthrough()` 的作用是原样传递输入。

`format_documents()` 的作用：

```text
list[Document]
→ 带来源、主题和正文的 context str
```

---

## 10. 相关性门槛与资料不足保护

练习文件：`day5_rag_guard.py`

向量数据库总能返回“最接近”的文档，但最接近不等于真正相关。

因此加入阈值：

```python
relevant_results = [
    (document, score)
    for document, score in candidate_results
    if score >= RELEVANCE_THRESHOLD
]
```

资料不足：

```text
没有结果通过门槛
→ Python 固定回答“根据现有资料无法回答”
→ 不调用 DeepSeek
```

资料足够：

```text
通过门槛的 Document
→ 生成 context
→ 调用 DeepSeek
→ Python 添加真实来源
```

来源应从实际使用的 metadata 提取：

```python
source = document.metadata["source"]
```

不要完全依赖模型自己生成来源，否则可能引用不存在的文件。

阈值不是通用常数，真实项目需要使用测试问题评估确定。

---

## 11. SQLiteVec 持久化向量库

练习文件：`day5_vector_store_persistent.py`

数据库文件：

```text
day5_vector_store.db
```

核心对象：

```python
vector_store = SQLiteVec(
    table="knowledge_chunks",
    connection=connection,
    embedding=embedding_model
)
```

主要数据表：

```text
knowledge_chunks
→ 正文、metadata、序列化向量

knowledge_chunks_vec
→ sqlite-vec 向量索引
```

首次建库判断：

```sql
SELECT COUNT(*) AS total
FROM knowledge_chunks
```

```text
数量为 0 → 生成并写入向量
数量大于 0 → 跳过写入，直接检索
```

程序关闭：

```python
connection.close()
```

只会关闭当前连接，不会删除数据库文件。

### 分数方向差异

```text
InMemoryVectorStore
→ 相似度越大越相关

SQLiteVec
→ 距离越小越相关
```

使用新的向量数据库前，必须确认返回值究竟是 similarity 还是 distance。

当前“数据库非空就跳过”的逻辑只适合入门。真实项目需要根据稳定文档 ID 实现新增、更新和删除同步。

---

## 12. 对话式 RAG

练习文件：`day5_conversational_rag.py`

追问经常依赖历史：

```text
第一轮：请介绍 RAG。
第二轮：它会先做什么？
```

直接检索“它会先做什么”通常无法确定“它”是谁。

因此先改写问题：

```text
历史消息 + 当前追问
→ LLM 问题改写
→ “RAG 会先做什么？”
→ Retriever
```

LangGraph 状态：

```python
class ConversationalRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    standalone_question: str
    context: str
    sources: list[str]
```

三个节点：

```text
rewrite_question
→ 把追问改成独立问题

retrieve
→ 使用独立问题检索

generate_answer
→ 根据 context 生成回答
```

对话检查点保存在：

```text
day5_conversational_rag.db
```

相同 `thread_id` 恢复旧对话，不同 `thread_id` 是独立会话。

### 缩写标准化

测试发现小型中文模型对短缩写 `RAG` 的检索不稳定：

```text
RAG
→ 可能没有匹配到 RAG 文档

检索增强生成
→ 能稳定匹配 RAG 文档
```

因此在检索前执行：

```text
RAG → RAG（检索增强生成）
```

这是 query normalization（查询标准化）的简单示例。

---

## 13. RAG 工具 Agent

最终代码：`day5_rag_tool_agent.py`

可用工具：

```text
search_knowledge
→ 检索本地知识库

add
→ 加法

multiply
→ 乘法
```

主 Agent 循环：

```text
用户消息
→ assistant
→ AIMessage
→ 是否包含 tool_calls？
├─ 否 → 最终回答
└─ 是 → ToolNode
         → 执行工具
         → ToolMessage
         → assistant
         → 可能继续调用工具
```

知识工具内部：

```text
检索问题
→ 缩写标准化
→ BGE 问题向量
→ SQLiteVec 距离查询
→ 距离门槛过滤
→ 返回正文和来源
```

计算示例：

```text
(20+4)*3
→ add(20, 4)
→ ToolMessage(24)
→ multiply(24, 3)
→ ToolMessage(72)
→ 最终回答
```

消息历史保存在：

```text
day5_rag_tool_agent.db
```

因此后续追问“刚才的结果是多少”可以从历史得到 72。

### 工具路由的重要限制

测试中模型曾直接回答“法国首都是巴黎”，没有检索知识库。

原因是旧提示词只要求编程问题调用知识工具。修改后要求事实问题先检索。

但提示词约束仍不是绝对安全边界：

```text
模型决定是否调用工具
→ 属于概率性判断
```

如果业务要求“任何事实都绝不能绕过知识库”，需要像 Day 4 一样增加 Python 确定性路由或输出验证。

---

## 14. Day 5 的四类数据

### 原始资料

```text
txt、PDF、网页、数据库记录
```

### Document

```text
page_content + metadata + id
```

### 向量

```text
list[float]
例如 BGE 的 512 维向量
```

### 消息

```text
HumanMessage
AIMessage
ToolMessage
```

它们不要混淆：

```text
Document 保存知识
向量用于比较语义
消息保存对话和工具调用过程
SQLite 文件负责跨进程持久化
```

---

## 15. Day 5 最终数据流

### 建立知识库

```text
原始资料
→ Document
→ TextSplitter
→ chunks
→ embed_documents()
→ 文档向量
→ SQLiteVec
→ day5_vector_store.db
```

### 单轮 RAG

```text
问题
→ embed_query()
→ 问题向量
→ 向量检索
→ 相关 Document
→ context
→ Prompt
→ DeepSeek
→ 回答
```

### 对话式 RAG

```text
历史消息 + 当前追问
→ 独立问题改写
→ 查询标准化
→ 向量检索
→ context
→ 回答
→ SQLite 保存消息
```

### RAG Agent

```text
用户请求
→ 模型判断
├─ 知识问题 → search_knowledge
├─ 加法 → add
├─ 乘法 → multiply
└─ 普通对话 → 直接回答
→ ToolMessage
→ 最终回答
```

---

## 16. 今天最重要的结论

1. `Document` 同时保存正文和来源信息。
2. 长文档需要切块，chunk 大小会影响检索效果。
3. Embedding 把文本转换为固定维度向量。
4. 测试 Embedding 不理解语义，真实 RAG 需要语义模型。
5. VectorStore 负责保存、比较和排序向量。
6. Retriever 提供统一的“问题 → Document”接口。
7. RAG 是检索和生成两部分组成的流程。
8. 向量库总会返回近邻，因此需要相关性门槛。
9. 来源最好由 Python 从 metadata 中生成。
10. SQLite 向量库可以跨进程保存资料和向量。
11. 相似度通常越大越相关，距离通常越小越相关。
12. 对话式 RAG 需要先把依赖历史的追问改写成独立问题。
13. 缩写和专业术语可能需要查询标准化。
14. RAG 可以封装成工具，交给 Agent 按需调用。
15. 模型的工具选择是概率性的，高风险场景仍需 Python 强制规则。

---

## 17. 练习文件顺序

```text
day5_document_basic.py
→ Document

day5_text_splitter.py
→ 文本切分

day5_embedding_basic.py
→ 测试向量

day5_similarity_basic.py
→ 手动计算相似度

day5_vector_store.py
→ 内存向量数据库

day5_retriever_basic.py
→ Retriever

day5_real_embedding.py
→ 真实中文 Embedding

day5_rag_basic.py
→ 第一条完整 RAG 链

day5_rag_guard.py
→ 相关性门槛和可靠来源

day5_vector_store_persistent.py
→ SQLite 持久化向量库

day5_conversational_rag.py
→ 对话式 RAG

day5_rag_tool_agent.py
→ RAG 与工具 Agent 整合
```

---

## 18. 下一阶段预览

Day 6 可以进入“真实文件知识库”：

```text
读取 txt / Markdown / PDF
→ Loader
→ 自动切分
→ 增量索引
→ metadata 过滤
→ 引用具体文件和页码
→ 构建自己的本地知识库问答程序
```

重点将从“代码中手写五个 Document”，升级为“自动读取目录中的真实文件”。
