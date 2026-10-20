# LangChain Day6 学习笔记

## 一、今天完成了什么

Day6 的目标是把 Day5 的演示型 RAG，升级成可以处理真实文件的本地知识库。

今天完成的完整流程：

```text
TXT / Markdown / PDF
    ↓
根据扩展名选择 Loader
    ↓
list[Document]
    ↓
统一 metadata
    ↓
RecursiveCharacterTextSplitter
    ↓
稳定 chunk_id
    ↓
文件指纹增量同步
    ↓
FastEmbed 中文向量
    ↓
SQLiteVec 持久化
    ↓
向量检索
    ↓
相关性阈值守门
    ↓
相邻 chunk 扩展
    ↓
大模型根据资料回答
    ↓
文件名和 PDF 页码引用
```

---

## 二、真实文件加载

### 1. TXT 和 Markdown

使用 `TextLoader`：

```python
loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

documents = loader.load()
```

TXT、Markdown 通常是一份文件对应一个 `Document`。

### 2. PDF

使用 `PyPDFLoader`：

```python
loader = PyPDFLoader(str(file_path))
documents = loader.load()
```

PDF 通常是一页对应一个 `Document`：

```text
一份两页 PDF
    ↓
list[Document]
    ├── 第 1 页 Document
    └── 第 2 页 Document
```

`PyPDFLoader` 的原始页码从 `0` 开始。转换为用户习惯的页码：

```python
page_number = original_page + 1
```

### 3. 根据文件类型选择 Loader

```python
suffix = file_path.suffix.lower()

if suffix in {".txt", ".md"}:
    loader = TextLoader(...)
elif suffix == ".pdf":
    loader = PyPDFLoader(...)
```

无论使用哪个 Loader，最后统一得到：

```python
list[Document]
```

---

## 三、统一 metadata

不同 Loader 产生的 metadata 结构不完全相同，所以需要统一：

```python
document.metadata = {
    "source": source,
    "file_name": file_path.name,
    "file_type": suffix,
    "title": file_path.stem,
    "page": page_number,
    "document_number": document_number
}
```

主要字段：

| 字段 | 类型 | 作用 |
|---|---|---|
| `source` | `str` | 相对于 knowledge 的文件路径 |
| `file_name` | `str` | 文件名 |
| `file_type` | `str` | `.txt`、`.md` 或 `.pdf` |
| `title` | `str` | 不含扩展名的文件名 |
| `page` | `int \| None` | PDF 页码；普通文本为 `None` |
| `start_index` | `int` | chunk 在原文或当前页中的起始位置 |
| `chunk_id` | `str` | 文本块的稳定业务编号 |

metadata 会跟随 `Document` 进入文本切分器和向量数据库，并在检索时返回。

---

## 四、文本切分与稳定 chunk_id

### 1. 文本切分

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    add_start_index=True,
    separators=["\n\n", "\n", "。", "；", "，", ""]
)

chunks = splitter.split_documents(documents)
```

`add_start_index=True` 会把文本位置写进：

```python
chunk.metadata["start_index"]
```

### 2. 稳定 ID

```python
identity_text = (
    f"source={chunk.metadata['source']}\n"
    f"start_index={chunk.metadata['start_index']}\n"
    f"content={chunk.page_content}"
)

digest = hashlib.sha256(
    identity_text.encode("utf-8")
).hexdigest()

chunk_id = f"chunk-{digest[:16]}"
```

如果来源、位置和正文都不变，生成的 ID 就不变。

正文发生变化时：

```text
旧 chunk_id 消失 → 删除旧向量
新 chunk_id 出现 → 生成新向量
```

---

## 五、文件指纹与增量索引

### 1. 文件指纹

```python
digest = hashlib.sha256()

with file_path.open("rb") as file:
    while data := file.read(64 * 1024):
        digest.update(data)

file_hash = digest.hexdigest()
```

文件内容只要变化，SHA-256 指纹通常就会变化。

### 2. 四种文件状态

```text
当前有，数据库没有             → 新增文件
当前有，数据库有，指纹不同     → 修改文件
当前有，数据库有，指纹相同     → 未变化文件
当前没有，数据库有             → 删除文件
```

处理方式：

| 状态 | 操作 |
|---|---|
| 新增 | 加载、切分、Embedding、写入 |
| 修改 | 删除旧 chunks，重新建立索引 |
| 未变化 | 跳过 Loader、切分和 Embedding |
| 删除 | 删除文件对应的全部旧 chunks |

### 3. 两张清单表

`file_manifest`：

```text
source → file_hash → indexed_at
```

负责判断文件是否发生变化。

`chunk_manifest`：

```text
chunk_id → vector_rowid → source
```

负责连接业务文本块、SQLite 数据行和来源文件。

---

## 六、SQLite Row 类型问题

必须在连接后设置：

```python
connection = sqlite3.connect(database_path)
connection.row_factory = sqlite3.Row
```

默认结果是普通元组：

```python
row[0]
row[1]
```

设置以后支持列名：

```python
row["text"]
row["metadata"]
```

SQLiteVec 内部会使用 `row["metadata"]`，所以缺少 `row_factory` 会出现：

```text
TypeError: tuple indices must be integers or slices, not str
```

---

## 七、邻近文本块扩展

向量检索可能只命中半句话：

```text
chunk 1：每个 Document 都应该保留 source、title、page 等
chunk 2：这样最终回答可以展示资料来源……
```

解决流程：

```text
向量命中核心 chunk
    ↓
使用 source + page 找到同一位置分组
    ↓
使用 start_index 找到前后顺序
    ↓
取前一个、当前、后一个 chunk
    ↓
使用 chunk_id 去重
    ↓
按原文顺序组成 context
```

`window_size=1` 表示：

```text
最多取前 1 个 + 当前块 + 后 1 个
```

为了避免跨页串联，PDF 邻居必须同时满足：

```text
source 相同
page 相同
```

---

## 八、相关性守门

SQLiteVec 当前返回距离：

```text
距离越小 → 越相关
距离越大 → 越不相关
```

判断逻辑：

```python
if best_distance > RELEVANCE_THRESHOLD:
    # 拒绝回答，不调用 LLM
else:
    # 扩展邻居并调用 LLM
```

通过当前评测集得到：

```python
RELEVANCE_THRESHOLD = 0.95
```

完整分支：

```text
问题 → 最佳检索结果 → 距离判断

距离 > 0.95
    → 根据当前知识库资料无法回答
    → 不调用聊天模型

距离 <= 0.95
    → 邻居扩展
    → 构造 context
    → 调用聊天模型
```

---

## 九、检索评测

评测数据：

```python
@dataclass
class EvaluationCase:
    question: str
    expected_sources: set[str] | None
```

知识库内问题可以有多个正确来源：

```python
expected_sources={
    "markdown/day6_langgraph.md",
    "txt/day6_langchain.txt"
}
```

知识库外问题：

```python
expected_sources=None
```

今天的小型评测结果：

```text
相关问题 Top-1 来源准确率：6/6 = 100%
阈值 0.95 守门准确率：10/10 = 100%
```

这里的 100% 只适用于当前 10 道题，不能代表真实环境中的所有问题。

---

## 十、引用来源和页码

来源位置格式化：

```python
def format_location(document: Document) -> str:
    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")

    if page is None:
        return source

    return f"{source}，第 {page} 页"
```

模型收到的资料：

```text
[资料 1 | pdf/day6_rag_test.pdf，第 2 页]
资料正文……
```

回答后，程序再从 metadata 中确定性提取来源：

```text
程序核验的参考位置：
- pdf/day6_rag_test.pdf，第 2 页
```

不要完全依赖模型自由生成文件名和页码。

---

## 十一、最终程序命令

最终整合文件：`day6_knowledge_cli.py`

### 同步知识库

```bash
python day06/python/day6_knowledge_cli.py sync
```

### 查看原始检索结果

```bash
python day06/python/day6_knowledge_cli.py search "什么是 RAG？"
```

### 执行完整 RAG 问答

```bash
python day06/python/day6_knowledge_cli.py ask "为什么要保留来源信息？"
```

### 查看数据库状态

```bash
python day06/python/day6_knowledge_cli.py status
```

---

## 十二、今天的文件

| 文件 | 作用 |
|---|---|
| `day6_text_loader.py` | 加载单个 TXT |
| `day6_directory_loader.py` | 加载真实目录 |
| `day6_chunk_ids.py` | 文本切分和稳定 ID |
| `day6_incremental_index.py` | chunk 级增量索引 |
| `day6_incremental_update.py` | 修改文件的安全演示 |
| `day6_file_fingerprint.py` | 文件级指纹检查 |
| `day6_multi_format_loader.py` | TXT、MD、PDF 多格式加载 |
| `day6_neighbor_retrieval.py` | 邻近文本块扩展 |
| `day6_neighbor_rag.py` | 邻居扩展后调用 LLM |
| `day6_relevance_guard.py` | 阈值守门 |
| `day6_retrieval_evaluation.py` | 检索和阈值评测 |
| `day6_knowledge_cli.py` | 最终整合程序 |
| `knowledge/day6_rag_test.pdf` | 两页中文 PDF 测试资料 |

---

## 十三、Day6 核心结论

1. Loader 把不同文件统一转换为 `Document`。
2. metadata 决定来源是否可追溯。
3. 稳定 ID 和文件指纹让增量更新成为可能。
4. 未变化文件不应重复加载、切分和生成向量。
5. PDF 通常按页生成 `Document`，页码要保留下来。
6. 向量检索命中的是文本块，不一定是完整语义段落。
7. 邻居扩展可以补全被切开的上下文。
8. 相关性阈值必须通过评测数据确定。
9. 无关问题应该在调用 LLM 前被拦截。
10. 引用位置应从 metadata 确定性生成。

---

## 十四、Day7 预告

Day7 可以进入 Agentic RAG：让 LangGraph 根据问题和检索结果自主选择流程。

预计流程：

```text
用户问题
    ↓
问题分类/路由
    ├── 普通聊天
    ├── 知识库检索
    ├── 计算工具
    └── 无法处理
         ↓
检索质量判断
    ├── 资料充分 → 生成答案
    ├── 资料不足 → 改写问题后重试
    └── 仍然不足 → 拒绝回答
```

这会把 Day3 的工具、Day4 的安全控制、Day5 的 RAG 和 Day6 的真实知识库组合到同一个 LangGraph 工作流中。

