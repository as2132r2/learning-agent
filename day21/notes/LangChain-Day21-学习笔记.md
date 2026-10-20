# LangChain Day21：浏览器文档上传与 Web RAG 知识库

## 1. 今天完成了什么

Day21 把浏览器、文档上传、LangChain Document、Embedding、SQLiteVec、检索守门和大模型流式回答连接成完整 Web RAG。

固定完成七步：

1. 浏览器使用 `multipart/form-data` 上传文件。
2. 文件安全校验、SHA-256去重与原子保存。
3. 加载TXT、Markdown、PDF并切分Document。
4. 真实中文Embedding与持久化增量向量索引。
5. 浏览器检索、距离阈值和来源展示。
6. 整合上传、索引、检索、引用和流式回答。
7. 整理笔记和最终调用链。

---

## 2. 最终系统结构

```text
浏览器
├── HTTP上传文档
└── WebSocket流式问答
        ↓
aiohttp服务端
├── 文件校验
├── 增量索引
├── 检索守门
└── DeepSeek流式生成
        ↓
文档处理管道
├── Loader
├── Document
└── TextSplitter
        ↓
BGE中文Embedding
        ↓
SQLiteVec向量数据库
```

---

## 3. multipart/form-data

浏览器不能把文件对象直接放进普通JSON，使用：

```javascript
const body = new FormData();
body.append("document", file);
```

随后：

```javascript
fetch("/upload", {method: "POST", body});
```

浏览器自动生成带boundary的 `multipart/form-data` 请求头，不应手动覆盖。

服务端通过：

```python
reader = await request.multipart()
field = await reader.next()
```

得到文件名、MIME和二进制内容。

---

## 4. 分块读取上传内容

```python
chunk = await field.read_chunk(64 * 1024)
```

每次读取64KB，可以在读取过程中：

```text
统计大小
更新SHA-256
写入临时文件
```

文件超过限制时可以尽早停止，而不是无限占用内存。

---

## 5. 文件安全校验

最终版允许：

```text
.txt
.md
.pdf
```

校验层次：

```text
扩展名
→ 声明MIME
→ 最大5MB
→ 文本UTF-8与NUL检查
→ PDF的%PDF-文件头
→ 空内容检查
→ SHA-256
```

扩展名和MIME都可能伪造，所以必须检查内容特征。

这仍然不是完整恶意文件扫描；生产环境还需要沙箱、病毒扫描和更严格的PDF解析限制。

---

## 6. SHA-256去重

```python
digest = hashlib.sha256(data).hexdigest()
```

相同内容得到相同哈希，不依赖客户端文件名。

最终名称：

```text
sha256-哈希前24位.扩展名
```

效果：

```text
文件名不同但内容相同
→ 识别为重复
→ 不重复保存
→ 增量索引跳过
```

---

## 7. 临时文件与原子保存

可靠保存流程：

```text
写入临时文件
→ 完成全部校验
→ os.replace(临时路径, 最终路径)
```

同一文件系统内的原子替换避免其他读取者看到半个文件。

校验失败时在 `finally` 中清理临时文件。

---

## 8. 文档加载器

```text
TXT/Markdown → TextLoader
PDF          → PyPDFLoader
```

PDF通常一页生成一个Document，因此三个输入文件可能产生四个或更多原始Document。

`PyPDFLoader`页码从0开始，程序对用户展示时加1。

---

## 9. Document结构

每个Document包含：

```python
Document(
    page_content="正文",
    metadata={...},
    id="可选编号",
)
```

统一metadata：

```text
source
title
file_type
page
document_index
start_index（切分后）
chunk_id（切分后）
```

metadata用于来源引用、过滤、调试和增量清理。

---

## 10. 文本切分

```python
RecursiveCharacterTextSplitter(
    chunk_size=180,
    chunk_overlap=30,
    add_start_index=True,
)
```

优先按段落、换行和中文标点寻找边界。

`chunk_overlap`保留相邻块的少量上下文，减少边界语义丢失。

---

## 11. 稳定chunk_id

输入：

```text
source + page + start_index + page_content
```

经过SHA-256生成：

```text
chunk-13bb58d59e5e6030cd2c
```

正文或位置变化后ID改变，为增量索引和清理提供稳定业务标识。

---

## 12. Embedding适配器

FastEmbed产生NumPy向量，LangChain要求：

```python
embed_documents(list[str]) -> list[list[float]]
embed_query(str) -> list[float]
```

`LocalChineseEmbeddings`负责 `.tolist()` 转换。

模型：

```text
BAAI/bge-small-zh-v1.5
```

模型在本地CPU运行，缓存位于 `.fastembed_cache`。

---

## 13. 向量数据库

SQLiteVec同时保存：

```text
Document正文
metadata
向量
```

查询过程：

```text
用户问题
→ 问题向量
→ 与文本块向量计算距离
→ 从小到大排序
→ Top-K Document + distance
```

---

## 14. 增量索引清单

`knowledge_files`：

```text
source
file_hash
indexed_at
```

`knowledge_chunk_manifest`：

```text
chunk_id
vector_rowid
source
```

用于判断：

```text
新增文件
修改文件
未变化文件
删除文件
```

未变化文件不重新加载、不切分、不生成Embedding。

---

## 15. 修改和删除文件

修改文件：

```text
文件哈希变化
→ 删除该source旧向量
→ 重新加载和切分
→ 生成新Embedding
→ 写入新清单
```

删除文件：

```text
磁盘source消失
→ 删除旧向量、正文和清单
→ 不再重建
```

---

## 16. 距离阈值守门

当前SQLiteVec距离越小通常越相关。

```python
allowed = best_distance <= 1.05
```

实际测试：

```text
什么是RAG？          0.9949 → 允许
今天北京天气怎么样？ 1.1617 → 拒绝
```

拒绝时不调用大模型，避免模型使用外部常识猜测。

---

## 17. 阈值的边界

阈值只能判断距离是否足够近，不能保证Top-1一定正确。

本次知识库真实出现：无关上传文本在“什么是RAG？”中排到相关文本之前。

后续提升方向：

```text
关键词+向量混合检索
Reranker重排
文档分类和metadata过滤
更好的Embedding
真实评测集
动态阈值
```

---

## 18. 为什么先展示检索结果

浏览器先收到 `retrieval`：

```text
正文
来源
PDF页码
chunk_id
距离
阈值结果
```

用户可以区分：

```text
资料找错了
还是模型基于正确资料答错了
```

这对调试RAG非常重要。

---

## 19. RAG上下文格式

```text
[资料1｜source，第1页]
正文……

[资料2｜source]
正文……

用户问题：……
```

系统提示词要求：

```text
只根据资料回答
关键结论使用[资料N]引用
资料不足时明确说明
不使用外部常识补充
```

---

## 20. index_lock

上传后会修改向量数据库，检索会读取向量数据库。

两者共用：

```python
index_lock = asyncio.Lock()
```

```text
同步索引中 → 检索等待
检索中     → 新同步等待
```

避免读取到旧向量已删除、新向量尚未写完的中间状态。

---

## 21. model_semaphore

```python
model_semaphore = asyncio.Semaphore(2)
```

最多两个RAG回答同时调用DeepSeek。

检索被拒绝的问题不会占用模型许可。

---

## 22. 后台RAG Task与取消

`ask`创建：

```python
asyncio.create_task(run_rag(...))
```

主WebSocket循环继续接收 `cancel`。

取消流程：

```text
cancel + request_id
→ active_task.cancel()
→ CancelledError
→ cancelled + partial_answer
→ release Semaphore
→ WebSocket保持连接
```

---

## 23. 两条最终链路

上传链路：

```text
File → multipart → 校验 → SHA-256 → 原子保存
→ sync → Document → chunks → Embedding → SQLiteVec
```

问答链路：

```text
question → WebSocket → Retrieval → threshold
├── reject → 不调用LLM
└── allow → context → DeepSeek stream → token → DOM + sources
```

---

## 24. 最终数据类型流动

```text
浏览器File
→ multipart bytes
→ 磁盘Path
→ list[Document]
→ list[Document chunks]
→ list[list[float]]
→ SQLiteVec

浏览器question string
→ Python str
→ query vector
→ list[tuple[Document, float]]
→ context str
→ list[BaseMessage]
→ AIMessageChunk
→ WebSocket JSON
→ JavaScript object
→ DOM textContent
```

---

## 25. 最终运行方式

```bash
python day21/python/day21_vector_index.py sync
python day21/python/day21_final_server.py
```

访问：

```text
http://127.0.0.1:8028
```

测试上传、相关问题、无关问题和取消。

---

## 26. 今天全部代码

```text
第1步：day21_upload_basic_server.py + HTML
第2步：day21_upload_validation_server.py + HTML
第3步：day21_document_pipeline.py
第4步：day21_vector_index.py
第5步：day21_browser_retrieval_server.py + HTML
第6步：day21_final_server.py + HTML + JavaScript + CSS
```

---

## 27. 当前最终版边界

1. MIME和文件头检查不能代替病毒扫描。
2. 上传接口尚未接入用户认证和知识库权限。
3. 索引锁是单进程内存锁，不适用于多进程部署。
4. 大文件目前仍会组合到内存中。
5. `langchain-community`出现迁移警告，后续应迁移独立集成包。
6. 当前没有Reranker和混合检索。
7. 当前没有为不同用户隔离独立知识库。
8. 生产环境需要任务队列异步索引、配额、审计和监控。

---

## 28. 最重要的结论

1. 上传成功不等于可以直接建立索引，必须先校验。
2. SHA-256既可去重，也可辅助增量判断。
3. Document正文用于检索，metadata用于来源和管理。
4. 稳定chunk_id让增量替换和删除成为可能。
5. 未变化文件必须跳过Embedding，节省时间和资源。
6. Top-1不是事实，检索结果必须可观察和评测。
7. 阈值能过滤明显无关问题，但不能替代重排。
8. RAG回答必须携带来源，让用户能够回查原文。
9. 索引写入和检索读取需要一致性协调。
10. 严格RAG应在资料不足时拒绝，而不是依赖模型常识猜测。

---

## 29. 自测题

1. 为什么不能只检查文件扩展名？
2. 为什么最终磁盘文件名不使用客户端原始文件名？
3. 原子保存解决什么问题？
4. PDF为什么可能产生多个Document？
5. `chunk_overlap`有什么作用？
6. 文件哈希和chunk_id分别解决什么问题？
7. 未变化文件为什么应跳过Embedding？
8. 为什么通过距离阈值仍可能检索错误？
9. `index_lock`与`model_semaphore`有什么区别？
10. 为什么最终回答需要引用来源？
