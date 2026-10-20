# Day22 最终代码详细解读

解读对象：`day22/python/day22_final_hybrid_rag.py`及配套浏览器文件。

## 1. 程序启动阶段

```python
PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY21_DIR = PROJECT_DIR / "day21"
DAY22_DIR = PROJECT_DIR / "day22"
```

`Path`对象用于跨平台路径拼接。最终服务复用Day21的文档目录和向量数据库，同时使用Day22的页面与FTS数据库。

```python
HOST, PORT = "127.0.0.1", 8029
RRF_CANDIDATE_COUNT = 5
RERANK_THRESHOLD = 95
MAX_CONTEXT_COUNT = 4
```

含义：

```text
只监听本机8029端口
RRF最多送5个候选给Reranker
相关性至少95分才能回答
生成上下文最多使用4段资料
```

---

## 2. 模型对象和两把并发锁

```python
llm = ChatOpenAI(..., temperature=0)
index_lock = asyncio.Lock()
model_semaphore = asyncio.Semaphore(2)
```

`llm`用于最终流式答案。Reranker模块内部会创建相同配置的结构化模型。

`index_lock`是互斥锁，同时只能有一个索引读写临界区。

`model_semaphore`是计数信号量，最多允许两条模型任务链同时执行。

---

## 3. 浏览器命令的数据类型

```python
class AskPayload(BaseModel):
    question: str | None
```

```python
class ClientCommand(BaseModel):
    version: Literal["1.0"]
    type: Literal["ask", "cancel"]
    request_id: str
    payload: AskPayload
```

浏览器必须发送：

```json
{
  "version": "1.0",
  "type": "ask",
  "request_id": "req-...",
  "payload": {"question": "什么是 RAG？"}
}
```

`Literal`拒绝未知协议版本和命令类型，`extra="forbid"`拒绝多余字段。

---

## 4. 每条连接的运行状态

```python
@dataclass
class ConnectionState:
    active_request_id: str | None
    active_task: asyncio.Task[None] | None
    send_lock: asyncio.Lock
```

一条WebSocket连接同一时间只运行一个问答Task。`send_lock`避免多个协程同时写同一WebSocket导致事件交叉。

---

## 5. 上传参数流动

```text
HTML input[type=file]
→ JavaScript File
→ FormData的document字段
→ POST /upload
→ request.multipart()
→ BodyPartReader
→ 多个bytes块
→ 完整bytes
```

服务端检查：

```text
filename后缀
Content-Type
最大5MB
非空
PDF文件头或UTF-8文本
NUL字节
```

然后：

```python
digest = hashlib.sha256(data).hexdigest()
```

以内容哈希命名，相同内容不会重复保存。

---

## 6. 双索引同步

```python
def sync_both_indexes() -> int:
    sync_vector_index()
    connection = open_fts_database()
    try:
        return rebuild_fts_index(connection)
    finally:
        connection.close()
```

返回值是FTS文本块数量，数据类型为`int`。

同步函数是阻塞函数，异步处理器使用：

```python
await asyncio.to_thread(sync_both_indexes)
```

把工作放进线程，避免阻塞aiohttp事件循环。

---

## 7. 本地候选召回

```python
def retrieve_rrf_candidates(question: str)
    -> tuple[list[RRFResult], dict[str, Any]]:
```

输入是用户问题`str`，返回：

```text
list[RRFResult]：融合后的前5个候选
dict：关键词、MATCH表达式和两路候选数量
```

内部调用：

```python
vector_results = vector_search(question)
terms, match_query, keyword_results = fts_search(question)
rrf_results = reciprocal_rank_fusion(...)
```

---

## 8. 为什么转换成dict

`RRFResult`和`RerankedResult`是Python dataclass，不能直接假设浏览器理解。

```python
rrf_to_dict(...)
reranked_to_dict(...)
```

将对象转换成只包含字符串、数字、列表和字典的JSON兼容数据，再交给`send_json()`。

---

## 9. run_rag第一阶段：RRF

```python
await send(..., "retrieving", ...)
```

通知浏览器开始检索。

```python
async with index_lock:
    rrf_results, diagnostics = await asyncio.to_thread(
        retrieve_rrf_candidates,
        question,
    )
```

加锁读取两套索引，阻塞检索在线程中运行。

随后发送：

```python
await send(..., "rrf_results", ...)
```

浏览器立即展示初排，不必等待后续模型评分。

---

## 10. run_rag第二阶段：Reranker

```python
await model_semaphore.acquire()
acquired = True
```

获得模型并发许可后：

```python
reranked_results, _ = await asyncio.to_thread(
    rerank_candidates,
    question,
    rrf_results,
)
```

`rerank_candidates`接收：

```text
question: str
candidates: list[RRFResult]
```

返回：

```text
list[RerankedResult]
RerankResponse
```

每个`RerankedResult`包含原RRF排名、RRF分数、模型相关性分数和理由。

---

## 11. 95分守门

```python
best_score = reranked_results[0].relevance_score
```

如果：

```python
best_score < RERANK_THRESHOLD
```

服务端发送`rejected`并`return`，生成模型完全不会被调用。

通过后只保留：

```python
item.relevance_score >= 95
```

因此不是“第一名通过就把其他低分资料全部带入上下文”，可以减少无关资料污染。

---

## 12. 上下文构造

```python
context_from_results(accepted_results)
```

输出：

```text
[资料1｜来源位置｜相关性95]
正文

[资料2｜来源位置｜相关性98]
正文
```

编号顺序与提示词中的`[资料1]`引用一致。

---

## 13. 最终答案生成

消息类型：

```python
list[SystemMessage | HumanMessage]
```

SystemMessage约束：

```text
只能根据参考资料回答
关键结论必须引用
资料不足不能补充外部常识
```

HumanMessage包含经过守门的上下文和用户问题。

流式调用：

```python
async for chunk in llm.astream(messages):
```

每个`chunk`是`AIMessageChunk`。字符串内容立即通过`token`事件发给浏览器，同时追加到`answer_parts: list[str]`。

结束时：

```python
"".join(answer_parts)
```

得到完整答案。

---

## 14. 异常、取消和资源释放

```python
except asyncio.CancelledError:
```

捕获取消并发送已有部分回答。

```python
except Exception as error:
```

把异常类型和消息返回前端。

```python
finally:
    if acquired:
        model_semaphore.release()
```

无论成功、拒绝、异常还是取消，都释放许可并清除当前连接状态。

注意：`asyncio.to_thread()`中的同步外部请求不能保证随外层Task立即停止；生产系统还应给模型请求设置超时，并使用支持异步取消的客户端。

---

## 15. WebSocket处理器

```python
async for raw in ws:
```

持续接收浏览器命令。

`ask`创建后台Task：

```python
asyncio.create_task(run_rag(...))
```

因此处理器可以继续接收`cancel`，不会被整个问答流程阻塞。

如果当前已有任务，新`ask`得到`busy`，防止同一连接出现两个回答同时写页面。

---

## 16. HTTP路由

```text
GET  /         HTML
GET  /app.js   JavaScript
GET  /app.css  CSS
POST /upload   文件上传
GET  /ws       WebSocket升级
```

aiohttp还会为GET资源自动生成HEAD路由，所以检查时显示9条路由。

---

## 17. JavaScript状态

```javascript
let socket = null;
let activeRequestId = null;
let answerElement = null;
let reconnectAttempt = 0;
```

`activeRequestId`把服务端事件与当前问题对应起来，也决定提问和取消按钮是否可用。

事件处理：

```text
rrf_results → 左侧展示初排指标
reranked    → 右侧展示精排分数和理由
rejected    → 显示守门拒绝原因
token       → 追加回答文本
done        → 恢复输入控件
```

页面用`textContent`显示模型和文档内容，不把它们当HTML执行，可降低XSS风险。

---

## 18. 一次完整调用的参数关系

```text
question: str
  ↓ vector_search(question)
list[dict]
  ↓ fts_search(question)
list[dict]
  ↓ reciprocal_rank_fusion(vector_results, keyword_results)
list[RRFResult]
  ↓ rerank_candidates(question, rrf_results)
list[RerankedResult]
  ↓ relevance_score >= 95
accepted_results
  ↓ context_from_results()
context: str
  ↓ llm.astream([SystemMessage, HumanMessage])
AsyncIterator[AIMessageChunk]
  ↓ WebSocket token事件
浏览器逐块显示
```

---

## 19. 启动和测试

```bash
python day22/python/day22_final_hybrid_rag.py
```

打开：

```text
http://127.0.0.1:8029
```

知识库内测试：

```text
什么是 RAG？
```

知识库外测试：

```text
法国的首都是哪里？
```

前者应通过守门并回答，后者应在Reranker之后被拒绝。
