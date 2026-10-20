# LangChain Day20：SQLite 持久化、多会话与幂等写入

## 1. 今天完成了什么

Day20 解决了 Day19 的主要限制：服务端重启后，内存中的会话和聊天记录会消失。

今天固定完成了七步：

1. SQLite 建表、事务写入、关闭连接后重新读取。
2. 使用 Repository 隔离业务逻辑和 SQL。
3. WebSocket 从数据库加载历史，并在回答成功后写回。
4. 一个客户端创建、列出、切换多个聊天会话。
5. WAL、`busy_timeout`、`BEGIN IMMEDIATE` 与 `request_id` 幂等。
6. 整合最终浏览器聊天应用。
7. 整理完整笔记和最终代码调用链。

最终文件：

```text
day20/
├── python/day20_final_repository.py
├── python/day20_final_server.py
├── html/day20_final.html
├── javascript/day20_final.js
├── css/day20_final.css
├── database/day20_final.db
└── notes/LangChain-Day20-学习笔记.md
```

---

## 2. 从内存状态到数据库

Day19 使用：

```python
sessions: dict[str, SessionState] = {}
```

它只能在当前 Python 进程中存在：

```text
WebSocket 重连 → 可以恢复
Python 服务重启 → 全部消失
```

Day20 使用 SQLite 文件：

```text
day20/database/day20_final.db
```

现在：

```text
WebSocket 重连 → 从数据库恢复
浏览器刷新 → 从数据库恢复
Python 服务重启 → 从数据库恢复
```

---

## 3. 最终系统分层

```text
HTML + CSS
→ 页面结构和样式

JavaScript
→ WebSocket、DOM、多会话、重连和按钮状态

aiohttp 服务端
→ 协议校验、Task、取消、限流和模型流

FinalRepository
→ SQL、事务、幂等和对象转换

SQLite
→ sessions + turns 持久化
```

分层的目的不是增加代码，而是让每层只负责一种问题。

---

## 4. SQLite 数据库文件

SQLite 不需要单独启动数据库服务器。

Python 直接连接文件：

```python
sqlite3.connect("day20_final.db")
```

一个数据库文件内部可以包含：

```text
表
索引
约束
数据
事务日志
```

适合本地应用、学习项目、单机服务和中小规模数据。

---

## 5. sessions 表

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

一行代表一个聊天窗口。

| 字段 | 作用 |
|---|---|
| `session_id` | 对话窗口编号 |
| `client_id` | 所属浏览器客户端 |
| `title` | 会话标题 |
| `created_at` | 创建时间 |
| `updated_at` | 最后更新时间 |

一个 `client_id` 可以拥有多个 `session_id`。

---

## 6. turns 表

```sql
CREATE TABLE turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id)
        REFERENCES sessions(session_id)
        ON DELETE CASCADE
);
```

一行代表一轮完整问答：

```text
question + answer
```

`request_id UNIQUE` 保证同一个请求最多保存一次。

`ON DELETE CASCADE` 表示删除会话时，它的问答记录也随之删除。

---

## 7. 为什么最终版按 turn 保存

前面的 `chat_messages` 是一行一条消息：

```text
user消息一行
assistant消息一行
```

最终版使用一行一轮：

```text
question和answer在同一行
```

优点：

1. `request_id` 可以直接约束整轮问答。
2. 不会出现只写入用户问题、没有助手回答的半轮数据。
3. 幂等查询和重放更简单。

缺点是对工具消息、多角色消息的扩展不如逐消息模型灵活。

---

## 8. Repository 模式

业务代码不再直接写：

```python
connection.execute("SELECT ...")
```

而是调用：

```python
repository.list_sessions(client_id)
repository.load_turns(session_id)
repository.save_turn(...)
```

完整关系：

```text
WebSocket业务代码
→ FinalRepository方法
→ 参数化SQL
→ SQLite
→ sqlite3.Row
→ SessionRecord / TurnRecord
→ WebSocket业务代码
```

---

## 9. SessionRecord 与 TurnRecord

Repository 返回冻结的 dataclass：

```python
@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    client_id: str
    title: str
    created_at: str
    updated_at: str
```

```python
@dataclass(frozen=True)
class TurnRecord:
    turn_id: int
    request_id: str
    session_id: str
    question: str
    answer: str
    created_at: str
```

`frozen=True` 表示查询结果对象创建后不能直接修改字段。

数据库行通过 `_session()` 和 `_turn()` 转成领域对象。

---

## 10. 参数化 SQL

正确写法：

```python
connection.execute(
    "SELECT * FROM sessions WHERE session_id = ?",
    (session_id,),
)
```

`?` 是参数占位符。

数据不会被当成 SQL 语法拼接，有助于防止 SQL 注入，并正确处理引号等字符。

单元素元组必须写成：

```python
(session_id,)
```

---

## 11. 事务

事务提供四个重要性质：

```text
一组操作全部成功
或者全部失败回滚
```

最终保存一轮问答时：

```text
BEGIN IMMEDIATE
→ 查询request_id
→ INSERT turns
→ UPDATE sessions.updated_at
→ COMMIT
```

发生异常：

```text
ROLLBACK
```

因此不会出现问答写入成功但会话更新时间没有更新的半完成状态。

---

## 12. WAL

初始化时执行：

```sql
PRAGMA journal_mode = WAL;
```

WAL 是 Write-Ahead Logging。

它把新写入先记录到日志文件，让读取和写入更适合并发进行。

但 SQLite 同一时刻仍然只有一个写事务，所以多个写入需要排队。

---

## 13. busy_timeout

```sql
PRAGMA busy_timeout = 10000;
```

表示数据库写锁被占用时最多等待10秒，而不是立即抛出：

```text
database is locked
```

它用于应对短时间锁竞争，不能解决无限阻塞或不合理的长事务。

---

## 14. BEGIN IMMEDIATE

```python
connection.execute("BEGIN IMMEDIATE")
```

它会在“先查询、再写入”之前取得写锁。

并发流程：

```text
线程A取得写锁
→ 查询req-1不存在
→ 插入req-1
→ commit

线程B等待写锁
→ 查询req-1已经存在
→ 返回原记录
```

避免两个线程同时看到“记录不存在”。

---

## 15. 什么是幂等

幂等意味着：

```text
同一个请求执行一次
和同一个请求执行多次
最终业务结果相同
```

浏览器可能因为网络、重连或重复点击，重复发送同一个 `request_id`。

服务端首先执行：

```python
repository.get_turn_by_request(request_id)
```

如果已经保存：

```text
直接返回duplicate事件
不再次调用模型
不产生重复费用
不重复写数据库
```

数据库还有：

```sql
UNIQUE(request_id)
```

作为最后一道保护。

---

## 16. 多会话关系

```text
client_id = browser-A
│
├── session-1：Python学习
│   ├── turn-1
│   └── turn-2
│
└── session-2：C语言学习
    ├── turn-3
    └── turn-4
```

切换会话时服务端必须检查：

```python
session.client_id == state.client_id
```

不能仅凭 `session_id` 就返回历史。

---

## 17. 浏览器 client_id

JavaScript 把客户端编号保存到：

```javascript
localStorage
```

随后连接：

```text
/ws?client_id=browser-xxx
```

服务端根据该编号查询 `sessions` 表。

注意：`client_id` 只是教学版会话隔离标识，不是安全的登录凭证。

生产环境必须使用 Day15 学习过的身份认证和授权。

---

## 18. 同步 SQLite 与异步服务

Python 标准库 `sqlite3` 是同步接口。

直接在 aiohttp 协程中执行可能阻塞事件循环。

最终服务端使用：

```python
await asyncio.to_thread(
    repository.load_turns,
    session_id,
)
```

数据流：

```text
事件循环
→ 把SQLite方法交给工作线程
→ 继续处理其他连接
→ 工作线程完成
→ 当前协程恢复
```

---

## 19. ConnectionState

```python
@dataclass
class ConnectionState:
    client_id: str
    current_session_id: str | None
    active_request_id: str | None
    active_task: asyncio.Task[None] | None
    send_lock: asyncio.Lock
```

它只保存当前连接的临时运行状态。

真正需要跨重启保存的会话和问答都在 SQLite 中。

---

## 20. Pydantic 命令协议

客户端消息统一为：

```json
{
  "version": "1.0",
  "type": "ask",
  "request_id": "req-xxx",
  "payload": {
    "question": "什么是SQLite？"
  }
}
```

允许的命令：

```text
create_session
switch_session
list_sessions
ask
cancel
status
ping
```

`Command.model_validate_json()` 同时完成 JSON 解析和字段校验。

---

## 21. 后台模型 Task

收到 `ask` 后：

```python
state.active_task = asyncio.create_task(
    run_model(...)
)
```

主 WebSocket 接收循环继续运行，因此模型生成期间仍可处理：

```text
cancel
status
ping
```

如果直接 `await run_model()`，当前连接在模型完成前无法接收控制命令。

---

## 22. Semaphore

```python
semaphore = asyncio.Semaphore(2)
```

整个服务最多同时执行两个真实模型流。

```text
请求A → 获得许可
请求B → 获得许可
请求C → queued并等待
```

成功、失败或取消都必须在 `finally` 中归还许可。

---

## 23. 事务式模型保存

模型生成期间：

```python
parts.append(text)
```

浏览器能实时看到部分回答，但数据库尚未写入。

只有模型完整结束后：

```python
repository.save_turn(
    request_id,
    session_id,
    question,
    complete_answer,
)
```

取消或断线时：

```text
部分回答可以显示
但不写入turns表
```

---

## 24. 最终服务端调用链

```text
浏览器连接/ws
→ websocket_handler()
→ list_sessions(client_id)
→ 创建ConnectionState
→ connected + sessions + turns
→ 收到TEXT
→ handle()
→ Pydantic校验
→ ask分支
→ create_task(run_model())
→ 查询request_id是否已存在
→ queued
→ semaphore.acquire()
→ load_turns()
→ 转换成LangChain消息
→ llm.astream()
→ token × N
→ save_turn()
→ done
→ semaphore.release()
```

---

## 25. 最终浏览器调用链

```text
加载HTML
→ 加载CSS和JavaScript
→ 从localStorage读取client_id
→ connect()
→ new WebSocket()
→ connected
→ renderSessions()
→ renderTurns()
→ 用户创建或切换会话
→ 用户提交问题
→ 生成request_id
→ ask
→ queued/start/token/done
→ 更新同一个DOM回答元素
→ 刷新会话列表
```

---

## 26. 完整数据类型流动

```text
浏览器question: string
→ JavaScript object
→ JSON string
→ WebSocket TEXT
→ Python raw: str
→ Pydantic Command
→ question: str
→ Repository.load_turns()
→ list[TurnRecord]
→ list[BaseMessage]
→ llm.astream()
→ AIMessageChunk
→ chunk.content: str
→ WebSocket token JSON
→ JavaScript object
→ DOM textContent
→ complete_answer: str
→ Repository.save_turn()
→ SQLite row
→ TurnRecord
→ done JSON
```

---

## 27. 最终数据库恢复流程

```text
Python服务停止
→ 内存状态消失
→ day20_final.db仍存在
→ 重启Python服务
→ repository.initialize()
→ 表已存在，不重复创建
→ 浏览器携带原client_id重连
→ list_sessions(client_id)
→ load_turns(session_id)
→ 浏览器恢复侧边栏和历史
```

---

## 28. 最终代码运行

```bash
python day20/python/day20_final_server.py
```

浏览器打开：

```text
http://127.0.0.1:8024
```

建议测试：

1. 创建两个会话并分别提问。
2. 来回切换，验证历史隔离。
3. 刷新浏览器，验证恢复。
4. 重启服务，验证数据库持久化。
5. 取消长回答，验证残缺回答不入库。
6. 打开无痕窗口，验证不同 `client_id` 隔离。

---

## 29. 今天全部代码

```text
第1步：day20_sqlite_basic.py

第2步：
├── day20_chat_repository.py
└── day20_repository_demo.py

第3步：
├── day20_persistent_ws_server.py
└── day20_persistent_ws_client.py

第4步：
├── day20_multi_session_server.py
└── day20_multi_session_client.py

第5步：
├── day20_safe_repository.py
└── day20_idempotency_demo.py

第6步最终版：
├── day20_final_repository.py
├── day20_final_server.py
├── day20_final.html
├── day20_final.js
└── day20_final.css
```

---

## 30. 当前最终版的边界

最终版已经具备：

- SQLite 持久化
- 多客户端数据隔离
- 多聊天窗口
- 流式输出
- 自动重连
- 取消控制
- 并发限流
- WAL与写锁等待
- 请求幂等
- 事务式保存

仍需注意：

1. `client_id` 不是安全认证。
2. SQLite 适合单机服务，不适合任意规模的多机写入。
3. 当前没有删除、重命名和分页界面。
4. 当前按最近6轮加载模型历史，没有精确按 Token 裁剪。
5. 模型生成完成但数据库写入前断线时，需要更完整的后台任务策略。
6. 生产环境应使用 HTTPS/WSS、认证、日志和监控。

---

## 31. 最重要的结论

1. 浏览器状态负责用户体验，数据库状态才负责可靠恢复。
2. Repository 将业务逻辑与 SQL 解耦。
3. 一个 `client_id` 可以拥有多个独立 `session_id`。
4. 同一个 `request_id` 必须只产生一个业务结果。
5. `UNIQUE`、事务和锁共同保证并发幂等。
6. 同步 SQLite 操作应通过 `asyncio.to_thread()` 离开事件循环。
7. 模型完整成功后才提交问答，取消的部分回答不入库。
8. `Semaphore` 控制模型并发，SQLite事务控制数据一致性。
9. WAL改善读写并发，但SQLite仍只有一个写事务。
10. 服务端重启后的恢复来自数据库，而不是内存对象。

---

## 32. 自测题

1. `client_id`、`session_id`、`request_id` 分别标识什么？
2. 为什么 `request_id` 需要唯一约束？
3. WAL 能否让 SQLite 同时执行多个写事务？
4. `busy_timeout` 解决什么问题？
5. `BEGIN IMMEDIATE` 为什么适合“先查再写”？
6. 为什么取消的回答不写入数据库？
7. 为什么 SQLite 操作使用 `asyncio.to_thread()`？
8. Repository 相比到处直接写 SQL 有什么好处？
9. 为什么服务端重启后仍能恢复对话？
10. 当前 `client_id` 为什么不能代替正式登录认证？
