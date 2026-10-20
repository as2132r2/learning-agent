# LangChain Day 2 学习笔记

> 适合零基础复习。学习目标：理解大模型为什么会“失忆”，掌握消息历史、会话隔离、LangGraph 状态管理、摘要压缩，以及使用 SQLite 实现程序重启后仍可恢复的持久化记忆。

## 1. 今天学到了什么

```text
模型调用默认无状态
        ↓
手动维护消息历史
        ↓
MessagesPlaceholder 插入历史
        ↓
RunnableWithMessageHistory 自动管理历史
        ↓
发现旧接口弃用警告
        ↓
LangGraph + MessagesState
        ↓
thread_id 隔离不同会话
        ↓
Checkpointer 保存与恢复状态
        ↓
限制发送给模型的历史长度
        ↓
摘要压缩 + 保留最近原文
        ↓
SQLite 实现跨程序重启的持久化记忆
```

最终程序：

```text
/path/to/learning-agent/
├── .env                         # 模型配置和 API Key
├── .venv/                       # Python 虚拟环境
├── day2_memory.py               # Day 2 最终练习代码
├── day2_memory.db               # SQLite 对话状态数据库
└── LangChain-Day2-学习笔记.md   # 本笔记
```

---

## 2. 为什么模型会“失忆”

下面是两次完全独立的调用：

```python
first_response = llm.invoke([
    HumanMessage(content="我叫小林，喜欢 Python。")
])

second_response = llm.invoke([
    HumanMessage(content="我叫什么？")
])
```

第二次调用没有包含第一次的消息，因此模型不知道用户叫小林。

```text
第一次请求：我叫小林
        ↓
模型回答

第二次请求：我叫什么？
        ↓
模型只看到当前问题
        ↓
无法知道“小林”
```

关键认识：

> 大模型接口默认是无状态的。Python 变量保存了上次返回值，不代表模型下次调用会自动收到它。

所谓“对话记忆”，本质上是程序在调用模型前，把相关历史消息再次放进输入。

---

## 3. 手动维护消息历史

最直接的方法是建立一个列表：

```python
history = [
    SystemMessage(content="你是一名友好的助手。")
]
```

每轮依次执行：

```python
history.append(
    HumanMessage(content=user_input)
)

response = llm.invoke(history)

history.append(response)
```

完整规律：

```text
保存 HumanMessage
        ↓
把完整 history 交给模型
        ↓
得到 AIMessage
        ↓
把 AIMessage 保存进 history
```

一轮正常对话会增加两条消息：

```text
HumanMessage + AIMessage = 2 条
```

如果初始列表中还有一条 `SystemMessage`，进行五轮对话后：

```text
1 + 5 × 2 = 11 条消息
```

### 3.1 为什么 `response` 可以直接加入历史

```python
response = llm.invoke(history)
```

返回的 `response` 本身就是 `AIMessage`，所以可以直接：

```python
history.append(response)
```

不需要重新包装成：

```python
AIMessage(content=response.content)
```

---

## 4. 命令行连续聊天

使用 `while True` 可以不断接收输入：

```python
while True:
    user_input = input("\n你：").strip()

    if not user_input:
        print("请输入内容，不要只按回车。")
        continue

    if user_input.lower() == "exit":
        break
```

重点：

- `input()`：等待用户输入。
- `.strip()`：删除输入前后的空格和换行。
- `if not user_input`：判断是否为空。
- `continue`：结束当前循环，重新等待输入。
- `.lower()`：把英文转为小写，使 `exit`、`EXIT`、`Exit` 都能退出。
- `break`：结束整个循环。

终端里出现 `^[` 通常表示按到了 `Esc`，程序会把它当作普通文本。它不是退出指令。

---

## 5. `MessagesPlaceholder`：在提示词中预留历史位置

提示词可以分成三个部分：

```text
固定系统规则
    +
以前的聊天历史
    +
当前用户问题
```

代码：

```python
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名简洁、友好的编程学习助手。"
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{question}"
    )
])

chain = prompt | llm
```

调用：

```python
response = chain.invoke({
    "history": history,
    "question": user_input
})
```

参数对应关系：

```text
history
   ↓
MessagesPlaceholder(variable_name="history")

user_input
   ↓
{question}
```

`MessagesPlaceholder` 不负责保存历史，只负责把传入的消息列表插入模板。

---

## 6. 从旧式历史管理过渡到 LangGraph

练习中使用过：

```python
RunnableWithMessageHistory
```

它能根据 `session_id` 自动保存 `HumanMessage` 和 `AIMessage`，但当前环境提示：

```text
RunnableWithMessageHistory is deprecated.
Use LangGraph's built-in persistence instead.
```

这不是运行错误，而是弃用警告：

- 当前代码还能运行。
- 官方推荐新项目使用 LangGraph 的状态持久化。
- 因此后续练习改用 `StateGraph` 和 Checkpointer。

---

## 7. LangGraph 的核心概念

可以把 LangGraph 理解成一个“带档案柜的工作流程”。

| 概念 | 程序中的对象 | 类比 |
|---|---|---|
| State | `State` | 当前对话档案 |
| Node | `chatbot` | 执行具体工作的人员 |
| Edge | `add_edge(...)` | 工作流路线 |
| Conditional Edge | `add_conditional_edges(...)` | 根据条件选择路线 |
| Checkpointer | `SqliteSaver` | 保存档案的管理员 |
| `thread_id` | `xiaolin` | 某一本档案的编号 |

基本流程：

```text
START
  ↓
chatbot
  ↓
条件判断
  ├── 不需要摘要 → END
  └── 需要摘要 → summarize_conversation → END
```

---

## 8. `MessagesState` 与自定义状态

LangGraph 内置的 `MessagesState` 包含：

```python
state["messages"]
```

为了额外保存摘要，可以继承它：

```python
class State(MessagesState):
    summary: str
```

最终状态可以想象为：

```python
{
    "messages": [
        HumanMessage(...),
        AIMessage(...)
    ],
    "summary": "小林正在学习C语言……"
}
```

职责分工：

```text
messages → 保存最近几轮原始消息
summary  → 保存更早对话的压缩信息
```

### 8.1 `MessagesState` 为什么不会直接覆盖消息

节点返回：

```python
return {
    "messages": [response]
}
```

对于普通字典，这看起来像覆盖；但 `MessagesState` 为消息字段配置了专门的合并规则，会根据消息 ID：

- 添加新消息；
- 更新已有消息；
- 响应 `RemoveMessage` 删除消息。

因此模型回答会被追加到历史中。

---

## 9. `chatbot` 节点

核心代码：

```python
def chatbot(state: State):
    summary = state.get("summary", "")

    system_content = """
你是一名简洁、友好的编程学习助手。
请根据用户的学习情况提供清晰、适合初学者的回答。
""".strip()

    if summary:
        system_content += f"""

以下是以前对话的摘要：
{summary}
"""

    messages = [
        SystemMessage(content=system_content),
        *state["messages"]
    ]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }
```

执行过程：

```text
读取 state.summary
        ↓
生成 SystemMessage
        ↓
加入 state.messages
        ↓
llm.invoke(messages)
        ↓
得到 AIMessage
        ↓
返回给 LangGraph 合并
```

### 9.1 `state.get("summary", "")`

如果存在摘要，就返回摘要；如果不存在，就返回空字符串，避免第一次对话时报错。

### 9.2 星号展开列表

```python
messages = [
    SystemMessage(...),
    *state["messages"]
]
```

假设：

```python
state["messages"] = [message_a, message_b]
```

展开后相当于：

```python
messages = [
    SystemMessage(...),
    message_a,
    message_b
]
```

---

## 10. 构建工作流

创建状态图：

```python
builder = StateGraph(State)
```

添加节点：

```python
builder.add_node("chatbot", chatbot)
builder.add_node(
    "summarize_conversation",
    summarize_conversation
)
```

添加固定路线：

```python
builder.add_edge(START, "chatbot")
builder.add_edge("summarize_conversation", END)
```

添加条件路线：

```python
builder.add_conditional_edges(
    "chatbot",
    should_summarize
)
```

最后编译：

```python
graph = builder.compile(
    checkpointer=checkpointer
)
```

`builder` 是流程设计，`graph` 才是可以通过 `invoke()` 执行的对象。

---

## 11. `thread_id`：会话隔离

配置：

```python
config = {
    "configurable": {
        "thread_id": "xiaolin"
    }
}
```

不同 `thread_id` 对应不同会话：

```text
Checkpointer
├── xiaolin  → 小林的消息和摘要
├── xiaoming → 小明的消息和摘要
└── teacher  → 另一份独立会话
```

判断规则：

```text
同一个数据库 + 相同 thread_id
→ 恢复并继续旧会话

同一个数据库 + 不同 thread_id
→ 使用另一份会话状态

不同数据库
→ 即使 thread_id 相同，也不是同一份记录
```

LangGraph 每次执行 `graph.invoke()` 时，都会让 Checkpointer 尝试查找该 `thread_id`：

```text
找到 → 恢复旧 State
没找到 → 从空 State 开始
```

不需要自己写 `if` 查询数据库。

---

## 12. `graph.invoke()` 的参数流动

代码：

```python
result = graph.invoke(
    {
        "messages": [
            HumanMessage(content=user_input)
        ]
    },
    config=config
)
```

第一个参数是本轮输入数据：

```python
{
    "messages": [HumanMessage(...)]
}
```

第二个参数是运行配置：

```python
config = {
    "configurable": {
        "thread_id": "xiaolin"
    }
}
```

内部过程：

```text
graph.invoke(新消息, config)
        ↓
Checkpointer 根据 thread_id 查询 SQLite
        ↓
恢复旧 messages 和 summary
        ↓
合并当前 HumanMessage
        ↓
调用 chatbot(state)
        ↓
添加新的 AIMessage
        ↓
执行条件路由
        ↓
保存最终 State
        ↓
返回 result
```

取得最新回答：

```python
latest_message = result["messages"][-1]
print(latest_message.content)
```

`[-1]` 表示列表最后一个元素。

---

## 13. 查看状态：`graph.get_state()`

```python
snapshot = graph.get_state(config)
```

这会读取指定 `thread_id` 的最新状态，但不会调用大模型。

读取字段：

```python
messages = snapshot.values.get(
    "messages",
    []
)

summary = snapshot.values.get(
    "summary",
    "暂无摘要"
)
```

区别：

```text
graph.invoke(...)     → 执行工作流，会改变状态
graph.get_state(...)  → 只查看状态，不调用模型
```

---

## 14. 为什么不能无限发送全部历史

如果每轮都向模型发送全部历史：

```text
对话越长
  ↓
输入 Token 越多
  ↓
费用和等待时间增加
  ↓
最终可能超过上下文窗口
```

最简单的限制方法：

```python
recent_messages = state["messages"][-6:]
```

它只把最近六条消息发送给模型，但状态里仍可能保存完整历史。

直接截断的缺点是：重要的早期信息可能完全消失。因此最终练习采用“摘要 + 最近原文”。

---

## 15. 条件摘要

判断函数：

```python
def should_summarize(state: State):
    if len(state["messages"]) > 6:
        return "summarize_conversation"

    return END
```

三轮对话通常产生六条消息。第四轮的实际过程：

```text
恢复前三轮：6条
        ↓
加入第四轮 HumanMessage：7条
        ↓
chatbot 生成 AIMessage：8条
        ↓
判断 8 > 6
        ↓
进入摘要节点
```

因此摘要发生在第四轮回答完成之后，不是在第四轮回答之前。

---

## 16. 生成和更新摘要

摘要节点会读取以前的摘要：

```python
old_summary = state.get("summary", "")
```

第一次摘要：

```text
当前原始对话
    ↓
生成第一份摘要
```

后续摘要：

```text
旧摘要 + 最近新对话
        ↓
生成更新后的摘要
```

摘要模型：

```python
summary_llm = llm.bind(max_tokens=250)
```

它与 `llm` 使用相同的模型配置，但调用时自动携带 `max_tokens=250`。

注意：

- `max_tokens` 限制输出 Token，不等于精确的汉字数量。
- “不超过 150 个汉字”是提示词要求，模型可能偶尔不严格遵守。
- 当前摘要仍是普通字符串，没有永久固定的字符长度。

---

## 17. `RemoveMessage` 与保留最近两轮

```python
messages_to_delete = [
    RemoveMessage(id=message.id)
    for message in state["messages"][:-4]
]
```

`[:-4]` 表示取出“除最后四条以外”的消息。

假设当前有八条：

```text
第1条  Human  第一轮问题   删除
第2条  AI     第一轮回答   删除
第3条  Human  第二轮问题   删除
第4条  AI     第二轮回答   删除
第5条  Human  第三轮问题   保留
第6条  AI     第三轮回答   保留
第7条  Human  第四轮问题   保留
第8条  AI     第四轮回答   保留
```

四条消息约等于最近两轮对话。

`RemoveMessage` 必须携带目标消息的 ID：

```python
RemoveMessage(id=message.id)
```

返回：

```python
return {
    "summary": response.content,
    "messages": messages_to_delete
}
```

LangGraph 会更新摘要，同时删除指定的旧消息。

---

## 18. `InMemorySaver` 与 `SqliteSaver`

### 18.1 临时内存

```python
checkpointer = InMemorySaver()
```

```text
程序运行中 → 有记忆
程序关闭   → 记忆消失
```

适合快速实验和单元测试。

### 18.2 SQLite 持久化

```python
connection = sqlite3.connect(
    "day2_memory.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(connection)
```

```text
程序运行中 → 状态写入数据库
程序关闭   → 数据库文件保留
再次启动   → 根据 thread_id 恢复状态
```

创建图时把保存器交给 LangGraph：

```python
graph = builder.compile(
    checkpointer=checkpointer
)
```

持久化身份由两部分共同决定：

```text
数据库文件 day2_memory.db
            +
thread_id = xiaolin
```

建议将数据库加入 `.gitignore`：

```gitignore
*.db
```

对话数据库可能包含个人信息，不应意外提交到远程仓库。

---

## 19. 第二次启动为什么还能回答

第一次运行：

```text
graph.invoke
    ↓
数据库里没有 xiaolin
    ↓
创建新状态
    ↓
对话、摘要
    ↓
保存进 day2_memory.db
```

第二次运行：

```text
graph.invoke
    ↓
在 day2_memory.db 找到 xiaolin
    ↓
恢复 summary 和 messages
    ↓
加入本轮问题
    ↓
模型根据恢复的信息回答
```

实际练习中，程序重新启动后仍正确恢复了：

```text
姓名：小林
学习内容：C语言
目标：单片机开发
时间：每天一小时
担心：指针太难
```

这说明 SQLite 持久化成功。

---

## 20. `try` / `finally` 与关闭数据库

```python
try:
    while True:
        ...
finally:
    connection.close()
```

无论程序正常退出还是中途发生异常，`finally` 都会尝试执行：

```python
connection.close()
```

这样可以及时释放 SQLite 数据库连接。

---

## 21. 最终程序的完整数据流

```text
用户输入 user_input
        ↓
HumanMessage(content=user_input)
        ↓
graph.invoke(input, config)
        ↓
SqliteSaver 根据 thread_id 查询数据库
        ↓
恢复旧 State，或创建空 State
        ↓
合并本轮 HumanMessage
        ↓
chatbot(state)
        ↓
SystemMessage + summary + messages
        ↓
llm.invoke(...)
        ↓
AIMessage 加入 State
        ↓
should_summarize(state)
        ↓
messages 是否超过6条？
   ├── 否 → 直接结束
   └── 是 → summary_llm 生成摘要
                 ↓
            RemoveMessage 删除旧原文
                 ↓
            保留最近4条消息
        ↓
SqliteSaver 写回数据库
        ↓
graph.invoke 返回 result
        ↓
result["messages"][-1].content
        ↓
打印模型最新回答
```

---

## 22. 常见问题与易错点

### 22.1 在终端直接输入 Python 导入语句

错误操作：

```bash
from langgraph.checkpoint.sqlite import SqliteSaver
```

Shell 会把 `from` 当成 Linux 命令。

正确做法：把它写进 `.py` 文件，再运行：

```bash
python day02/python/day2_memory.py
```

### 22.2 `RunnableWithMessageHistory` 弃用警告

这不是语法错误。当前版本建议改用 LangGraph 的 Checkpointer。

### 22.3 历史数量与预期不同

空输入如果没有被过滤，也会产生一轮对话。因此应在调用模型前判断：

```python
if not user_input:
    continue
```

### 22.4 改了 `thread_id` 后“失忆”

不同 `thread_id` 表示不同会话，这是正常的隔离效果。

### 22.5 换目录运行后找不到数据库

当前数据库路径是相对路径：

```python
"day2_memory.db"
```

它相对于终端当前目录。如果从另一个目录启动程序，可能打开另一个同名数据库。学习阶段应先进入项目目录：

```bash
cd /path/to/learning-agent
python day02/python/day2_memory.py
```

### 22.6 删除 `.db` 后记忆消失

SQLite 记忆就在数据库文件中。删除数据库文件会删除已保存的对话状态。

---

## 23. Day 2 自测题

### 问题 1

为什么两次独立的 `llm.invoke()` 不会自动形成记忆？

答案：因为模型接口默认无状态，第二次调用只能看到本次传入的消息。

### 问题 2

一轮正常对话通常产生几条历史消息？

答案：两条，一条 `HumanMessage` 和一条 `AIMessage`。

### 问题 3

`MessagesPlaceholder` 会自动保存历史吗？

答案：不会，它只负责把传入的消息列表插入提示词。

### 问题 4

`thread_id` 有什么作用？

答案：标识和隔离不同会话，Checkpointer 根据它查找对应状态。

### 问题 5

`graph.invoke()` 和 `graph.get_state()` 有什么区别？

答案：前者运行工作流并更新状态；后者只读取已有状态。

### 问题 6

为什么需要对历史进行摘要？

答案：避免输入 Token 无限增长，同时尽量保留早期的重要信息。

### 问题 7

`state["messages"][:-4]` 表示什么？

答案：选择除最后四条以外的消息，用于删除旧原文并保留最近两轮。

### 问题 8

程序重启后恢复记忆需要什么？

答案：使用相同数据库文件，并继续使用相同的 `thread_id`。

### 问题 9

系统消息为什么没有出现在保存的原始历史里？

答案：系统消息只在 `chatbot` 函数内部临时组合给模型，没有作为状态更新返回。

### 问题 10

`summary_llm = llm.bind(max_tokens=250)` 会创建另一个模型账号吗？

答案：不会。它基于同一个模型对象预先绑定摘要调用参数。

---

## 24. Day 2 最小记忆卡

如果时间有限，只复习下面这些：

```text
1. 大模型 API 默认无状态
   → 记忆来自程序重新传入历史

2. 一轮对话
   → HumanMessage + AIMessage

3. MessagesPlaceholder
   → 在提示词里插入消息列表，不负责保存

4. State
   → 当前工作流的数据

5. Node
   → 接收 State、完成工作、返回状态更新

6. Edge
   → 节点之间的执行路线

7. MessagesState
   → 自动合并消息并支持 RemoveMessage

8. thread_id
   → 会话编号，用于查找和隔离状态

9. Checkpointer
   → 保存和恢复 LangGraph 状态

10. InMemorySaver
    → 程序退出后记忆消失

11. SqliteSaver
    → 状态保存到 .db，程序重启后仍可恢复

12. graph.invoke(input, config)
    → 恢复状态、执行图、保存状态、返回结果

13. graph.get_state(config)
    → 只读取指定会话的状态

14. 摘要 + 最近原文
    → 在保留信息的同时控制上下文长度

15. RemoveMessage
    → 按消息 ID 删除旧原始消息
```

一句话总结 Day 2：

> 今天从“模型为什么失忆”出发，逐步实现了手动消息历史、会话隔离、LangGraph 状态管理、自动摘要和 SQLite 持久化，最终完成了一个程序重启后仍能继续对话的命令行助手。

---

## 25. Day 3 预告：Tools 与 Function Calling

Day 2 解决了“模型怎样记住上下文”，Day 3 将解决：

> 模型怎样调用 Python 函数，真正执行计算或其他操作？

模型擅长理解语言，但不应该依靠语言预测完成所有精确任务。例如：

```text
用户：“计算 23 × 47”
        ↓
模型判断应该调用 calculator
        ↓
生成工具调用参数 {a: 23, b: 47}
        ↓
Python 真正执行 calculator(23, 47)
        ↓
得到 1081
        ↓
ToolMessage 把结果交还模型
        ↓
模型组织最终回答
```

计划按以下顺序学习：

1. 什么是 Tool，以及模型为什么需要工具。
2. 使用 `@tool` 把 Python 函数定义成 LangChain 工具。
3. 使用 Pydantic 类型和说明定义可靠的工具参数。
4. 使用 `llm.bind_tools(...)` 把工具列表交给模型。
5. 查看 `AIMessage.tool_calls` 中的工具名、参数和调用 ID。
6. 在 Python 中执行模型选择的工具。
7. 使用 `ToolMessage` 把执行结果返回给模型。
8. 完成“判断工具 → 调用工具 → 返回结果 → 最终回答”的闭环。
9. 把工具节点加入 LangGraph。
10. 理解普通 Chain、Tool Calling 和 Agent 的区别。

Day 3 预计完成的小项目：

```text
一个带工具的命令行助手
├── 普通知识问题 → 模型直接回答
├── 数学问题     → 调用 calculator
└── 工具执行结果 → 返回模型生成最终回复
```

Day 3 学完后，路线将进入：

```text
Tool Calling
      ↓
Agent 自动循环
      ↓
Streaming 流式输出
      ↓
小型 Agent 项目
      ↓
RAG 文档问答
```
