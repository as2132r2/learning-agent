# LangChain Day 3 学习笔记

> 学习目标：理解 Tool、Function Calling、ToolMessage 和 Agent 工具循环，掌握多工具选择、错误处理、SQLite 记忆、摘要压缩，以及节点级和 Token 级流式输出。

## 1. 今天的学习路线

```text
普通 Python 函数
        ↓
@tool 包装为 StructuredTool
        ↓
llm.bind_tools() 把工具说明交给模型
        ↓
AIMessage.tool_calls 产生调用请求
        ↓
Python 执行真实工具
        ↓
ToolMessage 返回工具结果
        ↓
模型生成最终自然语言回答
        ↓
ToolNode 自动执行工具
        ↓
LangGraph 构建 Agent 循环
        ↓
错误处理、循环保护、持久化、摘要和流式输出
```

练习文件：

```text
day3_tool_basic.py       @tool 基础
day3_tool_calling.py     查看 tool_calls
day3_tool_execution.py   手动执行工具闭环
day3_multiple_tools.py   多工具通用分发
day3_tool_graph.py       ToolNode 与 Agent 循环
day3_tool_agent.py       最终综合版本
```

---

## 2. Tool 是什么

Tool 是经过规范描述、能被模型选择并由 Python 执行的函数。

```python
from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""

    return a * b
```

`@tool` 会读取：

- 函数名 `multiply`；
- Docstring 工具说明；
- 参数名 `a`、`b`；
- 参数类型 `int`；
- 返回类型 `int`。

包装结果不再是普通函数，而是：

```text
langchain_core.tools.structured.StructuredTool
```

工具对象主要包含：

```text
name         工具名称
description  工具用途说明
args         参数说明
args_schema  参数 JSON Schema
invoke()     执行工具
```

手动执行：

```python
result = multiply.invoke({
    "a": 23,
    "b": 47
})
```

---

## 3. 类型标注与 JSON Schema

定义：

```python
def multiply(a: int, b: int) -> int:
```

含义：

```text
a: int  → a 应该是整数
b: int  → b 应该是整数
-> int  → 返回值是整数
```

LangChain 会生成类似 Schema：

```python
{
    "type": "object",
    "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"}
    },
    "required": ["a", "b"]
}
```

模型根据这份 Schema 生成结构化参数，而不是随意拼接函数代码。

Docstring 同样重要：模型主要根据工具名、说明和参数 Schema 判断什么时候使用工具。

---

## 4. `bind_tools()` 做了什么

```python
llm_with_tools = llm.bind_tools([
    multiply
])
```

它不会执行工具，只把工具说明交给模型。

```text
StructuredTool
├── name
├── description
└── args_schema
        ↓
llm.bind_tools(...)
        ↓
模型知道“可以请求哪些工具”
```

需要区分：

```text
bind_tools()       告诉模型有哪些工具
AIMessage.tool_calls 模型提出调用请求
tool.invoke(args)  Python 真正执行工具
```

---

## 5. `AIMessage.tool_calls`

用户询问：

```text
请计算23乘以47。
```

模型可能返回：

```python
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "multiply",
            "args": {
                "a": 23,
                "b": 47
            },
            "id": "call_00_...",
            "type": "tool_call"
        }
    ]
)
```

此时：

```text
content = ""
finish_reason = "tool_calls"
```

表示模型暂停生成最终答案，等待程序执行工具。它尚未得到 `1081`。

工具调用字段：

| 字段 | 含义 |
|---|---|
| `name` | 要调用的工具名 |
| `args` | 传给工具的参数字典 |
| `id` | 本次调用的唯一编号 |
| `type` | 调用记录类型 |

---

## 6. 手动工具执行闭环

第一步，让模型生成请求：

```python
response = llm_with_tools.invoke(messages)
messages.append(response)
```

第二步，读取并执行工具：

```python
tool_call = response.tool_calls[0]

tool_result = multiply.invoke(
    tool_call["args"]
)
```

第三步，构造工具结果消息：

```python
tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call["id"]
)

messages.append(tool_message)
```

第四步，让模型生成最终回答：

```python
final_response = llm_with_tools.invoke(
    messages
)
```

完整顺序：

```text
SystemMessage
HumanMessage
AIMessage（包含 tool_calls）
ToolMessage（包含工具结果）
AIMessage（最终自然语言回答）
```

`AIMessage.tool_calls[].id` 必须和 `ToolMessage.tool_call_id` 对应。它们表示“这份结果属于哪次工具请求”。

---

## 7. 多工具与工具分发

工具列表：

```python
tools = [
    add,
    subtract,
    multiply,
    divide
]
```

工具列表交给模型：

```python
llm_with_tools = llm.bind_tools(tools)
```

手动分发时可以构造字典：

```python
tools_by_name = {
    current_tool.name: current_tool
    for current_tool in tools
}
```

生成：

```python
{
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide
}
```

模型返回名称后：

```python
selected_tool = tools_by_name[
    tool_call["name"]
]

tool_result = selected_tool.invoke(
    tool_call["args"]
)
```

---

## 8. `ToolNode` 自动执行工具

```python
tool_node = ToolNode(tools)
```

`ToolNode` 自动完成：

```text
读取 AIMessage.tool_calls
        ↓
按照 name 查找工具
        ↓
把 args 传给工具
        ↓
执行 tool.invoke()
        ↓
创建对应 ToolMessage
        ↓
返回消息状态更新
```

它替代了手写的工具字典、参数读取、循环执行和 `ToolMessage` 构造。

---

## 9. `tools_condition` 与路由

```python
route = tools_condition(state)
```

它不直接判断用户问题是不是数学题，而是检查最后一条 `AIMessage`：

```text
tool_calls 非空 → 返回 "tools"
tool_calls 为空 → 返回 END
```

职责分工：

```text
模型              决定是否提出工具调用
tools_condition   检查模型的决定并选择路线
ToolNode           真正执行模型请求的工具
```

自定义包装函数可以在保留原判断的同时打印或增加更多路线：

```python
def route_after_chatbot(state):
    route = tools_condition(state)

    if route == "tools":
        return "tools"

    return END
```

---

## 10. LangGraph Agent 循环

建图：

```python
builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
builder.add_edge("tools", "chatbot")
```

最关键的循环边：

```python
builder.add_edge("tools", "chatbot")
```

它让模型拿到工具结果后再次思考。

```text
START
  ↓
chatbot
  ↓
有 tool_calls？
  ├── 否 → END
  └── 是 → tools
               ↓
          执行并返回 ToolMessage
               ↓
            chatbot
```

这就是最小 Agent 循环：

```text
理解问题 → 选择行动 → 执行动作 → 观察结果 → 再次判断
```

---

## 11. 连续工具调用

问题：

```text
(20+4)*3
```

实际产生六条消息：

```text
1. HumanMessage：提出完整问题
2. AIMessage：请求 add(20, 4)
3. ToolMessage：返回 24.0
4. AIMessage：请求 multiply(24, 3)
5. ToolMessage：返回 72.0
6. AIMessage：最终回答 72
```

虽然有六条消息，但只是一轮用户对话。因此统计对话轮数时，不能简单用全部消息数量。

---

## 12. 工具错误处理

工具可以主动抛出异常：

```python
if b == 0:
    raise ValueError("除数不能为零")
```

错误处理函数：

```python
def handle_tool_error(error: Exception) -> str:
    return (
        "工具执行失败。"
        f"异常类型：{type(error).__name__}；"
        f"异常信息：{str(error)}。"
    )
```

交给 `ToolNode`：

```python
tool_node = ToolNode(
    tools,
    handle_tool_errors=handle_tool_error
)
```

数据流：

```text
divide(a=10, b=0)
        ↓
raise ValueError
        ↓
ToolNode 捕获异常
        ↓
handle_tool_error(error)
        ↓
错误 ToolMessage
        ↓
chatbot 向用户解释
```

工具错误和程序错误需要区分：这里将可预期的工具异常转成模型能够阅读的观察结果。

---

## 13. 防止 Agent 无限循环

配置：

```python
config = {
    "configurable": {
        "thread_id": "xiaolin-day3-summary"
    },
    "recursion_limit": 10
}
```

`recursion_limit` 与 `configurable` 同级。它限制的是单次图执行的节点步数，不等于工具调用次数。

例如两步计算可能经过：

```text
chatbot → tools → chatbot → tools → chatbot
```

大约是五个节点步骤。

捕获超限异常：

```python
except GraphRecursionError:
    print("本轮执行步骤过多")
```

---

## 14. SQLite 持久化记忆

```python
connection = sqlite3.connect(
    database_path,
    check_same_thread=False
)

checkpointer = SqliteSaver(connection)

graph = builder.compile(
    checkpointer=checkpointer
)
```

会话配置：

```python
config = {
    "configurable": {
        "thread_id": "xiaolin-day3-summary"
    }
}
```

每次执行时：

```text
graph 调用开始
      ↓
SqliteSaver 根据 thread_id 恢复旧 State
      ↓
执行模型和工具节点
      ↓
SqliteSaver 保存最终 State
```

同一数据库与相同 `thread_id` 会继续旧对话；改变其中任意一个都会得到另一份会话状态。

---

## 15. 工具对话的摘要压缩

最终状态：

```python
class State(MessagesState):
    summary: str
```

路由优先级：

```text
1. AIMessage 有 tool_calls → 先执行工具
2. 没有工具调用且对话过长 → 生成摘要
3. 其他情况 → END
```

必须优先处理工具。如果模型已经提出工具请求，却先进入摘要节点，会破坏正常工具调用顺序。

### 为什么按 `HumanMessage` 统计轮数

```python
sum(
    1
    for message in state["messages"]
    if isinstance(message, HumanMessage)
)
```

一次复杂计算可能产生多个模型消息和工具消息，但仍只有一个 `HumanMessage`。

### 为什么从倒数第二个 HumanMessage 开始保留

工具对话不能随意保留“最后四条”，否则可能保留 `ToolMessage`，却删除它对应的 `AIMessage.tool_calls`。

正确思路：找到倒数第二个用户问题的位置，从那里开始保留全部消息。这能保留最近两轮的完整工具调用链。

---

## 16. 输入清理

终端可能产生 `^?` 等控制字符。最终代码使用：

```python
def clean_terminal_input(raw_text: str) -> str:
    cleaned_text = "".join(
        character
        for character in raw_text
        if character.isprintable()
    )

    return cleaned_text.replace("^?", "").strip()
```

数据流：

```text
input() 原始字符串
       ↓
过滤不可打印字符
       ↓
删除可见的 ^?
       ↓
strip() 删除首尾空白
       ↓
HumanMessage
```

---

## 17. 两种流式输出

### 节点级流式输出

```python
graph.stream(
    input_data,
    config=config,
    stream_mode="updates"
)
```

每个节点完成后返回该节点产生的状态更新：

```text
chatbot 更新
tools 更新
chatbot 更新
summarize_conversation 更新
```

适合学习、调试和观察 Agent 路线。

### Token / 消息片段流式输出

```python
graph.stream(
    input_data,
    config=config,
    stream_mode="messages"
)
```

返回：

```python
(message_chunk, metadata)
```

其中：

- `message_chunk.content`：本次生成的文本片段；
- `metadata["langgraph_node"]`：片段来自哪个节点。

摘要节点同样会调用模型，所以最终代码只显示：

```python
metadata.get("langgraph_node") == "chatbot"
```

从而避免把内部摘要过程输出给用户。

---

## 18. `invoke`、节点流和 Token 流的区别

```text
graph.invoke(...)
→ 等整张图完成
→ 返回最终状态

graph.stream(..., stream_mode="updates")
→ 每完成一个节点返回一次更新

graph.stream(..., stream_mode="messages")
→ 模型生成消息片段时持续返回
```

三者不是三套不同的 Agent；它们只是观察同一张图的不同方式。

---

## 19. 最终项目完整数据流

```text
终端输入 str
    ↓
clean_terminal_input()
    ↓
HumanMessage(content=user_input)
    ↓
graph.stream(input, config, stream_mode="messages")
    ↓
SqliteSaver 根据 thread_id 恢复 State
    ↓
chatbot(state)
    ↓
SystemMessage + summary + messages
    ↓
llm_with_tools.invoke(...)
    ↓
AIMessage
    ↓
route_after_chatbot(state)
    ├── 有 tool_calls → ToolNode
    │                       ↓
    │                  ToolMessage
    │                       ↓
    │                    chatbot
    │
    ├── 无工具且对话过长 → summarize_conversation
    │                              ↓
    │                         更新摘要并删除旧消息
    │
    └── 无工具且无需摘要 → END
    ↓
SqliteSaver 保存最终 State
    ↓
最终回答片段打印到终端
```

---

## 20. 常见问题

### `content` 为什么为空

模型正在请求工具时，主要输出在 `tool_calls` 中，正文通常为空。

### 为什么工具结果要转成 `ToolMessage`

因为模型需要结构化地知道“这是某次工具调用的结果”，而不是把它误认为普通用户文本。

### 为什么最终调用仍用 `llm_with_tools`

因为模型拿到一个工具结果后，可能还需要继续调用第二个工具。

### 为什么计算结果是 `27.0`

工具参数和返回值标注为 `float`，所以 Python 返回浮点数。模型最终可以把它组织成自然的 `27`。

### 为什么最终答案有时显示两次

节点更新调试输出打印一次，面向用户的最终输出又打印一次。Token 流版本只保留面向用户的流式正文。

### 为什么重新运行还能记住

因为 SQLite 数据库和 `thread_id` 都没有改变。

---

## 21. Day 3 自测题

1. `@tool` 会从函数中提取哪些信息？
2. `bind_tools()` 会直接执行工具吗？
3. `AIMessage.tool_calls` 中的 `name`、`args`、`id` 分别是什么？
4. 为什么必须把工具调用的 `AIMessage` 放进消息历史？
5. `ToolMessage.tool_call_id` 为什么必须对应工具调用 ID？
6. 模型和 `tools_condition` 分别负责什么判断？
7. `ToolNode` 替代了哪些手写代码？
8. 为什么 `tools → chatbot` 这条边会形成 Agent 循环？
9. 为什么复杂计算不能按总消息数量计算对话轮数？
10. `recursion_limit` 限制的是工具次数还是图步骤？
11. `stream_mode="updates"` 和 `"messages"` 有什么区别？
12. 为什么流式输出要根据 `langgraph_node` 过滤摘要节点？

---

## 22. Day 3 最小记忆卡

```text
1. @tool
   → 把 Python 函数包装为 StructuredTool

2. bind_tools(tools)
   → 把工具说明和参数 Schema 交给模型

3. AIMessage.tool_calls
   → 模型提出的结构化工具调用请求

4. tool.invoke(args)
   → Python 真正执行函数

5. ToolMessage
   → 把工具结果与 tool_call_id 返回模型

6. tools_condition
   → 检查最后一条 AIMessage 是否有 tool_calls

7. ToolNode
   → 自动查找、执行工具并生成 ToolMessage

8. tools → chatbot
   → 让模型观察工具结果并继续思考

9. Agent
   → 模型选择行动、工具执行、模型观察并重复

10. SqliteSaver + thread_id
    → 保存并恢复不同会话状态

11. 摘要路由
    → 控制长期对话的上下文长度

12. stream_mode="updates"
    → 观察节点更新

13. stream_mode="messages"
    → 观察模型消息片段
```

一句话总结 Day 3：

> 今天把普通 Python 函数变成了模型可选择的 Tool，完成了工具请求、执行、结果回传和多步循环，并进一步加入错误处理、持久化记忆、摘要压缩及流式输出，形成了一个完整的本地工具 Agent。

---

## 23. Day 4 预告

Day 4 建议学习“Agent 的安全控制与人工确认”：

```text
模型提出工具调用
        ↓
判断工具是否有副作用
        ↓
只读工具 → 可以自动执行
写入、删除、支付等工具 → 等待人工确认
        ↓
批准后继续 / 拒绝后停止
```

计划内容：

1. 工具的只读操作与有副作用操作。
2. 为什么真实 Agent 不能自动执行所有工具。
3. LangGraph interrupt 暂停工作流。
4. 保存暂停状态并等待批准。
5. 使用 Command 恢复工作流。
6. 审批通过与拒绝两条路线。
7. 完成一个“转账模拟工具”或“文件写入模拟工具”的安全 Agent。

