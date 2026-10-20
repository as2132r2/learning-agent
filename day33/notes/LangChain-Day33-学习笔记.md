# Day33：毕业项目核心功能搭建

项目：编程学习助手。今天固定7步已完成，Day34加入笔记审批与可靠性处理，Day35完成毕业验收。

## 1. 输入与状态约定

文件：day33_project_contract.py。

UserRequest用Pydantic验证session_id和question，拒绝额外字段、空白问题和非法会话编号。

Route定义chat、calculate、knowledge、save_note四种路线。Evidence定义chunk_id、source和content。AcceptanceCase保存业务验收目标，不表示功能已经通过。

TypedDict仅描述状态类型，运行时输入验证依靠Pydantic。

## 2. LangGraph骨架

文件：day33_graph_basic.py。

StateGraph注册节点与边。START进入router，router写route，条件边读取route并选择业务节点，各分支结束后进入END。

messages使用Annotated[list[BaseMessage], add_messages]。节点返回一条新消息，reducer负责合并，不要求每次返回完整历史。

add_messages通常追加新消息，遇到相同消息ID时会更新对应消息，并非无条件列表拼接。

当前路由是关键词规则，不调用LLM。规则有优先级：保存笔记优先，其次计算，再是课程知识，最后聊天。它适合验证项目骨架，暂不处理复杂多意图规划。

## 3. 真实计算工具

文件：day33_tool_agent.py。

@tool把函数包装成工具；bind_tools把工具描述交给模型；模型返回tool_calls并不代表工具已经执行。

```text
HumanMessage
→ calculator返回AIMessage.tool_calls
→ tools_condition选择tools
→ ToolNode执行Python函数
→ ToolMessage保存结果与tool_call_id
→ calculator再次读取消息
→ 无tool_calls时结束
```

add和multiply限制输入与结果为有限数值。工具错误由ToolNode处理，递归步数上限20用于阻止无限循环；它不是墙钟时间限制。

## 4. 课程资料检索

文件：day33_knowledge_agent.py。

正常模式复用Day22的向量检索、FTS关键词检索、RRF融合和LLM精排。仅保留相关性达到95分的前3条资料。

```text
问题 → 混合召回 → RRF → 精排 → 阈值过滤
                                      ↓
                         有证据回答 / 无证据说明不足
```

知识回答模型只收到本轮问题和合格证据，要求引用[资料N]。程序检查引用编号存在且不越界，没有复刻完整语义支持度评测。

95分沿用旧评测基线，不能保证所有新问题都正确。展示的检索来源也不意味着回答使用了其中每一条。

## 5. 会话持久化

文件：day33_persistent_agent.py。

SqliteSaver在编译图时注入。session_id映射到configurable.thread_id，用来区分同一数据库中的会话。

```text
当前问题 + thread_id
→ 读取对应checkpoint
→ 合并当前HumanMessage
→ 执行图节点
→ 保存新checkpoint
```

普通字段每轮覆盖，消息字段按add_messages合并。evidence、answer、route、pending_note、approval每轮重置。

证据在持久化状态中存成普通字典，避免checkpoint恢复依赖自定义Pydantic类型。

检查点保存图状态，不等同于“只保存聊天文本”。evidence虽然没有写成消息，也可以出现在该轮checkpoint里。

## 6. 连续交互入口

文件：day33_learning_assistant.py。

一个SqliteSaver上下文覆盖整个聊天循环，避免每轮反复创建模型和数据库连接。

| 命令 | 作用 |
|---|---|
| /help | 查看帮助 |
| /session 编号 | 切换历史，不删除旧会话 |
| /history | 最近12条消息 |
| /status | 消息数量、最后路线、待执行节点 |
| exit | 退出并关闭连接 |

本地命令由CLI处理，不进入模型和历史。空白输入被拦截。

若snapshot.next非空，说明该会话还有待执行节点。当前入口拒绝追加新问题，以免把新轮次和未完成流程混合；自动恢复将在后续可靠性工作中处理。

## 当前完成与待完成

已实现：连续聊天入口、计算工具循环、检索接线、来源展示、会话保存与恢复、会话切换。

已用Fake验证：固定计算72、姓名记忆、会话隔离、有无证据分支、关闭后重新连接恢复、脚本化交互。

真实DeepSeek问答与真实检索路径尚未在本次验证中调用。保存笔记仍是明确说明未保存的占位节点，不能算审批已完成。

当前还有这些限制：关键词路由、未做追问检索改写、未做消息压缩、未做用户认证、未做引用语义审计。session_id隔离不是身份权限控制；知道另一会话编号的本机操作者可以切换过去。

## 复习命令

```bash
python day33/python/day33_project_contract.py
python day33/python/day33_graph_basic.py --demo
python day33/python/day33_tool_agent.py --demo
python day33/python/day33_knowledge_agent.py --demo
python day33/python/day33_persistent_agent.py --demo
python day33/python/day33_learning_assistant.py --check

# 正常交互，使用项目.env与已有课程索引
python day33/python/day33_learning_assistant.py xiaolin-capstone-001

# 离线固定样例，使用独立演示数据库
python day33/python/day33_learning_assistant.py --demo
```

## 自测问题

1. router和条件边分别负责什么？
2. tool_calls与ToolMessage有什么区别？
3. 哪些返回消息进入历史，哪些只在临时Prompt中使用？
4. 为什么每轮重置evidence，却不清空messages？
5. session_id怎样关联到SQLite中的历史？
6. 为什么知识回答不能直接用模型常识补足资料？
7. /session为什么不是认证机制？
8. 今天的Fake验证能证明什么、不能证明什么？
