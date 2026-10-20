import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    RemoveMessage
)
from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState
)
from langgraph.checkpoint.sqlite import SqliteSaver


# ==================================================
# 1. 加载配置并创建模型
# ==================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# 摘要最多生成约250个Token
summary_llm = llm.bind(max_tokens=250)


# ==================================================
# 2. 定义对话状态
# ==================================================

class State(MessagesState):
    # 除了消息列表，还保存一份旧对话摘要
    summary: str


# ==================================================
# 3. 定义聊天节点
# ==================================================

def chatbot(state: State):
    # 如果还没有摘要，就返回空字符串
    summary = state.get("summary", "")

    system_content = """
你是一名简洁、友好的编程学习助手。
请根据用户的学习情况提供清晰、适合初学者的回答。
""".strip()

    # 如果存在旧对话摘要，就把它加入系统消息
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


# ==================================================
# 4. 判断是否需要生成摘要
# ==================================================

def should_summarize(state: State):
    # 超过6条原始消息时生成摘要
    if len(state["messages"]) > 6:
        return "summarize_conversation"

    return END


# ==================================================
# 5. 生成或更新摘要
# ==================================================

def summarize_conversation(state: State):
    old_summary = state.get("summary", "")

    if old_summary:
        summary_request = f"""
这是以前保存的对话摘要：

{old_summary}

请结合最近的新对话更新摘要。

要求：
1. 不超过150个汉字
2. 保留用户姓名
3. 保留学习目标
4. 保留用户偏好
5. 保留已完成的学习内容
6. 保留尚未解决的问题
7. 删除寒暄和重复信息
8. 只输出更新后的摘要
"""
    else:
        summary_request = """
请总结以上对话。

要求：
1. 不超过150个汉字
2. 保留用户姓名
3. 保留学习目标
4. 保留用户偏好
5. 保留已完成的学习内容
6. 保留尚未解决的问题
7. 删除寒暄和重复信息
8. 只输出摘要
"""

    messages_for_summary = [
        *state["messages"],
        HumanMessage(content=summary_request)
    ]

    response = summary_llm.invoke(messages_for_summary)

    # 保留最后4条消息，也就是最近两轮对话
    messages_to_delete = [
        RemoveMessage(id=message.id)
        for message in state["messages"][:-4]
    ]

    return {
        "summary": response.content,
        "messages": messages_to_delete
    }


# ==================================================
# 6. 创建 LangGraph 工作流程
# ==================================================

builder = StateGraph(State)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_node(
    "summarize_conversation",
    summarize_conversation
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_conditional_edges(
    "chatbot",
    should_summarize
)

builder.add_edge(
    "summarize_conversation",
    END
)


# ==================================================
# 7. 创建 SQLite 持久化存储
# ==================================================

connection = sqlite3.connect(
    Path(__file__).resolve().parent.parent
    / "database"
    / "day2_memory.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(connection)

graph = builder.compile(
    checkpointer=checkpointer
)


# ==================================================
# 8. 设置当前会话编号
# ==================================================

config = {
    "configurable": {
        "thread_id": "xiaolin"
    }
}


# ==================================================
# 9. 开始聊天
# ==================================================

print("开始聊天，输入 exit 结束。")
print("当前会话：xiaolin")

try:
    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            print("请输入内容，不要只按回车。")
            continue

        if user_input.lower() == "exit":
            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config
        )

        latest_message = result["messages"][-1]

        print("助手：", latest_message.content)


    # ==============================================
    # 10. 查看最终保存状态
    # ==============================================

    snapshot = graph.get_state(config)

    messages = snapshot.values.get(
        "messages",
        []
    )

    summary = snapshot.values.get(
        "summary",
        "暂无摘要"
    )

    print("\n聊天结束")
    print("当前原始消息数量：", len(messages))
    print("对话摘要：", summary)

finally:
    # 无论程序是否正常结束，都关闭数据库连接
    connection.close()
