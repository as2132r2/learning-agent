import argparse
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph
)
from langgraph.types import (
    Command,
    interrupt
)


# ==================================================
# 1. 加载模型配置
# ==================================================

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# ==================================================
# 2. 定义只读工具
# ==================================================

@tool
def check_balance(
    account: str
) -> str:
    """
    查询模拟账户余额。

    这是只读操作，不会修改账户。
    只有用户明确要求查询余额时才使用。
    """

    print(
        f"[执行只读工具] "
        f"查询账户 {account}"
    )

    return (
        f"模拟账户 {account} "
        f"的余额是1000元。"
    )


# ==================================================
# 3. 定义高风险工具
# ==================================================

@tool
def transfer_money(
    to_account: str,
    amount: float
) -> str:
    """
    执行一笔模拟转账。

    用户提出具体转账请求时使用。
    Python程序会在真正执行前完成人工审批。
    模型只需要生成工具调用请求。
    """

    print(
        f"[执行高风险工具] "
        f"向 {to_account} "
        f"转账 {amount} 元"
    )

    # 这里只是模拟操作
    return (
        "模拟转账成功："
        f"已向 {to_account} "
        f"转账 {amount} 元。"
    )


# ==================================================
# 4. 创建工具集合
# ==================================================

tools = [
    check_balance,
    transfer_money
]


tools_by_name = {
    current_tool.name: current_tool
    for current_tool in tools
}


# 这些工具必须经过人工审批
approval_required_tools = {
    "transfer_money"
}


llm_with_tools = llm.bind_tools(
    tools
)


# ==================================================
# 5. 定义模型节点
# ==================================================

def assistant(
    state: MessagesState
):
    """让模型直接回答或生成工具调用请求。"""

    messages = [
        SystemMessage(
            content=(
                "你是一名模拟账户助手。"

                "用户明确查询余额时，"
                "调用 check_balance。"

                "用户提出包含账户和金额的"
                "具体转账请求时，"
                "调用 transfer_money。"

                "生成 tool_call 只是提出操作请求，"
                "不代表工具已经执行。"

                "Python程序会在高风险工具真正执行前"
                "完成人工审批，"
                "所以你不需要提前询问用户确认。"

                "除非用户明确要求，"
                "否则不要自行查询余额。"

                "不得猜测用户没有提供的关键参数。"

                "工具返回成功结果以前，"
                "不能声称操作已经成功。"
            )
        ),
        *state["messages"]
    ]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ==================================================
# 6. 判断模型是否请求工具
# ==================================================

def route_after_assistant(
    state: MessagesState
):
    """有工具请求就进入工具节点，否则结束。"""

    latest_message = (
        state["messages"][-1]
    )

    if latest_message.tool_calls:
        return "execute_tools"

    return END


# ==================================================
# 7. 执行工具节点
# ==================================================

def execute_tools(
    state: MessagesState
):
    """
    处理模型提出的工具调用。

    如果包含高风险工具：
        先 interrupt()。

    批准后：
        执行工具。

    拒绝后：
        不执行高风险工具，
        但仍返回 ToolMessage。
    """

    latest_message = (
        state["messages"][-1]
    )

    tool_calls = (
        latest_message.tool_calls
    )


    # ==============================================
    # 7.1 找出需要审批的工具调用
    # ==============================================

    risky_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call["name"]
        in approval_required_tools
    ]


    # ==============================================
    # 7.2 如果有高风险工具，暂停整个图
    # ==============================================

    approved = True

    if risky_tool_calls:
        approval_items = [
            {
                "tool_name": (
                    tool_call["name"]
                ),
                "arguments": (
                    tool_call["args"]
                ),
                "tool_call_id": (
                    tool_call["id"]
                )
            }
            for tool_call
            in risky_tool_calls
        ]

        decision = interrupt({
            "question": (
                "是否批准执行这些"
                "高风险工具？"
            ),
            "tool_calls": approval_items
        })

        approved = (
            str(decision).strip().lower()
            == "yes"
        )


    # ==============================================
    # 7.3 处理每一个工具调用
    # ==============================================

    tool_messages = []

    for tool_call in tool_calls:
        tool_name = (
            tool_call["name"]
        )

        tool_args = (
            tool_call["args"]
        )

        tool_call_id = (
            tool_call["id"]
        )

        requires_approval = (
            tool_name
            in approval_required_tools
        )


        # 高风险工具被拒绝
        if (
            requires_approval
            and not approved
        ):
            tool_result = (
                "用户拒绝了本次操作。"
                "工具没有执行，"
                "没有产生任何修改。"
            )


        # 已批准，或者工具本身是只读操作
        else:
            selected_tool = (
                tools_by_name[
                    tool_name
                ]
            )

            try:
                tool_result = (
                    selected_tool.invoke(
                        tool_args
                    )
                )

            except Exception as error:
                tool_result = (
                    "工具执行失败。"
                    f"异常类型："
                    f"{type(error).__name__}；"
                    f"异常信息：{str(error)}。"
                )


        tool_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )
        )


    return {
        "messages": tool_messages
    }


# ==================================================
# 8. 构建 LangGraph
# ==================================================

builder = StateGraph(
    MessagesState
)


builder.add_node(
    "assistant",
    assistant
)


builder.add_node(
    "execute_tools",
    execute_tools
)


builder.add_edge(
    START,
    "assistant"
)


builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "execute_tools": (
            "execute_tools"
        ),
        END: END
    }
)


# 工具执行或拒绝后，
# 回到模型节点生成最终回答
builder.add_edge(
    "execute_tools",
    "assistant"
)


# ==================================================
# 9. SQLite 持久化
# ==================================================

script_directory = (
    Path(__file__).resolve().parent
)


database_path = (
    script_directory.parent
    / "database"
    / "day4_agent_interrupt.db"
)


connection = sqlite3.connect(
    database_path,
    check_same_thread=False
)


checkpointer = SqliteSaver(
    connection
)


graph = builder.compile(
    checkpointer=checkpointer
)


# ==================================================
# 10. 读取命令行参数
# ==================================================

parser = argparse.ArgumentParser(
    description=(
        "带人工审批的模拟账户 Agent"
    )
)


parser.add_argument(
    "action",
    choices=[
        "start",
        "resume"
    ],
    help=(
        "start 发起请求；"
        "resume 恢复审批"
    )
)


parser.add_argument(
    "thread_id",
    help=(
        "工作流编号，"
        "例如 approval-001"
    )
)


parser.add_argument(
    "decision",
    nargs="?",
    choices=[
        "yes",
        "no"
    ],
    help=(
        "恢复时的审批结果："
        "yes 或 no"
    )
)


args = parser.parse_args()


config = {
    "configurable": {
        "thread_id": args.thread_id
    },
    "recursion_limit": 10
}


# ==================================================
# 11. 显示暂停信息
# ==================================================

def show_interrupt(
    result
):
    """显示 interrupt() 暴露的审批数据。"""

    interrupts = result.get(
        "__interrupt__",
        []
    )

    if not interrupts:
        return False

    approval_request = (
        interrupts[0].value
    )

    print("\n工作流已经暂停并保存。")

    print(
        "流程编号：",
        args.thread_id
    )

    print(
        "审批问题：",
        approval_request["question"]
    )


    for number, tool_call in enumerate(
        approval_request["tool_calls"],
        start=1
    ):
        print(
            f"\n待审批操作 {number}："
        )

        print(
            "工具名称：",
            tool_call["tool_name"]
        )

        print(
            "工具参数：",
            tool_call["arguments"]
        )

        print(
            "调用ID：",
            tool_call["tool_call_id"]
        )


    print(
        "\n稍后批准："
    )

    print(
        "python day04/python/day4_agent_interrupt.py "
        f"resume {args.thread_id} yes"
    )

    print(
        "\n稍后拒绝："
    )

    print(
        "python day04/python/day4_agent_interrupt.py "
        f"resume {args.thread_id} no"
    )

    return True


# ==================================================
# 12. 启动自然语言请求
# ==================================================

def start_workflow():
    """把用户问题交给模型和工作流。"""

    existing_snapshot = graph.get_state(
        config
    )

    if existing_snapshot.next:
        print(
            "该 thread_id 已有暂停中的流程。"
        )

        print(
            "请使用 resume 恢复。"
        )

        return

    user_input = input(
        "请输入请求："
    ).strip()

    if not user_input:
        print(
            "请求不能为空。"
        )

        return

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ]
        },
        config=config
    )


    # 高风险工具会在这里暂停
    if show_interrupt(result):
        return


    # 普通回答或只读工具会直接完成
    final_message = (
        result["messages"][-1]
    )

    print("\n助手：")
    print(
        final_message.content
    )


# ==================================================
# 13. 恢复暂停的工作流
# ==================================================

def resume_workflow():
    """批准或拒绝暂停的工具调用。"""

    if args.decision is None:
        print(
            "resume 必须提供 "
            "yes 或 no。"
        )

        return

    snapshot = graph.get_state(
        config
    )

    if not snapshot.next:
        print(
            "没有找到等待审批的流程。"
        )

        print(
            "请检查 thread_id。"
        )

        return

    print(
        "找到暂停中的工作流。"
    )

    print(
        "流程编号：",
        args.thread_id
    )

    print(
        "审批决定：",
        args.decision
    )

    result = graph.invoke(
        Command(
            resume=args.decision
        ),
        config=config
    )


    # 正常情况下，恢复后会完成工作流。
    # 如果未来出现第二个 interrupt，
    # 这里仍然可以再次显示暂停信息。
    if show_interrupt(result):
        return


    final_message = (
        result["messages"][-1]
    )

    print("\n助手：")
    print(
        final_message.content
    )


# ==================================================
# 14. 执行命令
# ==================================================

try:
    if args.action == "start":
        start_workflow()

    elif args.action == "resume":
        resume_workflow()


finally:
    connection.close()
