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

    Python程序会在真正执行前完成人工审批。
    模型只需要生成工具调用请求。
    """

    print(
        f"[执行高风险工具] "
        f"向 {to_account} "
        f"转账 {amount} 元"
    )

    return (
        "模拟转账成功："
        f"已向 {to_account} "
        f"转账 {amount} 元。"
    )


tools = [
    check_balance,
    transfer_money
]


tools_by_name = {
    current_tool.name: current_tool
    for current_tool in tools
}


approval_required_tools = {
    "transfer_money"
}


llm_with_tools = llm.bind_tools(
    tools
)


# ==================================================
# 4. 定义模型节点
# ==================================================

def assistant(
    state: MessagesState
):
    """生成普通回答或工具调用请求。"""

    messages = [
        SystemMessage(
            content=(
                "你是一名模拟账户助手。"

                "查询余额时调用 check_balance。"

                "用户提出具体转账请求时，"
                "调用 transfer_money。"

                "生成工具调用请求不代表已经执行。"

                "Python程序会在执行前让用户"
                "批准、拒绝或修改参数。"

                "模型不需要提前询问确认。"

                "不得猜测用户没有提供的关键参数。"

                "工具返回成功结果以前，"
                "不能声称操作已经成功。"

                "如果工具结果说明参数被用户修改，"
                "最终回答必须使用修改后的参数。"
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
# 5. 判断是否需要执行工具
# ==================================================

def route_after_assistant(
    state: MessagesState
):
    """有工具请求就进入执行节点。"""

    latest_message = (
        state["messages"][-1]
    )

    if latest_message.tool_calls:
        return "execute_tools"

    return END


# ==================================================
# 6. 人工审批并执行工具
# ==================================================

def execute_tools(
    state: MessagesState
):
    """
    对高风险工具暂停审批。

    支持：
        approve：按原参数执行
        reject：拒绝执行
        edit：修改参数后执行
    """

    latest_message = (
        state["messages"][-1]
    )

    tool_calls = (
        latest_message.tool_calls
    )


    # ==============================================
    # 6.1 找出高风险工具调用
    # ==============================================

    risky_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call["name"]
        in approval_required_tools
    ]


    # 默认决定是批准。
    # 如果只有只读工具，就不需要暂停。
    decision = {
        "action": "approve"
    }


    # ==============================================
    # 6.2 高风险工具触发 interrupt
    # ==============================================

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
                "请选择批准、拒绝"
                "或修改参数。"
            ),
            "allowed_decisions": [
                "yes",
                "no",
                "edit"
            ],
            "tool_calls": approval_items
        })


    action = decision.get(
        "action",
        "reject"
    )


    # ==============================================
    # 6.3 处理每个工具调用
    # ==============================================

    tool_messages = []

    for tool_call in tool_calls:
        tool_name = (
            tool_call["name"]
        )

        original_args = (
            tool_call["args"]
        )

        tool_call_id = (
            tool_call["id"]
        )

        requires_approval = (
            tool_name
            in approval_required_tools
        )

        # 默认使用模型生成的原始参数
        final_args = original_args

        parameters_were_edited = False


        # ==========================================
        # 拒绝高风险工具
        # ==========================================

        if (
            requires_approval
            and action == "reject"
        ):
            tool_result = (
                "用户拒绝了本次操作。"
                "工具没有执行，"
                "没有产生任何修改。"
            )


        # ==========================================
        # 修改高风险工具参数
        # ==========================================

        else:
            if (
                requires_approval
                and action == "edit"
            ):
                final_args = decision.get(
                    "new_args",
                    original_args
                )

                parameters_were_edited = True


            selected_tool = (
                tools_by_name[
                    tool_name
                ]
            )

            try:
                execution_result = (
                    selected_tool.invoke(
                        final_args
                    )
                )

                if parameters_were_edited:
                    tool_result = (
                        "用户修改了工具参数，"
                        "并批准按新参数执行。"
                        f"原参数：{original_args}；"
                        f"新参数：{final_args}；"
                        f"执行结果：{execution_result}"
                    )

                else:
                    tool_result = (
                        execution_result
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
# 7. 构建 LangGraph
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


builder.add_edge(
    "execute_tools",
    "assistant"
)


# ==================================================
# 8. 创建 SQLite 持久化
# ==================================================

script_directory = (
    Path(__file__).resolve().parent
)


database_path = (
    script_directory.parent
    / "database"
    / "day4_agent_edit.db"
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
# 9. 读取命令行参数
# ==================================================

parser = argparse.ArgumentParser(
    description=(
        "支持批准、拒绝和修改参数的"
        "模拟账户 Agent"
    )
)


parser.add_argument(
    "action",
    choices=[
        "start",
        "resume"
    ]
)


parser.add_argument(
    "thread_id",
    help=(
        "工作流编号，"
        "例如 edit-001"
    )
)


parser.add_argument(
    "decision",
    nargs="?",
    choices=[
        "yes",
        "no",
        "edit"
    ]
)


args = parser.parse_args()


config = {
    "configurable": {
        "thread_id": args.thread_id
    },
    "recursion_limit": 10
}


# ==================================================
# 10. 显示暂停信息
# ==================================================

def show_interrupt(
    result
):
    """显示等待审批的工具信息。"""

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
            "原始参数：",
            tool_call["arguments"]
        )

        print(
            "调用ID：",
            tool_call["tool_call_id"]
        )


    print("\n可以执行：")

    print(
        "python day04/python/day4_agent_edit.py "
        f"resume {args.thread_id} yes"
    )

    print(
        "python day04/python/day4_agent_edit.py "
        f"resume {args.thread_id} no"
    )

    print(
        "python day04/python/day4_agent_edit.py "
        f"resume {args.thread_id} edit"
    )

    return True


# ==================================================
# 11. 启动工作流
# ==================================================

def start_workflow():
    """接收自然语言请求并启动 Agent。"""

    existing_snapshot = graph.get_state(
        config
    )

    if existing_snapshot.next:
        print(
            "该 thread_id 已有暂停中的流程，"
            "请使用 resume。"
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
# 12. 创建恢复数据
# ==================================================

def create_resume_data():
    """
    根据 yes、no、edit，
    创建传给 Command(resume=...) 的字典。
    """

    if args.decision == "yes":
        return {
            "action": "approve"
        }


    if args.decision == "no":
        return {
            "action": "reject"
        }


    if args.decision == "edit":
        new_account = input(
            "请输入新的收款账户："
        ).strip()

        new_amount_text = input(
            "请输入新的转账金额："
        ).strip()

        try:
            new_amount = float(
                new_amount_text
            )

        except ValueError:
            print(
                "金额格式错误，"
                "请输入数字。"
            )

            return None


        if not new_account:
            print(
                "收款账户不能为空。"
            )

            return None


        return {
            "action": "edit",
            "new_args": {
                "to_account": new_account,
                "amount": new_amount
            }
        }


    return None


# ==================================================
# 13. 恢复工作流
# ==================================================

def resume_workflow():
    """恢复暂停的审批流程。"""

    if args.decision is None:
        print(
            "resume 必须提供 "
            "yes、no 或 edit。"
        )

        return

    snapshot = graph.get_state(
        config
    )

    if not snapshot.next:
        print(
            "没有找到等待审批的流程。"
        )

        return

    resume_data = create_resume_data()

    if resume_data is None:
        return

    print(
        "找到暂停中的工作流。"
    )

    print(
        "流程编号：",
        args.thread_id
    )

    print(
        "审批数据：",
        resume_data
    )

    result = graph.invoke(
        Command(
            resume=resume_data
        ),
        config=config
    )


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
