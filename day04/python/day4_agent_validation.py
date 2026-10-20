import argparse
import os
import re
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
# 1. 基础配置
# ==================================================

load_dotenv()


# 模拟付款账户
SOURCE_ACCOUNT = "account-001"

# 模拟单笔转账限额
MAX_TRANSFER_AMOUNT = 5000.0


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

    这是只读操作，不会修改账户数据。
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
    从固定模拟付款账户 account-001
    向指定账户执行模拟转账。

    Python程序会在真正执行前完成人工审批
    和参数校验。
    """

    print(
        f"[执行高风险工具] "
        f"从 {SOURCE_ACCOUNT} "
        f"向 {to_account} "
        f"转账 {amount} 元"
    )

    return (
        "模拟转账成功："
        f"已从 {SOURCE_ACCOUNT} "
        f"向 {to_account} "
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
# 4. 模型节点
# ==================================================

def assistant(
    state: MessagesState
):
    """生成普通回答或工具调用请求。"""

    messages = [
        SystemMessage(
            content=(
                "你是一名模拟账户助手。"

                "付款账户固定为 account-001。"

                "查询余额时使用 check_balance。"

                "用户提出具体转账请求时，"
                "使用 transfer_money。"

                "每次只提出一笔转账工具调用。"

                "生成 tool_call 只是操作提案，"
                "不表示已经执行。"

                "Python程序会负责人工审批"
                "和业务参数校验。"

                "工具返回成功以前，"
                "不能声称操作已经成功。"

                "如果工具结果表示参数校验失败，"
                "请向用户解释具体原因。"
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
# 5. 模型后的路由
# ==================================================

def route_after_assistant(
    state: MessagesState
):
    """有工具请求就进入工具处理节点。"""

    latest_message = (
        state["messages"][-1]
    )

    if latest_message.tool_calls:
        return "execute_tools"

    return END


# ==================================================
# 6. 转账参数校验
# ==================================================

def validate_transfer_args(
    tool_args: dict
):
    """
    检查转账参数。

    返回 None：
        参数合法。

    返回 str：
        参数不合法，字符串是错误原因。
    """

    to_account = tool_args.get(
        "to_account"
    )

    amount = tool_args.get(
        "amount"
    )


    # ==============================================
    # 6.1 校验收款账户
    # ==============================================

    if not isinstance(
        to_account,
        str
    ):
        return "收款账户必须是字符串。"


    to_account = to_account.strip()


    if not to_account:
        return "收款账户不能为空。"


    # 账户格式示例：
    # account-002
    # account-123
    account_pattern = (
        r"account-\d{3}"
    )


    if not re.fullmatch(
        account_pattern,
        to_account
    ):
        return (
            "收款账户格式不正确。"
            "正确示例：account-002。"
        )


    if to_account == SOURCE_ACCOUNT:
        return (
            "收款账户不能与付款账户相同。"
        )


    # ==============================================
    # 6.2 校验转账金额
    # ==============================================

    if not isinstance(
        amount,
        (int, float)
    ):
        return "转账金额必须是数字。"


    amount = float(amount)


    if amount <= 0:
        return (
            "转账金额必须大于0元。"
        )


    if amount > MAX_TRANSFER_AMOUNT:
        return (
            "转账金额超过单笔限额。"
            f"单笔最多允许"
            f"{MAX_TRANSFER_AMOUNT}元。"
        )


    # 所有校验通过
    return None


# ==================================================
# 7. 工具审批与执行节点
# ==================================================

def execute_tools(
    state: MessagesState
):
    """审批、修改、校验并执行工具。"""

    latest_message = (
        state["messages"][-1]
    )

    tool_calls = (
        latest_message.tool_calls
    )


    risky_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call["name"]
        in approval_required_tools
    ]


    # 只读工具默认批准
    decision = {
        "action": "approve"
    }


    # 高风险工具需要暂停
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

        final_args = original_args

        parameters_were_edited = False


        # ==========================================
        # 7.1 用户拒绝
        # ==========================================

        if (
            requires_approval
            and action == "reject"
        ):
            tool_result = (
                "用户拒绝了本次操作。"
                "工具没有执行。"
            )


        # ==========================================
        # 7.2 批准或修改参数
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


            # ======================================
            # 7.3 高风险工具执行前进行业务校验
            # ======================================

            validation_error = None

            if tool_name == "transfer_money":
                validation_error = (
                    validate_transfer_args(
                        final_args
                    )
                )


            if validation_error:
                print(
                    "[参数校验失败] ",
                    validation_error
                )

                tool_result = (
                    "工具没有执行。"
                    f"参数校验失败："
                    f"{validation_error}"
                )


            # ======================================
            # 7.4 校验通过后才真正执行
            # ======================================

            else:
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
                            "用户修改并批准了参数。"
                            f"原参数：{original_args}；"
                            f"最终参数：{final_args}；"
                            f"执行结果："
                            f"{execution_result}"
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
    / "day4_agent_validation.db"
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
# 10. 命令行参数
# ==================================================

parser = argparse.ArgumentParser(
    description=(
        "带审批和参数校验的"
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
    "thread_id"
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
# 11. 显示审批请求
# ==================================================

def show_interrupt(
    result
):
    """显示 interrupt 暴露的工具信息。"""

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


    print("\n可用命令：")

    print(
        "python day04/python/day4_agent_validation.py "
        f"resume {args.thread_id} yes"
    )

    print(
        "python day04/python/day4_agent_validation.py "
        f"resume {args.thread_id} no"
    )

    print(
        "python day04/python/day4_agent_validation.py "
        f"resume {args.thread_id} edit"
    )


    return True


# ==================================================
# 12. 启动工作流
# ==================================================

def start_workflow():
    """启动自然语言工具请求。"""

    snapshot = graph.get_state(
        config
    )

    if snapshot.next:
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


    print("\n助手：")

    print(
        result["messages"][-1].content
    )


# ==================================================
# 13. 创建恢复数据
# ==================================================

def create_resume_data():
    """把命令行决定转换成结构化恢复数据。"""

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
                "金额必须是数字。"
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
# 14. 恢复工作流
# ==================================================

def resume_workflow():
    """恢复等待审批的工作流。"""

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


    print("\n助手：")

    print(
        result["messages"][-1].content
    )


# ==================================================
# 15. 执行命令
# ==================================================

try:
    if args.action == "start":
        start_workflow()

    elif args.action == "resume":
        resume_workflow()


finally:
    connection.close()
