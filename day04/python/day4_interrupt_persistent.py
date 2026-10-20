import argparse
import sqlite3
from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import (
    END,
    START,
    StateGraph
)
from langgraph.types import (
    Command,
    interrupt
)


# ==================================================
# 1. 定义状态
# ==================================================

class TransferState(TypedDict):
    """模拟转账工作流的状态。"""

    to_account: str
    amount: float
    approved: NotRequired[bool]
    result: NotRequired[str]


# ==================================================
# 2. 人工审批节点
# ==================================================

def review_transfer(
    state: TransferState
):
    """暂停工作流并等待批准。"""

    print(
        "[进入审批节点] "
        "准备请求人工确认"
    )

    decision = interrupt({
        "question": "是否批准这笔模拟转账？",
        "to_account": state["to_account"],
        "amount": state["amount"]
    })

    print(
        "[审批节点恢复] "
        "收到决定：",
        decision
    )

    approved = (
        str(decision).strip().lower()
        == "yes"
    )

    return {
        "approved": approved
    }


# ==================================================
# 3. 审批后的条件路由
# ==================================================

def route_after_review(
    state: TransferState
):
    """根据审批结果选择下一节点。"""

    if state["approved"]:
        return "execute_transfer"

    return "reject_transfer"


# ==================================================
# 4. 执行节点
# ==================================================

def execute_transfer(
    state: TransferState
):
    """执行模拟转账。"""

    to_account = state["to_account"]
    amount = state["amount"]

    print(
        "[执行模拟转账] "
        f"向 {to_account} "
        f"转账 {amount} 元"
    )

    return {
        "result": (
            "模拟转账成功："
            f"已向 {to_account} "
            f"转账 {amount} 元。"
        )
    }


# ==================================================
# 5. 拒绝节点
# ==================================================

def reject_transfer(
    state: TransferState
):
    """拒绝后不执行转账。"""

    print(
        "[拒绝执行] "
        "没有执行模拟转账"
    )

    return {
        "result": (
            "用户拒绝了这笔模拟转账，"
            "没有产生任何修改。"
        )
    }


# ==================================================
# 6. 构建 LangGraph
# ==================================================

builder = StateGraph(
    TransferState
)


builder.add_node(
    "review_transfer",
    review_transfer
)


builder.add_node(
    "execute_transfer",
    execute_transfer
)


builder.add_node(
    "reject_transfer",
    reject_transfer
)


builder.add_edge(
    START,
    "review_transfer"
)


builder.add_conditional_edges(
    "review_transfer",
    route_after_review,
    {
        "execute_transfer": (
            "execute_transfer"
        ),
        "reject_transfer": (
            "reject_transfer"
        )
    }
)


builder.add_edge(
    "execute_transfer",
    END
)


builder.add_edge(
    "reject_transfer",
    END
)


# ==================================================
# 7. 创建 SQLite 持久化
# ==================================================

script_directory = (
    Path(__file__).resolve().parent
)


database_path = (
    script_directory.parent
    / "database"
    / "day4_interrupt_persistent.db"
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
# 8. 读取终端命令参数
# ==================================================

parser = argparse.ArgumentParser(
    description=(
        "演示可以跨程序重启的"
        "LangGraph 人工审批"
    )
)


parser.add_argument(
    "action",
    choices=[
        "start",
        "resume"
    ],
    help=(
        "start 创建审批；"
        "resume 恢复审批"
    )
)


parser.add_argument(
    "thread_id",
    help=(
        "本次审批流程的唯一编号，"
        "例如 transfer-001"
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
        "恢复时的审批决定："
        "yes 或 no"
    )
)


args = parser.parse_args()


# 根据命令参数创建配置
config = {
    "configurable": {
        "thread_id": args.thread_id
    }
}


# ==================================================
# 9. 启动新的审批流程
# ==================================================

def start_workflow():
    """创建一笔新的模拟转账并暂停等待审批。"""

    # 检查这个 thread_id
    # 是否已经有暂停中的流程
    existing_snapshot = graph.get_state(
        config
    )

    if existing_snapshot.next:
        print(
            "该 thread_id 已有暂停中的流程。"
        )

        print(
            "请使用 resume 恢复，"
            "不要重复 start。"
        )

        return

    to_account = input(
        "请输入收款账户："
    ).strip()

    amount_text = input(
        "请输入转账金额："
    ).strip()

    try:
        amount = float(
            amount_text
        )

    except ValueError:
        print(
            "金额格式错误，"
            "请输入数字。"
        )

        return

    initial_state: TransferState = {
        "to_account": to_account,
        "amount": amount
    }

    paused_result = graph.invoke(
        initial_state,
        config=config
    )

    interrupts = paused_result.get(
        "__interrupt__",
        []
    )

    if not interrupts:
        print(
            "工作流没有产生审批请求。"
        )

        return

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

    print(
        "收款账户：",
        approval_request["to_account"]
    )

    print(
        "转账金额：",
        approval_request["amount"]
    )

    print(
        "\n现在可以关闭程序。"
    )

    print(
        "稍后使用以下命令批准："
    )

    print(
        "python day04/python/day4_interrupt_persistent.py "
        f"resume {args.thread_id} yes"
    )

    print(
        "\n或者使用以下命令拒绝："
    )

    print(
        "python day04/python/day4_interrupt_persistent.py "
        f"resume {args.thread_id} no"
    )


# ==================================================
# 10. 恢复暂停的审批流程
# ==================================================

def resume_workflow():
    """根据相同 thread_id 恢复暂停的工作流。"""

    if args.decision is None:
        print(
            "resume 命令必须提供 "
            "yes 或 no。"
        )

        print(
            "示例："
        )

        print(
            "python day04/python/day4_interrupt_persistent.py "
            f"resume {args.thread_id} yes"
        )

        return

    snapshot = graph.get_state(
        config
    )

    # next 为空通常表示：
    # 没有找到暂停中的后续节点
    if not snapshot.next:
        print(
            "没有找到等待审批的流程。"
        )

        print(
            "请检查数据库文件和 "
            "thread_id 是否正确。"
        )

        return

    print(
        "找到暂停流程，准备恢复。"
    )

    print(
        "流程编号：",
        args.thread_id
    )

    print(
        "人工决定：",
        args.decision
    )

    final_state = graph.invoke(
        Command(
            resume=args.decision
        ),
        config=config
    )

    print("\n流程执行结束。")

    print(
        "是否批准：",
        final_state["approved"]
    )

    print(
        "最终结果：",
        final_state["result"]
    )


# ==================================================
# 11. 根据 action 选择操作
# ==================================================

try:
    if args.action == "start":
        start_workflow()

    elif args.action == "resume":
        resume_workflow()


finally:
    connection.close()
