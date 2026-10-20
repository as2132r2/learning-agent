import sqlite3
import uuid
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
    """
    to_account：
        收款账户。

    amount：
        转账金额。

    approved：
        用户是否批准。

    result：
        最终执行结果。
    """

    to_account: str
    amount: float

    # 第一次启动图时，
    # approved 和 result 还不存在
    approved: NotRequired[bool]
    result: NotRequired[str]


# ==================================================
# 2. 人工审批节点
# ==================================================

def review_transfer(
    state: TransferState
):
    """暂停工作流，等待人工审批。"""

    print(
        "[进入审批节点] "
        "准备请求人工确认"
    )

    # interrupt() 会暂停当前图的执行，
    # 并把字典返回给调用 graph.invoke() 的程序。
    #
    # 使用 Command(resume=...) 恢复时，
    # resume 中的值会成为 interrupt() 的返回值。
    decision = interrupt({
        "question": "是否批准这笔模拟转账？",
        "to_account": state["to_account"],
        "amount": state["amount"]
    })

    print(
        "[审批节点恢复] "
        "收到人工决定：",
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
# 3. 审批后的路由判断
# ==================================================

def route_after_review(
    state: TransferState
):
    """根据 approved 选择执行或拒绝节点。"""

    if state["approved"]:
        return "execute_transfer"

    return "reject_transfer"


# ==================================================
# 4. 执行转账节点
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
# 5. 拒绝转账节点
# ==================================================

def reject_transfer(
    state: TransferState
):
    """用户拒绝后，不执行转账。"""

    print(
        "[拒绝执行] "
        "没有调用转账操作"
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
# 7. 创建 SQLite Checkpointer
# ==================================================

script_directory = (
    Path(__file__).resolve().parent
)


database_path = (
    script_directory.parent
    / "database"
    / "day4_interrupt.db"
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
# 8. 为这一次流程创建 thread_id
# ==================================================

thread_id = str(
    uuid.uuid4()
)


config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# ==================================================
# 9. 第一次调用：运行到 interrupt()
# ==================================================

initial_state: TransferState = {
    "to_account": "account-002",
    "amount": 100.0
}


print(
    "开始一笔模拟转账。"
)


try:
    paused_result = graph.invoke(
        initial_state,
        config=config
    )


    # ==================================================
    # 10. 读取 interrupt() 暴露的审批信息
    # ==================================================

    interrupts = paused_result.get(
        "__interrupt__",
        []
    )


    if not interrupts:
        print(
            "流程没有暂停，"
            "未取得审批请求。"
        )

    else:
        interrupt_info = interrupts[0]

        # value 就是传给 interrupt(...)
        # 的那份字典
        approval_request = (
            interrupt_info.value
        )

        print(
            "\n流程已暂停。"
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


        # ==============================================
        # 11. 在图外取得人工决定
        # ==============================================

        human_decision = input(
            "请输入 yes 批准，"
            "输入 no 拒绝："
        ).strip().lower()


        # ==============================================
        # 12. 使用 Command 恢复原来的图
        # ==============================================

        final_state = graph.invoke(
            Command(
                resume=human_decision
            ),
            config=config
        )


        print(
            "\n流程执行结束。"
        )

        print(
            "是否批准：",
            final_state["approved"]
        )

        print(
            "最终结果：",
            final_state["result"]
        )


finally:
    connection.close()
