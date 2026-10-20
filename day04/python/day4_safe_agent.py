import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt


# ==================================================
# 1. 基础配置
# ==================================================

load_dotenv()

SOURCE_ACCOUNT = "account-001"
MAX_TRANSFER_AMOUNT = 5000.0

script_directory = Path(__file__).resolve().parent
database_path = (
    script_directory.parent / "database" / "day4_safe_agent.db"
)

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# ==================================================
# 2. 命令行参数
# ==================================================

parser = argparse.ArgumentParser(
    description="带审批、校验和审计日志的模拟账户 Agent"
)
parser.add_argument("action", choices=["start", "resume", "logs"])
parser.add_argument("thread_id")
parser.add_argument(
    "decision",
    nargs="?",
    choices=["yes", "no", "edit"]
)
args = parser.parse_args()

config = {
    "configurable": {"thread_id": args.thread_id},
    "recursion_limit": 10
}


# ==================================================
# 3. 数据库与审计日志
# ==================================================

# 同一个 SQLite 文件中：
# - SqliteSaver 保存 LangGraph 工作流状态
# - audit_logs 表保存我们自己的审计记录
connection = sqlite3.connect(
    database_path,
    check_same_thread=False
)

connection.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        tool_name TEXT,
        tool_call_id TEXT,
        original_args TEXT,
        final_args TEXT,
        detail TEXT
    )
""")

# tool_executions 是“执行凭证表”。
# 同一个 tool_call_id 只能成功写入一次。
connection.execute("""
    CREATE TABLE IF NOT EXISTS tool_executions (
        tool_call_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        final_args TEXT NOT NULL,
        result TEXT NOT NULL,
        executed_at TEXT NOT NULL
    )
""")
connection.commit()


def write_audit_log(
    event_type: str,
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    original_args: dict | None = None,
    final_args: dict | None = None,
    detail: str | None = None
) -> None:
    """向 audit_logs 表追加一条不可覆盖的事件记录。"""

    connection.execute(
        """
        INSERT INTO audit_logs (
            created_at,
            thread_id,
            event_type,
            tool_name,
            tool_call_id,
            original_args,
            final_args,
            detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            args.thread_id,
            event_type,
            tool_name,
            tool_call_id,
            json.dumps(original_args, ensure_ascii=False)
            if original_args is not None else None,
            json.dumps(final_args, ensure_ascii=False)
            if final_args is not None else None,
            detail
        )
    )
    connection.commit()


def show_audit_logs() -> None:
    """显示当前 thread_id 的全部审计记录。"""

    rows = connection.execute(
        """
        SELECT
            id,
            created_at,
            event_type,
            tool_name,
            original_args,
            final_args,
            detail
        FROM audit_logs
        WHERE thread_id = ?
        ORDER BY id
        """,
        (args.thread_id,)
    ).fetchall()

    if not rows:
        print("没有找到该流程的审计日志。")
        return

    print(f"流程 {args.thread_id} 的审计日志：")

    for row in rows:
        (
            log_id,
            created_at,
            event_type,
            tool_name,
            original_args,
            final_args,
            detail
        ) = row

        print("\n" + "-" * 50)
        print("日志编号：", log_id)
        print("时间 UTC：", created_at)
        print("事件类型：", event_type)
        print("工具名称：", tool_name or "-")
        print("原始参数：", original_args or "-")
        print("最终参数：", final_args or "-")
        print("说明：", detail or "-")


def find_previous_execution(tool_call_id: str) -> str | None:
    """根据工具调用 ID 查询它是否已经成功执行过。"""

    row = connection.execute(
        """
        SELECT result
        FROM tool_executions
        WHERE tool_call_id = ?
        """,
        (tool_call_id,)
    ).fetchone()

    return row[0] if row else None


def save_successful_execution(
    tool_call_id: str,
    tool_name: str,
    final_args: dict,
    result: str
) -> None:
    """保存成功执行凭证；重复 ID 不会产生第二条记录。"""

    connection.execute(
        """
        INSERT OR IGNORE INTO tool_executions (
            tool_call_id,
            thread_id,
            tool_name,
            final_args,
            result,
            executed_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            tool_call_id,
            args.thread_id,
            tool_name,
            json.dumps(final_args, ensure_ascii=False),
            result,
            datetime.now(timezone.utc).isoformat()
        )
    )
    connection.commit()


# ==================================================
# 4. 工具定义
# ==================================================

@tool
def check_balance(account: str) -> str:
    """查询模拟账户余额。这是只读操作。"""

    print(f"[执行只读工具] 查询账户 {account}")
    return f"模拟账户 {account} 的余额是1000元。"


@tool
def transfer_money(to_account: str, amount: float) -> str:
    """从 account-001 向指定账户执行模拟转账。"""

    print(
        f"[执行高风险工具] 从 {SOURCE_ACCOUNT} "
        f"向 {to_account} 转账 {amount} 元"
    )
    return (
        f"模拟转账成功：已从 {SOURCE_ACCOUNT} "
        f"向 {to_account} 转账 {amount} 元。"
    )


tools = [check_balance, transfer_money]
tools_by_name = {current_tool.name: current_tool for current_tool in tools}
approval_required_tools = {"transfer_money"}
llm_with_tools = llm.bind_tools(tools)


# ==================================================
# 5. 模型节点和路由
# ==================================================

def assistant(state: MessagesState):
    messages = [
        SystemMessage(
            content=(
                "你是一名模拟账户助手。付款账户固定为 account-001。"
                "查询余额使用 check_balance；具体转账使用 transfer_money。"
                "每次只提出一笔转账。tool_call 只是提案，不代表已执行。"
                "Python 会完成人工审批、参数校验和审计记录。"
                "收到工具结果前不能声称操作成功。"
            )
        ),
        *state["messages"]
    ]
    return {"messages": [llm_with_tools.invoke(messages)]}


def route_after_assistant(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "execute_tools"
    return END


# ==================================================
# 6. 确定性的业务参数校验
# ==================================================

def validate_transfer_args(tool_args: dict) -> str | None:
    to_account = tool_args.get("to_account")
    amount = tool_args.get("amount")

    if not isinstance(to_account, str):
        return "收款账户必须是字符串。"

    to_account = to_account.strip()

    if not re.fullmatch(r"account-\d{3}", to_account):
        return "收款账户格式不正确，正确示例为 account-002。"

    if to_account == SOURCE_ACCOUNT:
        return "收款账户不能与付款账户相同。"

    if not isinstance(amount, (int, float)):
        return "转账金额必须是数字。"

    amount = float(amount)

    if amount <= 0:
        return "转账金额必须大于0元。"

    if amount > MAX_TRANSFER_AMOUNT:
        return f"转账金额超过单笔限额 {MAX_TRANSFER_AMOUNT} 元。"

    return None


# ==================================================
# 7. 工具审批、校验、执行与审计节点
# ==================================================

def execute_tools(state: MessagesState):
    tool_calls = state["messages"][-1].tool_calls
    risky_tool_calls = [
        call for call in tool_calls
        if call["name"] in approval_required_tools
    ]

    decision = {"action": "approve"}

    if risky_tool_calls:
        approval_items = [
            {
                "tool_name": call["name"],
                "arguments": call["args"],
                "tool_call_id": call["id"]
            }
            for call in risky_tool_calls
        ]

        # interrupt 前可以安全记录“请求审批”。
        # 节点恢复时会从头重跑，所以用 tool_call_id 防止重复写入。
        for call in risky_tool_calls:
            already_logged = connection.execute(
                """
                SELECT 1 FROM audit_logs
                WHERE thread_id = ?
                  AND event_type = 'approval_requested'
                  AND tool_call_id = ?
                """,
                (args.thread_id, call["id"])
            ).fetchone()

            if not already_logged:
                write_audit_log(
                    "approval_requested",
                    tool_name=call["name"],
                    tool_call_id=call["id"],
                    original_args=call["args"],
                    detail="高风险工具等待人工审批。"
                )

        decision = interrupt({
            "question": "请选择批准、拒绝或修改参数。",
            "allowed_decisions": ["yes", "no", "edit"],
            "tool_calls": approval_items
        })

    action = decision.get("action", "reject")
    tool_messages = []

    for call in tool_calls:
        tool_name = call["name"]
        original_args = call["args"]
        tool_call_id = call["id"]
        requires_approval = tool_name in approval_required_tools
        final_args = original_args

        if requires_approval and action == "reject":
            tool_result = "用户拒绝了本次操作，工具没有执行。"
            write_audit_log(
                "rejected",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                original_args=original_args,
                detail="人工拒绝。"
            )

        else:
            if requires_approval and action == "edit":
                final_args = decision.get("new_args", original_args)
                write_audit_log(
                    "edited_and_approved",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    original_args=original_args,
                    final_args=final_args,
                    detail="人工修改参数并批准。"
                )
            elif requires_approval:
                write_audit_log(
                    "approved",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    original_args=original_args,
                    final_args=final_args,
                    detail="人工批准原始参数。"
                )
            else:
                write_audit_log(
                    "auto_approved",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    original_args=original_args,
                    final_args=final_args,
                    detail="只读工具无需人工审批。"
                )

            validation_error = None
            if tool_name == "transfer_money":
                validation_error = validate_transfer_args(final_args)

            if validation_error:
                print("[参数校验失败]", validation_error)
                tool_result = f"工具没有执行。参数校验失败：{validation_error}"
                write_audit_log(
                    "validation_failed",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    original_args=original_args,
                    final_args=final_args,
                    detail=validation_error
                )
            else:
                try:
                    # 幂等检查：相同 tool_call_id 如果已经执行成功，
                    # 直接复用旧结果，不再次调用有副作用的工具。
                    previous_result = find_previous_execution(tool_call_id)

                    if previous_result is not None:
                        execution_result = previous_result
                        execution_event = "execution_result_reused"
                        print("[幂等保护] 该工具调用已执行，复用原结果")
                        write_audit_log(
                            "duplicate_execution_prevented",
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            original_args=original_args,
                            final_args=final_args,
                            detail="检测到重复调用，没有再次执行工具。"
                        )
                    else:
                        execution_result = str(
                            tools_by_name[tool_name].invoke(final_args)
                        )
                        execution_event = "execution_succeeded"
                        save_successful_execution(
                            tool_call_id,
                            tool_name,
                            final_args,
                            execution_result
                        )

                    # 如果人工修改过参数，必须把“修改已获授权”
                    # 一并告诉模型。否则模型只能看到原始 tool_call
                    # 和不同参数的执行结果，可能误判为参数被篡改。
                    if requires_approval and action == "edit":
                        tool_result = (
                            "用户在人工审批阶段修改并批准了参数。"
                            f"原始提案参数：{original_args}；"
                            f"用户最终授权参数：{final_args}；"
                            f"工具执行结果：{execution_result}"
                        )
                    else:
                        tool_result = str(execution_result)

                    write_audit_log(
                        execution_event,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        original_args=original_args,
                        final_args=final_args,
                        detail=tool_result
                    )
                except Exception as error:
                    tool_result = (
                        f"工具执行失败。异常类型：{type(error).__name__}；"
                        f"异常信息：{error}。"
                    )
                    write_audit_log(
                        "execution_failed",
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        original_args=original_args,
                        final_args=final_args,
                        detail=tool_result
                    )

        tool_messages.append(
            ToolMessage(content=tool_result, tool_call_id=tool_call_id)
        )

    return {"messages": tool_messages}


# ==================================================
# 8. 构建带持久化的 LangGraph
# ==================================================

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("execute_tools", execute_tools)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {"execute_tools": "execute_tools", END: END}
)
builder.add_edge("execute_tools", "assistant")

checkpointer = SqliteSaver(connection)
graph = builder.compile(checkpointer=checkpointer)


# ==================================================
# 9. 暂停信息与恢复数据
# ==================================================

def show_interrupt(result) -> bool:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return False

    request = interrupts[0].value
    print("\n工作流已经暂停并保存。")
    print("流程编号：", args.thread_id)

    for number, call in enumerate(request["tool_calls"], start=1):
        print(f"\n待审批操作 {number}：")
        print("工具名称：", call["tool_name"])
        print("原始参数：", call["arguments"])

    print("\n可用命令：")
    for decision in ("yes", "no", "edit"):
        print(
            f"python day04/python/day4_safe_agent.py resume "
            f"{args.thread_id} {decision}"
        )
    return True


def create_resume_data() -> dict | None:
    if args.decision == "yes":
        return {"action": "approve"}
    if args.decision == "no":
        return {"action": "reject"}
    if args.decision == "edit":
        new_account = input("请输入新的收款账户：").strip()
        new_amount_text = input("请输入新的转账金额：").strip()
        try:
            new_amount = float(new_amount_text)
        except ValueError:
            print("金额必须是数字。")
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
# 10. 启动与恢复工作流
# ==================================================

def start_workflow() -> None:
    if graph.get_state(config).next:
        print("该 thread_id 已有暂停中的流程，请使用 resume。")
        return

    user_input = input("请输入请求：").strip()
    if not user_input:
        print("请求不能为空。")
        return

    write_audit_log("workflow_started", detail=user_input)
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    if not show_interrupt(result):
        print("\n助手：")
        print(result["messages"][-1].content)


def resume_workflow() -> None:
    if args.decision is None:
        print("resume 必须提供 yes、no 或 edit。")
        return

    if not graph.get_state(config).next:
        print("没有找到等待审批的流程。")
        return

    resume_data = create_resume_data()
    if resume_data is None:
        return

    print("找到暂停中的工作流。")
    print("审批数据：", resume_data)

    result = graph.invoke(Command(resume=resume_data), config=config)
    if not show_interrupt(result):
        print("\n助手：")
        print(result["messages"][-1].content)


# ==================================================
# 11. 程序入口
# ==================================================

try:
    if args.action == "start":
        start_workflow()
    elif args.action == "resume":
        resume_workflow()
    elif args.action == "logs":
        show_audit_logs()
finally:
    connection.close()
