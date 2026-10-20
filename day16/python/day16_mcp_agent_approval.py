"""Day16 第十四步：根据 MCP ToolAnnotations 自动建立 Agent 审批策略。

运行方式：
    python day16/python/day16_mcp_agent_approval.py

建议输入：
    请从 account-001 向 account-002 转账100元。

模型可以提出 transfer_money 调用，但 HumanInTheLoopMiddleware 会在
MCP tools/call 发送前暂停。用户批准后才会真正执行远程工具。
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Literal

import mcp_types as types
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


load_dotenv()


def create_llm() -> ChatOpenAI:
    """从 .env 创建 DeepSeek 模型。"""

    model = os.getenv("DEEPSEEK_MODEL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    missing = [
        name
        for name, value in {
            "DEEPSEEK_MODEL": model,
            "DEEPSEEK_API_KEY": api_key,
            "DEEPSEEK_BASE_URL": base_url,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(".env 缺少配置：" + "、".join(missing))

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


mcp_server = MCPServer(
    name="day16-approved-finance",
    title="Day16 带审批的模拟账户服务",
    version="1.0.0",
)


ACCOUNTS: dict[str, float] = {
    "account-001": 1000.0,
    "account-002": 200.0,
}


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def check_balance(
    account: Literal["account-001", "account-002"],
) -> dict[str, Any]:
    """查询指定模拟账户的余额。"""

    print(f"[执行只读 MCP 工具] 查询 {account}", file=sys.stderr)
    return {"account": account, "balance": ACCOUNTS[account]}


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    )
)
async def transfer_money(
    from_account: Literal["account-001", "account-002"],
    to_account: Literal["account-001", "account-002"],
    amount: float,
) -> dict[str, Any]:
    """在两个模拟账户之间转账。"""

    if from_account == to_account:
        raise ToolError("付款账户与收款账户不能相同。")
    if amount <= 0:
        raise ToolError("转账金额必须大于 0。")
    if ACCOUNTS[from_account] < amount:
        raise ToolError("付款账户余额不足。")

    print(
        f"[执行高风险 MCP 工具] {from_account} → {to_account}，{amount} 元",
        file=sys.stderr,
        flush=True,
    )
    ACCOUNTS[from_account] -= amount
    ACCOUNTS[to_account] += amount

    return {
        "status": "success",
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "balances": dict(ACCOUNTS),
    }


def get_annotations(tool: Any) -> dict[str, Any]:
    """从 MCPAdapter 生成的 LangChain Tool metadata 中读取注解。"""

    metadata = tool.metadata or {}
    mcp_metadata = metadata.get("mcp", {})
    tool_metadata = mcp_metadata.get("tool", {})
    annotations = tool_metadata.get("annotations", {})
    return annotations if isinstance(annotations, dict) else {}


def build_interrupt_policy(tools: list) -> dict[str, Any]:
    """根据 MCP 注解生成 HumanInTheLoopMiddleware 配置。"""

    policy: dict[str, Any] = {}

    for tool in tools:
        annotations = get_annotations(tool)
        is_read_only = annotations.get("read_only_hint") is True
        is_destructive = annotations.get("destructive_hint") is True

        if is_read_only and not is_destructive:
            policy[tool.name] = False
            print(f"[审批策略] {tool.name}：只读，自动执行")
        else:
            policy[tool.name] = {
                "allowed_decisions": ["approve", "reject"],
                "description": (
                    f"MCP 工具 {tool.name} 可能修改数据，执行前需要审批。"
                ),
            }
            print(f"[审批策略] {tool.name}：高风险，需要人工审批")

    return policy


def show_interrupts(interrupts: tuple) -> None:
    """显示 LangGraph 中断携带的待审批工具调用。"""

    print("\n" + "=" * 70)
    print("Agent 已暂停，等待人工审批。")

    for graph_interrupt in interrupts:
        request = graph_interrupt.value
        print("中断编号：", graph_interrupt.id)

        for index, action in enumerate(request["action_requests"], start=1):
            print(f"\n待审批操作 {index}")
            print("工具：", action["name"])
            print("参数：", action["args"])
            print("说明：", action.get("description"))


async def main() -> None:
    """运行一个带 MCP 注解驱动审批的 Agent。"""

    question = input("请输入请求：").strip()
    if not question:
        print("请求不能为空。")
        return

    llm = create_llm()
    config = {"configurable": {"thread_id": "day16-approval-001"}}

    async with MCPAdapter(mcp_server) as adapter:
        tools = await adapter.list_tools(cache_mode="refresh")
        interrupt_policy = build_interrupt_policy(tools)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "你是模拟账户助手。付款账户默认是 account-001。"
                "查询余额和转账都必须使用工具，不得声称执行未调用的操作。"
            ),
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on=interrupt_policy,
                )
            ],
            checkpointer=InMemorySaver(),
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config=config,
        )

        interrupts = result.get("__interrupt__", ())

        if interrupts:
            show_interrupts(interrupts)
            decision = input(
                "\n是否批准？输入 yes 批准，输入 no 拒绝："
            ).strip().lower()

            if decision == "yes":
                resume_data = {
                    "decisions": [{"type": "approve"}]
                }
            else:
                resume_data = {
                    "decisions": [
                        {
                            "type": "reject",
                            "message": "用户拒绝执行该高风险操作。",
                        }
                    ]
                }

            result = await agent.ainvoke(
                Command(resume=resume_data),
                config=config,
            )

    print("\n" + "=" * 70)
    print("助手最终回答：")
    print(result["messages"][-1].content)
    print("最终账户状态：", ACCOUNTS)

    print("\n本步安全数据流：")
    print("MCP tools/list → ToolAnnotations")
    print("→ build_interrupt_policy()")
    print("→ HumanInTheLoopMiddleware")
    print("→ 模型提出 transfer_money")
    print("→ LangGraph interrupt（尚未调用 MCP 工具）")
    print("→ 人工 approve / reject")
    print("→ approve 后才由 MCPAdapter 发送 tools/call")


if __name__ == "__main__":
    asyncio.run(main())
