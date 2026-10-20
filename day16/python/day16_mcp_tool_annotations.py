"""Day16 第十三步：读取 MCP Tool 安全注解并执行客户端审批策略。

运行方式：
    python day16/python/day16_mcp_tool_annotations.py

本步骤演示：
- check_balance 是只读、幂等工具，可以自动执行；
- transfer_money 会修改数据且非幂等，执行前必须人工确认；
- ToolAnnotations 是提示信息，不是自动生效的安全机制。
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import mcp_types as types
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


mcp_server = MCPServer(
    name="day16-safe-finance",
    title="Day16 安全工具服务",
    description="演示只读工具、高风险工具和客户端审批。",
    version="1.0.0",
)


# 模拟服务端账户数据。
ACCOUNTS: dict[str, float] = {
    "account-001": 1000.0,
    "account-002": 200.0,
}


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        title="查询账户余额",
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

    print(f"[执行只读工具] 查询 {account}")
    return {
        "account": account,
        "balance": ACCOUNTS[account],
    }


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        title="模拟转账",
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
        raise ToolError("付款账户和收款账户不能相同。")
    if amount <= 0:
        raise ToolError("转账金额必须大于 0。")
    if ACCOUNTS[from_account] < amount:
        raise ToolError("付款账户余额不足。")

    print(
        f"[执行高风险工具] {from_account} → {to_account}，{amount} 元"
    )
    ACCOUNTS[from_account] -= amount
    ACCOUNTS[to_account] += amount

    return {
        "status": "success",
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "from_balance": ACCOUNTS[from_account],
        "to_balance": ACCOUNTS[to_account],
    }


# 安全策略不能只相信服务端自我声明的 annotations，还要有客户端自己的规则。
HIGH_RISK_TOOL_NAMES = {"transfer_money"}


def show_tool_annotations(tool: types.Tool) -> None:
    """显示一个工具的 MCP 安全注解。"""

    annotations = tool.annotations

    print(f"\n工具：{tool.name}")
    print("说明：", tool.description)
    print("annotations 类型：", type(annotations))

    if annotations is None:
        print("该工具没有声明安全注解。")
        return

    print("read_only_hint：", annotations.read_only_hint)
    print("destructive_hint：", annotations.destructive_hint)
    print("idempotent_hint：", annotations.idempotent_hint)
    print("open_world_hint：", annotations.open_world_hint)


def requires_approval(tool: types.Tool) -> bool:
    """结合客户端 allowlist 和服务端注解判断是否需要审批。"""

    if tool.name in HIGH_RISK_TOOL_NAMES:
        return True

    annotations = tool.annotations
    if annotations is None:
        # 未声明风险信息时采取保守策略。
        return True

    return bool(
        annotations.destructive_hint
        or annotations.read_only_hint is not True
    )


def show_result(result: types.CallToolResult) -> None:
    """显示工具执行结果。"""

    print("is_error：", result.is_error)
    print("structured_content：", result.structured_content)

    for content in result.content:
        if isinstance(content, types.TextContent):
            print("text：", content.text)


async def main() -> None:
    """自动执行只读工具，审批后才执行转账工具。"""

    print("Day16 MCP Tool 安全注解练习已启动。")

    async with Client(mcp_server) as client:
        tools_result = await client.list_tools()
        tools_by_name = {tool.name: tool for tool in tools_result.tools}

        print("\n" + "=" * 70)
        print("服务端公开的工具及安全注解：")
        for tool in tools_result.tools:
            show_tool_annotations(tool)

        balance_tool = tools_by_name["check_balance"]
        transfer_tool = tools_by_name["transfer_money"]

        print("\n" + "=" * 70)
        print("准备调用 check_balance(account='account-001')")
        print("是否需要审批：", requires_approval(balance_tool))

        balance_result = await client.call_tool(
            "check_balance",
            arguments={"account": "account-001"},
        )
        show_result(balance_result)

        transfer_arguments = {
            "from_account": "account-001",
            "to_account": "account-002",
            "amount": 100,
        }

        print("\n" + "=" * 70)
        print("模型或程序提议调用 transfer_money")
        print("参数：", transfer_arguments)
        print("是否需要审批：", requires_approval(transfer_tool))

        decision = input("是否批准执行？输入 yes 批准，输入 no 拒绝：").strip()

        if decision.lower() == "yes":
            transfer_result = await client.call_tool(
                "transfer_money",
                arguments=transfer_arguments,
            )
            print("\n转账工具返回：")
            show_result(transfer_result)
        else:
            print("\n用户没有批准，客户端没有发送 tools/call 请求。")

        print("\n最终账户状态：", ACCOUNTS)

    print("\n" + "=" * 70)
    print("安全判断数据流：")
    print("MCP tools/list")
    print("→ Tool.annotations")
    print("→ 客户端风险规则 + HIGH_RISK_TOOL_NAMES")
    print("→ 只读工具自动执行")
    print("→ 高风险工具等待人工审批")
    print("→ 批准后才发送 tools/call")


if __name__ == "__main__":
    asyncio.run(main())
