"""Day16 最终整合（客户端）：访问受保护的完整 MCP 服务。

先在终端 1 启动服务端：
    python day16/python/day16_final_mcp_server.py

再在终端 2 运行只读客户端：
    python day16/python/day16_final_mcp_client.py reader

或者运行操作员客户端：
    python day16/python/day16_final_mcp_client.py operator

本客户端会：
    1. 把 Bearer Token 放入 HTTP 请求头；
    2. 建立 Streamable HTTP MCP 连接；
    3. 发现 Tool、Resource 和 Prompt；
    4. 查询当前身份和课程资料；
    5. 读取 Resource、渲染 Prompt、查询余额；
    6. 根据 ToolAnnotations 判断转账需要审批；
    7. 只有操作员输入 yes 后才发送转账请求。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import httpx2
import mcp_types as types
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


SERVER_URL = "http://127.0.0.1:8004/mcp"

TOKENS: dict[str, str] = {
    "reader": "day16-final-reader-token",
    "operator": "day16-final-operator-token",
}

# 即使服务端没有提供 annotations，客户端也知道这个名称属于高风险操作。
HIGH_RISK_TOOL_NAMES = {"transfer_money"}


def line() -> None:
    """打印分隔线，让终端结果更容易阅读。"""

    print("\n" + "=" * 72)


def show_call_result(title: str, result: types.CallToolResult) -> None:
    """统一显示 MCP 工具成功结果或工具错误。"""

    line()
    print(title)
    print("is_error：", result.is_error)
    print("structured_content：", result.structured_content)

    for content in result.content:
        if isinstance(content, types.TextContent):
            print("text：", content.text)


def get_text_from_resource(result: types.ReadResourceResult) -> str:
    """从 ReadResourceResult 中提取第一个文本资源。"""

    if not result.contents:
        return ""

    content = result.contents[0]
    if isinstance(content, types.TextResourceContents):
        return content.text

    return "[这是二进制资源，当前示例不显示 blob]"


def get_prompt_text(result: types.GetPromptResult) -> list[tuple[str, str]]:
    """把 MCP Prompt 的结构化消息转换成便于打印的数据。"""

    messages: list[tuple[str, str]] = []

    for message in result.messages:
        if isinstance(message.content, types.TextContent):
            messages.append((message.role, message.content.text))
        else:
            messages.append((message.role, str(message.content)))

    return messages


def requires_approval(tool: types.Tool) -> bool:
    """依据客户端规则和 MCP 安全注解判断是否需要人工审批。"""

    if tool.name in HIGH_RISK_TOOL_NAMES:
        return True

    annotations = tool.annotations

    # 没有安全声明时采用保守策略。
    if annotations is None:
        return True

    return bool(
        annotations.destructive_hint
        or annotations.read_only_hint is not True
    )


def show_catalogs(
    tools_result: types.ListToolsResult,
    resources_result: types.ListResourcesResult,
    prompts_result: types.ListPromptsResult,
) -> dict[str, types.Tool]:
    """显示服务端能力目录，并返回按名称索引的工具字典。"""

    line()
    print("服务端能力目录")

    tools_by_name = {tool.name: tool for tool in tools_result.tools}

    print("\nTools：")
    for tool in tools_result.tools:
        annotations = tool.annotations
        print(f"- {tool.name}：{tool.description}")
        if annotations is not None:
            print(
                "  "
                f"readOnly={annotations.read_only_hint}，"
                f"destructive={annotations.destructive_hint}，"
                f"idempotent={annotations.idempotent_hint}"
            )

    print("\nResources：")
    for resource in resources_result.resources:
        print(f"- {resource.uri}：{resource.title}")

    print("\nPrompts：")
    for prompt in prompts_result.prompts:
        print(f"- {prompt.name}：{prompt.title}")

    return tools_by_name


async def run_demo(client: Client[Any], role: str) -> None:
    """完成一次 Tool、Resource、Prompt 和审批的综合演示。"""

    # 发现阶段：客户端先询问服务端公开了哪些标准能力。
    tools_result = await client.list_tools(cache_mode="bypass")
    resources_result = await client.list_resources(cache_mode="bypass")
    prompts_result = await client.list_prompts(cache_mode="bypass")

    tools_by_name = show_catalogs(
        tools_result,
        resources_result,
        prompts_result,
    )

    # Tool 1：查询服务端解析出来的当前身份。
    identity_result = await client.call_tool("whoami")
    show_call_result("1. 调用 whoami", identity_result)

    identity: dict[str, Any] = identity_result.structured_content or {}
    current_scopes = set(identity.get("scopes", []))

    # Tool 2：通过工具查询课程知识。
    course_result = await client.call_tool(
        "search_course",
        arguments={"topic": "mcp"},
    )
    show_call_result("2. 调用 search_course(topic='mcp')", course_result)

    # Resource：通过 URI 直接读取资料。
    resource_result = await client.read_resource(
        "course://notes/rag",
        cache_mode="bypass",
    )
    line()
    print("3. 读取 Resource：course://notes/rag")
    print(get_text_from_resource(resource_result))

    # Prompt：让服务端渲染结构化消息，但这里不调用大模型。
    prompt_result = await client.get_prompt(
        "teach-topic",
        arguments={
            "topic": "MCP Tool、Resource 与 Prompt 的区别",
            "level": "LangChain 初学者",
        },
    )
    line()
    print("4. 获取 Prompt：teach-topic")
    for message_role, message_text in get_prompt_text(prompt_result):
        print(f"{message_role}：{message_text}")
    print("说明：这里只取得 Prompt，没有调用大模型。")

    # Tool 3：只读余额查询，根据注解可自动执行。
    balance_tool = tools_by_name["check_balance"]
    line()
    print("5. 准备调用 check_balance")
    print("是否需要人工审批：", requires_approval(balance_tool))

    balance_result = await client.call_tool(
        "check_balance",
        arguments={"account": "account-001"},
    )
    show_call_result("check_balance 返回", balance_result)

    # Tool 4：转账具有副作用，必须同时通过客户端审批和服务端授权。
    transfer_tool = tools_by_name["transfer_money"]
    transfer_arguments = {
        "from_account": "account-001",
        "to_account": "account-002",
        "amount": 100,
    }

    line()
    print("6. 程序提议调用 transfer_money")
    print("参数：", transfer_arguments)
    print("是否需要人工审批：", requires_approval(transfer_tool))

    if "account:write" not in current_scopes:
        print(f"当前角色 {role!r} 没有 account:write。")
        print("客户端提前阻止请求，没有询问审批，也没有发送 tools/call。")
        return

    decision = input("是否批准转账？输入 yes 批准，输入 no 拒绝：").strip()

    if decision.lower() != "yes":
        print("用户拒绝，客户端没有发送 transfer_money 请求。")
        return

    transfer_result = await client.call_tool(
        "transfer_money",
        arguments=transfer_arguments,
    )
    show_call_result("transfer_money 返回", transfer_result)


async def main() -> None:
    """读取角色、建立带认证头的 MCP 连接并运行综合演示。"""

    role = sys.argv[1].lower() if len(sys.argv) == 2 else "reader"
    token = TOKENS.get(role)

    if token is None:
        print("角色只能是 reader 或 operator。")
        print("示例：python day16/python/day16_final_mcp_client.py reader")
        return

    print("Day16 最终 MCP Client 已启动。")
    print("当前角色：", role)
    print("服务地址：", SERVER_URL)
    print("Authorization：Bearer ******（终端不显示完整令牌）")

    headers = {"Authorization": f"Bearer {token}"}

    # httpx2.AsyncClient 负责普通 HTTP 请求头。
    async with httpx2.AsyncClient(headers=headers) as http_client:
        # transport 把 MCP 消息放到 Streamable HTTP 上发送。
        transport = streamable_http_client(
            SERVER_URL,
            http_client=http_client,
        )

        try:
            # Client 负责 MCP 协议协商以及标准方法调用。
            async with Client(transport) as client:
                print("认证成功。")
                print("协商的 MCP 协议版本：", client.protocol_version)
                await run_demo(client, role)
        except Exception as error:
            print("\n连接、认证或 MCP 调用失败。")
            print("错误类型：", type(error).__name__)
            print("错误信息：", error)
            print("请确认最终服务端正在另一个终端的 8004 端口运行。")
            return

    line()
    print("本次综合数据流：")
    print("角色 → Token → Authorization 请求头")
    print("→ Streamable HTTP → TokenVerifier → AccessToken")
    print("→ list_tools / list_resources / list_prompts")
    print("→ Tool / Resource / Prompt 调用")
    print("→ 客户端风险判断 + 服务端 scope 检查")
    print("→ 业务结果 + SQLite 审计日志")


if __name__ == "__main__":
    asyncio.run(main())
