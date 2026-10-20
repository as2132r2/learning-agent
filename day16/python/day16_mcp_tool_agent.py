"""Day16 第八步：把 MCP Tool 转换成 LangChain Tool，并交给 Agent。

运行方式：
    python day16/python/day16_mcp_tool_agent.py

推荐测试问题：
    请计算 (20+4)*3。

本步骤使用 LangChain 1.4 内置的 MCPAdapter。它会把 MCP Server 发布的
工具转换成 LangChain BaseTool，随后 create_agent 可以让模型自主选择工具。
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from mcp.server import MCPServer


load_dotenv()


def create_llm() -> ChatOpenAI:
    """使用项目根目录 .env 中的 DeepSeek 配置创建模型。"""

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


# ---------------------------------------------------------------------------
# 第一层：MCP Server 发布工具
# ---------------------------------------------------------------------------

mcp_server = MCPServer(
    name="day16-calculator-server",
    title="Day16 MCP 计算器",
    description="向 Agent 提供加法和乘法工具。",
    version="1.0.0",
)


@mcp_server.tool()
async def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    print(f"[MCP Tool 执行 add] {a} + {b}", file=sys.stderr, flush=True)
    return a + b


@mcp_server.tool()
async def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(
        f"[MCP Tool 执行 multiply] {a} × {b}",
        file=sys.stderr,
        flush=True,
    )
    return a * b


def show_adapted_tools(tools: list) -> None:
    """显示 MCPAdapter 生成的 LangChain 工具。"""

    print("\n" + "=" * 70)
    print("MCPAdapter 转换后的 LangChain 工具：")

    for index, tool in enumerate(tools, start=1):
        print(f"\n工具 {index}")
        print("Python 类型：", type(tool))
        print("名称：", tool.name)
        print("说明：", tool.description)
        print("参数 Schema：", tool.args)
        print("metadata：", tool.metadata)


def show_message_flow(messages: list) -> None:
    """显示 Agent 产生的模型消息和工具消息。"""

    print("\n" + "=" * 70)
    print("完整消息流程：")

    for index, message in enumerate(messages, start=1):
        print(f"\n第 {index} 条：{type(message).__name__}")
        print("content：", repr(message.content))

        if isinstance(message, AIMessage) and message.tool_calls:
            print("tool_calls：", message.tool_calls)

        if isinstance(message, ToolMessage):
            print("tool_call_id：", message.tool_call_id)
            print("工具名称：", message.name)


async def main() -> None:
    """发现并转换 MCP 工具，然后让 Agent 自主完成用户问题。"""

    question = input("请输入问题：").strip()
    if not question:
        print("问题不能为空。")
        return

    llm = create_llm()

    # MCPAdapter 的上下文必须覆盖工具的整个使用周期。
    async with MCPAdapter(mcp_server) as adapter:
        # list_tools() 从 MCP Server 发现工具，并转换成 LangChain BaseTool。
        tools = await adapter.list_tools()
        show_adapted_tools(tools)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "你是一名简洁、可靠的计算助手。"
                "遇到数学运算必须使用可用工具，不能心算猜测。"
                "复杂表达式要按照运算顺序分步调用工具。"
            ),
        )

        print("\n" + "=" * 70)
        print("Agent 开始处理问题：", question)

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

    final_message = result["messages"][-1]

    print("\n" + "=" * 70)
    print("最终回答：")
    print(final_message.content)

    show_message_flow(result["messages"])

    print("\n" + "=" * 70)
    print("本步数据流：")
    print("@mcp_server.tool()")
    print("→ MCP Tool")
    print("→ MCPAdapter.list_tools()")
    print("→ LangChain BaseTool")
    print("→ create_agent(model, tools)")
    print("→ 模型生成 tool_calls")
    print("→ LangChain ToolNode 调用适配后的工具")
    print("→ MCPAdapter 调用 MCP Server")
    print("→ MCP Tool 执行并返回结果")
    print("→ ToolMessage 回到模型")
    print("→ 模型生成最终回答")


if __name__ == "__main__":
    asyncio.run(main())
