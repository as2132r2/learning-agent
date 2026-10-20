"""Day16 第十步：LangChain Agent 通过 HTTP 使用远程 MCP Tool。

终端一先启动 MCP Server：
    python day16/python/day16_mcp_http_server.py

终端二再启动本程序：
    python day16/python/day16_mcp_http_agent.py

推荐测试问题：
    请计算 (20+4)*3。
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI


load_dotenv()

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


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


def show_remote_tools(tools: list) -> None:
    """显示从远程 MCP Server 发现并转换的工具。"""

    print("\n" + "=" * 70)
    print("远程 MCP 工具已经转换为 LangChain 工具：")

    for index, tool in enumerate(tools, start=1):
        print(f"\n工具 {index}")
        print("类型：", type(tool))
        print("名称：", tool.name)
        print("说明：", tool.description)
        print("参数：", tool.args)

        metadata = tool.metadata or {}
        print("MCP metadata：", metadata.get("mcp"))


def show_message_flow(messages: list) -> None:
    """显示模型、用户和远程工具之间产生的消息。"""

    print("\n" + "=" * 70)
    print("完整消息流程：")

    for index, message in enumerate(messages, start=1):
        print(f"\n第 {index} 条：{type(message).__name__}")
        print("content：", repr(message.content))

        if isinstance(message, AIMessage) and message.tool_calls:
            print("tool_calls：", message.tool_calls)

        if isinstance(message, ToolMessage):
            print("工具名称：", message.name)
            print("tool_call_id：", message.tool_call_id)


async def main() -> None:
    """连接远程 MCP Server，让 Agent 自主选择并调用工具。"""

    question = input("请输入问题：").strip()
    if not question:
        print("问题不能为空。")
        return

    llm = create_llm()

    print("\n正在连接远程 MCP Server：", MCP_SERVER_URL)

    try:
        # URL 表明这是网络 MCP Server，不是进程内服务对象。
        async with MCPAdapter(MCP_SERVER_URL) as adapter:
            tools = await adapter.list_tools(cache_mode="refresh")
            show_remote_tools(tools)

            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=(
                    "你是一名简洁、可靠的计算助手。"
                    "遇到数学运算必须调用工具。"
                    "复杂表达式需要按照运算顺序分步调用工具。"
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
    except Exception as error:
        print("\n连接或执行失败。")
        print("错误类型：", type(error).__name__)
        print("错误信息：", error)
        print("请确认 day16_mcp_http_server.py 已在另一个终端运行。")
        return

    final_message = result["messages"][-1]

    print("\n" + "=" * 70)
    print("最终回答：")
    print(final_message.content)

    show_message_flow(result["messages"])

    print("\n" + "=" * 70)
    print("本步数据流：")
    print("用户问题")
    print("→ DeepSeek 生成 tool_calls")
    print("→ LangChain ToolNode")
    print("→ 适配后的 BaseTool")
    print("→ MCPAdapter")
    print("→ Streamable HTTP")
    print("→ 独立 MCP Server")
    print("→ MCP Tool 执行")
    print("→ HTTP 返回结果")
    print("→ ToolMessage")
    print("→ DeepSeek 生成最终回答")


if __name__ == "__main__":
    asyncio.run(main())
