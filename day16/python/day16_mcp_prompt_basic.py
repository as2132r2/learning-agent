"""Day16 第六步：使用正式 MCP SDK 发布和获取 Prompt。

运行方式：
    python day16/python/day16_mcp_prompt_basic.py

Prompt 不负责调用大模型。它负责根据参数生成可复用、可发现的消息模板，
客户端随后可以把这些消息交给任意兼容的大模型。
"""

from __future__ import annotations

import asyncio

import mcp_types as types
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver import AssistantMessage, Message, UserMessage


mcp_server = MCPServer(
    name="day16-prompt-service",
    title="Day16 提示词模板服务",
    description="向客户端提供编程教学和代码审查提示词。",
    version="1.0.0",
)


@mcp_server.prompt(
    name="teach-concept",
    title="讲解编程概念",
    description="根据学习者水平生成编程概念讲解任务。",
)
async def teach_concept(topic: str, level: str = "初学者") -> list[Message]:
    """生成编程概念教学提示词。"""

    return [
        AssistantMessage(
            "我会使用准确、循序渐进的方式讲解，并提供一个最小示例。"
        ),
        UserMessage(
            f"请面向{level}讲解“{topic}”。\n"
            "要求：\n"
            "1. 先用一句话给出定义；\n"
            "2. 再解释它解决什么问题；\n"
            "3. 给出一个简短的 Python 示例；\n"
            "4. 最后给出一道练习题。"
        ),
    ]


@mcp_server.prompt(
    name="review-code",
    title="审查代码",
    description="生成指定语言和关注点的代码审查提示词。",
)
async def review_code(
    language: str,
    focus: str = "正确性、安全性和可读性",
) -> UserMessage:
    """生成代码审查提示词。"""

    return UserMessage(
        f"你是一名资深 {language} 工程师。\n"
        f"请重点从“{focus}”审查用户接下来提供的代码。\n"
        "请按严重程度列出问题，并为每个问题给出修改建议。"
    )


def show_prompt_catalog(result: types.ListPromptsResult) -> None:
    """显示 Prompt 目录以及每个模板所需参数。"""

    print("\n" + "=" * 70)
    print("Prompt 数量：", len(result.prompts))

    for index, prompt in enumerate(result.prompts, start=1):
        print(f"\nPrompt {index}")
        print("名称：", prompt.name)
        print("标题：", prompt.title)
        print("说明：", prompt.description)
        print("参数：")

        for argument in prompt.arguments or []:
            print(
                f"- name={argument.name!r}，"
                f"required={argument.required}，"
                f"description={argument.description!r}"
            )


def show_rendered_prompt(title: str, result: types.GetPromptResult) -> None:
    """显示已经代入参数后的结构化 Prompt 消息。"""

    print("\n" + "=" * 70)
    print(title)
    print("返回对象类型：", type(result))
    print("模板说明：", result.description)
    print("消息数量：", len(result.messages))

    for index, message in enumerate(result.messages, start=1):
        print(f"\n消息 {index}")
        print("消息类型：", type(message))
        print("role：", message.role)
        print("content 类型：", type(message.content))

        if isinstance(message.content, types.TextContent):
            print("text：")
            print(message.content.text)
        else:
            print("content：", message.content)


async def main() -> None:
    """发现 Prompt，并使用不同参数获取渲染后的消息。"""

    print("Day16 MCP Prompt 客户端已启动。")

    client_info = types.Implementation(
        name="day16-prompt-client",
        version="1.0.0",
    )

    async with Client(
        mcp_server,
        client_info=client_info,
    ) as client:
        print("协议版本：", client.protocol_version)

        prompts_result = await client.list_prompts()
        show_prompt_catalog(prompts_result)

        teaching_prompt = await client.get_prompt(
            "teach-concept",
            arguments={
                "topic": "Python 协程",
                "level": "刚学完 Python 基础的初学者",
            },
        )
        show_rendered_prompt(
            "获取 teach-concept，并代入 topic、level",
            teaching_prompt,
        )

        review_prompt = await client.get_prompt(
            "review-code",
            arguments={
                "language": "Python",
                "focus": "异步并发和异常处理",
            },
        )
        show_rendered_prompt(
            "获取 review-code，并代入 language、focus",
            review_prompt,
        )

    print("\n" + "=" * 70)
    print("注意：本程序没有调用大模型，只生成了可以交给模型的消息。")
    print("\n本步数据流：")
    print("Prompt Python 函数")
    print("→ @mcp_server.prompt() 注册")
    print("→ Client.list_prompts() 发现模板和参数")
    print("→ Client.get_prompt(name, arguments) 传入参数")
    print("→ 服务端渲染 UserMessage / AssistantMessage")
    print("→ SDK 返回 GetPromptResult")
    print("→ 客户端可把 result.messages 交给大模型")


if __name__ == "__main__":
    asyncio.run(main())
