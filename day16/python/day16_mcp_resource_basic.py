"""Day16 第四步：使用正式 MCP SDK 发布和读取 Resource。

运行方式：
    python day16/python/day16_mcp_resource_basic.py

本步骤继续使用进程内连接，重点比较：
- Tool：让服务端执行某个动作；
- Resource：让客户端通过 URI 读取某份资料。
"""

from __future__ import annotations

import asyncio

import mcp_types as types
from mcp import Client
from mcp.server import MCPServer


mcp_server = MCPServer(
    name="day16-course-resources",
    title="Day16 课程资料服务",
    description="通过 MCP Resource 提供 LangChain 和 RAG 学习资料。",
    version="1.0.0",
)


# URI 是资源的稳定地址。客户端不需要知道资料实际来自文件、数据库还是 API。
@mcp_server.resource(
    "course://notes/langchain",
    name="langchain-note",
    title="LangChain 学习笔记",
    description="介绍 LangChain 的主要用途。",
    mime_type="text/plain",
)
async def read_langchain_note() -> str:
    """返回 LangChain 学习资料。"""

    print("[Resource Provider] 正在读取 LangChain 资料")
    return (
        "LangChain 是开发大语言模型应用的框架。\n"
        "它可以连接模型、提示词、工具、记忆和检索系统。"
    )


@mcp_server.resource(
    "course://notes/rag",
    name="rag-note",
    title="RAG 学习笔记",
    description="介绍 RAG 的基本工作流程。",
    mime_type="text/markdown",
)
async def read_rag_note() -> str:
    """返回 RAG 学习资料。"""

    print("[Resource Provider] 正在读取 RAG 资料")
    return (
        "# RAG\n\n"
        "RAG 的中文名称是检索增强生成。\n\n"
        "基本流程：问题 → 检索资料 → 组合上下文 → 模型生成回答。"
    )


def show_resource_list(result: types.ListResourcesResult) -> None:
    """逐项显示服务端公开的资源信息。"""

    print("\n" + "=" * 70)
    print("list_resources() 返回对象类型：", type(result))
    print("资源数量：", len(result.resources))

    for index, resource in enumerate(result.resources, start=1):
        print(f"\n资源 {index}")
        print("Python 类型：", type(resource))
        print("URI：", resource.uri)
        print("name：", resource.name)
        print("title：", resource.title)
        print("description：", resource.description)
        print("MIME 类型：", resource.mime_type)


def show_read_result(title: str, result: types.ReadResourceResult) -> None:
    """逐项显示 read_resource() 返回的资源正文。"""

    print("\n" + "=" * 70)
    print(title)
    print("返回对象类型：", type(result))
    print("contents 数量：", len(result.contents))

    for index, content in enumerate(result.contents, start=1):
        print(f"\n内容 {index}")
        print("内容对象类型：", type(content))
        print("URI：", content.uri)
        print("MIME 类型：", content.mime_type)

        if isinstance(content, types.TextResourceContents):
            print("文本正文：")
            print(content.text)
        else:
            print("这是二进制资源，内容保存在 blob 字段中。")


async def main() -> None:
    """连接资源服务，先发现资源，再读取两份资料。"""

    client_info = types.Implementation(
        name="day16-resource-client",
        version="1.0.0",
    )

    print("Day16 MCP Resource 客户端已启动。")
    print("连接方式：进程内连接")

    async with Client(
        mcp_server,
        client_info=client_info,
    ) as client:
        print("协议版本：", client.protocol_version)

        resources_result = await client.list_resources()
        show_resource_list(resources_result)

        langchain_result = await client.read_resource(
            "course://notes/langchain"
        )
        show_read_result("读取 course://notes/langchain", langchain_result)

        rag_result = await client.read_resource("course://notes/rag")
        show_read_result("读取 course://notes/rag", rag_result)

    print("\n" + "=" * 70)
    print("本步数据流：")
    print("Python 资源函数")
    print("→ @mcp_server.resource(uri) 注册")
    print("→ Client.list_resources() 发现 URI 和 metadata")
    print("→ Client.read_resource(uri) 请求正文")
    print("→ MCPServer 执行对应资源函数")
    print("→ SDK 返回 ReadResourceResult")


if __name__ == "__main__":
    asyncio.run(main())
