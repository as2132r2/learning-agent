"""Day16 第十七步：为 MCP Server 编写自动化契约测试。

运行方式：
    python day16/python/day16_mcp_contract_tests.py

契约测试关注客户端可观察到的接口：
- 工具是否存在；
- 工具参数 Schema 是否符合约定；
- 安全注解是否保留；
- 工具成功和失败结果是否正确；
- Resource 和 Prompt 是否能够发现、读取和渲染。

全部测试使用进程内 MCP Client，不访问网络，也不调用大模型。
"""

from __future__ import annotations

import unittest

import mcp_types as types
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver import UserMessage
from mcp.server.mcpserver.exceptions import ToolError


mcp_server = MCPServer(
    name="day16-contract-server",
    title="Day16 契约测试服务",
    version="1.0.0",
)


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def divide(a: float, b: float) -> float:
    """计算 a 除以 b；b 不能为零。"""

    if b == 0:
        raise ToolError("除数 b 不能为 0。")
    return a / b


@mcp_server.resource(
    "course://notes/mcp",
    name="mcp-note",
    title="MCP 学习笔记",
    mime_type="text/plain",
)
async def read_mcp_note() -> str:
    """返回 MCP 基础笔记。"""

    return "MCP 通过统一协议向大模型应用提供 Tool、Resource 和 Prompt。"


@mcp_server.prompt(
    name="explain-topic",
    title="解释主题",
    description="生成面向指定学习水平的解释请求。",
)
async def explain_topic(topic: str, level: str = "初学者") -> UserMessage:
    """返回课程讲解提示词。"""

    return UserMessage(f"请面向{level}解释“{topic}”。")


class MCPContractTests(unittest.IsolatedAsyncioTestCase):
    """验证客户端依赖的 MCP 公共契约。"""

    async def test_server_identity_and_protocol_version(self) -> None:
        """客户端应能协商当前协议并获得服务端身份。"""

        async with Client(mcp_server) as client:
            self.assertEqual(client.protocol_version, "2026-07-28")
            self.assertIsNotNone(client.server_info)
            assert client.server_info is not None
            self.assertEqual(client.server_info.name, "day16-contract-server")
            self.assertEqual(client.server_info.version, "1.0.0")

    async def test_tool_catalog_and_input_schema(self) -> None:
        """divide 必须存在，并要求 a、b 两个 number 参数。"""

        async with Client(mcp_server) as client:
            result = await client.list_tools(cache_mode="bypass")

        self.assertEqual([tool.name for tool in result.tools], ["divide"])

        tool = result.tools[0]
        properties = tool.input_schema["properties"]
        required = set(tool.input_schema["required"])

        self.assertEqual(properties["a"]["type"], "number")
        self.assertEqual(properties["b"]["type"], "number")
        self.assertEqual(required, {"a", "b"})

    async def test_tool_security_annotations(self) -> None:
        """divide 的只读、非破坏性和幂等注解不能丢失。"""

        async with Client(mcp_server) as client:
            tool = (await client.list_tools(cache_mode="bypass")).tools[0]

        self.assertIsNotNone(tool.annotations)
        assert tool.annotations is not None
        self.assertTrue(tool.annotations.read_only_hint)
        self.assertFalse(tool.annotations.destructive_hint)
        self.assertTrue(tool.annotations.idempotent_hint)
        self.assertFalse(tool.annotations.open_world_hint)

    async def test_tool_success_result(self) -> None:
        """divide(20, 4) 应返回结构化结果 5.0。"""

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "divide",
                arguments={"a": 20, "b": 4},
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, {"result": 5.0})

    async def test_tool_business_error(self) -> None:
        """除数为零应成为工具错误，而不是成功结果。"""

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "divide",
                arguments={"a": 20, "b": 0},
            )

        self.assertTrue(result.is_error)
        self.assertIsNone(result.structured_content)
        self.assertIsInstance(result.content[0], types.TextContent)
        assert isinstance(result.content[0], types.TextContent)
        self.assertIn("除数 b 不能为 0", result.content[0].text)

    async def test_resource_catalog_and_content(self) -> None:
        """MCP 笔记应能被发现并读取。"""

        async with Client(mcp_server) as client:
            catalog = await client.list_resources(cache_mode="bypass")
            result = await client.read_resource(
                "course://notes/mcp",
                cache_mode="bypass",
            )

        self.assertEqual(len(catalog.resources), 1)
        self.assertEqual(str(catalog.resources[0].uri), "course://notes/mcp")
        self.assertEqual(len(result.contents), 1)
        self.assertIsInstance(result.contents[0], types.TextResourceContents)
        assert isinstance(result.contents[0], types.TextResourceContents)
        self.assertIn("Tool、Resource 和 Prompt", result.contents[0].text)

    async def test_prompt_catalog_and_rendering(self) -> None:
        """Prompt 参数应该正确渲染成 user 消息。"""

        async with Client(mcp_server) as client:
            catalog = await client.list_prompts(cache_mode="bypass")
            result = await client.get_prompt(
                "explain-topic",
                arguments={
                    "topic": "MCP",
                    "level": "Python 初学者",
                },
            )

        self.assertEqual([prompt.name for prompt in catalog.prompts], ["explain-topic"])
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].role, "user")
        self.assertIsInstance(result.messages[0].content, types.TextContent)
        assert isinstance(result.messages[0].content, types.TextContent)
        self.assertEqual(
            result.messages[0].content.text,
            "请面向Python 初学者解释“MCP”。",
        )


if __name__ == "__main__":
    print("开始运行 Day16 MCP 契约测试……", flush=True)
    unittest.main(verbosity=2)
