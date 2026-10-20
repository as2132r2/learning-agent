"""Day16 第一步：理解 MCP 底层使用的 JSON-RPC 消息结构。

这是一个“教学模拟器”，重点观察：
1. 客户端怎样列出工具；
2. 客户端怎样请求调用工具；
3. 服务端怎样用相同的 id 返回结果；
4. 请求不合法时怎样返回标准 JSON-RPC 错误。

它还不是真正通过网络或 stdio 通信的 MCP Server，后续步骤会逐渐实现。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


# JSON 可以表示的任意值。这里用 Any，方便初学时阅读。
JsonValue = Any
JsonObject = dict[str, JsonValue]


def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""

    print(f"[服务端执行 multiply] {a} × {b}")
    return a * b


# 服务端内部的工具注册表：工具名 → 可调用的 Python 函数。
TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "multiply": multiply,
}


# 客户端通过 tools/list 看到的是工具说明，而不是 Python 函数本身。
TOOL_DESCRIPTIONS: list[JsonObject] = [
    {
        "name": "multiply",
        "description": "计算两个整数的乘积。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "第一个整数"},
                "b": {"type": "integer", "description": "第二个整数"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    }
]


def success_response(request_id: JsonValue, result: JsonObject) -> JsonObject:
    """创建成功响应；响应 id 必须和请求 id 一致。"""

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def error_response(
    request_id: JsonValue,
    code: int,
    message: str,
) -> JsonObject:
    """创建 JSON-RPC 错误响应。"""

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def validate_multiply_arguments(arguments: JsonObject) -> tuple[int, int]:
    """验证 multiply 参数，并返回经过验证的 a、b。"""

    if set(arguments) != {"a", "b"}:
        raise ValueError("multiply 只接受并且必须包含 a、b 两个参数。")

    a = arguments["a"]
    b = arguments["b"]

    # bool 是 int 的子类，因此需要额外排除 True 和 False。
    if isinstance(a, bool) or not isinstance(a, int):
        raise ValueError("参数 a 必须是整数。")
    if isinstance(b, bool) or not isinstance(b, int):
        raise ValueError("参数 b 必须是整数。")

    return a, b


def handle_jsonrpc(request: JsonObject) -> JsonObject:
    """接收一条 JSON-RPC 请求，并分派给对应的方法。"""

    request_id = request.get("id")

    if request.get("jsonrpc") != "2.0" or "id" not in request:
        return error_response(request_id, -32600, "Invalid Request")

    method = request.get("method")
    params = request.get("params", {})

    if not isinstance(params, dict):
        return error_response(request_id, -32602, "Invalid params")

    # 发现工具：客户端询问服务端提供哪些工具。
    if method == "tools/list":
        return success_response(request_id, {"tools": TOOL_DESCRIPTIONS})

    # 调用工具：客户端提供工具名和参数，服务端实际执行函数。
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments")

        if tool_name not in TOOL_FUNCTIONS:
            return error_response(request_id, -32602, "未知工具名称。")
        if not isinstance(arguments, dict):
            return error_response(request_id, -32602, "arguments 必须是对象。")

        try:
            # 当前只有一个工具；下一步会把验证和执行改成更通用的形式。
            a, b = validate_multiply_arguments(arguments)
            value = TOOL_FUNCTIONS[tool_name](a=a, b=b)
        except ValueError as error:
            return error_response(request_id, -32602, str(error))
        except Exception:
            # 不把内部异常细节直接泄露给客户端。
            return error_response(request_id, -32603, "Internal error")

        return success_response(
            request_id,
            {
                # content 适合模型读取；文本必须是字符串。
                "content": [{"type": "text", "text": str(value)}],
                # structuredContent 适合程序继续处理结构化结果。
                "structuredContent": {"result": value},
                "isError": False,
            },
        )

    return error_response(request_id, -32601, "Method not found")


def show_exchange(title: str, request: JsonObject) -> None:
    """打印一次客户端请求和服务端响应。"""

    print("\n" + "=" * 70)
    print(title)
    print("\n[客户端发送]")
    print(json.dumps(request, ensure_ascii=False, indent=2))

    # 这里是直接函数调用；真正 MCP 会通过 stdio 或 HTTP 传输消息。
    response = handle_jsonrpc(request)

    print("\n[服务端返回]")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    print(
        f"\n请求 id={request.get('id')!r}，"
        f"响应 id={response.get('id')!r}"
    )


def main() -> None:
    """构造四组请求，观察不同的数据流。"""

    client_meta = {
        "io.modelcontextprotocol/clientInfo": {
            "name": "day16-learning-client",
            "version": "1.0.0",
        }
    }

    show_exchange(
        "场景一：客户端发现服务端工具",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": client_meta},
        },
    )

    show_exchange(
        "场景二：客户端调用 multiply 工具",
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "multiply",
                "arguments": {"a": 23, "b": 47},
                "_meta": client_meta,
            },
        },
    )

    show_exchange(
        "场景三：请求一个不存在的方法",
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/delete_everything",
            "params": {"_meta": client_meta},
        },
    )

    show_exchange(
        "场景四：工具参数类型错误",
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "multiply",
                "arguments": {"a": "23", "b": 47},
                "_meta": client_meta,
            },
        },
    )

    print("\n" + "=" * 70)
    print("本步核心数据流：")
    print("客户端 dict → JSON-RPC 请求 → 方法分派 → Python 工具")
    print("Python 返回值 → JSON-RPC 响应 → 客户端按相同 id 对应请求")


if __name__ == "__main__":
    main()
