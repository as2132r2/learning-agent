"""Day16 第二步：用 stdio 在客户端进程和工具服务端进程之间传输消息。

运行方式：
    python day16/python/day16_stdio_transport.py

程序会自动完成以下工作：
1. 当前进程作为客户端；
2. 启动一个新的 Python 子进程作为服务端；
3. 客户端通过服务端 stdin 发送 JSON-RPC 请求；
4. 服务端通过 stdout 返回 JSON-RPC 响应；
5. 服务端运行日志通过 stderr 输出，避免污染协议消息。

这仍是教学版协议实现，下一步才会使用正式 MCP Python SDK。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO


JsonObject = dict[str, Any]


def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    # stdout 留给协议消息，所以服务端日志必须写入 stderr。
    print(f"[服务端执行 add] {a} + {b}", file=sys.stderr, flush=True)
    return a + b


def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(f"[服务端执行 multiply] {a} × {b}", file=sys.stderr, flush=True)
    return a * b


TOOL_FUNCTIONS: dict[str, Callable[[float, float], float]] = {
    "add": add,
    "multiply": multiply,
}


TOOL_DESCRIPTIONS: list[JsonObject] = [
    {
        "name": "add",
        "description": "计算两个数字的和。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "multiply",
        "description": "计算两个数字的乘积。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
]


def make_result(request_id: Any, result: JsonObject) -> JsonObject:
    """创建成功的 JSON-RPC 响应。"""

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: Any, code: int, message: str) -> JsonObject:
    """创建失败的 JSON-RPC 响应。"""

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def validate_number_arguments(arguments: Any) -> tuple[float, float]:
    """检查工具参数，并转换成两个浮点数。"""

    if not isinstance(arguments, dict) or set(arguments) != {"a", "b"}:
        raise ValueError("arguments 必须只包含 a 和 b。")

    a = arguments["a"]
    b = arguments["b"]

    if isinstance(a, bool) or not isinstance(a, (int, float)):
        raise ValueError("参数 a 必须是数字。")
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise ValueError("参数 b 必须是数字。")

    return float(a), float(b)


def dispatch(request: JsonObject) -> JsonObject:
    """服务端根据 method 分派一条请求。"""

    request_id = request.get("id")

    if request.get("jsonrpc") != "2.0" or "id" not in request:
        return make_error(request_id, -32600, "Invalid Request")

    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return make_result(request_id, {"tools": TOOL_DESCRIPTIONS})

    if method != "tools/call":
        return make_error(request_id, -32601, "Method not found")

    if not isinstance(params, dict):
        return make_error(request_id, -32602, "Invalid params")

    tool_name = params.get("name")
    tool_function = TOOL_FUNCTIONS.get(tool_name)
    if tool_function is None:
        return make_error(request_id, -32602, f"未知工具：{tool_name}")

    try:
        a, b = validate_number_arguments(params.get("arguments"))
        value = tool_function(a, b)
    except ValueError as error:
        return make_error(request_id, -32602, str(error))
    except Exception:
        return make_error(request_id, -32603, "Internal error")

    return make_result(
        request_id,
        {
            "content": [{"type": "text", "text": str(value)}],
            "structuredContent": {"result": value},
            "isError": False,
        },
    )


def run_server() -> None:
    """服务端循环：从 stdin 逐行读取请求，向 stdout 逐行写响应。"""

    print("[服务端进程已启动]", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                response = make_error(None, -32600, "Invalid Request")
            else:
                response = dispatch(request)
        except json.JSONDecodeError:
            response = make_error(None, -32700, "Parse error")

        # 每条消息独占一行，并立即刷新缓冲区。
        print(
            json.dumps(response, ensure_ascii=False),
            file=sys.stdout,
            flush=True,
        )

    print("[服务端输入已关闭，准备退出]", file=sys.stderr, flush=True)


def send_request(
    server_input: TextIO,
    server_output: TextIO,
    request: JsonObject,
) -> JsonObject:
    """客户端发送一条请求，并同步读取一条响应。"""

    request_text = json.dumps(request, ensure_ascii=False)
    print(f"\n[客户端发送] {request_text}")

    server_input.write(request_text + "\n")
    server_input.flush()

    response_text = server_output.readline()
    if response_text == "":
        raise RuntimeError("服务端意外退出，没有返回响应。")

    response = json.loads(response_text)
    print("[客户端收到]")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    print(
        f"[ID 对应] 请求 id={request['id']}，"
        f"响应 id={response.get('id')}"
    )
    return response


def run_client() -> None:
    """启动服务端子进程，然后完成工具发现和两次工具调用。"""

    script_path = Path(__file__).resolve()

    print("Day16 stdio 客户端已启动。")
    print("客户端 PID：", os.getpid())

    # 使用当前虚拟环境中的 Python，再次运行本文件并进入 --server 模式。
    process = subprocess.Popen(
        [sys.executable, str(script_path), "--server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    if process.stdin is None or process.stdout is None:
        raise RuntimeError("无法创建服务端通信管道。")

    print("服务端 PID：", process.pid)
    print("两个 PID 不同，说明它们是两个独立进程。")

    client_meta = {
        "io.modelcontextprotocol/clientInfo": {
            "name": "day16-stdio-client",
            "version": "1.0.0",
        }
    }

    try:
        send_request(
            process.stdin,
            process.stdout,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {"_meta": client_meta},
            },
        )

        add_response = send_request(
            process.stdin,
            process.stdout,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "add",
                    "arguments": {"a": 20, "b": 4},
                    "_meta": client_meta,
                },
            },
        )
        first_result = add_response["result"]["structuredContent"]["result"]

        send_request(
            process.stdin,
            process.stdout,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "multiply",
                    "arguments": {"a": first_result, "b": 3},
                    "_meta": client_meta,
                },
            },
        )
    finally:
        # 关闭 stdin 会让服务端的 for 循环自然结束。
        process.stdin.close()
        return_code = process.wait(timeout=5)
        process.stdout.close()
        print(f"\n服务端进程已退出，退出码：{return_code}")

    print("\n" + "=" * 70)
    print("本步数据流：")
    print("客户端 dict → JSON 文本 → 子进程 stdin")
    print("服务端 stdin → json.loads() → dispatch() → Python 工具")
    print("工具结果 → JSON 文本 → 子进程 stdout → 客户端")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
