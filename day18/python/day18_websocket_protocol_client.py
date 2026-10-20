"""Day18 第三步（客户端）：使用 version、request_id 和 payload 协议。

先启动服务端：
    python day18/python/day18_websocket_protocol_server.py

再运行客户端：
    python day18/python/day18_websocket_protocol_client.py
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from aiohttp import ClientSession, WSMsgType


WEBSOCKET_URL = "ws://127.0.0.1:8012/ws"
PROTOCOL_VERSION = "1.0"


def validate_server_message(value: Any) -> dict[str, Any]:
    """检查服务端消息是否包含统一协议的四个字段。"""

    if not isinstance(value, dict):
        raise ValueError("服务端消息不是 JSON object。")

    required_fields = {"version", "type", "request_id", "payload"}
    missing = required_fields - value.keys()

    if missing:
        raise ValueError(f"服务端消息缺少字段：{sorted(missing)}")

    if value["version"] != PROTOCOL_VERSION:
        raise ValueError(f"不支持协议版本：{value['version']!r}")

    if not isinstance(value["payload"], dict):
        raise ValueError("服务端 payload 不是 object。")

    return value


async def main() -> None:
    """发送带 request_id 的 ask，并只处理属于它的流。"""

    question = input("请输入问题：").strip()
    if not question:
        print("问题不能为空。")
        return

    request_id = "req-" + uuid.uuid4().hex[:12]

    print("Day18 WebSocket 协议客户端")
    print("地址：", WEBSOCKET_URL)
    print("协议版本：", PROTOCOL_VERSION)
    print("request_id：", request_id)
    print("=" * 72)

    text_parts: list[str] = []

    async with ClientSession() as session:
        async with session.ws_connect(
            WEBSOCKET_URL,
            heartbeat=20,
        ) as websocket:
            connected = validate_server_message(
                await websocket.receive_json()
            )
            print("连接消息：", connected)

            outgoing = {
                "version": PROTOCOL_VERSION,
                "type": "ask",
                "request_id": request_id,
                "payload": {
                    "question": question,
                },
            }

            print("\n发送 ask：", outgoing)
            await websocket.send_json(outgoing)

            async for raw_message in websocket:
                if raw_message.type != WSMsgType.TEXT:
                    if raw_message.type == WSMsgType.ERROR:
                        print("WebSocket 错误：", websocket.exception())
                        break
                    continue

                message = validate_server_message(raw_message.json())
                message_type = str(message["type"])
                incoming_request_id = message["request_id"]
                payload = message["payload"]

                # 连接级消息可以没有 request_id；请求级消息必须匹配。
                if (
                    incoming_request_id is not None
                    and incoming_request_id != request_id
                ):
                    print(
                        f"忽略其他请求的消息：{incoming_request_id}"
                    )
                    continue

                if message_type == "start":
                    print("\n助手：", end="", flush=True)

                elif message_type == "token":
                    text = str(payload.get("text", ""))
                    text_parts.append(text)
                    print(text, end="", flush=True)

                elif message_type == "done":
                    print("\n\n[done]", payload)
                    break

                elif message_type == "error":
                    print("\n\n[error]", payload)
                    break

            await websocket.close()

    complete_answer = "".join(text_parts)

    print("\n" + "=" * 72)
    print("完整回答：")
    print(complete_answer)

    print("\n协议数据流：")
    print("ask(version + type + request_id + payload)")
    print("→ Pydantic AskMessage 校验")
    print("→ start/token/done 保留相同 request_id")
    print("→ 客户端按 request_id 归属消息")


if __name__ == "__main__":
    asyncio.run(main())
