"""Day13 第五步配套客户端：聊天、查询历史和删除会话。"""

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 40


def request_json(request: Request) -> dict:
    """发送HTTP请求并解析JSON响应。"""

    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(base_url: str, thread_id: str, question: str) -> dict:
    body = json.dumps(
        {"thread_id": thread_id, "question": question},
        ensure_ascii=False
    ).encode("utf-8")

    request = Request(
        url=f"{base_url}/chat",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    return request_json(request)


def get_history(base_url: str, thread_id: str) -> dict:
    query = urlencode({"thread_id": thread_id})
    request = Request(
        url=f"{base_url}/history?{query}",
        method="GET"
    )
    return request_json(request)


def delete_history(base_url: str, thread_id: str) -> dict:
    query = urlencode({"thread_id": thread_id})
    request = Request(
        url=f"{base_url}/history?{query}",
        method="DELETE"
    )
    return request_json(request)


def print_history(result: dict) -> None:
    print(f"历史消息数量：{result['message_count']}")

    for index, message in enumerate(result["messages"], start=1):
        print(f"\n第{index}条消息")
        print("类型：", message["type"])
        print("内容：", message["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Day13 HTTP会话管理客户端")
    parser.add_argument(
        "thread_id",
        nargs="?",
        default="xiaolin-day13-session"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    print("Day13会话管理客户端已启动。")
    print("当前会话：", args.thread_id)
    print("直接输入文字：聊天")
    print("/history：查询历史")
    print("/delete：删除历史")
    print("exit：退出客户端")

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            print("请输入内容，不要只按回车。")
            continue

        if user_input.lower() == "exit":
            break

        try:
            if user_input == "/history":
                print_history(get_history(args.base_url, args.thread_id))
                continue

            if user_input == "/delete":
                result = delete_history(args.base_url, args.thread_id)
                print("会话已删除。")
                print("删除的消息数量：", result["deleted_message_count"])
                continue

            result = chat(args.base_url, args.thread_id, user_input)
            print("助手：", result["answer"])
            print("[当前消息数量：", result["message_count"], "]")

        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            print(f"HTTP错误：{error.code}")
            print(body)
        except URLError as error:
            print("无法连接服务器，请先启动day13_http_session_api.py。")
            print("具体原因：", error.reason)

    print("\n客户端已结束。")


if __name__ == "__main__":
    main()
