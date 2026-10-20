"""Day13 第三步配套客户端：测试带thread_id的多轮对话API。"""

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 40


def post_chat(
    base_url: str,
    thread_id: str,
    question: str
) -> dict:
    """向POST /chat发送一轮对话。"""

    request_body = json.dumps(
        {
            "thread_id": thread_id,
            "question": question
        },
        ensure_ascii=False
    ).encode("utf-8")

    request = Request(
        url=f"{base_url}/chat",
        data=request_body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )

    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        response_text = response.read().decode("utf-8")
        return json.loads(response_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day13 多轮对话HTTP客户端"
    )
    parser.add_argument(
        "thread_id",
        nargs="?",
        default="xiaolin-day13",
        help="会话编号，默认是xiaolin-day13"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="HTTP服务地址"
    )
    args = parser.parse_args()

    print("Day13 多轮对话客户端已启动，输入 exit 结束。")
    print("当前会话：", args.thread_id)
    print("服务器：", args.base_url)

    while True:
        question = input("\n你：").strip()

        if not question:
            print("请输入内容，不要只按回车。")
            continue

        if question.lower() == "exit":
            break

        try:
            result = post_chat(
                base_url=args.base_url,
                thread_id=args.thread_id,
                question=question
            )

            print("助手：", result["answer"])
            print(
                f"[request_id={result['request_id']} | "
                f"消息数量={result['message_count']}]"
            )

        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            print(f"服务器返回HTTP错误：{error.code}")
            print(error_body)

        except URLError as error:
            print("无法连接服务器，请先启动day13_http_memory_api.py。")
            print("具体原因：", error.reason)

    print("\n客户端已结束。")


if __name__ == "__main__":
    main()


# ============================================================
# 数据流
# ============================================================
#
# 终端输入 question + 命令行参数 thread_id
#                    ↓ json.dumps() / UTF-8编码
# POST /chat JSON请求
#                    ↓
# day13_http_memory_api.py
#                    ↓ JSON响应
# json.loads() → result字典 → 打印answer和message_count
