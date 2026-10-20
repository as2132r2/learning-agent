"""Day13 第一步配套客户端：用 Python 标准库测试本地 HTTP API。"""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 5


def get_json(path: str) -> dict:
    """发送 GET 请求，并把 JSON 响应转换成 Python 字典。"""

    url = f"{BASE_URL}{path}"

    with urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        response_text = response.read().decode("utf-8")
        return json.loads(response_text)


def post_json(path: str, payload: dict) -> dict:
    """发送带有 JSON 请求体的 POST 请求。"""

    url = f"{BASE_URL}{path}"
    request_body = json.dumps(
        payload,
        ensure_ascii=False
    ).encode("utf-8")

    request = Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )

    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        response_text = response.read().decode("utf-8")
        return json.loads(response_text)


def main() -> None:
    print("开始测试 Day13 HTTP API。")
    print(f"服务器地址：{BASE_URL}")

    try:
        health_result = get_json("/health")

        print("\n" + "=" * 60)
        print("GET /health 的响应：")
        print(json.dumps(health_result, ensure_ascii=False, indent=2))

        ask_result = post_json(
            "/ask",
            {"question": "什么是 LangGraph？"}
        )

        print("\n" + "=" * 60)
        print("POST /ask 的响应：")
        print(json.dumps(ask_result, ensure_ascii=False, indent=2))

    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"\n服务器返回 HTTP 错误：{error.code}")
        print(error_body)

    except URLError as error:
        print("\n无法连接服务器。")
        print("请先在另一个终端启动 day13_http_api_basic.py。")
        print("具体原因：", error.reason)


if __name__ == "__main__":
    main()


# ============================================================
# 数据流
# ============================================================
#
# Python字典
#    ↓ json.dumps() + encode("utf-8")
# JSON请求体（bytes）
#    ↓ Request + urlopen()
# HTTP服务器
#    ↓ HTTP响应
# JSON响应体（bytes）
#    ↓ read() + decode("utf-8") + json.loads()
# Python字典
