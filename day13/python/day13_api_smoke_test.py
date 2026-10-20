"""Day13 第九步：对运行中的HTTP API执行部署前冒烟测试。"""

import json
import os
import unittest
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()

BASE_URL = os.getenv("DAY13_TEST_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("DAY13_API_KEY", "").strip()
TIMEOUT_SECONDS = 5


def send(request: Request) -> tuple[int, dict]:
    """发送请求；无论成功还是HTTP错误，都返回状态码和JSON。"""

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body
    except HTTPError as error:
        body = json.loads(error.read().decode("utf-8"))
        return error.code, body


def authorized_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


class TestDay13ApiSmoke(unittest.TestCase):
    """只检查API边界，不发送有效模型问题。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not API_KEY:
            raise RuntimeError(".env中缺少DAY13_API_KEY。")

        # 先快速确认服务可连接，让失败原因比单个测试超时更清楚。
        try:
            urlopen(f"{BASE_URL}/health", timeout=TIMEOUT_SECONDS).close()
        except URLError as error:
            raise RuntimeError(
                "无法连接Day13 API，请先启动day13_deploy_api.py。"
            ) from error

    def test_01_health_is_public(self) -> None:
        request = Request(f"{BASE_URL}/health", method="GET")
        status, body = send(request)

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_02_chat_requires_token(self) -> None:
        request = Request(
            f"{BASE_URL}/chat",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        status, body = send(request)

        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_03_invalid_chat_input_returns_422(self) -> None:
        request_body = json.dumps({
            "thread_id": "smoke-test",
            "question": ""
        }).encode("utf-8")

        request = Request(
            f"{BASE_URL}/chat",
            data=request_body,
            headers={
                **authorized_headers(),
                "Content-Type": "application/json"
            },
            method="POST"
        )
        status, body = send(request)

        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_04_unknown_thread_returns_404(self) -> None:
        query = urlencode({
            "thread_id": "thread-that-does-not-exist-smoke-test"
        })
        request = Request(
            f"{BASE_URL}/history?{query}",
            headers=authorized_headers(),
            method="GET"
        )
        status, body = send(request)

        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "thread_not_found")


if __name__ == "__main__":
    print("开始运行Day13 API冒烟测试……")
    print("测试地址：", BASE_URL)
    print("本测试不会调用大模型。\n")
    unittest.main(verbosity=2)


# ============================================================
# 测试数据流
# ============================================================
# unittest → HTTP请求 → 正在运行的Day13 API
#                         ↓
#                    HTTP状态码 + JSON
#                         ↓
#              assertEqual()判断是否符合预期
