"""Day25 第 2 步配套客户端：测试反馈 HTTP API。

先在终端 1 启动服务端：
    python day25/python/day25_feedback_api.py

再在终端 2 运行客户端：
    python day25/python/day25_feedback_api_client.py
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8032"


def request_json(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """发送 HTTP 请求并解析 JSON 响应。"""

    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = Request(
        url=f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        response_body = json.loads(error.read().decode("utf-8"))
        return error.code, response_body


def show_result(title: str, status: int, body: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("HTTP 状态码：", status)
    print(json.dumps(body, ensure_ascii=False, indent=2))


def main() -> None:
    print("Day25 反馈 API 客户端")
    print("服务地址：", BASE_URL)

    try:
        status, body = request_json("GET", "/health")
        show_result("1. 健康检查", status, body)

        feedback = {
            "request_id": "req-day25-http-001",
            "session_id": "xiaolin-day25",
            "user_id": "user-xiaolin",
            "question": "为什么 RAG 要保留来源？",
            "answer": "保留来源可以让用户核对证据，并支持后续审计。",
            "rating": "up",
            "reason": "结论清楚，但还可以增加例子。",
            "answer_version": "day25-api-v1",
            "source_ids": ["chunk-rag-source-001"],
            "pipeline_metrics": {
                "model_calls": 2,
                "total_tokens": 1800,
                "risk": "medium",
            },
        }
        status, body = request_json("POST", "/feedback", feedback)
        show_result("2. 提交点赞反馈", status, body)

        # 模拟用户改变意见。request_id 和 user_id 不变，所以执行 UPDATE。
        feedback["rating"] = "down"
        feedback["reason"] = "缺少具体来源示例。"
        status, body = request_json("POST", "/feedback", feedback)
        show_result("3. 把同一条反馈修改为点踩", status, body)

        status, body = request_json("GET", "/feedback?rating=down")
        show_result("4. 只查询点踩记录", status, body)

        status, body = request_json("GET", "/feedback/statistics")
        show_result("5. 查询反馈统计", status, body)

    except URLError as error:
        print("\n连接服务失败：", error.reason)
        print("请先在另一个终端运行：")
        print("python day25/python/day25_feedback_api.py")
        raise SystemExit(1) from error

    print("\n" + "=" * 80)
    print("客户端测试结束。")
    print("本步数据流：")
    print("Python客户端 → JSON/HTTP → aiohttp路由")
    print("→ FeedbackInput → SQLite UPSERT → JSON响应")


if __name__ == "__main__":
    main()
