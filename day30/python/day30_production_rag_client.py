"""Day30 第 6 步配套客户端：验证最终生产化RAG服务。

先在终端1启动服务：
    python day30/python/day30_production_rag_service.py

再在终端2运行：
    python day30/python/day30_production_rag_client.py "什么是 RAG？"
"""

from __future__ import annotations

import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8050"
TIMEOUT_SECONDS = 40


def request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        text = error.read().decode("utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"error": text}
        return error.code, data


def print_response(title: str, status: int, data: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("HTTP状态码：", status)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"
    print("Day30最终生产化RAG服务客户端")
    print("服务地址：", BASE_URL)
    print("问题：", question)

    try:
        live_status, live = request_json("GET", "/livez")
        print_response("1. 存活检查", live_status, live)

        ready_status, ready = request_json("GET", "/readyz")
        print_response("2. 就绪检查", ready_status, ready)
        if ready_status != 200:
            print("服务尚未就绪，停止发送模型请求。")
            raise SystemExit(1)

        ask_status, answer = request_json(
            "POST",
            "/ask",
            {"question": question},
        )
        print_response("3. RAG问答", ask_status, answer)
        request_id = answer.get("request_id")
        if ask_status != 200 or not isinstance(request_id, str):
            raise SystemExit(1)

        detail_status, detail = request_json(
            "GET",
            f"/requests/{request_id}",
        )
        print_response("4. 持久化请求详情", detail_status, detail)

        ops_status, ops = request_json("GET", "/ops/status")
        print_response("5. 运维状态", ops_status, ops)
    except URLError as error:
        print("\n连接服务失败：", error)
        print("请先在另一个终端启动day30_production_rag_service.py。")
        raise SystemExit(1) from error

    print("\n" + "=" * 100)
    print("本次端到端验证完成：")
    print("健康检查 → RAG请求 → 混合检索 → 精排 → 回答")
    print("→ 数据库持久化 → 按request_id查询 → 运维状态统计")


if __name__ == "__main__":
    main()
