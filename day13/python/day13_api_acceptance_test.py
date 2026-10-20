"""Day13 第十步：对运行中的API执行完整端到端验收。"""

import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv


load_dotenv()

BASE_URL = os.getenv("DAY13_TEST_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("DAY13_API_KEY", "").strip()
THREAD_ID = f"acceptance-{uuid4().hex[:10]}"
TIMEOUT_SECONDS = 60

if not API_KEY:
    raise RuntimeError(".env中缺少DAY13_API_KEY。")


def request_json(request: Request) -> tuple[int, dict]:
    """发送请求，统一返回状态码和JSON字典。"""

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return (
                response.status,
                json.loads(response.read().decode("utf-8"))
            )
    except HTTPError as error:
        return (
            error.code,
            json.loads(error.read().decode("utf-8"))
        )


def headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }


def chat(question: str) -> tuple[int, dict]:
    body = json.dumps(
        {"thread_id": THREAD_ID, "question": question},
        ensure_ascii=False
    ).encode("utf-8")
    return request_json(Request(
        f"{BASE_URL}/chat",
        data=body,
        headers=headers(),
        method="POST"
    ))


def history(method: str = "GET") -> tuple[int, dict]:
    query = urlencode({"thread_id": THREAD_ID})
    return request_json(Request(
        f"{BASE_URL}/history?{query}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        method=method
    ))


def require(condition: bool, message: str) -> None:
    """验收条件失败时立即终止并说明原因。"""

    if not condition:
        raise AssertionError(message)


print("开始执行Day13端到端验收。")
print("服务地址：", BASE_URL)
print("临时会话：", THREAD_ID)
print("注意：本测试会调用模型两次。")

try:
    print("\n[1/5] 创建第一轮对话")
    status, first = chat("请记住：我的测试代号是青竹。只需简短确认。")
    require(status == 200, f"第一轮状态码应为200，实际为{status}：{first}")
    require(first["message_count"] == 2, "第一轮消息数量应为2。")
    print("通过，回答：", first["answer"])

    print("\n[2/5] 追问以验证记忆")
    status, second = chat("我的测试代号是什么？只回答代号。")
    require(status == 200, f"第二轮状态码应为200，实际为{status}：{second}")
    require(second["message_count"] == 4, "第二轮消息数量应为4。")
    require("青竹" in second["answer"], "模型没有正确恢复测试代号。")
    print("通过，回答：", second["answer"])

    print("\n[3/5] 查询持久化历史")
    status, saved = history("GET")
    require(status == 200, f"历史查询应为200，实际为{status}：{saved}")
    require(saved["message_count"] == 4, "数据库历史应有4条消息。")
    print("通过，数据库中共有4条消息。")

    print("\n[4/5] 删除临时会话")
    status, deleted = history("DELETE")
    require(status == 200, f"删除请求应为200，实际为{status}：{deleted}")
    require(deleted["deleted"] is True, "删除结果应为true。")
    print("通过，临时会话已经删除。")

    print("\n[5/5] 确认会话已经不存在")
    status, missing = history("GET")
    require(status == 404, f"删除后查询应为404，实际为{status}：{missing}")
    require(missing["error"] == "thread_not_found", "错误类型不正确。")
    print("通过，删除后返回404。")

except Exception:
    # 验收中途失败也尝试清理临时thread_id。
    history("DELETE")
    raise

print("\n" + "=" * 60)
print("Day13端到端验收全部通过。")
print("测试产生的临时会话已经清理。")


# 客户端 → 认证 → 两轮模型对话 → SQLite历史
#       → 查询历史 → 删除历史 → 404确认 → 验收完成
