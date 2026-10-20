"""Day13 第六步客户端：自动携带Bearer Token。"""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("DAY13_API_KEY", "").strip()
BASE_URL = "http://127.0.0.1:8000"
THREAD_ID = "xiaolin-day13-auth"

if not API_KEY:
    raise RuntimeError("缺少DAY13_API_KEY，请先在.env中设置。")


def send_chat(question: str) -> dict:
    body = json.dumps(
        {"thread_id": THREAD_ID, "question": question},
        ensure_ascii=False
    ).encode("utf-8")

    request = Request(
        f"{BASE_URL}/chat",
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )

    with urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


print("Bearer认证客户端已启动，输入exit结束。")
print("当前会话：", THREAD_ID)

while True:
    question = input("\n你：").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue

    try:
        result = send_chat(question)
        print("助手：", result["answer"])
        print("消息数量：", result["message_count"])
    except HTTPError as error:
        print("HTTP错误：", error.code)
        print(error.read().decode("utf-8", errors="replace"))

print("客户端已结束。")
