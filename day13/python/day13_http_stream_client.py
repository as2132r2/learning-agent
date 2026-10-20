"""Day13 第七步客户端：逐条读取SSE事件并实时打印。"""

import json
import os
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("DAY13_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError("缺少DAY13_API_KEY，请先在.env中设置。")

question = input("请输入问题：").strip()
body = json.dumps(
    {"question": question},
    ensure_ascii=False
).encode("utf-8")

request = Request(
    "http://127.0.0.1:8000/stream",
    data=body,
    headers={
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "text/event-stream"
    },
    method="POST"
)

print("\n助手：", end="", flush=True)

with urlopen(request, timeout=60) as response:
    current_event = ""

    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
            continue

        if not line.startswith("data: "):
            continue

        data = json.loads(line.removeprefix("data: "))

        if current_event == "token":
            print(data["delta"], end="", flush=True)
        elif current_event == "done":
            print("\n\n[流式响应完成，request_id=" + data["request_id"] + "]")
        elif current_event == "error":
            print("\n[错误]", data["message"])
