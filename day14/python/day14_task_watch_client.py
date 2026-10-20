"""Day14 第十三步观察客户端：订阅一个已有task_id。"""

import argparse
import json
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser(description="订阅已有后台任务的SSE状态")
parser.add_argument("task_id", help="需要观察的任务编号")
parser.add_argument(
    "--base-url",
    default="http://127.0.0.1:8000"
)
args = parser.parse_args()

request = Request(
    f"{args.base_url}/tasks/{args.task_id}/events",
    headers={"Accept": "text/event-stream"},
    method="GET"
)

print("开始观察任务：", args.task_id)

with urlopen(request, timeout=60) as stream:
    current_event = ""

    for raw_line in stream:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
            continue

        if not line.startswith("data: "):
            continue

        payload = json.loads(line.removeprefix("data: "))

        if current_event == "status":
            print("状态：", payload["status"])
            if payload["answer"]:
                print("回答：", payload["answer"])
            if payload["error"]:
                print("错误：", payload["error"])
        elif current_event == "heartbeat":
            print("心跳：连接正常")
        elif current_event == "done":
            print("任务结束：", payload["status"])
