"""Day14 第十二步客户端：提交任务并接收SSE状态推送。"""

import json
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"

question = input("请输入后台任务问题：").strip()
body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")

submit_request = Request(
    f"{BASE_URL}/tasks",
    data=body,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST"
)

with urlopen(submit_request, timeout=10) as submit_response:
    submitted = json.loads(submit_response.read().decode("utf-8"))

task_id = submitted["task_id"]
print("任务已提交：", task_id)
print("开始等待服务端主动推送，不再循环发送GET请求。\n")

events_request = Request(
    f"{BASE_URL}/tasks/{task_id}/events",
    headers={"Accept": "text/event-stream"},
    method="GET"
)

with urlopen(events_request, timeout=60) as stream:
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
            print("服务端推送状态：", payload["status"])
            if payload["answer"] is not None:
                print("回答：", payload["answer"])
            if payload["error"] is not None:
                print("错误：", payload["error"])

        elif current_event == "heartbeat":
            print("收到心跳，连接仍然存活。")

        elif current_event == "done":
            print("推送结束，最终状态：", payload["status"])

        elif current_event == "error":
            print("推送错误：", payload)
