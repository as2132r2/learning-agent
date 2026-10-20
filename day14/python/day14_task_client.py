"""Day14 第八步客户端：提交后台任务并查询状态。"""

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"


def request_json(request: Request) -> tuple[int, dict]:
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


question = input("请输入后台任务问题：").strip()
body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")

status, job = request_json(Request(
    f"{BASE_URL}/tasks",
    data=body,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST"
))

if status != 202:
    raise RuntimeError(f"提交失败：HTTP {status}，{job}")

task_id = job["task_id"]
print("任务已接受，HTTP状态码：", status)
print("task_id：", task_id)

while True:
    time.sleep(0.3)
    status, job = request_json(Request(
        f"{BASE_URL}/tasks/{task_id}",
        method="GET"
    ))
    print("当前状态：", job["status"])

    if job["status"] in {"succeeded", "failed", "cancelled"}:
        break

print("\n最终任务数据：")
print(json.dumps(job, ensure_ascii=False, indent=2))
