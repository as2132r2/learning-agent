"""Day14 第九步：观察任务状态并取消运行中的后台任务。"""

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 0.02


def request_json(request: Request) -> tuple[int, dict]:
    """发送请求，HTTP错误也转换成状态码和JSON。"""

    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def submit_task(question: str) -> tuple[int, dict]:
    body = json.dumps(
        {"question": question},
        ensure_ascii=False
    ).encode("utf-8")

    return request_json(Request(
        f"{BASE_URL}/tasks",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    ))


def get_task(task_id: str) -> tuple[int, dict]:
    return request_json(Request(
        f"{BASE_URL}/tasks/{task_id}",
        method="GET"
    ))


def cancel_task(task_id: str) -> tuple[int, dict]:
    return request_json(Request(
        f"{BASE_URL}/tasks/{task_id}",
        method="DELETE"
    ))


question = (
    "请分五点解释异步任务队列的工作过程，"
    "每一点都给出一句简短说明。"
)

print("提交一个后台模型任务……")
submit_status, submitted = submit_task(question)

if submit_status != 202:
    raise RuntimeError(
        f"任务提交失败：HTTP {submit_status}，{submitted}"
    )

task_id = submitted["task_id"]
print("task_id：", task_id)
print("提交时状态：", submitted["status"])

last_status = None

while True:
    query_status, job = get_task(task_id)

    if query_status != 200:
        raise RuntimeError(
            f"查询失败：HTTP {query_status}，{job}"
        )

    if job["status"] != last_status:
        print("观察到状态：", job["status"])
        last_status = job["status"]

    if job["status"] == "running":
        print("任务正在运行，现在发送DELETE取消请求。")
        cancel_status, cancel_result = cancel_task(task_id)
        print("取消接口状态码：", cancel_status)
        print("取消接口响应状态：", cancel_result.get("status"))
        break

    if job["status"] in {"succeeded", "failed", "cancelled"}:
        print("任务在取消请求之前已经结束。")
        print("最终状态：", job["status"])
        print("这是正常的并发竞态，不应强行覆盖最终结果。")
        break

    time.sleep(POLL_INTERVAL_SECONDS)

# DELETE只发出取消请求；再次查询才能确认Worker最终处理后的状态。
time.sleep(0.1)
final_status_code, final_job = get_task(task_id)

print("\n最终查询HTTP状态码：", final_status_code)
print("最终任务数据：")
print(json.dumps(final_job, ensure_ascii=False, indent=2))


# POST /tasks → pending
# GET轮询 → running
# DELETE /tasks/{id} → running_operation.cancel()
# Worker捕获CancelledError → cancelled
# GET再次确认最终状态
