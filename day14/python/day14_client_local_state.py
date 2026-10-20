"""Day14 第十一步：客户端保存局部任务索引，并与服务端状态同步。"""

import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
DAY14_DIRECTORY = Path(__file__).resolve().parents[1]
CLIENT_STATE_PATH = DAY14_DIRECTORY / "database" / "day14_client_state.json"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_local_state() -> dict:
    """读取客户端本地索引；首次运行时返回空结构。"""

    if not CLIENT_STATE_PATH.exists():
        return {"tasks": {}}

    try:
        return json.loads(CLIENT_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"客户端状态文件损坏：{error}") from error


def save_local_state(state: dict) -> None:
    """先写临时文件，再原子替换正式文件。"""

    CLIENT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = CLIENT_STATE_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    temporary_path.replace(CLIENT_STATE_PATH)


def request_json(request: Request) -> tuple[int, dict]:
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def submit_remote_task(question: str) -> tuple[int, dict]:
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


def get_remote_task(task_id: str) -> tuple[int, dict]:
    return request_json(Request(
        f"{BASE_URL}/tasks/{task_id}",
        method="GET"
    ))


def submit_command(state: dict) -> None:
    question = input("请输入问题：").strip()
    if not question:
        print("问题不能为空。")
        return

    status_code, remote_job = submit_remote_task(question)
    if status_code != 202:
        print("提交失败：", status_code, remote_job)
        return

    task_id = remote_job["task_id"]

    # 只保存客户端展示和重新定位任务所需的局部字段。
    state["tasks"][task_id] = {
        "task_id": task_id,
        "question_preview": question[:60],
        "last_known_status": remote_job["status"],
        "last_synced_at": now_text()
    }
    save_local_state(state)

    print("任务已提交：", task_id)
    print("局部索引已保存：", CLIENT_STATE_PATH)


def list_command(state: dict) -> None:
    tasks = list(state["tasks"].values())

    if not tasks:
        print("客户端本地还没有任务索引。")
        return

    print(f"客户端本地记录了{len(tasks)}个任务：")

    for item in tasks:
        print("\n任务ID：", item["task_id"])
        print("问题摘要：", item["question_preview"])
        print("最后已知状态：", item["last_known_status"])
        print("最后同步时间：", item["last_synced_at"])


def sync_command(state: dict) -> None:
    """逐个向服务端查询，以服务端结果更新本地缓存。"""

    if not state["tasks"]:
        print("没有可以同步的本地任务。")
        return

    for task_id, local_item in list(state["tasks"].items()):
        status_code, remote_job = get_remote_task(task_id)

        if status_code == 200:
            old_status = local_item["last_known_status"]
            new_status = remote_job["status"]
            local_item["last_known_status"] = new_status
            local_item["last_synced_at"] = now_text()
            print(f"{task_id}: {old_status} → {new_status}")

            if new_status == "succeeded":
                print("回答：", remote_job["answer"])

        elif status_code == 404:
            # 不擅自删除本地索引，保留记录并标记服务端已不存在。
            local_item["last_known_status"] = "missing_on_server"
            local_item["last_synced_at"] = now_text()
            print(f"{task_id}: 服务端已不存在")

        else:
            print(f"{task_id}: 同步失败，HTTP {status_code}")

    save_local_state(state)
    print("同步后的局部状态已保存。")


def main() -> None:
    state = load_local_state()

    print("Day14客户端局部状态练习。")
    print("本地索引：", CLIENT_STATE_PATH)
    print("命令：submit、list、sync、exit")

    while True:
        command = input("\n命令：").strip().lower()

        try:
            if command == "submit":
                submit_command(state)
            elif command == "list":
                list_command(state)
            elif command == "sync":
                sync_command(state)
            elif command == "exit":
                break
            else:
                print("请输入submit、list、sync或exit。")
        except URLError as error:
            print("无法连接服务端：", error.reason)

    print("客户端已关闭，本地索引仍然保留。")


if __name__ == "__main__":
    main()


# 客户端JSON：局部索引、最后已知状态、界面恢复依据
# 服务端SQLite：完整问题、权威状态、回答、错误、时间信息
# sync命令：服务端权威状态 → 更新客户端局部缓存
