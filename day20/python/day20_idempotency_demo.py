"""Day20 第五步：并发提交和幂等写入演示。

运行：python day20/python/day20_idempotency_demo.py
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from day20_safe_repository import SafeChatRepository, SaveResult


def main() -> None:
    repository = SafeChatRepository()
    mode = repository.initialize()

    client_id = "client-" + uuid.uuid4().hex[:8]
    session_id = "session-" + uuid.uuid4().hex[:10]
    duplicate_request_id = "req-duplicate-" + uuid.uuid4().hex[:8]

    repository.create_session_if_missing(
        session_id,
        client_id,
        "幂等与并发测试",
    )

    print("Day20 SQLite 并发安全与幂等测试")
    print("数据库：", repository.database_path)
    print("journal_mode：", mode)
    print("session_id：", session_id)
    print("重复 request_id：", duplicate_request_id)
    print("=" * 72)

    def submit_duplicate(worker_number: int) -> tuple[int, SaveResult]:
        result = repository.save_turn_idempotent(
            request_id=duplicate_request_id,
            session_id=session_id,
            question="重复提交会保存几次？",
            answer="相同 request_id 只能保存一次。",
        )
        return worker_number, result

    print("5个线程同时提交同一个request_id：")
    with ThreadPoolExecutor(max_workers=5) as executor:
        duplicate_results = list(executor.map(submit_duplicate, range(1, 6)))

    for worker_number, result in duplicate_results:
        print(
            f"线程{worker_number}：created={result.created} "
            f"turn_id={result.turn.turn_id}"
        )

    created_count = sum(
        1 for _, result in duplicate_results if result.created
    )
    unique_turn_ids = {
        result.turn.turn_id for _, result in duplicate_results
    }

    print("\n重复提交统计：")
    print("真正创建次数：", created_count)
    print("所有线程看到的turn_id：", sorted(unique_turn_ids))
    print("数据库当前轮数：", repository.count_turns(session_id))

    print("\n" + "=" * 72)
    print("4个线程分别提交不同request_id：")

    unique_request_ids = [
        "req-unique-" + uuid.uuid4().hex[:8]
        for _ in range(4)
    ]

    def submit_unique(request_id: str) -> SaveResult:
        return repository.save_turn_idempotent(
            request_id=request_id,
            session_id=session_id,
            question=f"问题：{request_id}",
            answer=f"回答：{request_id}",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        unique_results = list(executor.map(submit_unique, unique_request_ids))

    for result in unique_results:
        print(
            f"request_id={result.turn.request_id} "
            f"created={result.created} turn_id={result.turn.turn_id}"
        )

    turns = repository.load_turns(session_id, limit=20)
    print("\n最终数据库内容：")
    for turn in turns:
        print(asdict(turn))

    print("\n最终轮数：", repository.count_turns(session_id))
    print("预期轮数：5（1个重复请求 + 4个不同请求）")

    print("\n" + "=" * 72)
    print("关键保证：")
    print("WAL → 读取与写入更适合并发")
    print("busy_timeout → 写锁繁忙时等待，而不是立即失败")
    print("BEGIN IMMEDIATE → 先取得写锁，再执行先查后写")
    print("UNIQUE(request_id) → 数据库提供最后一道重复保护")


if __name__ == "__main__":
    main()
