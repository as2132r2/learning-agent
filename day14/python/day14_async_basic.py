"""Day14 第一步：理解同步执行和asyncio异步并发。"""

import asyncio
import time


TASK_NAMES = ["读取资料", "调用模型", "保存结果"]
WAIT_SECONDS = 1


def synchronous_task(name: str, seconds: int) -> str:
    """同步任务：time.sleep期间会阻塞当前线程。"""

    print(f"[同步开始] {name}")
    time.sleep(seconds)
    print(f"[同步完成] {name}")
    return f"{name}完成"


async def asynchronous_task(name: str, seconds: int) -> str:
    """异步任务：等待期间把执行权交还给事件循环。"""

    print(f"[异步开始] {name}")
    await asyncio.sleep(seconds)
    print(f"[异步完成] {name}")
    return f"{name}完成"


def run_synchronously() -> list[str]:
    """三个任务依次执行。"""

    results = []

    for task_name in TASK_NAMES:
        result = synchronous_task(task_name, WAIT_SECONDS)
        results.append(result)

    return results


async def run_asynchronously() -> list[str]:
    """创建三个协程，并交给asyncio.gather并发等待。"""

    coroutines = [
        asynchronous_task(task_name, WAIT_SECONDS)
        for task_name in TASK_NAMES
    ]

    results = await asyncio.gather(*coroutines)
    return list(results)


def main() -> None:
    print("=" * 60)
    print("第一部分：同步执行")

    sync_start = time.perf_counter()
    sync_results = run_synchronously()
    sync_elapsed = time.perf_counter() - sync_start

    print("同步结果：", sync_results)
    print(f"同步总耗时：{sync_elapsed:.2f}秒")

    print("\n" + "=" * 60)
    print("第二部分：异步并发")

    async_start = time.perf_counter()
    async_results = asyncio.run(run_asynchronously())
    async_elapsed = time.perf_counter() - async_start

    print("异步结果：", async_results)
    print(f"异步总耗时：{async_elapsed:.2f}秒")

    print("\n" + "=" * 60)
    print("结论：")
    print("同步执行：三个1秒任务依次等待，大约需要3秒。")
    print("异步并发：三个任务同时等待，大约只需要1秒。")
    print("asyncio适合模型API、HTTP、数据库等I/O等待任务。")


if __name__ == "__main__":
    main()


# ============================================================
# 同步数据流
# ============================================================
# 任务1开始 → 等1秒 → 任务1完成
#                    ↓
# 任务2开始 → 等1秒 → 任务2完成
#                    ↓
# 任务3开始 → 等1秒 → 任务3完成
#
# ============================================================
# 异步数据流
# ============================================================
# asyncio.run()创建事件循环
#              ↓
# asyncio.gather(协程1, 协程2, 协程3)
#              ↓
# 三个协程开始 → 各自await等待 → 事件循环调度其他协程
#              ↓
# 三个结果按传入gather的顺序组成列表
