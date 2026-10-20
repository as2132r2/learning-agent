"""Day29 第 4 步：批量导出、队列、背压与优雅关闭。

运行：
    python day29/python/day29_batch_exporter.py

输出：
    day29/logs/day29_exported_spans.jsonl
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from day29_traceparent_basic import create_child_context, create_root_context


DAY29_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = DAY29_DIR / "logs" / "day29_exported_spans.jsonl"


@dataclass(frozen=True)
class ExportableSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    status: str
    duration_ms: float
    ended_at: str
    attributes: dict[str, Any]


@dataclass
class ExporterMetrics:
    offered: int = 0
    accepted: int = 0
    dropped: int = 0
    exported: int = 0
    export_batches: int = 0
    export_failures: int = 0
    max_observed_queue_size: int = 0


class JsonlBatchSink:
    """模拟较慢的远程遥测后端。"""

    def __init__(self, path: Path, export_delay_seconds: float = 0.08) -> None:
        self.path = path
        self.export_delay_seconds = export_delay_seconds

    async def export(self, batch: list[ExportableSpan]) -> None:
        # 模拟网络传输和后端确认耗时。
        await asyncio.sleep(self.export_delay_seconds)
        lines = "".join(
            json.dumps(asdict(span), ensure_ascii=False, default=str) + "\n"
            for span in batch
        )
        # 文件写入很小；生产中可改为 asyncio.to_thread 或异步 HTTP 客户端。
        with self.path.open("a", encoding="utf-8") as file:
            file.write(lines)


class BatchSpanExporter:
    """有界队列 + 后台批处理 + 非阻塞生产者。"""

    def __init__(
        self,
        sink: JsonlBatchSink,
        *,
        max_queue_size: int = 8,
        max_batch_size: int = 4,
        schedule_delay_seconds: float = 0.05,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size必须大于0。")
        if not 0 < max_batch_size <= max_queue_size:
            raise ValueError("max_batch_size必须在1到max_queue_size之间。")
        if schedule_delay_seconds <= 0:
            raise ValueError("schedule_delay_seconds必须大于0。")

        self.sink = sink
        self.max_batch_size = max_batch_size
        self.schedule_delay_seconds = schedule_delay_seconds
        self.queue: asyncio.Queue[ExportableSpan | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.metrics = ExporterMetrics()
        self.worker_task: asyncio.Task[None] | None = None
        self._accepting = False

    async def start(self) -> None:
        if self.worker_task is not None:
            raise RuntimeError("Exporter已经启动。")
        self._accepting = True
        self.worker_task = asyncio.create_task(
            self._worker(),
            name="day29-span-export-worker",
        )

    def emit(self, span: ExportableSpan) -> bool:
        """业务路径非阻塞入队；队列满时丢弃最新 Span。"""

        if not self._accepting:
            raise RuntimeError("Exporter未启动或已经关闭。")

        self.metrics.offered += 1
        try:
            self.queue.put_nowait(span)
        except asyncio.QueueFull:
            self.metrics.dropped += 1
            return False

        self.metrics.accepted += 1
        self.metrics.max_observed_queue_size = max(
            self.metrics.max_observed_queue_size,
            self.queue.qsize(),
        )
        return True

    async def _export_batch(self, batch: list[ExportableSpan]) -> None:
        try:
            await self.sink.export(batch)
        except Exception as error:
            self.metrics.export_failures += 1
            print(f"[导出失败] {type(error).__name__}: {error}")
        else:
            self.metrics.exported += len(batch)
            self.metrics.export_batches += 1
            print(
                f"[批量导出] batch={self.metrics.export_batches} "
                f"size={len(batch)} queue_remaining={self.queue.qsize()}"
            )
        finally:
            # 每个成功取出的普通元素都必须恰好调用一次 task_done()。
            for _ in batch:
                self.queue.task_done()

    async def _worker(self) -> None:
        """后台收集一批 Span；达到大小或时间阈值后导出。"""

        stopping = False
        while not stopping:
            first = await self.queue.get()
            if first is None:
                self.queue.task_done()
                break

            batch = [first]
            deadline = time.monotonic() + self.schedule_delay_seconds

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    break

                if item is None:
                    self.queue.task_done()
                    stopping = True
                    break
                batch.append(item)

            await self._export_batch(batch)

    async def shutdown(self) -> None:
        """停止接收，排队写入哨兵，并等待此前已接受 Span 导出完成。"""

        if self.worker_task is None:
            return
        self._accepting = False

        # 队列可能已满，所以这里使用 await，而不是 put_nowait。
        # 哨兵进入队尾，Worker 会先处理之前已接受的数据。
        await self.queue.put(None)
        await self.queue.join()
        await self.worker_task
        self.worker_task = None


def make_span(index: int) -> ExportableSpan:
    root = create_root_context(sampled=True)
    child = create_child_context(root)
    return ExportableSpan(
        trace_id=child.trace_id,
        span_id=child.span_id,
        parent_span_id=root.span_id,
        name="rag.generate" if index % 2 else "rag.retrieve",
        kind="INTERNAL",
        status="UNSET",
        duration_ms=round(8.0 + index * 0.7, 3),
        ended_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
        attributes={"demo.sequence": index, "rag.document.count": index % 4 + 1},
    )


async def run_normal_load() -> ExporterMetrics:
    print("第一部分：正常流量，Worker来得及消费")
    print("-" * 92)

    exporter = BatchSpanExporter(
        JsonlBatchSink(OUTPUT_PATH, export_delay_seconds=0.03),
        max_queue_size=8,
        max_batch_size=4,
        schedule_delay_seconds=0.04,
    )
    await exporter.start()

    for index in range(1, 9):
        accepted = exporter.emit(make_span(index))
        print(
            f"生产Span {index:02d} | accepted={accepted} "
            f"queue={exporter.queue.qsize()}"
        )
        await asyncio.sleep(0.025)

    await exporter.shutdown()
    print("正常流量指标：")
    print(json.dumps(asdict(exporter.metrics), ensure_ascii=False, indent=2))
    return exporter.metrics


async def run_overload() -> ExporterMetrics:
    print("\n" + "=" * 92)
    print("第二部分：突发流量超过有界队列")
    print("-" * 92)

    exporter = BatchSpanExporter(
        JsonlBatchSink(OUTPUT_PATH, export_delay_seconds=0.08),
        max_queue_size=8,
        max_batch_size=4,
        schedule_delay_seconds=0.05,
    )
    await exporter.start()

    # 故意不 await，让25条Span在Worker获得执行机会前快速冲击容量8的队列。
    accepted_ids: list[int] = []
    dropped_ids: list[int] = []
    for index in range(101, 126):
        if exporter.emit(make_span(index)):
            accepted_ids.append(index)
        else:
            dropped_ids.append(index)

    print("被队列接受的序号：", accepted_ids)
    print("因队列已满被丢弃：", dropped_ids)
    print("调用shutdown前queue size：", exporter.queue.qsize())

    await exporter.shutdown()
    print("过载流量指标：")
    print(json.dumps(asdict(exporter.metrics), ensure_ascii=False, indent=2))
    print("shutdown后queue size：", exporter.queue.qsize())
    return exporter.metrics


def count_exported_file() -> int:
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


async def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("", encoding="utf-8")

    print("Day29 批量导出、队列、背压与优雅关闭")
    print("输出文件：", OUTPUT_PATH)
    print("=" * 92)

    normal = await run_normal_load()
    overload = await run_overload()
    file_count = count_exported_file()

    print("\n" + "=" * 92)
    print("最终一致性检查：")
    expected_exported = normal.exported + overload.exported
    print("两个场景exported之和：", expected_exported)
    print("JSONL实际行数：", file_count)
    print("是否一致：", expected_exported == file_count)
    print("过载时业务生产者是否等待慢后端：否，emit()使用put_nowait。")
    print("已被接受的Span是否在关闭前全部处理：", overload.accepted == overload.exported)

    print("\n本步数据流：")
    print(
        "业务协程产生Span\n"
        "→ emit()非阻塞put_nowait\n"
        "   ├→ 队列有空间：accepted\n"
        "   └→ 队列已满：drop newest + dropped指标\n"
        "→ 后台Worker按batch size或等待时间组批\n"
        "→ 慢速Sink批量写入JSONL\n"
        "→ shutdown停止接收 + 哨兵入队\n"
        "→ Worker处理完已接受Span → queue.join → 任务结束"
    )


if __name__ == "__main__":
    asyncio.run(main())
