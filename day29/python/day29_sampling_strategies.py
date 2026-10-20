"""Day29 第 3 步：Head Sampling 与 Parent-Based Sampling。

运行：
    python day29/python/day29_sampling_strategies.py

本程序只实现教学采样器，不依赖 OpenTelemetry SDK。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

from day29_traceparent_basic import TraceContext


TRACE_ID_BITS = 128
TRACE_ID_SPACE = 1 << TRACE_ID_BITS


class SamplingDecision(StrEnum):
    DROP = "DROP"
    RECORD_AND_SAMPLE = "RECORD_AND_SAMPLE"


@dataclass(frozen=True)
class SamplingResult:
    decision: SamplingDecision
    sampler: str
    reason: str
    trace_id: str
    inherited_from_parent: bool = False

    @property
    def sampled(self) -> bool:
        return self.decision is SamplingDecision.RECORD_AND_SAMPLE


class Sampler(Protocol):
    """所有教学采样器必须实现的最小接口。"""

    @property
    def description(self) -> str: ...

    def should_sample(
        self,
        trace_id: str,
        parent_context: TraceContext | None = None,
    ) -> SamplingResult: ...


def validate_trace_id(trace_id: str) -> None:
    if len(trace_id) != 32:
        raise ValueError("trace_id必须是32位十六进制字符串。")
    try:
        numeric = int(trace_id, 16)
    except ValueError as error:
        raise ValueError("trace_id包含非十六进制字符。") from error
    if numeric == 0:
        raise ValueError("trace_id不能全部为0。")


class AlwaysOnSampler:
    @property
    def description(self) -> str:
        return "AlwaysOn"

    def should_sample(
        self,
        trace_id: str,
        parent_context: TraceContext | None = None,
    ) -> SamplingResult:
        del parent_context
        validate_trace_id(trace_id)
        return SamplingResult(
            SamplingDecision.RECORD_AND_SAMPLE,
            self.description,
            "AlwaysOn保留所有Trace。",
            trace_id,
        )


class AlwaysOffSampler:
    @property
    def description(self) -> str:
        return "AlwaysOff"

    def should_sample(
        self,
        trace_id: str,
        parent_context: TraceContext | None = None,
    ) -> SamplingResult:
        del parent_context
        validate_trace_id(trace_id)
        return SamplingResult(
            SamplingDecision.DROP,
            self.description,
            "AlwaysOff丢弃所有普通Trace。",
            trace_id,
        )


class TraceIdRatioBasedSampler:
    """根据 Trace ID 数值稳定决定是否命中采样比例。"""

    def __init__(self, ratio: float) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio必须在0到1之间。")
        self.ratio = ratio
        self.threshold = int(ratio * TRACE_ID_SPACE)

    @property
    def description(self) -> str:
        return f"TraceIdRatioBased({self.ratio:.3f})"

    def should_sample(
        self,
        trace_id: str,
        parent_context: TraceContext | None = None,
    ) -> SamplingResult:
        del parent_context
        validate_trace_id(trace_id)
        numeric = int(trace_id, 16)
        sampled = numeric < self.threshold
        return SamplingResult(
            (
                SamplingDecision.RECORD_AND_SAMPLE
                if sampled
                else SamplingDecision.DROP
            ),
            self.description,
            (
                "Trace ID落在采样阈值内。"
                if sampled
                else "Trace ID落在采样阈值外。"
            ),
            trace_id,
        )


class ParentBasedSampler:
    """存在父上下文时继承决定；根 Trace 使用 root_sampler。"""

    def __init__(self, root_sampler: Sampler) -> None:
        self.root_sampler = root_sampler

    @property
    def description(self) -> str:
        return f"ParentBased({self.root_sampler.description})"

    def should_sample(
        self,
        trace_id: str,
        parent_context: TraceContext | None = None,
    ) -> SamplingResult:
        validate_trace_id(trace_id)
        if parent_context is None:
            root_result = self.root_sampler.should_sample(trace_id)
            return SamplingResult(
                root_result.decision,
                self.description,
                f"没有父上下文，使用根采样器：{root_result.reason}",
                trace_id,
                inherited_from_parent=False,
            )

        decision = (
            SamplingDecision.RECORD_AND_SAMPLE
            if parent_context.sampled
            else SamplingDecision.DROP
        )
        return SamplingResult(
            decision,
            self.description,
            (
                "继承父上下文的sampled=1。"
                if parent_context.sampled
                else "继承父上下文的sampled=0。"
            ),
            trace_id,
            inherited_from_parent=True,
        )


def deterministic_trace_id(index: int) -> str:
    """使用固定输入生成可重复的32位Trace ID。"""

    return hashlib.sha256(f"day29-trace-{index}".encode()).hexdigest()[:32]


def make_parent(trace_id: str, sampled: bool, remote: bool = True) -> TraceContext:
    flags = 0x01 if sampled else 0x00
    return TraceContext(
        version="00",
        trace_id=trace_id,
        span_id=hashlib.sha256(f"span-{trace_id}".encode()).hexdigest()[:16],
        trace_flags=flags,
        is_remote=remote,
    )


def evaluate_sampler(sampler: Sampler, count: int) -> dict[str, object]:
    results = [
        sampler.should_sample(deterministic_trace_id(index))
        for index in range(1, count + 1)
    ]
    sampled_count = sum(result.sampled for result in results)
    return {
        "sampler": sampler.description,
        "total": count,
        "sampled": sampled_count,
        "dropped": count - sampled_count,
        "actual_rate": round(sampled_count / count, 4),
    }


def demonstrate_stability() -> None:
    sampler = TraceIdRatioBasedSampler(0.10)
    trace_id = deterministic_trace_id(42)
    repeated = [sampler.should_sample(trace_id).decision for _ in range(5)]

    print("第二部分：同一个Trace ID的决定必须稳定")
    print("-" * 96)
    print("trace_id：", trace_id)
    print("连续5次决定：", [decision.value for decision in repeated])
    print("是否完全一致：", len(set(repeated)) == 1)
    print()
    print("不能每次调用random.random()重新决定，否则同一Trace的不同服务可能断裂。")


def demonstrate_parent_based() -> None:
    sampler = ParentBasedSampler(TraceIdRatioBasedSampler(0.10))
    trace_id = deterministic_trace_id(5000)
    sampled_parent = make_parent(trace_id, sampled=True)
    dropped_parent = make_parent(trace_id, sampled=False)

    root = sampler.should_sample(trace_id)
    inherited_sampled = sampler.should_sample(trace_id, sampled_parent)
    inherited_dropped = sampler.should_sample(trace_id, dropped_parent)

    print("\n" + "=" * 96)
    print("第三部分：Parent-Based Sampling")
    print("-" * 96)
    print("没有父上下文：")
    print(json.dumps(asdict(root), ensure_ascii=False, indent=2))
    print("\n父上下文 sampled=1：")
    print(json.dumps(asdict(inherited_sampled), ensure_ascii=False, indent=2))
    print("\n父上下文 sampled=0：")
    print(json.dumps(asdict(inherited_dropped), ensure_ascii=False, indent=2))


def demonstrate_error_limitation() -> None:
    print("\n" + "=" * 96)
    print("第四部分：Head Sampling无法预知未来错误")
    print("-" * 96)

    sampler = TraceIdRatioBasedSampler(0.10)
    dropped_result: SamplingResult | None = None
    for index in range(1, 100):
        candidate = sampler.should_sample(deterministic_trace_id(index))
        if not candidate.sampled:
            dropped_result = candidate
            break

    if dropped_result is None:
        raise RuntimeError("演示数据中没有找到被丢弃的Trace。")

    print("请求开始时决定：", dropped_result.decision.value)
    print("随后业务发生：TimeoutError")
    print("问题：Head Sampling在请求开始时不知道未来会失败。")
    print("结论：若必须保留所有错误Trace，需要额外策略，例如：")
    print("- 本地始终保留错误日志和指标；")
    print("- 使用Tail Sampling在Trace完成后决定；")
    print("- 或对高风险入口提高采样率。")


def main() -> None:
    count = 10_000
    samplers: list[Sampler] = [
        AlwaysOnSampler(),
        AlwaysOffSampler(),
        TraceIdRatioBasedSampler(0.01),
        TraceIdRatioBasedSampler(0.10),
        TraceIdRatioBasedSampler(0.50),
    ]

    print("Day29 Head Sampling 与 Parent-Based Sampling")
    print("样本数量：", count)
    print("=" * 96)
    print("第一部分：固定Trace ID集合上的采样结果")
    print("-" * 96)
    for sampler in samplers:
        report = evaluate_sampler(sampler, count)
        print(
            f"{report['sampler']:<28} "
            f"sampled={report['sampled']:<6} "
            f"dropped={report['dropped']:<6} "
            f"actual_rate={report['actual_rate']:.2%}"
        )

    print()
    print("比例不是保证每100条恰好命中10条；样本增大后才接近配置值。")

    demonstrate_stability()
    demonstrate_parent_based()
    demonstrate_error_limitation()

    print("\n" + "=" * 96)
    print("本步数据流：")
    print(
        "请求开始 → 获得trace_id\n"
        "→ 有父上下文？\n"
        "   ├→ 有：ParentBased继承父sampled位\n"
        "   └→ 无：根采样器根据trace_id稳定决定\n"
        "→ RECORD_AND_SAMPLE：记录并导出完整Trace\n"
        "→ DROP：不导出完整Trace，但业务请求仍然正常执行\n"
        "→ trace_flags把决定传播给下游"
    )


if __name__ == "__main__":
    main()
