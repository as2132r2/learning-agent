"""Day24 第一步：记录可审计RAG各阶段耗时、模型调用次数和Token。

运行：
    python day24/python/day24_pipeline_metrics.py

指定问题：
    python day24/python/day24_pipeline_metrics.py "什么是 RAG？"

注意：候选正文会发送给配置的DeepSeek API。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from langchain_community.callbacks.manager import get_openai_callback


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY22_PYTHON = PROJECT_DIR / "day22" / "python"
DAY23_PYTHON = PROJECT_DIR / "day23" / "python"
for directory in (DAY22_PYTHON, DAY23_PYTHON):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from day22_final_hybrid_rag import retrieve_rrf_candidates  # noqa: E402
from day22_llm_reranker import rerank_candidates  # noqa: E402
from day23_answer_quality import (  # noqa: E402
    build_final_quality_report,
    evaluate_answer_quality,
)
from day23_citation_audit import (  # noqa: E402
    audit_citations,
    generate_structured_answer,
)
from day23_claim_groundedness import (  # noqa: E402
    build_groundedness_report,
    evaluate_claim_support,
)
from day23_structured_citations import (  # noqa: E402
    build_readable_answer,
    select_evidence,
)


T = TypeVar("T")


@dataclass
class StageMetric:
    """一个流水线阶段的可观测数据。"""

    name: str
    kind: str
    elapsed_seconds: float
    model_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    status: str
    detail: str = ""


@dataclass
class PipelineMetrics:
    """聚合整条流水线的所有阶段指标。"""

    question: str
    stages: list[StageMetric] = field(default_factory=list)

    @property
    def total_elapsed_seconds(self) -> float:
        return sum(stage.elapsed_seconds for stage in self.stages)

    @property
    def total_model_calls(self) -> int:
        return sum(stage.model_calls for stage in self.stages)

    @property
    def total_input_tokens(self) -> int:
        return sum(stage.input_tokens for stage in self.stages)

    @property
    def total_output_tokens(self) -> int:
        return sum(stage.output_tokens for stage in self.stages)

    @property
    def total_tokens(self) -> int:
        return sum(stage.total_tokens for stage in self.stages)


def measure_local_stage(
    metrics: PipelineMetrics,
    name: str,
    function: Callable[..., T],
    *args: Any,
    detail: str = "",
) -> T:
    """测量不调用LLM的本地阶段。"""

    started = time.perf_counter()
    try:
        result = function(*args)
    except Exception as error:
        elapsed = time.perf_counter() - started
        metrics.stages.append(
            StageMetric(
                name=name,
                kind="local",
                elapsed_seconds=elapsed,
                model_calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                status="failed",
                detail=f"{type(error).__name__}: {error}",
            )
        )
        raise

    elapsed = time.perf_counter() - started
    metrics.stages.append(
        StageMetric(
            name=name,
            kind="local",
            elapsed_seconds=elapsed,
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            status="success",
            detail=detail,
        )
    )
    return result


def measure_model_stage(
    metrics: PipelineMetrics,
    name: str,
    function: Callable[..., T],
    *args: Any,
    detail: str = "",
) -> T:
    """在独立回调作用域中统计一个模型阶段。"""

    started = time.perf_counter()
    with get_openai_callback() as callback:
        try:
            result = function(*args)
        except Exception as error:
            elapsed = time.perf_counter() - started
            metrics.stages.append(
                StageMetric(
                    name=name,
                    kind="model",
                    elapsed_seconds=elapsed,
                    model_calls=int(callback.successful_requests),
                    input_tokens=int(callback.prompt_tokens),
                    output_tokens=int(callback.completion_tokens),
                    total_tokens=int(callback.total_tokens),
                    status="failed",
                    detail=f"{type(error).__name__}: {error}",
                )
            )
            raise

    elapsed = time.perf_counter() - started
    metrics.stages.append(
        StageMetric(
            name=name,
            kind="model",
            elapsed_seconds=elapsed,
            model_calls=int(callback.successful_requests),
            input_tokens=int(callback.prompt_tokens),
            output_tokens=int(callback.completion_tokens),
            total_tokens=int(callback.total_tokens),
            status="success",
            detail=detail,
        )
    )
    return result


def print_metrics(metrics: PipelineMetrics) -> None:
    """打印逐阶段指标表和总计。"""

    print("\n" + "=" * 112)
    print("流水线阶段指标：")
    print(
        f"{'阶段':<22} {'类型':<8} {'耗时(秒)':>10} "
        f"{'调用':>7} {'输入Token':>12} {'输出Token':>12} "
        f"{'总Token':>11} {'状态':>9}"
    )
    print("-" * 112)
    for stage in metrics.stages:
        print(
            f"{stage.name:<22} {stage.kind:<8} "
            f"{stage.elapsed_seconds:>10.3f} "
            f"{stage.model_calls:>7} "
            f"{stage.input_tokens:>12} "
            f"{stage.output_tokens:>12} "
            f"{stage.total_tokens:>11} "
            f"{stage.status:>9}"
        )
        if stage.detail:
            print("  说明：", stage.detail)

    print("-" * 112)
    print(f"总耗时（阶段之和）：{metrics.total_elapsed_seconds:.3f}秒")
    print("模型调用总数：", metrics.total_model_calls)
    print("输入Token总数：", metrics.total_input_tokens)
    print("输出Token总数：", metrics.total_output_tokens)
    print("Token总数：", metrics.total_tokens)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or (
        "什么是 RAG？它的工作流程是什么？"
    )
    metrics = PipelineMetrics(question=question)
    pipeline_started = time.perf_counter()

    print("Day24 可审计RAG阶段耗时与Token观测")
    print("问题：", question)
    print("=" * 88)

    rrf_results, diagnostics = measure_local_stage(
        metrics,
        "混合检索与RRF",
        retrieve_rrf_candidates,
        question,
        detail="本地Embedding、SQLiteVec、FTS5和RRF，不调用LLM。",
    )
    print("RRF候选数：", len(rrf_results))
    print("关键词：", diagnostics["terms"])

    reranked_results, _ = measure_model_stage(
        metrics,
        "Reranker精排",
        rerank_candidates,
        question,
        rrf_results,
        detail="一次批量调用，对全部RRF候选评分。",
    )
    evidence = measure_local_stage(
        metrics,
        "证据阈值过滤",
        select_evidence,
        reranked_results,
        detail="只保留相关性达到Day22阈值的资料。",
    )
    print("Reranker最高分：", reranked_results[0].relevance_score)
    print("合格证据数：", len(evidence))

    if not evidence:
        print("没有合格证据，流水线提前结束。")
        print_metrics(metrics)
        return

    answer = measure_model_stage(
        metrics,
        "结构化回答生成",
        generate_structured_answer,
        question,
        evidence,
        detail="生成answer、claims和citation_numbers。",
    )
    rendered_answer = build_readable_answer(answer)

    citation_audit = measure_local_stage(
        metrics,
        "确定性引用审计",
        audit_citations,
        answer,
        len(evidence),
        rendered_answer,
        detail="检查覆盖率、重复、越界和渲染一致性。",
    )
    if not citation_audit.passed:
        print("引用审计失败，流水线提前结束：", citation_audit.errors)
        print_metrics(metrics)
        return

    groundedness_response = measure_model_stage(
        metrics,
        "Groundedness评估",
        evaluate_claim_support,
        answer,
        evidence,
        detail="批量判断每条claim是否被其引用正文支持。",
    )
    groundedness_report = measure_local_stage(
        metrics,
        "忠实度报告计算",
        build_groundedness_report,
        answer,
        groundedness_response,
        detail="校验编号并计算严格证据支持率。",
    )

    quality_response = measure_model_stage(
        metrics,
        "相关性与完整性评估",
        evaluate_answer_quality,
        question,
        answer,
        evidence,
        detail="检查是否切题、是否遗漏资料可支持的重要要点。",
    )
    final_report = measure_local_stage(
        metrics,
        "最终质量报告计算",
        build_final_quality_report,
        citation_audit,
        groundedness_report,
        quality_response,
        detail="使用最短板原则生成最终通过/失败判断。",
    )

    real_elapsed = time.perf_counter() - pipeline_started
    print("\n" + "=" * 88)
    print("最终回答：")
    print(rendered_answer)
    print("\n最终质量：", "通过" if final_report.passed else "失败")
    print("综合最短板：", final_report.overall_score)

    print_metrics(metrics)
    print(f"程序实际墙钟总耗时：{real_elapsed:.3f}秒")
    print("当前为串行流水线，所以阶段耗时之和应接近墙钟总耗时。")

    print("\n本步数据流：")
    print("每个阶段 → perf_counter耗时")
    print("每个模型阶段 → 独立Token回调作用域")
    print("→ StageMetric列表 → PipelineMetrics聚合")
    print("下一步将缓存可复用结果，观察减少了多少真实调用和Token。")


if __name__ == "__main__":
    main()
