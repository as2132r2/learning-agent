"""Day24 第三步：对比两个独立评估器的串行与并行执行。

运行：
    python day24/python/day24_parallel_evaluators.py

说明：为公平比较，串行组和并行组都会真实调用模型，不使用缓存。
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
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
from day22_llm_reranker import RerankedResult, rerank_candidates  # noqa: E402
from day23_answer_quality import (  # noqa: E402
    AnswerQualityResponse,
    evaluate_answer_quality,
)
from day23_citation_audit import (  # noqa: E402
    audit_citations,
    generate_structured_answer,
)
from day23_claim_groundedness import (  # noqa: E402
    GroundednessResponse,
    evaluate_claim_support,
)
from day23_structured_citations import (  # noqa: E402
    StructuredRAGAnswer,
    build_readable_answer,
    select_evidence,
)


T = TypeVar("T")
MAX_EVALUATOR_CONCURRENCY = 2


@dataclass
class ModelCallResult:
    """一次模型调用的返回值和性能指标。"""

    name: str
    value: Any
    elapsed_seconds: float
    model_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ComparisonResult:
    """串行组或并行组的整体观测结果。"""

    mode: str
    wall_seconds: float
    calls: list[ModelCallResult]

    @property
    def individual_seconds_sum(self) -> float:
        return sum(call.elapsed_seconds for call in self.calls)

    @property
    def total_model_calls(self) -> int:
        return sum(call.model_calls for call in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.calls)


def measured_model_call(
    name: str,
    function: Callable[..., T],
    *args: Any,
) -> ModelCallResult:
    """在线程内部记录单次模型调用的耗时与Token。"""

    started = time.perf_counter()
    with get_openai_callback() as callback:
        value = function(*args)
    elapsed = time.perf_counter() - started

    return ModelCallResult(
        name=name,
        value=value,
        elapsed_seconds=elapsed,
        model_calls=int(callback.successful_requests),
        input_tokens=int(callback.prompt_tokens),
        output_tokens=int(callback.completion_tokens),
        total_tokens=int(callback.total_tokens),
    )


async def run_sequential(
    question: str,
    answer: StructuredRAGAnswer,
    evidence: list[RerankedResult],
) -> ComparisonResult:
    """等待第一个评估完成后，才启动第二个评估。"""

    started = time.perf_counter()
    groundedness = await asyncio.to_thread(
        measured_model_call,
        "Groundedness",
        evaluate_claim_support,
        answer,
        evidence,
    )
    quality = await asyncio.to_thread(
        measured_model_call,
        "相关性与完整性",
        evaluate_answer_quality,
        question,
        answer,
        evidence,
    )

    return ComparisonResult(
        mode="sequential",
        wall_seconds=time.perf_counter() - started,
        calls=[groundedness, quality],
    )


async def run_parallel(
    question: str,
    answer: StructuredRAGAnswer,
    evidence: list[RerankedResult],
) -> ComparisonResult:
    """同时创建两个Task，让两个外部模型请求重叠等待。"""

    semaphore = asyncio.Semaphore(MAX_EVALUATOR_CONCURRENCY)

    async def limited_call(
        name: str,
        function: Callable[..., T],
        *args: Any,
    ) -> ModelCallResult:
        async with semaphore:
            return await asyncio.to_thread(
                measured_model_call,
                name,
                function,
                *args,
            )

    started = time.perf_counter()
    groundedness_task = asyncio.create_task(
        limited_call(
            "Groundedness",
            evaluate_claim_support,
            answer,
            evidence,
        ),
        name="groundedness-evaluator",
    )
    quality_task = asyncio.create_task(
        limited_call(
            "相关性与完整性",
            evaluate_answer_quality,
            question,
            answer,
            evidence,
        ),
        name="answer-quality-evaluator",
    )

    print("\n并行组创建的Task：")
    print("-", groundedness_task.get_name(), type(groundedness_task))
    print("-", quality_task.get_name(), type(quality_task))

    calls = await asyncio.gather(
        groundedness_task,
        quality_task,
    )
    return ComparisonResult(
        mode="parallel",
        wall_seconds=time.perf_counter() - started,
        calls=list(calls),
    )


def print_comparison_group(result: ComparisonResult) -> None:
    title = "串行组" if result.mode == "sequential" else "并行组"
    print("\n" + "=" * 88)
    print(title)
    for call in result.calls:
        print("\n阶段：", call.name)
        print("单调用耗时：", f"{call.elapsed_seconds:.3f}秒")
        print("模型调用数：", call.model_calls)
        print("输入Token：", call.input_tokens)
        print("输出Token：", call.output_tokens)
        print("Token总数：", call.total_tokens)
    print("\n两个单调用耗时之和：", f"{result.individual_seconds_sum:.3f}秒")
    print("该组实际墙钟耗时：", f"{result.wall_seconds:.3f}秒")
    print("该组模型调用总数：", result.total_model_calls)
    print("该组Token总数：", result.total_tokens)


async def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or (
        "什么是 RAG？它的工作流程是什么？"
    )

    print("Day24 独立评估器串行与并行对比")
    print("问题：", question)
    print("评估器最大并发数：", MAX_EVALUATOR_CONCURRENCY)
    print("说明：本程序不使用Day24模型缓存。")
    print("=" * 88)

    print("准备共同输入：检索、精排并生成一份结构化回答……")
    rrf_results, _ = await asyncio.to_thread(
        retrieve_rrf_candidates,
        question,
    )
    reranked_results, _ = await asyncio.to_thread(
        rerank_candidates,
        question,
        rrf_results,
    )
    evidence = select_evidence(reranked_results)
    if not evidence:
        print("没有合格证据，无法进行评估器对比。")
        return

    answer = await asyncio.to_thread(
        generate_structured_answer,
        question,
        evidence,
    )
    rendered_answer = build_readable_answer(answer)
    citation_audit = audit_citations(
        answer,
        len(evidence),
        rendered_answer,
    )
    if not citation_audit.passed:
        print("引用结构审计失败：", citation_audit.errors)
        return

    print("共同输入准备完成。")
    print("结论数量：", len(answer.claims))
    print("证据数量：", len(evidence))

    print("\n开始串行组……")
    sequential = await run_sequential(question, answer, evidence)
    print_comparison_group(sequential)

    print("\n开始并行组……")
    parallel = await run_parallel(question, answer, evidence)
    print_comparison_group(parallel)

    groundedness_value = parallel.calls[0].value
    quality_value = parallel.calls[1].value
    print("\n" + "=" * 88)
    print("并行返回值类型：")
    print("Groundedness：", type(groundedness_value))
    print("回答质量：", type(quality_value))
    assert isinstance(groundedness_value, GroundednessResponse)
    assert isinstance(quality_value, AnswerQualityResponse)

    saved_seconds = sequential.wall_seconds - parallel.wall_seconds
    speedup = (
        sequential.wall_seconds / parallel.wall_seconds
        if parallel.wall_seconds > 0
        else 0.0
    )
    saved_rate = (
        saved_seconds / sequential.wall_seconds
        if sequential.wall_seconds > 0
        else 0.0
    )

    print("\n性能对比：")
    print("串行墙钟耗时：", f"{sequential.wall_seconds:.3f}秒")
    print("并行墙钟耗时：", f"{parallel.wall_seconds:.3f}秒")
    print("节省时间：", f"{saved_seconds:.3f}秒")
    print("耗时下降比例：", f"{saved_rate:.1%}")
    print("加速倍数：", f"{speedup:.2f}x")
    print("串行Token：", sequential.total_tokens)
    print("并行Token：", parallel.total_tokens)
    print("结论：并行减少等待时间，但不会自动减少模型调用和Token。")

    print("\n本步数据流：")
    print("同一份question + answer + evidence")
    print("├→ groundedness-evaluator Task")
    print("└→ answer-quality-evaluator Task")
    print("→ asyncio.gather保持传入顺序返回两个ModelCallResult")


if __name__ == "__main__":
    asyncio.run(main())
