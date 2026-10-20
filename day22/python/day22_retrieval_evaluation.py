"""Day22 第五步：评测向量检索、RRF和LLM Reranker。

运行：
    python day22/python/day22_retrieval_evaluation.py

注意：程序会对每道评测题调用一次外部LLM Reranker。
"""

from __future__ import annotations

from dataclasses import dataclass

from day22_llm_reranker import rerank_candidates
from day22_retrieval_compare import fts_search, vector_search
from day22_rrf_fusion import reciprocal_rank_fusion


EXPECTED_SOURCE = "day21/documents/day21_sample.txt"
RRF_CANDIDATE_COUNT = 5
THRESHOLDS = list(range(50, 100, 5))


@dataclass(frozen=True)
class EvaluationCase:
    """一条人工标注的评测题。"""

    question: str
    expected_source: str | None

    @property
    def should_answer(self) -> bool:
        return self.expected_source is not None


@dataclass
class EvaluationResult:
    """一题在三种排序方案下的评测结果。"""

    case: EvaluationCase
    vector_source: str
    rrf_source: str
    reranker_source: str
    reranker_score: int
    reranker_reason: str


EVALUATION_CASES = [
    EvaluationCase("什么是 RAG？", EXPECTED_SOURCE),
    EvaluationCase("检索增强生成的工作过程是什么？", EXPECTED_SOURCE),
    EvaluationCase("浏览器上传文件通常使用什么格式？", EXPECTED_SOURCE),
    EvaluationCase("服务端收到上传文件后要检查哪些内容？", EXPECTED_SOURCE),
    EvaluationCase("法国的首都是哪里？", None),
    EvaluationCase("红烧肉应该怎么做？", None),
]


def evaluate_case(case: EvaluationCase) -> EvaluationResult:
    """执行完整检索链路并记录三层排序的Top-1。"""

    vector_results = vector_search(case.question)
    _, _, keyword_results = fts_search(case.question)

    rrf_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
    )[:RRF_CANDIDATE_COUNT]

    if not vector_results:
        raise RuntimeError("向量检索没有返回候选。")
    if not rrf_results:
        raise RuntimeError("RRF没有返回候选。")

    reranked_results, _ = rerank_candidates(
        case.question,
        rrf_results,
    )
    if not reranked_results:
        raise RuntimeError("Reranker没有返回候选。")

    return EvaluationResult(
        case=case,
        vector_source=str(vector_results[0]["location"]),
        rrf_source=rrf_results[0].location,
        reranker_source=reranked_results[0].location,
        reranker_score=reranked_results[0].relevance_score,
        reranker_reason=reranked_results[0].reason,
    )


def source_is_correct(actual: str, expected: str) -> bool:
    """location可能带页码，所以使用前缀判断来源文件。"""

    return actual.startswith(expected)


def source_accuracy(
    results: list[EvaluationResult],
    field_name: str,
) -> tuple[int, int, float]:
    """只对知识库内问题计算Top-1来源准确率。"""

    related = [result for result in results if result.case.should_answer]
    correct = sum(
        source_is_correct(
            str(getattr(result, field_name)),
            str(result.case.expected_source),
        )
        for result in related
    )
    total = len(related)
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy


def gate_accuracy(
    results: list[EvaluationResult],
    threshold: int,
) -> tuple[int, int, float]:
    """分数达到阈值则允许回答，计算允许/拒绝判断准确率。"""

    correct = 0
    for result in results:
        predicted_answer = result.reranker_score >= threshold
        if predicted_answer == result.case.should_answer:
            correct += 1

    total = len(results)
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy


def choose_threshold(
    results: list[EvaluationResult],
) -> tuple[int, int, int, float]:
    """选择准确率最高的阈值；并列时优先选择更高、更保守的阈值。"""

    scored = [
        (threshold, *gate_accuracy(results, threshold))
        for threshold in THRESHOLDS
    ]
    return max(scored, key=lambda item: (item[3], item[0]))


def main() -> None:
    print("Day22 混合检索与Reranker评测")
    print("评测问题数量：", len(EVALUATION_CASES))
    print("相关问题数量：", sum(c.should_answer for c in EVALUATION_CASES))
    print("无关问题数量：", sum(not c.should_answer for c in EVALUATION_CASES))
    print("每题会调用一次LLM Reranker。")
    print("=" * 96)

    results: list[EvaluationResult] = []
    for number, case in enumerate(EVALUATION_CASES, start=1):
        print(f"\n[{number}/{len(EVALUATION_CASES)}] {case.question}")
        print("预期：", case.expected_source or "应该拒绝回答")
        result = evaluate_case(case)
        results.append(result)

        print("向量Top-1：", result.vector_source)
        print("RRF Top-1：", result.rrf_source)
        print("Reranker Top-1：", result.reranker_source)
        print("Reranker最高分：", result.reranker_score)
        print("理由：", result.reranker_reason)

    print("\n" + "=" * 96)
    print("相关问题Top-1来源准确率：")
    for label, field_name in [
        ("纯向量检索", "vector_source"),
        ("RRF融合", "rrf_source"),
        ("LLM Reranker", "reranker_source"),
    ]:
        correct, total, accuracy = source_accuracy(results, field_name)
        print(f"{label:<16} {correct}/{total} = {accuracy:.1%}")

    print("\n" + "=" * 96)
    print("Reranker守门阈值评测：")
    print("最高相关性分数 >= 阈值：允许回答")
    print("最高相关性分数 <  阈值：拒绝回答\n")

    for threshold in THRESHOLDS:
        correct, total, accuracy = gate_accuracy(results, threshold)
        print(
            f"阈值 {threshold:>2}："
            f"{correct}/{total} = {accuracy:.1%}"
        )

    threshold, correct, total, accuracy = choose_threshold(results)
    print("\n推荐阈值：", threshold)
    print(f"当前评测集守门准确率：{correct}/{total} = {accuracy:.1%}")

    print("\n逐题守门结果：")
    for result in results:
        allowed = result.reranker_score >= threshold
        expected = "允许" if result.case.should_answer else "拒绝"
        actual = "允许" if allowed else "拒绝"
        mark = "正确" if actual == expected else "错误"
        print(
            f"- {mark} | 预期{expected}，实际{actual} | "
            f"分数={result.reranker_score} | {result.case.question}"
        )

    print("\n" + "=" * 96)
    print("结论：不能只看一次漂亮的检索结果，必须使用带标注的问题集评测。")
    print("本次阈值只适用于当前知识库、Embedding模型、提示词和评测题。")
    print("资料或模型变化后，应重新运行评测，不能永久照搬该阈值。")


if __name__ == "__main__":
    main()
