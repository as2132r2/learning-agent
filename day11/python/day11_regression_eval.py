"""Day11 第四步：用黄金数据集做确定性的回归评测。"""

from dataclasses import dataclass


# ============================================================
# 1. 模拟知识库和检索结果
# ============================================================

@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    topic: str
    text: str
    keywords: tuple[str, ...]


KNOWLEDGE_BASE = [
    KnowledgeChunk(
        source="day6_rag_test.pdf",
        topic="RAG",
        text="RAG 会先检索资料，再把资料和问题一起交给模型。",
        keywords=("RAG", "检索增强", "检索")
    ),
    KnowledgeChunk(
        source="day6_langgraph.md",
        topic="LangGraph",
        text="LangGraph 使用节点、边和状态组织工作流。",
        keywords=("LangGraph", "节点", "边", "状态", "工作流")
    ),
    KnowledgeChunk(
        source="day6_c_language.txt",
        topic="C语言",
        text="C语言接近底层，常用于操作系统、驱动和嵌入式开发。",
        keywords=("C语言", "操作系统", "驱动", "嵌入式")
    )
]


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_sources: tuple[str, ...]
    should_reject: bool


@dataclass(frozen=True)
class EvaluationResult:
    case: EvaluationCase
    top_source: str | None
    should_reject: bool
    passed: bool


def retrieve(question: str, top_k: int = 1) -> list[KnowledgeChunk]:
    """用关键词重叠模拟一个确定性的检索器。"""

    scored: list[tuple[int, KnowledgeChunk]] = []

    for chunk in KNOWLEDGE_BASE:
        score = sum(
            1 for keyword in chunk.keywords
            if keyword.lower() in question.lower()
        )

        if score > 0:
            scored.append((score, chunk))

    scored.sort(
        key=lambda item: (-item[0], item[1].source)
    )

    return [chunk for _, chunk in scored[:top_k]]


def should_reject(results: list[KnowledgeChunk]) -> bool:
    """没有任何相关资料时拒答。"""

    return len(results) == 0


# ============================================================
# 2. 黄金评测集
# ============================================================

EVALUATION_DATASET = [
    EvaluationCase(
        question="什么是 RAG？",
        expected_sources=("day6_rag_test.pdf",),
        should_reject=False
    ),
    EvaluationCase(
        question="LangGraph 的节点有什么作用？",
        expected_sources=("day6_langgraph.md",),
        should_reject=False
    ),
    EvaluationCase(
        question="C语言适合开发什么？",
        expected_sources=("day6_c_language.txt",),
        should_reject=False
    ),
    EvaluationCase(
        question="法国的首都是哪里？",
        expected_sources=(),
        should_reject=True
    ),
    EvaluationCase(
        question="今天北京天气怎么样？",
        expected_sources=(),
        should_reject=True
    ),
    EvaluationCase(
        question="红烧肉应该怎么做？",
        expected_sources=(),
        should_reject=True
    )
]


# ============================================================
# 3. 执行评测
# ============================================================

def evaluate_case(case: EvaluationCase) -> EvaluationResult:
    results = retrieve(case.question)
    top_source = results[0].source if results else None
    actual_rejection = should_reject(results)

    source_ok = (
        top_source in case.expected_sources
        if not case.should_reject
        else top_source is None
    )
    rejection_ok = actual_rejection == case.should_reject

    return EvaluationResult(
        case=case,
        top_source=top_source,
        should_reject=actual_rejection,
        passed=source_ok and rejection_ok
    )


def run_evaluation() -> list[EvaluationResult]:
    print("评测问题数量：", len(EVALUATION_DATASET))
    results: list[EvaluationResult] = []

    for number, case in enumerate(EVALUATION_DATASET, start=1):
        result = evaluate_case(case)
        results.append(result)

        print("\n" + "-" * 60)
        print(f"第 {number} 题：{case.question}")
        print("预期来源：", case.expected_sources or "应该拒答")
        print("Top-1 来源：", result.top_source or "无")
        print("实际决定：", "拒答" if result.should_reject else "允许回答")
        print("结果：", "通过" if result.passed else "失败")

    return results


results = run_evaluation()
passed_count = sum(1 for result in results if result.passed)
accuracy = passed_count / len(results) * 100

print("\n" + "=" * 60)
print("回归评测汇总")
print("通过数量：", f"{passed_count}/{len(results)}")
print(f"总体准确率：{accuracy:.1f}%")

if passed_count != len(results):
    raise SystemExit("回归评测失败：不能继续发布当前版本。")

print("质量门禁：通过，可以继续后续流程。")


# ============================================================
# 数据流
# ============================================================
#
# EvaluationCase 黄金问题
#          ↓
# retrieve() 检索结果
#          ↓
# 来源判断 + 拒答判断
#          ↓
# EvaluationResult
#          ↓
# 总体准确率和质量门禁
#
# 评测集的作用：
# 修改检索算法、阈值或提示词后，重新运行同一批问题，
# 如果准确率下降，立即发现回归问题。
