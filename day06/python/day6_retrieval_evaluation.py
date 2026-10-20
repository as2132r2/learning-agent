import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ============================================================
# 1. 配置与模型
# ============================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
database_path = day_directory / "database" / "day6_knowledge_cli.db"
cache_directory = project_directory / ".fastembed_cache"


class LocalChineseEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str):
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


embedding_model = LocalChineseEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir=str(cache_directory)
)


# ============================================================
# 2. 定义一条评测数据的数据类型
# ============================================================

@dataclass
class EvaluationCase:
    question: str

    # 知识库可以回答时填写一个或多个正确来源；
    # 知识库不应该回答时填写 None。
    expected_sources: set[str] | None


evaluation_cases: list[EvaluationCase] = [
    EvaluationCase(
        question="什么是 RAG？",
        expected_sources={
            "pdf/day6_rag_test.pdf",
            "txt/day6_langchain.txt"
        }
    ),
    EvaluationCase(
        question="为什么 RAG 要保留来源信息？",
        expected_sources={"pdf/day6_rag_test.pdf"}
    ),
    EvaluationCase(
        question="LangGraph 的节点有什么作用？",
        expected_sources={
            "markdown/day6_langgraph.md",
            "txt/day6_langchain.txt"
        }
    ),
    EvaluationCase(
        question="LangGraph 如何保存工作流状态？",
        expected_sources={
            "markdown/day6_langgraph.md",
            "txt/day6_langchain.txt"
        }
    ),
    EvaluationCase(
        question="C语言适合开发什么？",
        expected_sources={"txt/day6_c_language.txt"}
    ),
    EvaluationCase(
        question="学习 C 语言要注意哪些内存问题？",
        expected_sources={"txt/day6_c_language.txt"}
    ),
    EvaluationCase(
        question="法国的首都在哪里？",
        expected_sources=None
    ),
    EvaluationCase(
        question="今天北京天气怎么样？",
        expected_sources=None
    ),
    EvaluationCase(
        question="红烧肉应该怎么做？",
        expected_sources=None
    ),
    EvaluationCase(
        question="推荐一部科幻电影。",
        expected_sources=None
    )
]


# ============================================================
# 3. 打开向量数据库
# ============================================================

if not database_path.exists():
    print("数据库不存在，请先运行：python day06/python/day6_knowledge_cli.py sync")
    raise SystemExit(1)

connection = sqlite3.connect(database_path)
connection.row_factory = sqlite3.Row
connection.enable_load_extension(True)
sqlite_vec.load(connection)
connection.enable_load_extension(False)

vector_store = SQLiteVec(
    table="knowledge_chunks",
    connection=connection,
    embedding=embedding_model
)


# ============================================================
# 4. 批量运行评测问题
# ============================================================

@dataclass
class EvaluationResult:
    case: EvaluationCase
    retrieved_document: Document
    distance: float


evaluation_results: list[EvaluationResult] = []

print("评测问题数量：", len(evaluation_cases))
print("\n" + "=" * 75)

for number, case in enumerate(evaluation_cases, start=1):
    document, distance = vector_store.similarity_search_with_score(
        case.question,
        k=1
    )[0]

    result = EvaluationResult(
        case=case,
        retrieved_document=document,
        distance=distance
    )
    evaluation_results.append(result)

    actual_source = document.metadata.get("source", "未知来源")
    source_correct = (
        case.expected_sources is not None
        and actual_source in case.expected_sources
    )

    expected_text = (
        sorted(case.expected_sources)
        if case.expected_sources is not None
        else "应该拒绝"
    )

    print(f"第 {number} 题：{case.question}")
    print("预期：", expected_text)
    print("Top-1 来源：", actual_source)
    print("Top-1 距离：", round(distance, 4))

    if case.expected_sources is None:
        print("来源判断：这是无关问题，等待阈值守门")
    else:
        print("来源判断：", "正确" if source_correct else "错误")

    print("-" * 75)


# ============================================================
# 5. 统计相关问题的 Top-1 来源准确率
# ============================================================

answerable_results = [
    result
    for result in evaluation_results
    if result.case.expected_sources is not None
]

correct_source_count = sum(
    result.retrieved_document.metadata.get("source")
    in result.case.expected_sources
    for result in answerable_results
)

source_accuracy = (
    correct_source_count / len(answerable_results)
    if answerable_results
    else 0.0
)

print("\n相关问题 Top-1 来源准确率：")
print(
    f"{correct_source_count}/{len(answerable_results)} "
    f"= {source_accuracy:.1%}"
)


# ============================================================
# 6. 扫描多个候选阈值
# ============================================================

candidate_thresholds = [
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20
]

threshold_scores: list[tuple[float, int, float]] = []

print("\n" + "=" * 75)
print("阈值评测：")
print("距离 <= 阈值：允许回答")
print("距离 >  阈值：拒绝回答")
print()

for threshold in candidate_thresholds:
    correct_count = 0

    for result in evaluation_results:
        predicted_answerable = result.distance <= threshold
        expected_answerable = result.case.expected_sources is not None

        if predicted_answerable == expected_answerable:
            correct_count += 1

    accuracy = correct_count / len(evaluation_results)
    threshold_scores.append((threshold, correct_count, accuracy))

    print(
        f"阈值 {threshold:.2f}："
        f"{correct_count}/{len(evaluation_results)} "
        f"= {accuracy:.1%}"
    )


# ============================================================
# 7. 选出本评测集上的最佳阈值
# ============================================================

# max() 先比较准确率；准确率相同时，选择更小、更保守的阈值。
best_threshold, best_correct_count, best_accuracy = max(
    threshold_scores,
    key=lambda item: (item[2], -item[0])
)

print("\n" + "=" * 75)
print("当前小型评测集上的推荐阈值：", best_threshold)
print(
    "守门判断准确率：",
    f"{best_correct_count}/{len(evaluation_results)} "
    f"= {best_accuracy:.1%}"
)


# ============================================================
# 8. 使用最佳阈值逐题显示最终决策
# ============================================================

print("\n逐题守门结果：")

for result in evaluation_results:
    allowed = result.distance <= best_threshold
    decision = "允许回答" if allowed else "拒绝回答"

    print(
        f"- {decision} | "
        f"距离={result.distance:.4f} | "
        f"{result.case.question}"
    )

connection.close()

print("\n" + "=" * 75)
print("说明：这个阈值只适用于当前模型、当前知识库和当前评测问题。")
print("增加真实用户问题后，应当重新运行评测并调整阈值。")
