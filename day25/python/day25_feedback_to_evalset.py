"""Day25 第 4 步：从线上反馈生成可复现的离线评测样本。

运行：
    python day25/python/day25_feedback_to_evalset.py

输出：
    day25/datasets/day25_feedback_eval_candidates.jsonl

注意：点赞/点踩是弱标签，不等于经过人工确认的标准答案。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from day25_feedback_linkage import (
    AnswerRun,
    LinkedFeedback,
    SourceTrace,
    create_schema,
    open_database,
    save_answer_run,
    save_linked_feedback,
)


DAY25_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = DAY25_DIR / "datasets"
DATASET_PATH = DATASET_DIR / "day25_feedback_eval_candidates.jsonl"
DATASET_VERSION = "day25-feedback-eval-v1"

ReviewStatus = Literal["candidate_pass", "needs_review"]


@dataclass(slots=True)
class EvaluationCandidate:
    """从用户反馈生成的候选评测样本。"""

    sample_id: str
    dataset_version: str
    request_id: str
    question: str
    observed_answer: str
    observed_answer_version: str
    source_ids: list[str]
    source_paths: list[str]
    user_rating: str
    user_reason: str | None
    review_status: ReviewStatus
    expected_behavior: str
    metrics: dict[str, int | str | bool]


def seed_demonstration_records(connection: sqlite3.Connection) -> None:
    """写入三种反馈示例，让本练习可以独立运行。"""

    demonstrations = [
        (
            AnswerRun(
                request_id="req-day25-eval-up-001",
                session_id="xiaolin-day25",
                user_id="user-xiaolin",
                question="什么是 RAG？",
                answer=(
                    "RAG 是检索增强生成：先检索外部资料，"
                    "再把资料和问题交给大语言模型。[资料1]"
                ),
                answer_version="rag-pipeline-v1",
                risk_level="low",
                model_name="deepseek-chat",
                model_calls=2,
                input_tokens=1600,
                output_tokens=220,
                latency_ms=1850,
                cache_hits=0,
                quality_passed=True,
                sources=[
                    SourceTrace(
                        chunk_id="chunk-rag-definition-001",
                        source_path="day21/documents/day21_sample.txt",
                        citation_number=1,
                        reranker_score=98,
                        excerpt="RAG 的中文名称是检索增强生成。",
                    )
                ],
            ),
            LinkedFeedback(
                request_id="req-day25-eval-up-001",
                user_id="user-xiaolin",
                rating="up",
                reason="定义和流程都很清楚。",
            ),
        ),
        (
            AnswerRun(
                request_id="req-day25-eval-down-001",
                session_id="xiaolin-day25",
                user_id="user-xiaolin",
                question="为什么 RAG 要保留来源信息？",
                answer="保留来源信息很重要。",
                answer_version="rag-pipeline-v1",
                risk_level="medium",
                model_name="deepseek-chat",
                model_calls=3,
                input_tokens=2350,
                output_tokens=65,
                latency_ms=3100,
                cache_hits=0,
                quality_passed=True,
                sources=[
                    SourceTrace(
                        chunk_id="chunk-source-metadata-001",
                        source_path="day06/documents/pdf/day6_rag_test.pdf",
                        citation_number=1,
                        reranker_score=96,
                        excerpt=(
                            "每个 Document 都应该保留 source、title、page 等来源信息。"
                        ),
                    )
                ],
            ),
            LinkedFeedback(
                request_id="req-day25-eval-down-001",
                user_id="user-xiaolin",
                rating="down",
                reason="只说重要，没有解释为什么，也没有显示引用。",
            ),
        ),
        (
            AnswerRun(
                request_id="req-day25-eval-neutral-001",
                session_id="xiaolin-day25",
                user_id="user-xiaolin",
                question="LangGraph 的节点有什么作用？",
                answer="节点负责执行工作流中的一个处理步骤。[资料1]",
                answer_version="rag-pipeline-v2",
                risk_level="low",
                model_name="deepseek-chat",
                model_calls=2,
                input_tokens=1420,
                output_tokens=105,
                latency_ms=4800,
                cache_hits=1,
                quality_passed=True,
                sources=[
                    SourceTrace(
                        chunk_id="chunk-langgraph-node-001",
                        source_path="day06/documents/markdown/day6_langgraph.md",
                        citation_number=1,
                        reranker_score=97,
                        excerpt="LangGraph 中的节点负责处理状态数据。",
                    )
                ],
            ),
            LinkedFeedback(
                request_id="req-day25-eval-neutral-001",
                user_id="user-xiaolin",
                rating="neutral",
                reason="回答正确，但等待时间有点长。",
            ),
        ),
    ]

    for answer_run, feedback in demonstrations:
        save_answer_run(connection, answer_run)
        save_linked_feedback(connection, feedback)


def stable_sample_id(request_id: str, user_id: str) -> str:
    """用稳定输入生成稳定 ID；重复导出不会改变。"""

    raw = f"{DATASET_VERSION}|{request_id}|{user_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"eval-{digest}"


def decide_review_status(rating: str) -> ReviewStatus:
    """差评和中立样本必须人工复核；点赞样本也只是候选通过。"""

    return "candidate_pass" if rating == "up" else "needs_review"


def expected_behavior_from_feedback(rating: str, reason: str | None) -> str:
    """生成评测要求，但不把原回答复制成所谓标准答案。"""

    if rating == "down":
        detail = reason or "用户认为回答不满足需求。"
        return f"新版本应修复该负面反馈：{detail}"
    if rating == "neutral":
        detail = reason or "回答仍有可改进之处。"
        return f"新版本应保持正确性并改善体验：{detail}"
    return "新版本至少应保持原回答的正确性、相关性和引用可核验性。"


def read_candidates(connection: sqlite3.Connection) -> list[EvaluationCandidate]:
    """JOIN 三张表，构造候选评测样本。"""

    rows = connection.execute(
        """
        SELECT
            ar.request_id,
            ar.question,
            ar.answer,
            ar.answer_version,
            ar.risk_level,
            ar.model_name,
            ar.model_calls,
            ar.input_tokens,
            ar.output_tokens,
            ar.latency_ms,
            ar.cache_hits,
            ar.quality_passed,
            lf.user_id,
            lf.rating,
            lf.reason
        FROM answer_runs AS ar
        INNER JOIN linked_feedback AS lf
            ON lf.request_id = ar.request_id
        ORDER BY ar.request_id, lf.user_id
        """
    ).fetchall()

    candidates: list[EvaluationCandidate] = []
    for row in rows:
        source_rows = connection.execute(
            """
            SELECT chunk_id, source_path
            FROM answer_sources
            WHERE request_id = ?
            ORDER BY citation_number
            """,
            (row["request_id"],),
        ).fetchall()

        candidates.append(
            EvaluationCandidate(
                sample_id=stable_sample_id(row["request_id"], row["user_id"]),
                dataset_version=DATASET_VERSION,
                request_id=row["request_id"],
                question=row["question"],
                observed_answer=row["answer"],
                observed_answer_version=row["answer_version"],
                source_ids=[source["chunk_id"] for source in source_rows],
                source_paths=[source["source_path"] for source in source_rows],
                user_rating=row["rating"],
                user_reason=row["reason"],
                review_status=decide_review_status(row["rating"]),
                expected_behavior=expected_behavior_from_feedback(
                    row["rating"], row["reason"]
                ),
                metrics={
                    "risk_level": row["risk_level"],
                    "model_name": row["model_name"],
                    "model_calls": row["model_calls"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "latency_ms": row["latency_ms"],
                    "cache_hits": row["cache_hits"],
                    "quality_passed": bool(row["quality_passed"]),
                },
            )
        )
    return candidates


def write_jsonl(candidates: list[EvaluationCandidate]) -> str:
    """按稳定顺序写 JSONL，并返回整个文件的 SHA-256。"""

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True)
        for candidate in candidates
    ]
    content = "\n".join(lines) + ("\n" if lines else "")
    DATASET_PATH.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def summarize(candidates: list[EvaluationCandidate]) -> dict[str, Any]:
    """统计数据集构成。"""

    ratings = {"up": 0, "down": 0, "neutral": 0}
    review_statuses = {"candidate_pass": 0, "needs_review": 0}
    for candidate in candidates:
        ratings[candidate.user_rating] += 1
        review_statuses[candidate.review_status] += 1
    return {
        "samples": len(candidates),
        "ratings": ratings,
        "review_statuses": review_statuses,
    }


def main() -> None:
    print("Day25 从反馈生成候选评测集")
    print("数据集版本：", DATASET_VERSION)
    print("输出文件：", DATASET_PATH)
    print("=" * 88)

    with open_database() as connection:
        create_schema(connection)
        seed_demonstration_records(connection)
        candidates = read_candidates(connection)

    dataset_hash = write_jsonl(candidates)
    statistics = summarize(candidates)

    print("候选样本数量：", statistics["samples"])
    print("评价分布：", statistics["ratings"])
    print("人工复核状态：", statistics["review_statuses"])
    print("数据集 SHA-256：", dataset_hash)

    print("\n" + "=" * 88)
    print("逐条候选样本：")
    for candidate in candidates:
        print("\n样本：", candidate.sample_id)
        print("问题：", candidate.question)
        print("用户评价：", candidate.user_rating)
        print("评价原因：", candidate.user_reason)
        print("复核状态：", candidate.review_status)
        print("期望行为：", candidate.expected_behavior)
        print("来源：", candidate.source_paths)

    print("\n" + "=" * 88)
    print("本步数据流：")
    print("answer_runs + answer_sources + linked_feedback")
    print("→ SQL JOIN → EvaluationCandidate")
    print("→ 稳定sample_id + 确定性排序 + JSON sort_keys")
    print("→ JSONL评测候选集 + 文件SHA-256")
    print("\n重要：用户反馈是弱标签，needs_review 样本需人工确认后才能成为金标。")


if __name__ == "__main__":
    main()
