"""Day24 第二步：为四个模型阶段增加SQLite持久化缓存。

清空缓存并观察第一次MISS、第二次HIT：
    python day24/python/day24_pipeline_cache.py --clear-cache

保留已有缓存直接运行：
    python day24/python/day24_pipeline_cache.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY22_PYTHON = PROJECT_DIR / "day22" / "python"
DAY23_PYTHON = PROJECT_DIR / "day23" / "python"
DAY24_PYTHON = PROJECT_DIR / "day24" / "python"
for directory in (DAY22_PYTHON, DAY23_PYTHON, DAY24_PYTHON):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from day22_final_hybrid_rag import retrieve_rrf_candidates  # noqa: E402
from day22_llm_reranker import (  # noqa: E402
    RerankedResult,
    RerankResponse,
    rerank_candidates,
)
from day23_answer_quality import (  # noqa: E402
    AnswerQualityResponse,
    build_final_quality_report,
    evaluate_answer_quality,
)
from day23_citation_audit import (  # noqa: E402
    audit_citations,
    generate_structured_answer,
)
from day23_claim_groundedness import (  # noqa: E402
    GroundednessResponse,
    build_groundedness_report,
    evaluate_claim_support,
)
from day23_structured_citations import (  # noqa: E402
    StructuredRAGAnswer,
    build_readable_answer,
    select_evidence,
)
from day24_pipeline_metrics import (  # noqa: E402
    PipelineMetrics,
    StageMetric,
    measure_local_stage,
    measure_model_stage,
    print_metrics,
)


T = TypeVar("T")
DATABASE_PATH = PROJECT_DIR / "day24" / "database" / "day24_model_cache.db"
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "unknown-model")
PROMPT_VERSION = "day24-v1"


class ModelCache:
    """使用稳定哈希键保存模型阶段的JSON结果。"""

    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_cache (
                cache_key TEXT PRIMARY KEY,
                stage_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.connection.commit()

    def make_key(self, stage_name: str, input_data: Any) -> str:
        """规范化JSON后计算SHA-256，输入变化必然产生新键。"""

        canonical = json.dumps(
            {
                "stage_name": stage_name,
                "model_name": MODEL_NAME,
                "prompt_version": PROMPT_VERSION,
                "input": input_data,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM model_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None

        now = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            UPDATE model_cache
            SET hit_count = hit_count + 1,
                last_accessed_at = ?
            WHERE cache_key = ?
            """,
            (now, cache_key),
        )
        self.connection.commit()
        return json.loads(row["payload_json"])

    def put(
        self,
        cache_key: str,
        stage_name: str,
        payload: dict[str, Any],
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO model_cache (
                cache_key, stage_name, model_name, prompt_version,
                payload_json, created_at, last_accessed_at, hit_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                cache_key,
                stage_name,
                MODEL_NAME,
                PROMPT_VERSION,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        self.connection.commit()

    def clear(self) -> int:
        count = self.connection.execute(
            "SELECT COUNT(*) AS count FROM model_cache"
        ).fetchone()["count"]
        self.connection.execute("DELETE FROM model_cache")
        self.connection.commit()
        return int(count)

    def count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) AS count FROM model_cache"
            ).fetchone()["count"]
        )

    def close(self) -> None:
        self.connection.close()


def cached_model_stage(
    metrics: PipelineMetrics,
    cache: ModelCache,
    stage_name: str,
    input_data: Any,
    function: Callable[..., T],
    function_args: tuple[Any, ...],
    serializer: Callable[[T], dict[str, Any]],
    deserializer: Callable[[dict[str, Any]], T],
) -> tuple[T, bool]:
    """命中时反序列化；未命中时真实调用模型并写入缓存。"""

    cache_key = cache.make_key(stage_name, input_data)
    started = time.perf_counter()
    cached_payload = cache.get(cache_key)

    if cached_payload is not None:
        result = deserializer(cached_payload)
        metrics.stages.append(
            StageMetric(
                name=stage_name,
                kind="cache",
                elapsed_seconds=time.perf_counter() - started,
                model_calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                status="cache_hit",
                detail=f"key={cache_key[:12]}，没有调用模型。",
            )
        )
        return result, True

    result = measure_model_stage(
        metrics,
        stage_name,
        function,
        *function_args,
        detail=f"cache miss，key={cache_key[:12]}，调用模型后写入缓存。",
    )
    cache.put(cache_key, stage_name, serializer(result))
    return result, False


def serialize_reranker(
    result: tuple[list[RerankedResult], RerankResponse],
) -> dict[str, Any]:
    reranked, response = result
    return {
        "reranked": [asdict(item) for item in reranked],
        "response": response.model_dump(mode="json"),
    }


def deserialize_reranker(
    payload: dict[str, Any],
) -> tuple[list[RerankedResult], RerankResponse]:
    return (
        [RerankedResult(**item) for item in payload["reranked"]],
        RerankResponse.model_validate(payload["response"]),
    )


def serialize_pydantic(result: Any) -> dict[str, Any]:
    return {"value": result.model_dump(mode="json")}


def deserialize_answer(payload: dict[str, Any]) -> StructuredRAGAnswer:
    return StructuredRAGAnswer.model_validate(payload["value"])


def deserialize_groundedness(payload: dict[str, Any]) -> GroundednessResponse:
    return GroundednessResponse.model_validate(payload["value"])


def deserialize_quality(payload: dict[str, Any]) -> AnswerQualityResponse:
    return AnswerQualityResponse.model_validate(payload["value"])


def evidence_key_data(evidence: list[RerankedResult]) -> list[dict[str, Any]]:
    """把证据内容纳入键；正文变化时旧缓存自动失效。"""

    return [asdict(item) for item in evidence]


def run_pipeline(question: str, cache: ModelCache, run_number: int) -> PipelineMetrics:
    metrics = PipelineMetrics(question=question)
    started = time.perf_counter()

    print("\n" + "#" * 112)
    print(f"第 {run_number} 次运行")
    print("#" * 112)

    rrf_results, _ = measure_local_stage(
        metrics,
        "混合检索与RRF",
        retrieve_rrf_candidates,
        question,
        detail="本地检索不进入模型缓存。",
    )

    reranker_input = {
        "question": question,
        "candidates": [asdict(item) for item in rrf_results],
    }
    reranker_result, reranker_hit = cached_model_stage(
        metrics,
        cache,
        "Reranker精排",
        reranker_input,
        rerank_candidates,
        (question, rrf_results),
        serialize_reranker,
        deserialize_reranker,
    )
    reranked_results, _ = reranker_result
    evidence = select_evidence(reranked_results)

    print("Reranker缓存：", "HIT" if reranker_hit else "MISS")
    print("最高分：", reranked_results[0].relevance_score)
    print("合格证据数：", len(evidence))
    if not evidence:
        print("无合格证据，提前结束。")
        print_metrics(metrics)
        return metrics

    evidence_data = evidence_key_data(evidence)
    answer, answer_hit = cached_model_stage(
        metrics,
        cache,
        "结构化回答生成",
        {"question": question, "evidence": evidence_data},
        generate_structured_answer,
        (question, evidence),
        serialize_pydantic,
        deserialize_answer,
    )
    rendered_answer = build_readable_answer(answer)
    citation_audit = audit_citations(answer, len(evidence), rendered_answer)
    if not citation_audit.passed:
        print("引用审计失败：", citation_audit.errors)
        print_metrics(metrics)
        return metrics

    groundedness_response, groundedness_hit = cached_model_stage(
        metrics,
        cache,
        "Groundedness评估",
        {
            "answer": answer.model_dump(mode="json"),
            "evidence": evidence_data,
        },
        evaluate_claim_support,
        (answer, evidence),
        serialize_pydantic,
        deserialize_groundedness,
    )
    groundedness_report = build_groundedness_report(
        answer,
        groundedness_response,
    )

    quality_response, quality_hit = cached_model_stage(
        metrics,
        cache,
        "相关性与完整性评估",
        {
            "question": question,
            "answer": answer.model_dump(mode="json"),
            "evidence": evidence_data,
        },
        evaluate_answer_quality,
        (question, answer, evidence),
        serialize_pydantic,
        deserialize_quality,
    )
    final_report = build_final_quality_report(
        citation_audit,
        groundedness_report,
        quality_response,
    )

    real_elapsed = time.perf_counter() - started
    hit_count = sum(
        [reranker_hit, answer_hit, groundedness_hit, quality_hit]
    )
    print("结构化回答缓存：", "HIT" if answer_hit else "MISS")
    print("Groundedness缓存：", "HIT" if groundedness_hit else "MISS")
    print("质量评估缓存：", "HIT" if quality_hit else "MISS")
    print("本轮缓存命中：", f"{hit_count}/4")
    print("最终质量：", "通过" if final_report.passed else "失败")
    print_metrics(metrics)
    print(f"本轮墙钟耗时：{real_elapsed:.3f}秒")
    return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day24模型阶段缓存练习")
    parser.add_argument("question", nargs="*", help="知识库问题")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="运行前清空SQLite模型缓存",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    question = " ".join(arguments.question).strip() or (
        "什么是 RAG？它的工作流程是什么？"
    )

    print("Day24 SQLite模型阶段缓存")
    print("问题：", question)
    print("数据库：", DATABASE_PATH)
    print("模型：", MODEL_NAME)
    print("提示词版本：", PROMPT_VERSION)

    cache = ModelCache(DATABASE_PATH)
    try:
        if arguments.clear_cache:
            removed = cache.clear()
            print("运行前清除缓存条数：", removed)
        print("启动时缓存条数：", cache.count())

        first = run_pipeline(question, cache, run_number=1)
        second = run_pipeline(question, cache, run_number=2)

        print("\n" + "=" * 112)
        print("两次运行对比：")
        print(
            f"第一次：调用={first.total_model_calls}，"
            f"Token={first.total_tokens}，"
            f"阶段耗时={first.total_elapsed_seconds:.3f}秒"
        )
        print(
            f"第二次：调用={second.total_model_calls}，"
            f"Token={second.total_tokens}，"
            f"阶段耗时={second.total_elapsed_seconds:.3f}秒"
        )
        print("结束时缓存条数：", cache.count())
        print("第二次命中缓存时，模型调用和新增Token应当降为0。")
        print("修改问题、正文、模型名或PROMPT_VERSION都会自动产生新键。")
    finally:
        cache.close()


if __name__ == "__main__":
    main()
