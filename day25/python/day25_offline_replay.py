"""Day25 第 5 步：离线回放反馈样本，对比基线版本与候选版本。

真实回放（会调用模型）：
    python day25/python/day25_offline_replay.py

只检查输入，不调用模型：
    python day25/python/day25_offline_replay.py --dry-run

输出报告：
    day25/reports/day25_version_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from day25_feedback_linkage import create_schema, open_database


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY25_DIR = PROJECT_DIR / "day25"
DATASET_PATH = DAY25_DIR / "datasets" / "day25_feedback_eval_candidates.jsonl"
REPORT_DIR = DAY25_DIR / "reports"
REPORT_PATH = REPORT_DIR / "day25_version_comparison.json"

BASELINE_VERSION = "observed-production-answer"
CANDIDATE_VERSION = "day25-candidate-v2"


class PairwiseEvaluation(BaseModel):
    """成对评估模型必须返回的结构化结果。"""

    baseline_score: int = Field(ge=0, le=100)
    candidate_score: int = Field(ge=0, le=100)
    baseline_strengths: list[str] = Field(default_factory=list)
    baseline_weaknesses: list[str] = Field(default_factory=list)
    candidate_strengths: list[str] = Field(default_factory=list)
    candidate_weaknesses: list[str] = Field(default_factory=list)
    feedback_resolved: bool
    winner: Literal["baseline", "candidate", "tie"]
    reason: str = ""


@dataclass(slots=True)
class ModelUsage:
    """一次或一组模型调用的耗时与 Token。"""

    calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    elapsed_seconds: float


@dataclass(slots=True)
class ReplayResult:
    """一个反馈样本的完整版本对比结果。"""

    sample_id: str
    request_id: str
    question: str
    user_rating: str
    user_reason: str | None
    expected_behavior: str
    baseline_version: str
    baseline_answer: str
    candidate_version: str
    candidate_answer: str
    evidence: list[dict[str, object]]
    evaluation: dict[str, object]
    generation_usage: ModelUsage
    evaluation_usage: ModelUsage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day25 离线回放与版本对比")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查回放样本和证据，不调用模型。",
    )
    parser.add_argument(
        "--include-positive",
        action="store_true",
        help="除 needs_review 外，也回放点赞样本。",
    )
    return parser.parse_args()


def build_llm() -> ChatOpenAI:
    """根据项目 .env 创建模型。"""

    load_dotenv(PROJECT_DIR / ".env")
    model = os.getenv("DEEPSEEK_MODEL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    if not model or not api_key or not base_url:
        raise RuntimeError(
            ".env 缺少 DEEPSEEK_MODEL、DEEPSEEK_API_KEY 或 DEEPSEEK_BASE_URL。"
        )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def load_dataset(include_positive: bool) -> list[dict[str, object]]:
    """读取 JSONL；默认只回放需要复核的反馈样本。"""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "没有找到候选评测集，请先运行 day25_feedback_to_evalset.py。"
        )

    samples: list[dict[str, object]] = []
    for line_number, line in enumerate(
        DATASET_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是对象。")
        if include_positive or value.get("review_status") == "needs_review":
            samples.append(value)
    return samples


def load_evidence(request_id: str) -> list[dict[str, object]]:
    """从数据库读取当时真正关联的证据，而不是重新检索。"""

    with open_database() as connection:
        create_schema(connection)
        rows = connection.execute(
            """
            SELECT citation_number, chunk_id, source_path,
                   reranker_score, excerpt
            FROM answer_sources
            WHERE request_id = ?
            ORDER BY citation_number
            """,
            (request_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def format_evidence(evidence: list[dict[str, object]]) -> str:
    """把结构化证据转换成模型上下文。"""

    blocks: list[str] = []
    for item in evidence:
        blocks.append(
            "\n".join(
                [
                    f"[资料{item['citation_number']}]",
                    f"chunk_id: {item['chunk_id']}",
                    f"来源: {item['source_path']}",
                    f"正文: {item['excerpt']}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "（没有可用证据）"


def generate_candidate_answer(
    llm: ChatOpenAI,
    sample: dict[str, object],
    evidence: list[dict[str, object]],
) -> tuple[str, ModelUsage]:
    """候选 v2 根据原证据和反馈要求重新回答。"""

    messages = [
        SystemMessage(
            content=(
                "你是候选版RAG回答器。只能依据给定资料回答；"
                "重要结论必须标注[资料N]；不得编造资料外事实；"
                "应针对历史反馈改进，但不要在答案中提到评测或历史反馈。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：\n{sample['question']}\n\n"
                f"需要改善的行为：\n{sample['expected_behavior']}\n\n"
                f"参考资料：\n{format_evidence(evidence)}\n\n"
                "请给出简洁、完整、带引用的新回答。"
            )
        ),
    ]

    started = time.perf_counter()
    with get_openai_callback() as callback:
        response = llm.invoke(messages)
    elapsed = time.perf_counter() - started

    usage = ModelUsage(
        calls=1,
        input_tokens=callback.prompt_tokens,
        output_tokens=callback.completion_tokens,
        total_tokens=callback.total_tokens,
        elapsed_seconds=round(elapsed, 3),
    )
    return str(response.content), usage


def evaluate_pair(
    llm: ChatOpenAI,
    sample: dict[str, object],
    evidence: list[dict[str, object]],
    candidate_answer: str,
) -> tuple[PairwiseEvaluation, ModelUsage]:
    """在相同证据下对基线回答和候选回答进行成对比较。"""

    evaluator = llm.with_structured_output(PairwiseEvaluation)
    messages = [
        SystemMessage(
            content=(
                "你是严格的RAG版本评估器。只依据问题、证据和反馈要求判断。"
                "分别评价两个回答的正确性、相关性、完整性、引用可核验性，"
                "以及是否解决历史反馈。不要因为回答更长就给更高分。"
            )
        ),
        HumanMessage(
            content=(
                f"问题：\n{sample['question']}\n\n"
                f"历史用户评价：{sample['user_rating']}\n"
                f"历史评价原因：{sample.get('user_reason')}\n"
                f"期望行为：{sample['expected_behavior']}\n\n"
                f"证据：\n{format_evidence(evidence)}\n\n"
                f"回答A（baseline）：\n{sample['observed_answer']}\n\n"
                f"回答B（candidate）：\n{candidate_answer}\n\n"
                "请独立评分并选出更好的版本；质量相当时选择tie。"
            )
        ),
    ]

    started = time.perf_counter()
    with get_openai_callback() as callback:
        result = evaluator.invoke(messages)
    elapsed = time.perf_counter() - started
    if not isinstance(result, PairwiseEvaluation):
        raise TypeError("评估器没有返回 PairwiseEvaluation。")

    usage = ModelUsage(
        calls=1,
        input_tokens=callback.prompt_tokens,
        output_tokens=callback.completion_tokens,
        total_tokens=callback.total_tokens,
        elapsed_seconds=round(elapsed, 3),
    )
    return result, usage


def empty_usage() -> ModelUsage:
    return ModelUsage(0, 0, 0, 0, 0.0)


def dry_run_result(
    sample: dict[str, object],
    evidence: list[dict[str, object]],
) -> ReplayResult:
    """构造不调用模型的检查结果。"""

    return ReplayResult(
        sample_id=str(sample["sample_id"]),
        request_id=str(sample["request_id"]),
        question=str(sample["question"]),
        user_rating=str(sample["user_rating"]),
        user_reason=(
            str(sample["user_reason"]) if sample.get("user_reason") is not None else None
        ),
        expected_behavior=str(sample["expected_behavior"]),
        baseline_version=BASELINE_VERSION,
        baseline_answer=str(sample["observed_answer"]),
        candidate_version=CANDIDATE_VERSION,
        candidate_answer="（dry-run：没有调用候选模型）",
        evidence=evidence,
        evaluation={
            "winner": "not_evaluated",
            "reason": "dry-run 只验证数据流。",
        },
        generation_usage=empty_usage(),
        evaluation_usage=empty_usage(),
    )


def run_replay(
    samples: list[dict[str, object]],
    dry_run: bool,
) -> list[ReplayResult]:
    """依次回放样本，避免一次练习产生过高并发和费用。"""

    llm = None if dry_run else build_llm()
    results: list[ReplayResult] = []

    for index, sample in enumerate(samples, start=1):
        request_id = str(sample["request_id"])
        evidence = load_evidence(request_id)
        print(f"\n[{index}/{len(samples)}] {sample['question']}")
        print("request_id：", request_id)
        print("证据数量：", len(evidence))
        print("历史反馈：", sample["user_rating"], sample.get("user_reason"))

        if dry_run:
            results.append(dry_run_result(sample, evidence))
            print("dry-run：跳过候选回答生成和模型评估。")
            continue

        assert llm is not None
        candidate_answer, generation_usage = generate_candidate_answer(
            llm, sample, evidence
        )
        evaluation, evaluation_usage = evaluate_pair(
            llm, sample, evidence, candidate_answer
        )

        result = ReplayResult(
            sample_id=str(sample["sample_id"]),
            request_id=request_id,
            question=str(sample["question"]),
            user_rating=str(sample["user_rating"]),
            user_reason=(
                str(sample["user_reason"])
                if sample.get("user_reason") is not None
                else None
            ),
            expected_behavior=str(sample["expected_behavior"]),
            baseline_version=BASELINE_VERSION,
            baseline_answer=str(sample["observed_answer"]),
            candidate_version=CANDIDATE_VERSION,
            candidate_answer=candidate_answer,
            evidence=evidence,
            evaluation=evaluation.model_dump(),
            generation_usage=generation_usage,
            evaluation_usage=evaluation_usage,
        )
        results.append(result)

        print("候选回答：", candidate_answer)
        print(
            "评分：baseline=",
            evaluation.baseline_score,
            "candidate=",
            evaluation.candidate_score,
        )
        print("胜者：", evaluation.winner)
        print("反馈是否解决：", evaluation.feedback_resolved)

    return results


def build_summary(results: list[ReplayResult]) -> dict[str, object]:
    """聚合版本胜负、调用次数和 Token。"""

    winners = {"baseline": 0, "candidate": 0, "tie": 0, "not_evaluated": 0}
    model_calls = 0
    total_tokens = 0
    for result in results:
        winner = str(result.evaluation.get("winner", "not_evaluated"))
        winners[winner] = winners.get(winner, 0) + 1
        model_calls += result.generation_usage.calls + result.evaluation_usage.calls
        total_tokens += (
            result.generation_usage.total_tokens + result.evaluation_usage.total_tokens
        )

    evaluated = len(results) - winners["not_evaluated"]
    candidate_win_rate = (
        round(winners["candidate"] / evaluated * 100, 1) if evaluated else None
    )
    return {
        "samples": len(results),
        "evaluated_samples": evaluated,
        "winners": winners,
        "candidate_win_rate": candidate_win_rate,
        "model_calls": model_calls,
        "total_tokens": total_tokens,
    }


def save_report(results: list[ReplayResult], dry_run: bool) -> dict[str, object]:
    """保存机器可读的版本对比报告。"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "day25-replay-report-v1",
        "dry_run": dry_run,
        "baseline_version": BASELINE_VERSION,
        "candidate_version": CANDIDATE_VERSION,
        "summary": build_summary(results),
        "results": [asdict(result) for result in results],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    print("Day25 离线反馈回放与版本对比")
    print("基线版本：", BASELINE_VERSION)
    print("候选版本：", CANDIDATE_VERSION)
    print("模式：", "dry-run" if args.dry_run else "真实模型回放")
    print("=" * 96)

    samples = load_dataset(args.include_positive)
    if not samples:
        raise RuntimeError("没有符合回放条件的样本。")
    print("准备回放的样本数：", len(samples))

    results = run_replay(samples, dry_run=args.dry_run)
    report = save_report(results, dry_run=args.dry_run)

    print("\n" + "=" * 96)
    print("版本对比汇总：")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("报告文件：", REPORT_PATH)

    print("\n本步数据流：")
    print("反馈候选JSONL → 筛选needs_review → 按request_id读取原证据")
    print("→ 候选v2重新回答 → PairwiseEvaluation比较baseline/candidate")
    print("→ ReplayResult列表 → 版本胜率、调用数、Token与JSON报告")
    print("\n注意：候选胜出不等于可以直接上线，还需要更大的金标集和人工抽检。")


if __name__ == "__main__":
    main()
