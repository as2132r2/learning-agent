"""Day35 第1步：固定毕业验收场景与证据要求。

python day35/python/day35_acceptance_plan.py

生成验收计划，不执行模型、审批或写入，不把计划标记为已通过。
后续自动验收器可以直接导入CASES或读取生成的JSON。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


DAY35_DIR = Path(__file__).resolve().parents[1]
PLAN_PATH = DAY35_DIR / "evaluation" / "acceptance_plan.json"
PLAN_VERSION = "day35-v1"


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    capability: str
    steps: tuple[str, ...]
    checks: tuple[str, ...]
    evidence: tuple[str, ...]
    validation_modes: tuple[str, ...] = ("fake",)


CASES = (
    AcceptanceCase(
        "G01", "姓名记忆",
        ("在session-a说：我叫小林。", "同会话问：我叫什么名字？"),
        ("回复提到小林。", "两轮用户与助手消息保存在session-a。"),
        ("最终回答", "checkpoint.messages"), ("fake", "live"),
    ),
    AcceptanceCase(
        "G02", "会话隔离",
        ("在session-a登记小林。", "切换session-b，问：我叫什么名字？"),
        ("session-b历史不包含session-a的姓名登记。", "回复不声称已知小林的身份。"),
        ("两个会话checkpoint", "session-b回答"), ("fake", "live"),
    ),
    AcceptanceCase(
        "G03", "工具真实执行",
        ("输入：请计算(20+4)*3。",),
        ("执行add(20,4)和multiply(24,3)。", "工具结果依次24和72，最终答案为72。"),
        ("AIMessage.tool_calls", "匹配ID的ToolMessage", "最终回答"), ("fake", "live"),
    ),
    AcceptanceCase(
        "G04", "知识回答与引用",
        ("输入：根据课程资料解释RAG。",),
        ("有非空合格证据。", "回答解释检索增强生成，引用编号真实存在。"),
        ("evidence来源和正文", "回答及引用编号"), ("fake", "live"),
    ),
    AcceptanceCase(
        "G05", "无证据拒答",
        ("让检索返回空证据，提出资料中没有的问题。",),
        ("回复资料不足。", "不调用知识回答模型。"),
        ("空evidence", "模型调用计数", "拒答文本"),
    ),
    AcceptanceCase(
        "G06", "批准后写入",
        ("创建标题和正文明确的/note提案。", "查看暂停内容后批准该流程。"),
        ("批准前无笔记文件。", "批准后文件与审批快照正文一致。", "聊天历史如实记录已保存。"),
        ("审批checkpoint", "Markdown文件字节", "聊天事件"),
    ),
    AcceptanceCase(
        "G07", "拒绝不写入",
        ("创建独立笔记提案。", "拒绝该流程。"),
        ("状态rejected。", "该提案不产生笔记文件。", "拒绝后不能重试写入。"),
        ("流程最终状态", "文件集合", "重试拒绝结果"),
    ),
    AcceptanceCase(
        "G08", "重启恢复",
        ("记录姓名并创建待审批提案。", "关闭数据库连接与图，重新打开。", "问姓名，恢复原审批。"),
        ("相同会话恢复姓名历史。", "恢复前后note_id与快照哈希一致。", "决定作用于原提案。"),
        ("重建前后checkpoint", "恢复结果"), ("fake", "live"),
    ),
    AcceptanceCase(
        "G09", "参数拦截",
        ("向工具传入数字字符串、True、inf、未知字段和超范围数值。",),
        ("每组非法参数均被运行时校验拒绝。", "合法输入仍可计算72。"),
        ("ValidationError", "合法工具结果"),
    ),
    AcceptanceCase(
        "G10", "临时故障与幂等",
        ("批准时模拟EBUSY。", "解除模拟故障后重试同一流程。", "再次验证相同内容写入。"),
        ("故障时为failed，不能宣称saved。", "重试保留原快照且仅产生一份笔记。", "不同内容冲突不能覆盖。"),
        ("错误分类", "重试计数", "文件内容及数量"),
    ),
    AcceptanceCase(
        "G11", "审批归属与快照保护",
        ("尝试跨会话审批或重试。", "提交不匹配快照哈希。", "重复批准已完成流程。"),
        ("全部拒绝，不能新增错误笔记。",),
        ("拒绝异常", "审批状态", "文件集合"),
    ),
    AcceptanceCase(
        "G12", "故障边界与上限",
        ("模拟权限不足。", "另一流程持续模拟临时错误并重试。"),
        ("权限不足不能直接重试。", "人工重试最多2次，第3次被拒绝。"),
        ("error_type/retryable", "retry_count", "拒绝异常"),
    ),
)


def validate_plan() -> None:
    ids = [case.case_id for case in CASES]
    if ids != [f"G{index:02d}" for index in range(1, 13)]:
        raise ValueError("验收场景必须保持G01到G12唯一且连续。")
    for case in CASES:
        if not case.steps or not case.checks or not case.evidence:
            raise ValueError(f"{case.case_id}缺少步骤、判据或证据。")
        if not set(case.validation_modes) <= {"fake", "live"}:
            raise ValueError("未知验证模式。")


def save_plan(path: Path, plan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(plan, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    validate_plan()
    plan = {
        "plan_version": PLAN_VERSION,
        "project": "编程学习助手",
        "scope": "Day33-Day34既有毕业项目，不新增产品能力",
        "case_count": len(CASES),
        "execution_status": "not_run",
        "result_policy": {
            "passed": "已有证据满足判据。",
            "failed": "运行后不满足判据。",
            "blocked": "环境、网络或配置阻止验证，不算通过。",
            "not_run": "尚未执行，不算通过。",
        },
        "cases": [asdict(case) for case in CASES],
    }
    save_plan(PLAN_PATH, plan)
    print("Day35 第1步：固定毕业验收场景")
    print("计划版本：", PLAN_VERSION)
    for case in CASES:
        print(f"\n{case.case_id} {case.capability} | 验证模式：{', '.join(case.validation_modes)}")
        for check in case.checks:
            print("  判据：", check)
        print("  证据：", "；".join(case.evidence))
    print("\n场景数量：", len(CASES))
    print("执行状态：not_run；本步只固定计划，不宣称通过。")
    print("计划文件：", PLAN_PATH)
    print("下一步使用Fake组件自动验证，之后单独记录真实模型演示结果。")


if __name__ == "__main__":
    main()
