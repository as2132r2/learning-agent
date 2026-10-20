"""Day26 第 2 步：版本配置、提示词快照与实验冻结。

运行：
    python day26/python/day26_variant_config.py

本步复用第 1 步的实验和稳定分流结果。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from day26_stable_assignment import (
    EXPERIMENT,
    Experiment,
    assign_user,
    create_schema,
    open_database,
    save_experiment,
)


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """一个实验版本的完整可复现配置。"""

    variant: str
    prompt_version: str
    system_prompt: str
    model_name: str
    temperature: float
    max_evidence_count: int
    reranker_threshold: int


class ConfigFrozenError(RuntimeError):
    """尝试修改已冻结配置时抛出。"""


VARIANT_CONFIGS = (
    VariantConfig(
        variant="control",
        prompt_version="rag-answer-control-v1",
        system_prompt=(
            "你是RAG问答助手。只能依据参考资料回答；"
            "资料不足时明确说明无法回答；重要结论标注[资料N]。"
        ),
        model_name="deepseek-chat",
        temperature=0.0,
        max_evidence_count=3,
        reranker_threshold=95,
    ),
    VariantConfig(
        variant="candidate",
        prompt_version="rag-answer-candidate-v2",
        system_prompt=(
            "你是严谨、简洁的RAG问答助手。只能依据参考资料回答；"
            "先直接回答核心问题，再补充必要解释；每条可验证结论必须标注[资料N]；"
            "资料不足时明确拒绝，不得用模型常识补全。"
        ),
        model_name="deepseek-chat",
        temperature=0.0,
        max_evidence_count=3,
        reranker_threshold=95,
    ),
)


def create_config_schema(connection) -> None:
    """创建不可变版本配置快照表。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiment_variant_configs (
            experiment_id       TEXT NOT NULL,
            allocation_version  TEXT NOT NULL,
            variant             TEXT NOT NULL,
            prompt_version      TEXT NOT NULL,
            config_json         TEXT NOT NULL,
            config_hash         TEXT NOT NULL,
            frozen_at           TEXT NOT NULL,
            PRIMARY KEY (experiment_id, allocation_version, variant),
            FOREIGN KEY (experiment_id)
                REFERENCES experiments(experiment_id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_variant_config_hash
        ON experiment_variant_configs(
            experiment_id, allocation_version, config_hash
        );
        """
    )
    connection.commit()


def canonical_config(config: VariantConfig) -> str:
    """生成键顺序和空格都固定的规范 JSON。"""

    return json.dumps(
        asdict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def config_hash(config: VariantConfig) -> str:
    """配置任何字段变化都会产生不同 SHA-256。"""

    canonical = canonical_config(config)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_configs(
    experiment: Experiment,
    configs: tuple[VariantConfig, ...],
) -> None:
    """确保实验分流版本与运行配置一一对应。"""

    expected = {variant.name for variant in experiment.variants}
    actual = {config.variant for config in configs}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"版本配置与实验定义不一致；缺少={missing}，多余={unexpected}。"
        )

    prompt_versions = [config.prompt_version for config in configs]
    if len(prompt_versions) != len(set(prompt_versions)):
        raise ValueError("每个实验版本必须使用不同的 prompt_version。")

    for config in configs:
        if not 0 <= config.temperature <= 2:
            raise ValueError(f"{config.variant} 的 temperature 超出0～2。")
        if config.max_evidence_count <= 0:
            raise ValueError(f"{config.variant} 的证据数量必须大于0。")
        if not 0 <= config.reranker_threshold <= 100:
            raise ValueError(f"{config.variant} 的阈值必须在0～100。")


def freeze_variant_configs(
    connection,
    experiment: Experiment,
    configs: tuple[VariantConfig, ...],
) -> dict[str, str]:
    """首次写入配置；以后只允许完全相同的配置再次注册。"""

    validate_configs(experiment, configs)
    frozen_at = datetime.now().astimezone().isoformat(timespec="seconds")
    hashes: dict[str, str] = {}

    with connection:
        for config in configs:
            canonical = canonical_config(config)
            digest = config_hash(config)
            hashes[config.variant] = digest

            old = connection.execute(
                """
                SELECT config_hash
                FROM experiment_variant_configs
                WHERE experiment_id = ?
                  AND allocation_version = ?
                  AND variant = ?
                """,
                (
                    experiment.experiment_id,
                    experiment.allocation_version,
                    config.variant,
                ),
            ).fetchone()

            if old is not None:
                if old["config_hash"] != digest:
                    raise ConfigFrozenError(
                        f"{config.variant} 在 {experiment.allocation_version} "
                        "中的配置已经冻结；请创建新的 allocation_version。"
                    )
                continue

            connection.execute(
                """
                INSERT INTO experiment_variant_configs (
                    experiment_id, allocation_version, variant,
                    prompt_version, config_json, config_hash, frozen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.allocation_version,
                    config.variant,
                    config.prompt_version,
                    canonical,
                    digest,
                    frozen_at,
                ),
            )
    return hashes


def load_frozen_config(
    connection,
    experiment_id: str,
    allocation_version: str,
    variant: str,
) -> tuple[VariantConfig, str]:
    """读取配置快照，并重新计算哈希防止数据库内容意外损坏。"""

    row = connection.execute(
        """
        SELECT config_json, config_hash
        FROM experiment_variant_configs
        WHERE experiment_id = ?
          AND allocation_version = ?
          AND variant = ?
        """,
        (experiment_id, allocation_version, variant),
    ).fetchone()
    if row is None:
        raise LookupError(f"没有找到版本 {variant} 的冻结配置。")

    raw: dict[str, Any] = json.loads(row["config_json"])
    config = VariantConfig(**raw)
    actual_hash = config_hash(config)
    if actual_hash != row["config_hash"]:
        raise RuntimeError("配置内容与保存的哈希不一致，可能已损坏或被修改。")
    return config, actual_hash


def resolve_user_config(connection, user_id: str) -> tuple[str, VariantConfig, str]:
    """稳定分组后，加载该用户真正应执行的冻结配置。"""

    assignment = assign_user(connection, EXPERIMENT, user_id)
    config, digest = load_frozen_config(
        connection,
        assignment.experiment_id,
        assignment.allocation_version,
        assignment.variant,
    )
    return assignment.variant, config, digest


def main() -> None:
    print("Day26 版本配置、提示词快照与实验冻结")
    print("实验：", EXPERIMENT.experiment_id)
    print("分配版本：", EXPERIMENT.allocation_version)
    print("=" * 96)

    with open_database() as connection:
        create_schema(connection)
        create_config_schema(connection)
        save_experiment(connection, EXPERIMENT)

        hashes = freeze_variant_configs(connection, EXPERIMENT, VARIANT_CONFIGS)
        print("冻结配置完成：")
        for config in VARIANT_CONFIGS:
            print(f"\n版本：{config.variant}")
            print("prompt_version：", config.prompt_version)
            print("model：", config.model_name)
            print("temperature：", config.temperature)
            print("config_hash：", hashes[config.variant])
            print("system_prompt：", config.system_prompt)

        print("\n" + "=" * 96)
        print("根据稳定分组解析真实配置：")
        for user_id in ("user-xiaolin", "user-leo"):
            variant, config, digest = resolve_user_config(connection, user_id)
            print(
                f"- {user_id} → {variant} → {config.prompt_version} "
                f"→ hash={digest[:12]}"
            )

        print("\n" + "=" * 96)
        print("模拟在相同 allocation_version 中偷偷修改 candidate 提示词：")
        modified = tuple(
            VariantConfig(
                variant=config.variant,
                prompt_version=config.prompt_version,
                system_prompt=(
                    config.system_prompt + " 回答必须控制在100字以内。"
                    if config.variant == "candidate"
                    else config.system_prompt
                ),
                model_name=config.model_name,
                temperature=config.temperature,
                max_evidence_count=config.max_evidence_count,
                reranker_threshold=config.reranker_threshold,
            )
            for config in VARIANT_CONFIGS
        )
        try:
            freeze_variant_configs(connection, EXPERIMENT, modified)
        except ConfigFrozenError as error:
            print("修改被拒绝：", error)
        else:
            raise AssertionError("已冻结配置不应该允许修改。")

    print("\n" + "=" * 96)
    print("本步数据流：")
    print("VariantConfig → 规范JSON → SHA-256 config_hash")
    print("→ experiment_variant_configs不可变快照")
    print("用户Assignment.variant → load_frozen_config() → 真实模型与Prompt参数")
    print("配置要变化 → 新建allocation_version，而不是覆盖旧快照")


if __name__ == "__main__":
    main()
