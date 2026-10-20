"""Day12 第一步：配置校验、环境变量和敏感信息脱敏。"""

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv


class ConfigError(ValueError):
    """应用配置不符合要求。"""


def mask_secret(secret: str, visible_prefix: int = 3) -> str:
    """只显示密钥前几位，避免日志泄露完整密钥。"""

    if not secret:
        return "<empty>"

    if len(secret) <= visible_prefix:
        return "*" * len(secret)

    return secret[:visible_prefix] + "***"


@dataclass(frozen=True)
class AppConfig:
    """模型应用运行所需的集中配置。"""

    model: str
    api_key: str
    base_url: str
    timeout: float
    max_retries: int
    environment: str

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None
    ) -> "AppConfig":
        """从环境变量创建配置；传入environ便于测试。"""

        values = environ if environ is not None else os.environ

        required_names = [
            "DEEPSEEK_MODEL",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL"
        ]

        missing_names = [
            name for name in required_names
            if not values.get(name, "").strip()
        ]

        if missing_names:
            raise ConfigError(
                "缺少必填环境变量：" + ", ".join(missing_names)
            )

        model = values["DEEPSEEK_MODEL"].strip()
        api_key = values["DEEPSEEK_API_KEY"].strip()
        base_url = values["DEEPSEEK_BASE_URL"].strip().rstrip("/")
        environment = values.get("APP_ENV", "development").strip()

        try:
            timeout = float(values.get("LLM_TIMEOUT", "20"))
            max_retries = int(values.get("LLM_MAX_RETRIES", "0"))
        except ValueError as error:
            raise ConfigError(
                "LLM_TIMEOUT 必须是数字，LLM_MAX_RETRIES必须是整数。"
            ) from error

        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(
                "DEEPSEEK_BASE_URL 必须以 http:// 或 https:// 开头。"
            )

        if timeout <= 0:
            raise ConfigError("LLM_TIMEOUT 必须大于0。")

        if max_retries < 0:
            raise ConfigError("LLM_MAX_RETRIES 不能是负数。")

        if environment not in {"development", "testing", "production"}:
            raise ConfigError(
                "APP_ENV 只能是 development、testing 或 production。"
            )

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            environment=environment
        )

    def safe_dict(self) -> dict[str, object]:
        """生成可以安全写入日志的配置，不包含完整API密钥。"""

        return {
            "model": self.model,
            "api_key": mask_secret(self.api_key),
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "environment": self.environment
        }


# ============================================================
# 1. 读取项目 .env 并显示安全配置
# ============================================================

load_dotenv()

print("读取项目环境配置……")

try:
    config = AppConfig.from_environment()
except ConfigError as error:
    print("配置错误：", error)
    raise SystemExit(1) from error

print("校验通过。用于日志的安全配置：")
print(config.safe_dict())
print("注意：日志中不会显示完整 API Key。")


# ============================================================
# 2. 使用隔离环境演示错误配置
# ============================================================

print("\n" + "=" * 60)
print("错误配置演示")

invalid_environment = {
    "DEEPSEEK_MODEL": "demo-model",
    # 故意不提供DEEPSEEK_API_KEY。
    "DEEPSEEK_BASE_URL": "https://example.com/v1",
    "LLM_TIMEOUT": "20",
    "LLM_MAX_RETRIES": "0",
    "APP_ENV": "testing"
}

try:
    AppConfig.from_environment(invalid_environment)
except ConfigError as error:
    print("正确捕获配置错误：", error)


# ============================================================
# 3. 脱敏示例
# ============================================================

print("\n" + "=" * 60)
print("密钥脱敏示例")

for secret in [
    "sk-1234567890abcdef",
    "abc",
    ""
]:
    print(repr(secret), "→", mask_secret(secret))


# ============================================================
# 数据流
# ============================================================
#
# .env / 系统环境变量
#          ↓ load_dotenv()
#     os.environ
#          ↓ AppConfig.from_environment()
#     必填项检查 + 类型转换 + 范围检查
#          ├── 失败 → ConfigError，程序不启动
#          └── 成功 → AppConfig
#                         ├── 创建ChatOpenAI
#                         └── safe_dict()写入日志
#
# API Key只用于真正创建模型客户端，不能直接print到日志。
