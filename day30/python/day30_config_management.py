"""Day30 第 1 步：配置分层、类型校验与密钥隔离。

运行：
    python day30/python/day30_config_management.py

命令行覆盖示例：
    python day30/python/day30_config_management.py \
        --environment staging --port 8060 --max-concurrency 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY30_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = DAY30_DIR / "config" / "app.json"
DOTENV_PATH = PROJECT_ROOT / ".env"


DEFAULTS: dict[str, Any] = {
    "app_name": "rag-service",
    "environment": "development",
    "host": "127.0.0.1",
    "port": 8000,
    "log_level": "INFO",
    "database_path": "day30/database/app.db",
    "request_timeout_seconds": 30.0,
    "max_concurrency": 4,
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "api_key": None,
}

# 外部变量名 → 程序内部统一字段名。
ENV_MAPPING = {
    "APP_ENV": "environment",
    "APP_HOST": "host",
    "APP_PORT": "port",
    "APP_LOG_LEVEL": "log_level",
    "APP_DATABASE_PATH": "database_path",
    "APP_REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
    "APP_MAX_CONCURRENCY": "max_concurrency",
    "DEEPSEEK_MODEL": "model",
    "DEEPSEEK_BASE_URL": "base_url",
    "DEEPSEEK_API_KEY": "api_key",
}


class ConfigurationError(ValueError):
    """配置缺失、类型错误或取值不安全。"""


class SecretValue:
    """默认显示为遮盖值，只有明确调用 reveal() 才能读取原文。"""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not value.strip():
            raise ConfigurationError("密钥不能为空字符串。")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def fingerprint(self) -> str:
        """用于比较密钥版本，不暴露密钥原文。"""

        return hashlib.sha256(self._value.encode()).hexdigest()[:10]

    def __str__(self) -> str:
        return "********"

    def __repr__(self) -> str:
        return "SecretValue('********')"


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    environment: str
    host: str
    port: int
    log_level: str
    database_path: Path
    request_timeout_seconds: float
    max_concurrency: int
    model: str
    base_url: str
    api_key: SecretValue

    def safe_dict(self) -> dict[str, Any]:
        """只返回允许写日志和健康接口的安全配置。"""

        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level,
            "database_path": str(self.database_path),
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_concurrency": self.max_concurrency,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "********",
            "api_key_fingerprint": self.api_key.fingerprint(),
        }


@dataclass(frozen=True)
class LoadedConfig:
    config: AppConfig
    provenance: dict[str, str]


def read_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"配置文件JSON无效：{path}，第{error.lineno}行。"
        ) from error
    if not isinstance(data, dict):
        raise ConfigurationError("配置文件顶层必须是JSON对象。")

    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        raise ConfigurationError(f"配置文件包含未知字段：{unknown}")
    if "api_key" in data:
        raise ConfigurationError("禁止把api_key写入app.json，请使用环境变量。")
    return data


def mapped_environment(source: dict[str, Any]) -> dict[str, Any]:
    """只读取白名单变量，不把整个系统环境混入配置。"""

    result: dict[str, Any] = {}
    for external_name, internal_name in ENV_MAPPING.items():
        value = source.get(external_name)
        if value is not None and str(value).strip() != "":
            result[internal_name] = value
    return result


def apply_layer(
    merged: dict[str, Any],
    provenance: dict[str, str],
    values: dict[str, Any],
    source_name: str,
) -> None:
    for key, value in values.items():
        if key not in DEFAULTS:
            continue
        merged[key] = value
        provenance[key] = source_name


def parse_int(name: str, value: Any) -> int:
    try:
        # bool 是 int 的子类，但不应该被当成端口或并发数。
        if isinstance(value, bool):
            raise ValueError
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name}必须是整数，当前值={value!r}") from error


def parse_float(name: str, value: Any) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name}必须是数字，当前值={value!r}") from error


def build_config(values: dict[str, Any]) -> AppConfig:
    environment = str(values["environment"]).lower()
    if environment not in {"development", "testing", "staging", "production"}:
        raise ConfigurationError(
            "environment必须是development/testing/staging/production。"
        )

    port = parse_int("port", values["port"])
    if not 1 <= port <= 65535:
        raise ConfigurationError("port必须在1到65535之间。")

    max_concurrency = parse_int("max_concurrency", values["max_concurrency"])
    if not 1 <= max_concurrency <= 1000:
        raise ConfigurationError("max_concurrency必须在1到1000之间。")

    timeout = parse_float(
        "request_timeout_seconds", values["request_timeout_seconds"]
    )
    if not 0.1 <= timeout <= 600:
        raise ConfigurationError("request_timeout_seconds必须在0.1到600之间。")

    log_level = str(values["log_level"]).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("log_level取值无效。")

    api_key = values.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ConfigurationError(
            "缺少DEEPSEEK_API_KEY。请在项目.env或系统环境变量中配置。"
        )

    database_path = Path(str(values["database_path"]))
    if not database_path.is_absolute():
        database_path = (PROJECT_ROOT / database_path).resolve()

    base_url = str(values["base_url"]).rstrip("/")
    if not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise ConfigurationError(
            "base_url必须使用HTTPS；只有本机地址允许HTTP。"
        )

    return AppConfig(
        app_name=str(values["app_name"]),
        environment=environment,
        host=str(values["host"]),
        port=port,
        log_level=log_level,
        database_path=database_path,
        request_timeout_seconds=timeout,
        max_concurrency=max_concurrency,
        model=str(values["model"]),
        base_url=base_url,
        api_key=SecretValue(api_key),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day30配置管理练习")
    parser.add_argument("--environment")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level")
    parser.add_argument("--request-timeout-seconds", type=float)
    parser.add_argument("--max-concurrency", type=int)
    return parser.parse_args()


def command_line_values(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "environment": arguments.environment,
            "host": arguments.host,
            "port": arguments.port,
            "log_level": arguments.log_level,
            "request_timeout_seconds": arguments.request_timeout_seconds,
            "max_concurrency": arguments.max_concurrency,
        }.items()
        if value is not None
    }


def load_config(arguments: argparse.Namespace) -> LoadedConfig:
    merged = dict(DEFAULTS)
    provenance = {key: "default" for key in DEFAULTS}

    apply_layer(
        merged, provenance, read_json_config(CONFIG_PATH), f"json:{CONFIG_PATH.name}"
    )
    apply_layer(
        merged,
        provenance,
        mapped_environment(dict(dotenv_values(DOTENV_PATH))),
        f"dotenv:{DOTENV_PATH.name}",
    )
    apply_layer(
        merged,
        provenance,
        mapped_environment(dict(os.environ)),
        "system_environment",
    )
    apply_layer(
        merged,
        provenance,
        command_line_values(arguments),
        "command_line",
    )
    return LoadedConfig(build_config(merged), provenance)


def main() -> None:
    try:
        loaded = load_config(parse_arguments())
    except ConfigurationError as error:
        print("配置加载失败：", error)
        raise SystemExit(2) from error

    print("Day30 配置分层、类型校验与密钥隔离")
    print("配置文件：", CONFIG_PATH)
    print(".env文件：", DOTENV_PATH)
    print("=" * 100)
    print("最终安全配置（可以写日志）：")
    print(json.dumps(loaded.config.safe_dict(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    print("每个字段最终来自哪一层：")
    for key in DEFAULTS:
        print(f"{key:<28} ← {loaded.provenance[key]}")

    print("\n" + "=" * 100)
    print("SecretValue显示行为：")
    print("str(api_key)：", str(loaded.config.api_key))
    print("repr(api_key)：", repr(loaded.config.api_key))
    print("安全指纹：", loaded.config.api_key.fingerprint())
    print("真实密钥只应在创建模型客户端时显式调用 reveal()，本程序不打印它。")

    print("\n本步数据流：")
    print(
        "代码默认值\n"
        "→ app.json覆盖非敏感配置\n"
        "→ 项目.env覆盖本地配置和密钥\n"
        "→ 系统环境变量覆盖部署配置\n"
        "→ 命令行覆盖本次启动参数\n"
        "→ 类型转换 + 范围校验 + 安全规则\n"
        "→ AppConfig不可变配置对象\n"
        "→ safe_dict用于日志，SecretValue仅显式reveal给模型客户端"
    )


if __name__ == "__main__":
    main()
