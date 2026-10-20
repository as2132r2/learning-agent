"""Day12 第二步：结构化日志中的敏感信息递归脱敏。"""

import json
import logging
import re
from copy import deepcopy
from typing import Any


# ============================================================
# 1. 敏感字段和脱敏函数
# ============================================================

SECRET_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token"
}

SECRET_VALUE_PATTERNS = [
    # OpenAI/DeepSeek风格的sk-开头密钥。
    re.compile(r"sk-[A-Za-z0-9_-]+"),

    # Bearer Token。
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE)
]


def mask_value(value: Any) -> str:
    """隐藏一个敏感值，只保留极少量前缀。"""

    text = str(value)

    if len(text) <= 3:
        return "***"

    return text[:3] + "***"


def redact_sensitive_data(
    data: Any,
    current_key: str | None = None
) -> Any:
    """递归处理dict、list、tuple和普通文本中的敏感信息。"""

    if current_key and current_key.lower() in SECRET_FIELD_NAMES:
        return mask_value(data)

    if isinstance(data, dict):
        return {
            key: redact_sensitive_data(value, key)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]

    if isinstance(data, tuple):
        return tuple(redact_sensitive_data(item) for item in data)

    if isinstance(data, str):
        redacted = data

        for pattern in SECRET_VALUE_PATTERNS:
            redacted = pattern.sub(
                lambda match: mask_value(match.group(0)),
                redacted
            )

        return redacted

    return data


# ============================================================
# 2. 结构化日志格式器和脱敏过滤器
# ============================================================

class JsonFormatter(logging.Formatter):
    """把日志记录转换成一行JSON，方便后续采集和检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "data": getattr(record, "data", {})
        }

        return json.dumps(
            redact_sensitive_data(payload),
            ensure_ascii=False
        )


class SensitiveDataFilter(logging.Filter):
    """在日志进入Formatter前递归处理额外字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "data"):
            record.data = redact_sensitive_data(
                deepcopy(record.data)
            )

        return True


def build_logger() -> logging.Logger:
    logger = logging.getLogger("day12.security")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SensitiveDataFilter())
    logger.addHandler(handler)

    return logger


# ============================================================
# 3. 演示普通字段、嵌套字段和文本中的密钥
# ============================================================

logger = build_logger()

logger.info(
    "模型请求开始",
    extra={
        "request_id": "req-001",
        "data": {
            "model": "deepseek-chat",
            "api_key": "sk-live-1234567890",
            "timeout": 20,
            "user": {
                "name": "小林",
                "password": "my-password"
            }
        }
    }
)

logger.info(
    "工具调用参数已记录",
    extra={
        "request_id": "req-001",
        "data": {
            "tool": "search_knowledge",
            "args": {
                "keyword": "RAG",
                "authorization": "Bearer abc.def.ghi"
            }
        }
    }
)

logger.warning(
    "文本中出现疑似密钥：sk-inline-secret-999"
)


# ============================================================
# 4. 验证原始数据没有被脱敏函数修改
# ============================================================

original = {
    "api_key": "sk-original-secret",
    "nested": {
        "token": "token-value"
    }
}

safe_copy = redact_sensitive_data(original)

print("\n原始数据：", original)
print("安全副本：", safe_copy)
print("原始数据保持不变：", original["api_key"] == "sk-original-secret")


# ============================================================
# 数据流
# ============================================================
#
# logger.info(..., data={...})
#          ↓
# SensitiveDataFilter
#          ↓ 递归处理字段名和文本模式
# JsonFormatter
#          ↓
# 一行JSON日志
#
# 敏感字段：按字段名替换
# 文本密钥：按正则表达式替换
# 原始业务数据：deepcopy后处理，不直接修改原对象
