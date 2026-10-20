"""Day29 第 5 步：遥测脱敏、属性限制与数据保留。

运行：
    python day29/python/day29_telemetry_governance.py

数据库：
    day29/database/day29_telemetry_governance.db
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DAY29_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = DAY29_DIR / "database" / "day29_telemetry_governance.db"

MAX_ATTRIBUTES = 12
MAX_STRING_LENGTH = 100
MAX_LIST_ITEMS = 5
MAX_EVENTS = 4
RETENTION_DAYS = 30
REDACTED = "[REDACTED]"

# 按完整键段匹配，不能用宽泛的 "token" 子串；否则 input_tokens
# 这类安全的计数指标也会被误判。
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:^|[._-])"
    r"(?:password|passwd|secret|api[_-]?key|authorization|cookie|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|token)"
    r"(?:$|[._-])",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE)

SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
}


@dataclass(frozen=True)
class TelemetryLimits:
    max_attributes: int = MAX_ATTRIBUTES
    max_string_length: int = MAX_STRING_LENGTH
    max_list_items: int = MAX_LIST_ITEMS
    max_events: int = MAX_EVENTS


@dataclass
class GovernanceStats:
    redacted_fields: int = 0
    redacted_values: int = 0
    truncated_strings: int = 0
    truncated_lists: int = 0
    dropped_attributes: int = 0
    dropped_events: int = 0


@dataclass(frozen=True)
class RawSpan:
    span_id: str
    trace_id: str
    name: str
    ended_at: str
    attributes: dict[str, Any]
    events: list[dict[str, Any]]


@dataclass(frozen=True)
class GovernedSpan:
    span_id: str
    trace_id: str
    name: str
    ended_at: str
    attributes: dict[str, Any]
    events: list[dict[str, Any]]
    governance: GovernanceStats


def pseudonymize(value: str, namespace: str = "day29-course") -> str:
    """生成稳定假名；它仍属于可关联数据，不等于完全匿名。"""

    digest = hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()
    return f"user-hash-{digest[:16]}"


def sanitize_url(raw_url: str, stats: GovernanceStats) -> str:
    """保留URL结构，但清除敏感查询参数值。"""

    try:
        parts = urlsplit(raw_url)
    except ValueError:
        stats.redacted_values += 1
        return REDACTED

    clean_query: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in SENSITIVE_QUERY_KEYS:
            clean_query.append((key, REDACTED))
            stats.redacted_fields += 1
        else:
            clean_query.append((key, value))

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(clean_query),
            "",  # 不保存fragment。
        )
    )


def sanitize_string(
    value: str,
    stats: GovernanceStats,
    limits: TelemetryLimits,
) -> str:
    cleaned, bearer_count = BEARER_PATTERN.subn(REDACTED, value)
    cleaned, email_count = EMAIL_PATTERN.subn("[EMAIL]", cleaned)
    stats.redacted_values += bearer_count + email_count

    if len(cleaned) > limits.max_string_length:
        stats.truncated_strings += 1
        cleaned = cleaned[: limits.max_string_length] + "…[TRUNCATED]"
    return cleaned


def sanitize_value(
    key: str,
    value: Any,
    stats: GovernanceStats,
    limits: TelemetryLimits,
) -> Any:
    if SENSITIVE_KEY_PATTERN.search(key):
        stats.redacted_fields += 1
        return REDACTED

    if key in {"url.full", "http.url"} and isinstance(value, str):
        value = sanitize_url(value, stats)

    if isinstance(value, str):
        return sanitize_string(value, stats, limits)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) > limits.max_list_items:
            stats.truncated_lists += 1
            items = items[: limits.max_list_items]
        return [
            sanitize_value(f"{key}[]", item, stats, limits)
            for item in items
        ]

    # 复杂任意对象不直接进入遥测；先转字符串再受长度和内容规则限制。
    return sanitize_string(str(value), stats, limits)


def sanitize_attributes(
    attributes: dict[str, Any],
    stats: GovernanceStats,
    limits: TelemetryLimits,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if len(sanitized) >= limits.max_attributes:
            stats.dropped_attributes += 1
            continue
        sanitized[key] = sanitize_value(key, value, stats, limits)
    return sanitized


def sanitize_events(
    events: list[dict[str, Any]],
    stats: GovernanceStats,
    limits: TelemetryLimits,
) -> list[dict[str, Any]]:
    kept = events[: limits.max_events]
    stats.dropped_events += max(0, len(events) - len(kept))

    result: list[dict[str, Any]] = []
    for event in kept:
        result.append(
            {
                "name": sanitize_string(
                    str(event.get("name", "event")), stats, limits
                ),
                "timestamp": event.get("timestamp"),
                "attributes": sanitize_attributes(
                    dict(event.get("attributes") or {}),
                    stats,
                    limits,
                ),
            }
        )
    return result


def govern_span(
    raw: RawSpan,
    limits: TelemetryLimits = TelemetryLimits(),
) -> GovernedSpan:
    stats = GovernanceStats()
    attributes = dict(raw.attributes)

    # user.id需要跨请求聚合，因此使用稳定假名，而不是保存原始用户ID。
    if "user.id" in attributes:
        attributes["user.id"] = pseudonymize(str(attributes["user.id"]))

    return GovernedSpan(
        span_id=raw.span_id,
        trace_id=raw.trace_id,
        name=sanitize_string(raw.name, stats, limits),
        ended_at=raw.ended_at,
        attributes=sanitize_attributes(attributes, stats, limits),
        events=sanitize_events(raw.events, stats, limits),
        governance=stats,
    )


def open_database() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_schema() -> None:
    with open_database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS governed_spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                events_json TEXT NOT NULL,
                governance_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_governed_spans_ended_at
            ON governed_spans(ended_at);
            """
        )


def save_span(span: GovernedSpan) -> None:
    """数据库只接收已经通过治理的 Span 类型。"""

    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO governed_spans (
                span_id, trace_id, name, ended_at,
                attributes_json, events_json, governance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(span_id) DO UPDATE SET
                trace_id = excluded.trace_id,
                name = excluded.name,
                ended_at = excluded.ended_at,
                attributes_json = excluded.attributes_json,
                events_json = excluded.events_json,
                governance_json = excluded.governance_json
            """,
            (
                span.span_id,
                span.trace_id,
                span.name,
                span.ended_at,
                json.dumps(span.attributes, ensure_ascii=False),
                json.dumps(span.events, ensure_ascii=False),
                json.dumps(asdict(span.governance), ensure_ascii=False),
            ),
        )


def delete_expired_spans(now: datetime, retention_days: int) -> tuple[int, str]:
    if retention_days <= 0:
        raise ValueError("retention_days必须大于0。")
    cutoff = now - timedelta(days=retention_days)
    cutoff_text = cutoff.isoformat(timespec="seconds")
    with open_database() as connection:
        cursor = connection.execute(
            "DELETE FROM governed_spans WHERE ended_at < ?",
            (cutoff_text,),
        )
        return cursor.rowcount, cutoff_text


def load_saved_spans() -> list[dict[str, Any]]:
    with open_database() as connection:
        rows = connection.execute(
            "SELECT * FROM governed_spans ORDER BY ended_at"
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["attributes"] = json.loads(item.pop("attributes_json"))
        item["events"] = json.loads(item.pop("events_json"))
        item["governance"] = json.loads(item.pop("governance_json"))
        result.append(item)
    return result


def make_sensitive_span(now: datetime) -> RawSpan:
    attributes: dict[str, Any] = {
        "http.request.method": "POST",
        "url.full": (
            "https://rag.example/ask?topic=rag&access_token=top-secret-token"
        ),
        "user.id": "user-xiaolin@example.com",
        "user.email": "xiaolin@example.com",
        "authorization": "Bearer REDACTED_EXAMPLE_TOKEN",
        "api_key": "sk-course-secret-value",
        "request.password": "123456",
        "rag.question": (
            "用户 xiaolin@example.com 想知道什么是RAG。"
            + "这是一段故意构造的超长问题正文。" * 8
        ),
        "rag.source_ids": [f"chunk-{index}" for index in range(1, 10)],
        "http.response.status_code": 200,
        "rag.input_tokens": 420,
        "rag.output_tokens": 96,
        "extra.attribute.1": "将被保留或受数量限制",
        "extra.attribute.2": "超过最大属性数时会被丢弃",
    }
    events = [
        {
            "name": f"pipeline.event.{index}",
            "timestamp": now.isoformat(timespec="seconds"),
            "attributes": {
                "sequence": index,
                "message": f"联系邮箱 xiaolin@example.com，Bearer secret-{index}",
            },
        }
        for index in range(1, 7)
    ]
    return RawSpan(
        span_id="span-sensitive-demo",
        trace_id="trace-sensitive-demo",
        name="POST /ask/user-xiaolin@example.com",
        ended_at=now.isoformat(timespec="seconds"),
        attributes=attributes,
        events=events,
    )


def seed_retention_examples(now: datetime) -> None:
    examples = [
        ("span-40-days-old", 40),
        ("span-10-days-old", 10),
    ]
    for span_id, age_days in examples:
        raw = RawSpan(
            span_id=span_id,
            trace_id=f"trace-{age_days}-days-old",
            name="retention.demo",
            ended_at=(now - timedelta(days=age_days)).isoformat(timespec="seconds"),
            attributes={"demo.age_days": age_days},
            events=[],
        )
        save_span(govern_span(raw))


def main() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    create_schema()

    raw = make_sensitive_span(now)
    governed = govern_span(raw)
    save_span(governed)
    seed_retention_examples(now)

    print("Day29 遥测脱敏、属性限制与数据保留")
    print("数据库：", DATABASE_PATH)
    print("=" * 100)
    print("原始敏感属性：")
    print(json.dumps(raw.attributes, ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    print("治理后允许写入数据库的 Span：")
    print(json.dumps(asdict(governed), ensure_ascii=False, indent=2))

    print("\n治理统计：")
    print(json.dumps(asdict(governed.governance), ensure_ascii=False, indent=2))
    print("原始属性数量：", len(raw.attributes))
    print("保存属性数量：", len(governed.attributes))
    print("原始事件数量：", len(raw.events))
    print("保存事件数量：", len(governed.events))

    before = load_saved_spans()
    deleted, cutoff = delete_expired_spans(now, RETENTION_DAYS)
    after = load_saved_spans()

    print("\n" + "=" * 100)
    print("30天数据保留策略：")
    print("清理截止时间：", cutoff)
    print("清理前Span：", [item["span_id"] for item in before])
    print("删除数量：", deleted)
    print("清理后Span：", [item["span_id"] for item in after])
    print("40天旧数据是否删除：", "span-40-days-old" not in {item["span_id"] for item in after})
    print("10天数据是否保留：", "span-10-days-old" in {item["span_id"] for item in after})

    print("\n" + "=" * 100)
    print("本步数据流：")
    print(
        "原始Span\n"
        "→ 敏感键整字段REDACTED\n"
        "→ URL敏感查询参数REDACTED\n"
        "→ 邮箱和Bearer模式脱敏\n"
        "→ user.id稳定假名化\n"
        "→ 字符串、列表、属性、事件数量限制\n"
        "→ GovernedSpan才允许写入SQLite\n"
        "→ 定时按ended_at删除超过保留期的数据"
    )


if __name__ == "__main__":
    main()
