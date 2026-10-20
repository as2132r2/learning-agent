"""Day12 第五步：把工具授权决定写入SQLite审计日志。"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


Role = Literal["user", "operator", "admin"]


class PermissionDeniedError(PermissionError):
    pass


class ToolInputError(ValueError):
    pass


@dataclass(frozen=True)
class UserIdentity:
    user_id: str
    role: Role


@dataclass(frozen=True)
class ToolRequest:
    request_id: str
    tool_name: str
    arguments: dict


@dataclass(frozen=True)
class AuditEvent:
    request_id: str
    created_at: str
    user_id: str
    role: str
    tool_name: str
    arguments_summary: str
    decision: str
    reason: str


class AuditStore:
    """将每次授权决定保存到SQLite。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS permission_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_summary TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
            """)

    def write(self, event: AuditEvent) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO permission_audit_logs (
                    request_id,
                    created_at,
                    user_id,
                    role,
                    tool_name,
                    arguments_summary,
                    decision,
                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.request_id,
                    event.created_at,
                    event.user_id,
                    event.role,
                    event.tool_name,
                    event.arguments_summary,
                    event.decision,
                    event.reason
                )
            )

    def read_all(self) -> list[tuple]:
        with sqlite3.connect(self.database_path) as connection:
            return connection.execute(
                """
                SELECT request_id, created_at, user_id, role,
                       tool_name, arguments_summary, decision, reason
                FROM permission_audit_logs
                ORDER BY id
                """
            ).fetchall()


class PermissionPolicy:
    allowed_tools: dict[Role, set[str]] = {
        "user": {"check_balance"},
        "operator": {"check_balance", "request_transfer"},
        "admin": {
            "check_balance",
            "request_transfer",
            "approve_transfer"
        }
    }

    def check(
        self,
        identity: UserIdentity,
        request: ToolRequest
    ) -> None:
        if request.tool_name not in self.allowed_tools[identity.role]:
            raise PermissionDeniedError(
                f"角色{identity.role}无权使用{request.tool_name}"
            )

        if request.tool_name in {
            "request_transfer",
            "approve_transfer"
        }:
            amount = request.arguments.get("amount")

            if not isinstance(amount, (int, float)) or isinstance(amount, bool):
                raise ToolInputError("amount必须是数字")

            if amount <= 0 or amount > 5_000:
                raise ToolInputError("amount必须大于0且不超过5000")


def summarize_arguments(arguments: dict) -> str:
    """审计日志只保存参数摘要，不保存可能敏感的全部数据。"""

    safe_arguments = {
        key: "<redacted>"
        if key in {"password", "token", "api_key"}
        else value
        for key, value in arguments.items()
    }

    return json.dumps(
        safe_arguments,
        ensure_ascii=False,
        sort_keys=True
    )


def authorize_and_audit(
    identity: UserIdentity,
    request: ToolRequest,
    policy: PermissionPolicy,
    audit_store: AuditStore
) -> str:
    try:
        policy.check(identity, request)
    except (PermissionDeniedError, ToolInputError) as error:
        decision = "denied"
        reason = str(error)
        result = f"拒绝：{reason}"
    else:
        decision = "approved"
        reason = "角色和参数检查均通过"
        result = f"允许执行工具：{request.tool_name}"

    audit_store.write(AuditEvent(
        request_id=request.request_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        user_id=identity.user_id,
        role=identity.role,
        tool_name=request.tool_name,
        arguments_summary=summarize_arguments(request.arguments),
        decision=decision,
        reason=reason
    ))

    return result


project_directory = Path(__file__).resolve().parent.parent.parent
audit_database = (
    project_directory
    / "day12"
    / "database"
    / "day12_permission_audit.db"
)

audit_store = AuditStore(audit_database)
policy = PermissionPolicy()

cases = [
    (
        UserIdentity("xiaolin", "user"),
        ToolRequest("audit-001", "check_balance", {"account": "account-001"})
    ),
    (
        UserIdentity("xiaolin", "user"),
        ToolRequest(
            "audit-002",
            "request_transfer",
            {"to_account": "account-002", "amount": 100}
        )
    ),
    (
        UserIdentity("operator-001", "operator"),
        ToolRequest(
            "audit-003",
            "request_transfer",
            {"to_account": "account-002", "amount": 100}
        )
    ),
    (
        UserIdentity("admin-001", "admin"),
        ToolRequest(
            "audit-004",
            "approve_transfer",
            {"to_account": "account-002", "amount": 6_000}
        )
    )
]

print("审计数据库：", audit_database)

for identity, request in cases:
    print(
        request.request_id,
        "→",
        authorize_and_audit(identity, request, policy, audit_store)
    )

print("\n当前审计记录：")

for row in audit_store.read_all():
    print(dict(zip(
        [
            "request_id",
            "created_at",
            "user_id",
            "role",
            "tool_name",
            "arguments_summary",
            "decision",
            "reason"
        ],
        row
    )))


# ============================================================
# 数据流
# ============================================================
#
# ToolRequest
#      ↓
# PermissionPolicy.check()
#      ├── denied → 记录denied和原因
#      └── approved → 记录approved和原因
#                         ↓
#                 AuditStore.write()
#                         ↓
#                 SQLite审计表
#
# 审计记录回答：谁、何时、想调用什么、参数摘要、结果和原因。
