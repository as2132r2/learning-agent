"""Day12 第十步：整合输入校验、权限、配额、幂等和审计。"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


Role = Literal["user", "operator", "admin"]


class SecureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_id: str = Field(
        min_length=4,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$"
    )
    idempotency_key: str = Field(
        min_length=4,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$"
    )
    tool_name: Literal["check_balance", "request_transfer"]
    arguments: dict[str, Any]


class Identity(BaseModel):
    user_id: str
    role: Role


class GatewayError(Exception):
    pass


class AuditStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS secure_gateway_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
            """)

    def write(
        self,
        request: SecureRequest,
        identity: Identity,
        decision: str,
        reason: str
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO secure_gateway_audit (
                    created_at, request_id, user_id, role,
                    tool_name, decision, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    request.request_id,
                    identity.user_id,
                    identity.role,
                    request.tool_name,
                    decision,
                    reason
                )
            )


class SecureToolGateway:
    allowed_tools: dict[Role, set[str]] = {
        "user": {"check_balance"},
        "operator": {"check_balance", "request_transfer"},
        "admin": {"check_balance", "request_transfer"}
    }
    quota_limits: dict[Role, int] = {
        "user": 3,
        "operator": 5,
        "admin": 10
    }

    def __init__(self, audit_store: AuditStore) -> None:
        self.audit_store = audit_store
        self.quota_usage: dict[str, int] = {}
        self.idempotency_records: dict[str, tuple[str, dict]] = {}
        self.side_effect_count = 0

    @staticmethod
    def request_fingerprint(request: SecureRequest) -> str:
        payload = request.model_dump(
            exclude={"request_id", "idempotency_key"}
        )
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def check_quota(self, identity: Identity) -> None:
        used = self.quota_usage.get(identity.user_id, 0)
        limit = self.quota_limits[identity.role]

        if used >= limit:
            raise GatewayError(
                f"用户配额已用完：{used}/{limit}"
            )

        self.quota_usage[identity.user_id] = used + 1

    def check_permission(
        self,
        identity: Identity,
        request: SecureRequest
    ) -> None:
        if request.tool_name not in self.allowed_tools[identity.role]:
            raise GatewayError(
                f"角色{identity.role}无权调用{request.tool_name}"
            )

        if request.tool_name == "request_transfer":
            amount = request.arguments.get("amount")

            if not isinstance(amount, (int, float)) or isinstance(amount, bool):
                raise GatewayError("amount必须是数字")

            if amount <= 0 or amount > 5_000:
                raise GatewayError("amount必须大于0且不超过5000")

    def execute(
        self,
        identity: Identity,
        request: SecureRequest
    ) -> dict:
        fingerprint = self.request_fingerprint(request)
        old_record = self.idempotency_records.get(
            request.idempotency_key
        )

        if old_record is not None:
            old_fingerprint, old_result = old_record

            if old_fingerprint != fingerprint:
                raise GatewayError("幂等键对应了不同的请求参数")

            print("[幂等命中] 返回之前的工具结果。")
            return old_result

        if request.tool_name == "check_balance":
            return {"status": "success", "balance": 1000}

        self.side_effect_count += 1
        result = {
            "status": "success",
            "transfer_id": f"transfer-{self.side_effect_count:03d}",
            "message": "模拟转账成功"
        }
        self.idempotency_records[request.idempotency_key] = (
            fingerprint,
            result
        )
        return result

    def handle(
        self,
        identity: Identity,
        raw_request: dict[str, Any]
    ) -> dict:
        request: SecureRequest | None = None

        try:
            request = SecureRequest.model_validate(raw_request)
            self.check_quota(identity)
            self.check_permission(identity, request)
            result = self.execute(identity, request)
        except (ValidationError, GatewayError) as error:
            if request is None:
                # 无法通过结构校验时，使用原始字段中的安全摘要。
                print("[安全网关] 请求结构校验失败：", error)
                return {"status": "rejected", "reason": str(error)}

            self.audit_store.write(
                request,
                identity,
                "denied",
                str(error)
            )
            print("[安全网关] 拒绝：", error)
            return {"status": "rejected", "reason": str(error)}
        else:
            self.audit_store.write(
                request,
                identity,
                "approved",
                "所有安全检查通过"
            )
            print("[安全网关] 允许：", result)
            return result


project_directory = Path(__file__).resolve().parent.parent.parent
gateway = SecureToolGateway(AuditStore(
    project_directory
    / "day12"
    / "database"
    / "day12_secure_gateway.db"
))

operator = Identity(user_id="operator-001", role="operator")
user = Identity(user_id="xiaolin", role="user")

print("=" * 60)
print("实验一：操作员执行合法转账")
gateway.handle(operator, {
    "request_id": "secure-001",
    "idempotency_key": "payment-001",
    "tool_name": "request_transfer",
    "arguments": {
        "to_account": "account-002",
        "amount": 100
    }
})

print("\n实验二：相同幂等键重复提交")
gateway.handle(operator, {
    "request_id": "secure-002",
    "idempotency_key": "payment-001",
    "tool_name": "request_transfer",
    "arguments": {
        "to_account": "account-002",
        "amount": 100
    }
})

print("\n实验三：普通用户越权")
gateway.handle(user, {
    "request_id": "secure-003",
    "idempotency_key": "payment-003",
    "tool_name": "request_transfer",
    "arguments": {
        "to_account": "account-002",
        "amount": 100
    }
})

print("\n实验四：额外字段被拒绝")
gateway.handle(user, {
    "request_id": "secure-004",
    "idempotency_key": "query-004",
    "tool_name": "check_balance",
    "arguments": {"account": "account-001"},
    "is_admin": True
})

print("\n实际产生的转账副作用次数：", gateway.side_effect_count)


# ============================================================
# 数据流
# ============================================================
#
# 原始tool_calls dict
#       ↓ 1. Pydantic结构校验
# SecureRequest
#       ↓ 2. 用户配额检查
#       ↓ 3. 角色权限检查
#       ↓ 4. 幂等键检查
#       ↓ 5. 执行工具副作用
#       ↓ 6. 写入approved/denied审计日志
#
# 任意安全层失败，都不会执行高风险工具。
