"""Day12 第七步：使用幂等键避免副作用工具被重复执行。"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


class IdempotencyConflictError(Exception):
    """同一个幂等键被用于不同业务参数。"""


class TransferError(Exception):
    """转账请求不合法。"""


@dataclass(frozen=True)
class TransferRequest:
    idempotency_key: str
    from_account: str
    to_account: str
    amount: float


@dataclass(frozen=True)
class TransferResult:
    transfer_id: str
    status: str
    message: str


class IdempotencyStore:
    """保存幂等键、请求指纹和首次执行结果。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS idempotency_records (
                    idempotency_key TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)

    @staticmethod
    def make_request_hash(request: TransferRequest) -> str:
        payload = {
            "from_account": request.from_account,
            "to_account": request.to_account,
            "amount": request.amount
        }
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def find(self, key: str) -> tuple[str, str] | None:
        with sqlite3.connect(self.database_path) as connection:
            return connection.execute(
                """
                SELECT request_hash, result_json
                FROM idempotency_records
                WHERE idempotency_key = ?
                """,
                (key,)
            ).fetchone()

    def save(
        self,
        key: str,
        request_hash: str,
        result: TransferResult
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO idempotency_records (
                    idempotency_key,
                    request_hash,
                    result_json
                )
                VALUES (?, ?, ?)
                """,
                (
                    key,
                    request_hash,
                    json.dumps(result.__dict__, ensure_ascii=False)
                )
            )


class TransferService:
    def __init__(self, store: IdempotencyStore) -> None:
        self.store = store
        self.actual_side_effect_count = 0

    def transfer(self, request: TransferRequest) -> TransferResult:
        if request.amount <= 0:
            raise TransferError("金额必须大于0。")

        if request.from_account == request.to_account:
            raise TransferError("转出账户和收款账户不能相同。")

        request_hash = self.store.make_request_hash(request)
        previous = self.store.find(request.idempotency_key)

        if previous is not None:
            previous_hash, previous_result_json = previous

            if previous_hash != request_hash:
                raise IdempotencyConflictError(
                    "同一个幂等键不能对应不同的转账参数。"
                )

            print("[幂等命中] 重复请求，直接返回首次结果。")
            data = json.loads(previous_result_json)
            return TransferResult(**data)

        # 只有找不到幂等记录时，才执行真实副作用。
        self.actual_side_effect_count += 1
        transfer_id = f"transfer-{self.actual_side_effect_count:03d}"
        result = TransferResult(
            transfer_id=transfer_id,
            status="success",
            message=(
                f"已从{request.from_account}向{request.to_account} "
                f"转账{request.amount}元。"
            )
        )

        self.store.save(request.idempotency_key, request_hash, result)
        print("[首次执行] 已产生一次模拟转账副作用。")
        return result


project_directory = Path(__file__).resolve().parent.parent.parent
store = IdempotencyStore(
    project_directory / "day12" / "database" / "day12_idempotency.db"
)
service = TransferService(store)

request = TransferRequest(
    idempotency_key="payment-001",
    from_account="account-001",
    to_account="account-002",
    amount=100
)

print("=" * 60)
print("第一次提交")
print(service.transfer(request))

print("\n第二次重复提交")
print(service.transfer(request))

print("\n同一个幂等键但修改金额")
conflicting_request = TransferRequest(
    idempotency_key="payment-001",
    from_account="account-001",
    to_account="account-002",
    amount=200
)

try:
    service.transfer(conflicting_request)
except IdempotencyConflictError as error:
    print("正确拒绝参数冲突：", error)

print("\n实际产生的副作用次数：", service.actual_side_effect_count)


# ============================================================
# 数据流
# ============================================================
#
# TransferRequest + idempotency_key
#             ↓
# 查询SQLite记录
#   ├── 不存在 → 执行一次副作用 → 保存结果
#   ├── 存在且请求指纹相同 → 返回首次结果，不重复执行
#   └── 存在但请求指纹不同 → 拒绝，防止篡改
#
# 幂等键应该由业务请求方生成，并在网络重试时保持不变。
