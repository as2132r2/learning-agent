"""Day15 第十三步：只读查看最近的认证审计事件。"""

import argparse
import sqlite3

from day15_auth_audit_api import AUDIT_DATABASE_PATH


parser = argparse.ArgumentParser(description="查看认证审计日志")
parser.add_argument("--limit", type=int, default=20)
args = parser.parse_args()

if not AUDIT_DATABASE_PATH.exists():
    raise SystemExit("审计数据库尚不存在，请先运行认证审计API并尝试登录。")

connection = sqlite3.connect(AUDIT_DATABASE_PATH)
connection.row_factory = sqlite3.Row

try:
    rows = connection.execute(
        """
        SELECT id, occurred_at, event_type, user_id,
               client_ip, outcome, detail
        FROM auth_audit
        ORDER BY id DESC
        LIMIT ?
        """,
        (max(1, args.limit),)
    ).fetchall()
finally:
    connection.close()

print(f"最近{len(rows)}条认证审计事件：")

for row in rows:
    print("\n" + "-" * 60)
    print("编号：", row["id"])
    print("UTC时间：", row["occurred_at"])
    print("事件：", row["event_type"])
    print("用户：", row["user_id"])
    print("来源IP：", row["client_ip"])
    print("结果：", row["outcome"])
    print("说明：", row["detail"])
