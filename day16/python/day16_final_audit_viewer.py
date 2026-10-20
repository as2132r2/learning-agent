"""Day16 最终整合：查看 MCP 服务端的 SQLite 审计日志。

查看全部日志：
    python day16/python/day16_final_audit_viewer.py all

按用户筛选：
    python day16/python/day16_final_audit_viewer.py subject operator-leo

按操作筛选：
    python day16/python/day16_final_audit_viewer.py action transfer_money

查看统计信息：
    python day16/python/day16_final_audit_viewer.py stats

审计查看器只读取数据库，不会修改账户和审计记录。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "day16" / "database" / "day16_final_mcp.db"


def open_database() -> sqlite3.Connection:
    """打开 SQLite，并让查询结果支持 row['字段名'] 访问。"""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "审计数据库不存在。请先运行 day16_final_mcp_server.py。"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def parse_arguments_json(raw_json: str) -> dict[str, Any]:
    """把数据库中的 JSON 字符串还原成 Python 字典。"""

    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError:
        return {"无法解析的原始数据": raw_json}

    if isinstance(value, dict):
        return value

    return {"value": value}


def query_logs(
    connection: sqlite3.Connection,
    *,
    subject: str | None = None,
    action: str | None = None,
) -> list[sqlite3.Row]:
    """使用可选条件查询日志，并避免拼接用户输入到 SQL。"""

    sql = """
        SELECT id, created_at, subject, action, arguments_json, result
        FROM audit_logs
    """
    conditions: list[str] = []
    parameters: list[str] = []

    if subject is not None:
        conditions.append("subject = ?")
        parameters.append(subject)

    if action is not None:
        conditions.append("action = ?")
        parameters.append(action)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY id ASC"

    cursor = connection.execute(sql, parameters)
    return cursor.fetchall()


def show_logs(rows: list[sqlite3.Row]) -> None:
    """逐条显示审计事件。"""

    print("查询到的日志数量：", len(rows))

    if not rows:
        print("没有符合条件的审计记录。")
        return

    for row in rows:
        arguments = parse_arguments_json(row["arguments_json"])

        print("\n" + "-" * 72)
        print("日志编号：", row["id"])
        print("UTC 时间：", row["created_at"])
        print("用户身份：", row["subject"])
        print("操作名称：", row["action"])
        print("操作参数：", arguments)
        print("执行结果：", row["result"])


def show_statistics(connection: sqlite3.Connection) -> None:
    """分别按照操作、用户和结果统计审计日志。"""

    total = connection.execute(
        "SELECT COUNT(*) AS count FROM audit_logs"
    ).fetchone()["count"]

    print("日志总数：", total)

    print("\n按操作统计：")
    action_rows = connection.execute(
        """
        SELECT action, COUNT(*) AS count
        FROM audit_logs
        GROUP BY action
        ORDER BY count DESC, action ASC
        """
    ).fetchall()
    for row in action_rows:
        print(f"- {row['action']}：{row['count']} 次")

    print("\n按用户统计：")
    subject_rows = connection.execute(
        """
        SELECT subject, COUNT(*) AS count
        FROM audit_logs
        GROUP BY subject
        ORDER BY count DESC, subject ASC
        """
    ).fetchall()
    for row in subject_rows:
        print(f"- {row['subject']}：{row['count']} 次")

    print("\n按结果统计：")
    result_rows = connection.execute(
        """
        SELECT result, COUNT(*) AS count
        FROM audit_logs
        GROUP BY result
        ORDER BY count DESC, result ASC
        """
    ).fetchall()
    for row in result_rows:
        print(f"- {row['result']}：{row['count']} 次")


def show_usage() -> None:
    """显示所有支持的命令。"""

    print("用法：")
    print("  python day16/python/day16_final_audit_viewer.py all")
    print(
        "  python day16/python/day16_final_audit_viewer.py "
        "subject operator-leo"
    )
    print(
        "  python day16/python/day16_final_audit_viewer.py "
        "action transfer_money"
    )
    print("  python day16/python/day16_final_audit_viewer.py stats")


def main() -> None:
    """解析命令行参数并执行对应的只读查询。"""

    mode = sys.argv[1].lower() if len(sys.argv) >= 2 else "all"

    print("Day16 MCP 审计日志查看器")
    print("数据库：", DATABASE_PATH)
    print("查询模式：", mode)
    print("说明：本程序只读数据库，不会修改任何记录。")
    print("=" * 72)

    try:
        with open_database() as connection:
            if mode == "all" and len(sys.argv) == 2:
                show_logs(query_logs(connection))
            elif mode == "all" and len(sys.argv) == 1:
                show_logs(query_logs(connection))
            elif mode == "subject" and len(sys.argv) == 3:
                show_logs(
                    query_logs(connection, subject=sys.argv[2])
                )
            elif mode == "action" and len(sys.argv) == 3:
                show_logs(
                    query_logs(connection, action=sys.argv[2])
                )
            elif mode == "stats" and len(sys.argv) == 2:
                show_statistics(connection)
            else:
                print("命令参数不正确。\n")
                show_usage()
    except (FileNotFoundError, sqlite3.Error) as error:
        print("读取审计数据库失败。")
        print("错误类型：", type(error).__name__)
        print("错误信息：", error)
        return

    print("\n" + "=" * 72)
    print("审计数据流：")
    print("MCP 工具执行")
    print("→ 服务端 write_audit_log()")
    print("→ JSON 参数 + 身份 + 操作 + 结果")
    print("→ SQLite audit_logs 表")
    print("→ 审计查看器执行参数化 SELECT")
    print("→ 日志明细或统计结果")


if __name__ == "__main__":
    main()
