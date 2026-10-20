"""Day15 第十步：使用scrypt安全存储和验证用户密码。"""

import argparse
import getpass
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path


DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "day15_users.db"
)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_BYTES = 32


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            user_id TEXT PRIMARY KEY,
            password_hash BLOB NOT NULL,
            password_salt BLOB NOT NULL,
            scrypt_n INTEGER NOT NULL,
            scrypt_r INTEGER NOT NULL,
            scrypt_p INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()


def derive_password_key(
    password: str,
    salt: bytes,
    n: int,
    r: int,
    p: int
) -> bytes:
    """使用内存密集型scrypt从密码派生固定长度密钥。"""

    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=KEY_BYTES
    )


def validate_new_password(password: str) -> None:
    if len(password) < 10:
        raise ValueError("密码至少需要10个字符。")
    if password.isspace():
        raise ValueError("密码不能全部是空白字符。")


def register_user(
    connection: sqlite3.Connection,
    user_id: str,
    password: str
) -> None:
    """为每个用户生成独立Salt，只保存派生结果。"""

    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id不能为空。")

    validate_new_password(password)
    salt = secrets.token_bytes(SALT_BYTES)
    password_hash = derive_password_key(
        password,
        salt,
        SCRYPT_N,
        SCRYPT_R,
        SCRYPT_P
    )

    try:
        connection.execute(
            """
            INSERT INTO users(
                user_id, password_hash, password_salt,
                scrypt_n, scrypt_r, scrypt_p, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                password_hash,
                salt,
                SCRYPT_N,
                SCRYPT_R,
                SCRYPT_P,
                now_text()
            )
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        raise ValueError("该user_id已经注册。") from error

    print("注册成功：", user_id)
    print("数据库只保存Salt和scrypt派生结果，不保存密码原文。")


def verify_user(
    connection: sqlite3.Connection,
    user_id: str,
    password: str
) -> bool:
    """使用数据库中的同一Salt和参数重新派生并恒定时间比较。"""

    row = connection.execute(
        """
        SELECT password_hash, password_salt,
               scrypt_n, scrypt_r, scrypt_p
        FROM users
        WHERE user_id = ?
        """,
        (user_id.strip(),)
    ).fetchone()

    if row is None:
        # 对不存在用户也运行一次scrypt，减少明显的时间差异。
        fake_salt = b"\x00" * SALT_BYTES
        derive_password_key(
            password,
            fake_salt,
            SCRYPT_N,
            SCRYPT_R,
            SCRYPT_P
        )
        return False

    saved_hash, salt, n, r, p = row
    candidate_hash = derive_password_key(password, salt, n, r, p)
    return hmac.compare_digest(candidate_hash, saved_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day15密码认证练习")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("user_id")

    login_parser = subparsers.add_parser("login")
    login_parser.add_argument("user_id")

    args = parser.parse_args()
    connection = sqlite3.connect(DATABASE_PATH)
    initialize_database(connection)

    try:
        if args.command == "register":
            password = getpass.getpass("请输入新密码：")
            confirmation = getpass.getpass("请再次输入：")

            if not hmac.compare_digest(password, confirmation):
                raise ValueError("两次输入的密码不一致。")

            register_user(connection, args.user_id, password)

        elif args.command == "login":
            password = getpass.getpass("请输入密码：")
            authenticated = verify_user(
                connection,
                args.user_id,
                password
            )

            if authenticated:
                print("登录验证成功。")
            else:
                print("用户名或密码错误。")

    except ValueError as error:
        print("操作失败：", error)
        raise SystemExit(1) from error
    finally:
        connection.close()

    print("数据库：", DATABASE_PATH)


if __name__ == "__main__":
    main()


# 注册：password + random salt → scrypt → password_hash → SQLite
# 登录：输入password + saved salt → scrypt → candidate_hash
#       → compare_digest(candidate_hash, saved_hash)
