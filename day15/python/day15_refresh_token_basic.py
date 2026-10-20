"""Day15 第八步：Refresh Token存储、轮换与重放防护。"""

import hashlib
import os
import secrets
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

from day15_signed_token_basic import issue_token, verify_token


load_dotenv()

DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "day15_refresh_tokens.db"
)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

ACCESS_TOKEN_TTL = 60
REFRESH_TOKEN_TTL = 7 * 24 * 60 * 60


class RefreshTokenError(Exception):
    """Refresh Token无效、过期或被重复使用。"""


def load_secret() -> bytes:
    text = os.getenv("DAY15_TOKEN_SECRET", "").strip()
    if len(text) < 32:
        raise RuntimeError("DAY15_TOKEN_SECRET至少需要32个字符。")
    return text.encode("utf-8")


def hash_refresh_token(token: str) -> str:
    """数据库只保存摘要，不保存原始Refresh Token。"""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_tokens(
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            used_at INTEGER,
            replaced_by_hash TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    connection.commit()


def create_refresh_token(
    connection: sqlite3.Connection,
    user_id: str
) -> str:
    """生成高熵随机令牌，保存摘要并只返回一次原文。"""

    raw_token = secrets.token_urlsafe(48)
    token_hash = hash_refresh_token(raw_token)
    now = int(time.time())

    connection.execute(
        """
        INSERT INTO refresh_tokens(
            token_hash, user_id, expires_at, created_at
        ) VALUES (?, ?, ?, ?)
        """,
        (token_hash, user_id, now + REFRESH_TOKEN_TTL, now)
    )
    connection.commit()
    return raw_token


def issue_token_pair(
    connection: sqlite3.Connection,
    user_id: str
) -> tuple[str, str]:
    """签发短期Access Token和长期Refresh Token。"""

    access_token = issue_token(
        user_id=user_id,
        secret=load_secret(),
        ttl_seconds=ACCESS_TOKEN_TTL
    )
    refresh_token = create_refresh_token(connection, user_id)
    return access_token, refresh_token


def rotate_refresh_token(
    connection: sqlite3.Connection,
    old_raw_token: str
) -> tuple[str, str]:
    """消费旧Refresh Token并签发全新的令牌对。"""

    old_hash = hash_refresh_token(old_raw_token)
    now = int(time.time())

    # IMMEDIATE事务防止两个并发刷新请求同时消费同一令牌。
    connection.execute("BEGIN IMMEDIATE")

    try:
        row = connection.execute(
            """
            SELECT user_id, expires_at, used_at
            FROM refresh_tokens
            WHERE token_hash = ?
            """,
            (old_hash,)
        ).fetchone()

        if row is None:
            raise RefreshTokenError("Refresh Token不存在。")

        user_id, expires_at, used_at = row

        if used_at is not None:
            raise RefreshTokenError(
                "Refresh Token已经使用过，疑似重放请求。"
            )

        if now >= expires_at:
            raise RefreshTokenError("Refresh Token已经过期。")

        new_refresh_token = secrets.token_urlsafe(48)
        new_hash = hash_refresh_token(new_refresh_token)

        connection.execute(
            """
            UPDATE refresh_tokens
            SET used_at = ?, replaced_by_hash = ?
            WHERE token_hash = ?
            """,
            (now, new_hash, old_hash)
        )
        connection.execute(
            """
            INSERT INTO refresh_tokens(
                token_hash, user_id, expires_at, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (new_hash, user_id, now + REFRESH_TOKEN_TTL, now)
        )
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    new_access_token = issue_token(
        user_id=user_id,
        secret=load_secret(),
        ttl_seconds=ACCESS_TOKEN_TTL
    )
    return new_access_token, new_refresh_token


def main() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    initialize_database(connection)

    try:
        print("第一步：模拟登录并签发令牌对")
        access_token, refresh_token = issue_token_pair(
            connection,
            "user-xiaolin"
        )
        access_payload = verify_token(access_token, load_secret())

        print("Access Token有效秒数：", ACCESS_TOKEN_TTL)
        print("Access Token用户：", access_payload["sub"])
        print("Refresh Token有效秒数：", REFRESH_TOKEN_TTL)
        print("Refresh Token原文：", refresh_token)
        print("数据库保存的摘要：", hash_refresh_token(refresh_token))

        print("\n第二步：使用Refresh Token换取新令牌对")
        new_access_token, new_refresh_token = rotate_refresh_token(
            connection,
            refresh_token
        )
        new_payload = verify_token(new_access_token, load_secret())

        print("刷新成功，新Access Token用户：", new_payload["sub"])
        print("新Refresh Token：", new_refresh_token)
        print("旧Refresh Token与新令牌是否相同：", refresh_token == new_refresh_token)

        print("\n第三步：再次使用旧Refresh Token，模拟重放")
        try:
            rotate_refresh_token(connection, refresh_token)
        except RefreshTokenError as error:
            print("重放请求已拒绝：", error)

    finally:
        connection.close()

    print("\n数据库：", DATABASE_PATH)


if __name__ == "__main__":
    main()


# 登录 → Access Token(短期) + Refresh Token(长期)
# Refresh原文只交给客户端；服务端只保存SHA-256摘要
# 刷新 → 旧Refresh标记used → 新Access + 新Refresh
# 重复使用旧Refresh → 检测重放并拒绝
