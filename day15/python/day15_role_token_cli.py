"""Day15 第五步：签发包含user/admin角色的令牌。"""

import argparse
import json
import os
import time
from uuid import uuid4

from dotenv import load_dotenv

from day15_signed_token_basic import base64url_encode, sign_payload


load_dotenv()


def load_secret() -> bytes:
    text = os.getenv("DAY15_TOKEN_SECRET", "").strip()
    if len(text) < 32:
        raise RuntimeError("DAY15_TOKEN_SECRET至少需要32个字符。")
    return text.encode("utf-8")


def issue_role_token(
    user_id: str,
    role: str,
    ttl_seconds: int
) -> str:
    """把身份和角色一起放入受签名保护的载荷。"""

    now = int(time.time())
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid4().hex
    }
    payload_text = base64url_encode(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True
        ).encode("utf-8")
    )
    return f"{payload_text}.{sign_payload(payload_text, load_secret())}"


parser = argparse.ArgumentParser(description="Day15角色令牌签发工具")
parser.add_argument("user_id")
parser.add_argument("--role", choices=["user", "admin"], default="user")
parser.add_argument("--ttl", type=int, default=3600)
args = parser.parse_args()

if args.ttl <= 0:
    raise SystemExit("--ttl必须大于0。")

token = issue_role_token(args.user_id, args.role, args.ttl)
print("角色令牌签发成功。")
print("用户：", args.user_id)
print("角色：", args.role)
print("有效期：", args.ttl, "秒")
print("\n访问令牌：")
print(token)
