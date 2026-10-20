"""Day15 第六步：签发包含细粒度权限Scope的令牌。"""

import argparse
import json
import os
import time
from uuid import uuid4

from dotenv import load_dotenv

from day15_signed_token_basic import base64url_encode, sign_payload


load_dotenv()

ALLOWED_SCOPES = {
    "task:create",
    "task:read:own",
    "task:cancel:own",
    "task:read:any",
    "task:delete:any"
}


def load_secret() -> bytes:
    text = os.getenv("DAY15_TOKEN_SECRET", "").strip()
    if len(text) < 32:
        raise RuntimeError("DAY15_TOKEN_SECRET至少需要32个字符。")
    return text.encode("utf-8")


def issue_scope_token(
    user_id: str,
    scopes: list[str],
    ttl_seconds: int
) -> str:
    unknown_scopes = set(scopes) - ALLOWED_SCOPES
    if unknown_scopes:
        raise RuntimeError(
            "不允许的Scope：" + ", ".join(sorted(unknown_scopes))
        )

    now = int(time.time())
    payload = {
        "sub": user_id,
        "scopes": sorted(set(scopes)),
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


parser = argparse.ArgumentParser(description="Day15 Scope令牌签发工具")
parser.add_argument("user_id")
parser.add_argument(
    "--scope",
    action="append",
    required=True,
    choices=sorted(ALLOWED_SCOPES),
    help="可重复使用，为令牌添加多个权限"
)
parser.add_argument("--ttl", type=int, default=3600)
args = parser.parse_args()

if args.ttl <= 0:
    raise SystemExit("--ttl必须大于0。")

token = issue_scope_token(args.user_id, args.scope, args.ttl)
print("Scope令牌签发成功。")
print("用户：", args.user_id)
print("权限：", sorted(set(args.scope)))
print("有效期：", args.ttl, "秒")
print("\n访问令牌：")
print(token)
