"""Day15 第一步：用标准库理解签名访问令牌。"""

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any
from uuid import uuid4


class TokenError(Exception):
    """所有令牌验证错误的基类。"""


class InvalidTokenError(TokenError):
    """令牌格式或签名无效。"""


class ExpiredTokenError(TokenError):
    """令牌已经过期。"""


def base64url_encode(data: bytes) -> str:
    """生成适合放入URL和HTTP Header的无填充Base64文本。"""

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(text: str) -> bytes:
    """补回Base64填充并解码。"""

    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def sign_payload(payload_text: str, secret: bytes) -> str:
    """使用HMAC-SHA256为载荷生成签名。"""

    signature_bytes = hmac.new(
        key=secret,
        msg=payload_text.encode("ascii"),
        digestmod=hashlib.sha256
    ).digest()
    return base64url_encode(signature_bytes)


def issue_token(user_id: str, secret: bytes, ttl_seconds: int) -> str:
    """签发包含用户身份和过期时间的令牌。"""

    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid4().hex
    }

    payload_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")

    payload_text = base64url_encode(payload_bytes)
    signature_text = sign_payload(payload_text, secret)
    return f"{payload_text}.{signature_text}"


def verify_token(token: str, secret: bytes) -> dict[str, Any]:
    """验证格式、签名、载荷结构和过期时间。"""

    try:
        payload_text, received_signature = token.split(".", maxsplit=1)
    except ValueError as error:
        raise InvalidTokenError("令牌必须由载荷和签名两部分组成。") from error

    expected_signature = sign_payload(payload_text, secret)

    if not hmac.compare_digest(received_signature, expected_signature):
        raise InvalidTokenError("令牌签名无效，内容可能被篡改。")

    try:
        payload = json.loads(base64url_decode(payload_text).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidTokenError("令牌载荷不是有效JSON。") from error

    if not isinstance(payload, dict):
        raise InvalidTokenError("令牌载荷必须是JSON对象。")

    if not isinstance(payload.get("sub"), str) or not payload["sub"]:
        raise InvalidTokenError("令牌缺少有效的sub用户编号。")

    if not isinstance(payload.get("exp"), int):
        raise InvalidTokenError("令牌缺少有效的exp过期时间。")

    if int(time.time()) >= payload["exp"]:
        raise ExpiredTokenError("令牌已经过期。")

    return payload


def tamper_user_id(token: str, new_user_id: str) -> str:
    """模拟客户端只修改载荷但无法生成新签名。"""

    payload_text, old_signature = token.split(".", maxsplit=1)
    payload = json.loads(base64url_decode(payload_text).decode("utf-8"))
    payload["sub"] = new_user_id

    changed_payload_text = base64url_encode(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True
        ).encode("utf-8")
    )

    # 攻击者保留旧签名，因此服务端重新计算后会发现不一致。
    return f"{changed_payload_text}.{old_signature}"


def show_verification(name: str, token: str, secret: bytes) -> None:
    print(f"\n{name}：")

    try:
        payload = verify_token(token, secret)
        print("验证结果：通过")
        print("用户编号sub：", payload["sub"])
        print("签发时间iat：", payload["iat"])
        print("过期时间exp：", payload["exp"])
        print("令牌编号jti：", payload["jti"])
    except TokenError as error:
        print("验证结果：拒绝")
        print("错误类型：", type(error).__name__)
        print("错误信息：", error)


def main() -> None:
    # 教学演示每次启动生成随机密钥；真实服务必须从安全环境变量读取。
    secret = secrets.token_bytes(32)

    valid_token = issue_token(
        user_id="user-xiaolin",
        secret=secret,
        ttl_seconds=300
    )
    tampered_token = tamper_user_id(valid_token, "user-admin")
    expired_token = issue_token(
        user_id="user-xiaolin",
        secret=secret,
        ttl_seconds=-1
    )

    print("原始有效令牌：")
    print(valid_token)

    payload_text = valid_token.split(".", maxsplit=1)[0]
    visible_payload = json.loads(
        base64url_decode(payload_text).decode("utf-8")
    )

    print("\n不需要密钥也能看到载荷：")
    print(json.dumps(visible_payload, ensure_ascii=False, indent=2))
    print("结论：签名提供防篡改，不提供内容加密。")

    show_verification("测试1：原始令牌", valid_token, secret)
    show_verification("测试2：把用户改成admin", tampered_token, secret)
    show_verification("测试3：已经过期的令牌", expired_token, secret)


if __name__ == "__main__":
    main()


# user_id + iat + exp + jti
#          ↓ JSON + Base64URL（可读，不是加密）
# payload_text
#          ↓ HMAC-SHA256(secret, payload_text)
# signature
#          ↓
# token = payload_text.signature
