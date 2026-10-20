"""Day15 第二步：使用持久密钥签发和验证访问令牌。"""

import argparse
import os
import secrets

from dotenv import load_dotenv

from day15_signed_token_basic import (
    TokenError,
    issue_token,
    verify_token,
)


load_dotenv()


def load_secret() -> bytes:
    """从环境变量读取服务端签名密钥。"""

    secret_text = os.getenv("DAY15_TOKEN_SECRET", "").strip()

    if not secret_text:
        raise RuntimeError(
            "缺少DAY15_TOKEN_SECRET。请先运行generate-secret，"
            "再把结果写入项目根目录.env。"
        )

    if len(secret_text) < 32:
        raise RuntimeError(
            "DAY15_TOKEN_SECRET至少需要32个字符。"
        )

    return secret_text.encode("utf-8")


def generate_secret_command() -> None:
    """生成适合放进.env的随机密钥，但不自动修改文件。"""

    secret_text = secrets.token_urlsafe(48)

    print("请把下面这一整行复制到项目根目录.env：")
    print()
    print(f"DAY15_TOKEN_SECRET={secret_text}")
    print()
    print("不要把真实密钥提交到Git或发送给客户端。")


def issue_command(user_id: str, ttl_seconds: int) -> None:
    """为指定用户签发令牌。"""

    if not user_id.strip():
        raise RuntimeError("user_id不能为空。")

    if ttl_seconds <= 0:
        raise RuntimeError("ttl_seconds必须大于0。")

    token = issue_token(
        user_id=user_id.strip(),
        secret=load_secret(),
        ttl_seconds=ttl_seconds
    )

    print("令牌签发成功。")
    print("用户：", user_id)
    print("有效期：", ttl_seconds, "秒")
    print("\n访问令牌：")
    print(token)


def verify_command(token: str) -> None:
    """使用相同环境密钥验证令牌。"""

    # 允许从终端复制时意外带入首尾空格或换行；令牌内部不能有空白。
    token = token.strip()

    if not token:
        raise RuntimeError("token不能为空。")

    try:
        payload = verify_token(token, load_secret())
    except TokenError as error:
        print("令牌验证失败。")
        print("错误类型：", type(error).__name__)
        print("错误信息：", error)
        raise SystemExit(1) from error

    print("令牌验证通过。")
    print("用户sub：", payload["sub"])
    print("签发时间iat：", payload["iat"])
    print("过期时间exp：", payload["exp"])
    print("令牌编号jti：", payload["jti"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day15签名访问令牌工具"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    subparsers.add_parser(
        "generate-secret",
        help="生成新的服务端签名密钥"
    )

    issue_parser = subparsers.add_parser(
        "issue",
        help="为用户签发访问令牌"
    )
    issue_parser.add_argument("user_id")
    issue_parser.add_argument(
        "--ttl",
        type=int,
        default=3600,
        help="有效秒数，默认3600"
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="验证访问令牌"
    )
    verify_parser.add_argument("token")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-secret":
        generate_secret_command()
    elif args.command == "issue":
        issue_command(args.user_id, args.ttl)
    elif args.command == "verify":
        verify_command(args.token)


if __name__ == "__main__":
    main()


# generate-secret → 人工写入.env
# .env → load_dotenv() → DAY15_TOKEN_SECRET
# issue(user_id) → token
# 程序退出和重启
# verify(token) + 相同secret → 仍然通过
