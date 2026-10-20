"""Day29 第 1 步：W3C traceparent 格式与上下文传播。

运行：
    python day29/python/day29_traceparent_basic.py

traceparent 基本格式：
    version-trace_id-parent_id-trace_flags
    00-32位十六进制-16位十六进制-2位十六进制
"""

from __future__ import annotations

import re
import secrets
from dataclasses import asdict, dataclass


TRACEPARENT_PATTERN = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-"
    r"(?P<trace_flags>[0-9a-f]{2})$"
)

ZERO_TRACE_ID = "0" * 32
ZERO_SPAN_ID = "0" * 16
SAMPLED_FLAG = 0x01


class InvalidTraceparentError(ValueError):
    """外部 traceparent 不符合 W3C 基本规则。"""


@dataclass(frozen=True)
class TraceContext:
    """从 traceparent 得到或为新 Trace 创建的标准上下文。"""

    version: str
    trace_id: str
    span_id: str
    trace_flags: int
    is_remote: bool

    @property
    def sampled(self) -> bool:
        """最低位为 1 表示调用方希望记录并导出此 Trace。"""

        return bool(self.trace_flags & SAMPLED_FLAG)

    def to_traceparent(self) -> str:
        """序列化为可以放入 HTTP 请求头的字符串。"""

        return (
            f"{self.version}-{self.trace_id}-"
            f"{self.span_id}-{self.trace_flags:02x}"
        )


def random_trace_id() -> str:
    """生成 16 字节，也就是 32 位十六进制 Trace ID。"""

    while True:
        value = secrets.token_hex(16)
        if value != ZERO_TRACE_ID:
            return value


def random_span_id() -> str:
    """生成 8 字节，也就是 16 位十六进制 Span ID。"""

    while True:
        value = secrets.token_hex(8)
        if value != ZERO_SPAN_ID:
            return value


def parse_traceparent(header_value: str) -> TraceContext:
    """解析并严格校验一个 W3C traceparent 请求头。"""

    # 标准要求使用小写十六进制；这里不自动 lower()，避免掩盖非法输入。
    match = TRACEPARENT_PATTERN.fullmatch(header_value.strip())
    if match is None:
        raise InvalidTraceparentError(
            "格式必须是 00-32位小写trace_id-16位小写parent_id-2位flags。"
        )

    version = match.group("version")
    trace_id = match.group("trace_id")
    parent_id = match.group("parent_id")
    flags_text = match.group("trace_flags")

    # 本练习只实现当前版本 00。
    if version != "00":
        raise InvalidTraceparentError(f"暂不支持traceparent版本：{version}")
    if trace_id == ZERO_TRACE_ID:
        raise InvalidTraceparentError("trace_id不能全部为0。")
    if parent_id == ZERO_SPAN_ID:
        raise InvalidTraceparentError("parent_id不能全部为0。")

    return TraceContext(
        version=version,
        trace_id=trace_id,
        span_id=parent_id,
        trace_flags=int(flags_text, 16),
        is_remote=True,
    )


def create_root_context(*, sampled: bool) -> TraceContext:
    """没有合法上游上下文时，创建一条新的根 Trace。"""

    return TraceContext(
        version="00",
        trace_id=random_trace_id(),
        span_id=random_span_id(),
        trace_flags=SAMPLED_FLAG if sampled else 0,
        is_remote=False,
    )


def create_child_context(parent: TraceContext) -> TraceContext:
    """创建子 Span：保持 trace_id 和 flags，只更换 span_id。"""

    return TraceContext(
        version=parent.version,
        trace_id=parent.trace_id,
        span_id=random_span_id(),
        trace_flags=parent.trace_flags,
        is_remote=False,
    )


def extract_or_create_context(
    traceparent_header: str | None,
    *,
    default_sampled: bool = True,
) -> tuple[TraceContext, str]:
    """提取合法远程上下文；缺失或非法时创建新 Trace。"""

    if traceparent_header is None:
        return create_root_context(sampled=default_sampled), "missing_created_new"

    try:
        return parse_traceparent(traceparent_header), "valid_remote_parent"
    except InvalidTraceparentError:
        # 边界入口不能继续使用非法上游 ID，但仍可创建全新的安全 Trace。
        return create_root_context(sampled=default_sampled), "invalid_created_new"


def demonstrate_valid_remote_parent() -> None:
    print("第一部分：接收合法的上游 traceparent")
    print("-" * 92)

    incoming_header = (
        "00-4bf92f3577b34da6a3ce929d0e0e4736-"
        "00f067aa0ba902b7-01"
    )
    remote_parent = parse_traceparent(incoming_header)
    server_span = create_child_context(remote_parent)
    downstream_span = create_child_context(server_span)

    print("收到的请求头：", incoming_header)
    print("解析后的远程父上下文：", asdict(remote_parent))
    print("是否采样：", remote_parent.sampled)
    print()
    print("本服务创建 Server Span：")
    print("trace_id保持不变：", server_span.trace_id == remote_parent.trace_id)
    print("span_id已经更换：", server_span.span_id != remote_parent.span_id)
    print("父 span_id：", remote_parent.span_id)
    print("当前 span_id：", server_span.span_id)
    print()
    print("调用下游服务时发送的新 traceparent：")
    print(downstream_span.to_traceparent())
    print("下游仍属于同一条Trace：", downstream_span.trace_id == remote_parent.trace_id)
    print("采样决定继续传播：", downstream_span.sampled)


def demonstrate_missing_and_invalid() -> None:
    print("\n" + "=" * 92)
    print("第二部分：缺失或非法请求头")
    print("-" * 92)

    examples: list[tuple[str, str | None]] = [
        ("没有请求头", None),
        (
            "trace_id全为0",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        ),
        (
            "parent_id全为0",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        ),
        (
            "使用大写十六进制",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01",
        ),
        ("长度错误", "00-1234-abcd-01"),
    ]

    for label, raw_header in examples:
        context, action = extract_or_create_context(raw_header)
        print(f"\n{label}：")
        print("处理动作：", action)
        print("生成/提取的 traceparent：", context.to_traceparent())
        print("is_remote：", context.is_remote)


def demonstrate_sampling_flag() -> None:
    print("\n" + "=" * 92)
    print("第三部分：trace_flags 的最低位")
    print("-" * 92)

    sampled = create_root_context(sampled=True)
    not_sampled = create_root_context(sampled=False)

    print("采样 Trace：", sampled.to_traceparent())
    print("flags=01，sampled：", sampled.sampled)
    print()
    print("不采样 Trace：", not_sampled.to_traceparent())
    print("flags=00，sampled：", not_sampled.sampled)
    print()
    print("注意：sampled是传播给下游的采样决定，不是权限或安全标记。")


def main() -> None:
    print("Day29 W3C traceparent 基础练习")
    print("格式：version-trace_id-parent_id-trace_flags")
    print("=" * 92)

    demonstrate_valid_remote_parent()
    demonstrate_missing_and_invalid()
    demonstrate_sampling_flag()

    print("\n" + "=" * 92)
    print("本步数据流：")
    print(
        "上游HTTP请求\n"
        "→ traceparent字符串\n"
        "→ 正则格式检查 + 非零ID检查 + 版本检查\n"
        "→ TraceContext(remote parent)\n"
        "→ 本服务创建新span_id，保持trace_id和flags\n"
        "→ 调用下游时序列化新的traceparent\n"
        "→ 下游继续同一条分布式Trace"
    )


if __name__ == "__main__":
    main()
