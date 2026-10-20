"""Day12 第六步：按用户和角色实施独立请求配额。"""

from dataclasses import dataclass
from typing import Literal


Role = Literal["user", "operator", "admin"]


class QuotaExceededError(Exception):
    """当前用户在时间窗口内的额度已经用完。"""


@dataclass(frozen=True)
class Identity:
    user_id: str
    role: Role


class ManualClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds
        print(f"[时钟] 当前时间：{self.seconds:.0f}秒")


class UserQuotaLimiter:
    """固定时间窗口限流：每个用户分别维护自己的计数。"""

    limits_per_window: dict[Role, int] = {
        "user": 3,
        "operator": 5,
        "admin": 10
    }

    def __init__(
        self,
        clock: ManualClock,
        window_seconds: float = 60.0
    ) -> None:
        self.clock = clock
        self.window_seconds = window_seconds
        self.usage: dict[str, tuple[float, int]] = {}

    def acquire(self, identity: Identity) -> dict[str, int | float]:
        now = self.clock.now()
        window_start, used = self.usage.get(identity.user_id, (now, 0))

        # 时间窗口结束后，为该用户清零，而不是清零所有用户。
        if now - window_start >= self.window_seconds:
            window_start = now
            used = 0

        limit = self.limits_per_window[identity.role]

        if used >= limit:
            raise QuotaExceededError(
                f"用户{identity.user_id}在当前窗口内已使用{used}/{limit}次。"
            )

        used += 1
        self.usage[identity.user_id] = (window_start, used)

        return {
            "used": used,
            "limit": limit,
            "remaining": limit - used,
            "window_start": window_start
        }


def call_agent(
    identity: Identity,
    limiter: UserQuotaLimiter,
    question: str
) -> None:
    print(f"\n用户={identity.user_id} 角色={identity.role}")

    try:
        quota = limiter.acquire(identity)
    except QuotaExceededError as error:
        print("限流拒绝：", error)
        return

    print("允许调用模型：", question)
    print("配额状态：", quota)


clock = ManualClock()
limiter = UserQuotaLimiter(clock, window_seconds=60.0)

xiaolin = Identity("xiaolin", "user")
operator = Identity("operator-001", "operator")

print("=" * 60)
print("实验一：普通用户每窗口最多3次")

for number in range(1, 5):
    call_agent(
        xiaolin,
        limiter,
        f"小林的第{number}个问题"
    )

print("\n" + "=" * 60)
print("实验二：不同用户拥有独立额度")

call_agent(operator, limiter, "操作员的第1个问题")
call_agent(xiaolin, limiter, "小林继续提问")

print("\n" + "=" * 60)
print("实验三：时间窗口结束后恢复额度")

clock.advance(60)
call_agent(xiaolin, limiter, "新窗口中的第1个问题")


# ============================================================
# 数据流
# ============================================================
#
# Identity(user_id, role) + question
#              ↓
# usage[user_id]
#              ↓
# 当前时间窗口是否超额？
#       ├── 是 → QuotaExceededError
#       └── 否 → used + 1 → 允许调用模型
#
# 角色额度：user=3，operator=5，admin=10。
# 不同用户的计数相互独立，窗口结束后单独恢复。
