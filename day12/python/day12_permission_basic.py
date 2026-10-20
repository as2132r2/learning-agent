"""Day12 第四步：在Agent工具执行前进行角色和最小权限检查。"""

from dataclasses import dataclass
from typing import Literal


Role = Literal["user", "operator", "admin"]


class PermissionDeniedError(PermissionError):
    """当前角色没有执行指定工具的权限。"""


class ToolInputError(ValueError):
    """工具参数不符合安全规则。"""


@dataclass(frozen=True)
class UserIdentity:
    user_id: str
    role: Role


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    arguments: dict


class PermissionPolicy:
    """集中管理角色、工具和风险等级。"""

    # user只能查询，operator可以查询和发起审批，admin拥有全部权限。
    allowed_tools: dict[Role, set[str]] = {
        "user": {"check_balance", "search_knowledge"},
        "operator": {
            "check_balance",
            "search_knowledge",
            "request_transfer"
        },
        "admin": {
            "check_balance",
            "search_knowledge",
            "request_transfer",
            "approve_transfer"
        }
    }

    high_risk_tools = {
        "request_transfer",
        "approve_transfer"
    }

    def check(self, identity: UserIdentity, request: ToolRequest) -> None:
        allowed = self.allowed_tools.get(identity.role, set())

        if request.tool_name not in allowed:
            raise PermissionDeniedError(
                f"角色 {identity.role} 无权执行工具 "
                f"{request.tool_name}。"
            )

        if request.tool_name in self.high_risk_tools:
            self.check_high_risk_arguments(request)

    @staticmethod
    def check_high_risk_arguments(request: ToolRequest) -> None:
        to_account = request.arguments.get("to_account")
        amount = request.arguments.get("amount")

        if not isinstance(to_account, str) or not to_account.strip():
            raise ToolInputError("to_account 必须是非空字符串。")

        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            raise ToolInputError("amount 必须是数字。")

        if amount <= 0:
            raise ToolInputError("amount 必须大于0。")

        if amount > 5_000:
            raise ToolInputError("单笔转账不能超过5000元。")


def execute_tool(
    identity: UserIdentity,
    request: ToolRequest,
    policy: PermissionPolicy
) -> str:
    """统一入口：先授权和参数校验，再执行工具。"""

    print(
        f"\n[授权检查] 用户={identity.user_id} "
        f"角色={identity.role} 工具={request.tool_name}"
    )

    policy.check(identity, request)
    print("[授权结果] 通过，开始执行工具。")

    if request.tool_name == "check_balance":
        return "模拟余额：1000元。"

    if request.tool_name == "search_knowledge":
        return "模拟检索结果：RAG 会先检索资料再生成回答。"

    if request.tool_name == "request_transfer":
        return (
            f"已创建转账审批：向{request.arguments['to_account']} "
            f"转账{request.arguments['amount']}元。"
        )

    if request.tool_name == "approve_transfer":
        return "管理员已批准这笔转账。"

    raise ValueError(f"未知工具：{request.tool_name}")


def run_case(
    title: str,
    identity: UserIdentity,
    request: ToolRequest,
    policy: PermissionPolicy
) -> None:
    print("\n" + "=" * 60)
    print(title)

    try:
        result = execute_tool(identity, request, policy)
    except PermissionDeniedError as error:
        print("拒绝原因：权限不足。")
        print(error)
    except ToolInputError as error:
        print("拒绝原因：工具参数不安全。")
        print(error)
    else:
        print("工具结果：", result)


policy = PermissionPolicy()
normal_user = UserIdentity("xiaolin", "user")
operator = UserIdentity("operator-001", "operator")
admin = UserIdentity("admin-001", "admin")

run_case(
    "实验一：普通用户查询余额",
    normal_user,
    ToolRequest("check_balance", {"account": "account-001"}),
    policy
)

run_case(
    "实验二：普通用户尝试转账",
    normal_user,
    ToolRequest(
        "request_transfer",
        {"to_account": "account-002", "amount": 100}
    ),
    policy
)

run_case(
    "实验三：操作员发起合法转账审批",
    operator,
    ToolRequest(
        "request_transfer",
        {"to_account": "account-002", "amount": 100}
    ),
    policy
)

run_case(
    "实验四：操作员尝试直接批准转账",
    operator,
    ToolRequest(
        "approve_transfer",
        {"to_account": "account-002", "amount": 100}
    ),
    policy
)

run_case(
    "实验五：管理员批准合法转账",
    admin,
    ToolRequest(
        "approve_transfer",
        {"to_account": "account-002", "amount": 100}
    ),
    policy
)

run_case(
    "实验六：管理员尝试超额转账",
    admin,
    ToolRequest(
        "request_transfer",
        {"to_account": "account-002", "amount": 6_000}
    ),
    policy
)


# ============================================================
# 数据流
# ============================================================
#
# AIMessage.tool_calls
#        ↓
# ToolRequest
#        ↓
# PermissionPolicy.check()
#   ├── 角色无权限 → PermissionDeniedError
#   ├── 参数不安全 → ToolInputError
#   └── 通过 → execute_tool()
#
# 模型可以提出工具调用，但最终是否执行由Python策略决定。
