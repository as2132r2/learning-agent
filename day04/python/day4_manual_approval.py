import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import tool


# ==================================================
# 1. 加载模型配置
# ==================================================

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# ==================================================
# 2. 定义只读工具
# ==================================================

@tool
def check_balance(
    account: str
) -> str:
    """
    查询模拟账户的余额。

    这是只读工具，不会修改账户数据。
    只有用户明确要求查询余额时才使用。
    """

    print(
        f"[执行只读工具] "
        f"查询账户：{account}"
    )

    return (
        f"模拟账户 {account} "
        f"的余额是1000元。"
    )


# ==================================================
# 3. 定义有副作用的工具
# ==================================================

@tool
def transfer_money(
    to_account: str,
    amount: float
) -> str:
    """
    执行一笔模拟转账。

    用户提出具体转账请求时使用。
    Python程序会在真正执行本工具前完成人工审批。
    模型只需要生成工具调用请求，不需要自行询问确认。
    """

    print(
        f"[执行高风险工具] "
        f"向 {to_account} "
        f"转账 {amount} 元"
    )

    # 只是模拟，不会发生真实转账
    return (
        "模拟转账成功："
        f"已向 {to_account} "
        f"转账 {amount} 元。"
    )


# ==================================================
# 4. 创建工具列表和工具字典
# ==================================================

tools = [
    check_balance,
    transfer_money
]


tools_by_name = {
    current_tool.name: current_tool
    for current_tool in tools
}


# 只有这些工具需要人工审批
approval_required_tools = {
    "transfer_money"
}


# ==================================================
# 5. 给模型绑定工具
# ==================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ==================================================
# 6. 接收用户请求
# ==================================================

user_input = input(
    "请输入请求："
).strip()


messages = [
    SystemMessage(
        content=(
            "你是一名模拟账户助手。"

            "当用户明确要求查询余额时，"
            "调用 check_balance。"

            "当用户提出包含收款账户和金额的"
            "具体转账请求时，"
            "直接生成 transfer_money 工具调用请求。"

            "生成工具调用请求不代表工具已经执行。"
            "Python程序会在真正执行转账工具前"
            "向用户显示具体参数并请求批准。"

            "因此，你不需要在生成 tool_call 之前"
            "再次向用户询问确认。"

            "除非用户明确要求查询余额，"
            "否则不要自行调用 check_balance。"

            "不得猜测用户没有提供的账户、"
            "金额或其他关键参数。"

            "工具返回成功结果之前，"
            "不能声称操作已经成功。"
        )
    ),
    HumanMessage(
        content=user_input
    )
]


# ==================================================
# 7. 第一次调用模型
# ==================================================

response = llm_with_tools.invoke(
    messages
)


# 保存包含 tool_calls 的 AIMessage
messages.append(
    response
)


print("\n模型正文：")
print(repr(response.content))


print("\n模型请求的工具：")
print(response.tool_calls)


# ==================================================
# 8. 处理模型请求
# ==================================================

if not response.tool_calls:
    print("\n助手：")
    print(response.content)

else:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n--------------------")

        print(
            "模型提议调用工具：",
            tool_name
        )

        print(
            "模型提议的参数：",
            tool_args
        )

        selected_tool = (
            tools_by_name[tool_name]
        )

        # ==========================================
        # 高风险工具需要人工批准
        # ==========================================

        if tool_name in approval_required_tools:
            print(
                "\n注意：该工具会产生副作用。"
            )

            print(
                "即将执行的操作："
            )

            print(
                f"工具：{tool_name}"
            )

            print(
                f"参数：{tool_args}"
            )

            approval = input(
                "是否批准执行？"
                "输入 yes 批准，"
                "输入 no 拒绝："
            ).strip().lower()

            if approval == "yes":
                tool_result = (
                    selected_tool.invoke(
                        tool_args
                    )
                )

            else:
                tool_result = (
                    "用户拒绝了本次操作。"
                    "工具没有执行，"
                    "没有产生任何修改。"
                )

        # ==========================================
        # 只读工具可以自动执行
        # ==========================================

        else:
            tool_result = (
                selected_tool.invoke(
                    tool_args
                )
            )

        print(
            "\n工具处理结果：",
            tool_result
        )

        # 无论批准还是拒绝，
        # 都必须返回对应的 ToolMessage
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )
        )


    # ==================================================
    # 9. 第二次调用模型
    # ==================================================

    final_response = (
        llm_with_tools.invoke(
            messages
        )
    )

    print("\n====================")

    print("助手：")
    print(final_response.content)
