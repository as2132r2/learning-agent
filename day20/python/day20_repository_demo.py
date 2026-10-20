"""Day20 第二步：调用 ChatRepository 的完整演示程序。

运行：python day20/python/day20_repository_demo.py
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

from day20_chat_repository import ChatRepository


def main() -> None:
    repository = ChatRepository()
    repository.initialize()

    client_id = "browser-demo-" + uuid.uuid4().hex[:8]
    session_id = "session-demo-" + uuid.uuid4().hex[:10]

    print("Day20 Repository 演示")
    print("数据库：", repository.database_path)
    print("Repository类型：", type(repository))
    print("=" * 70)

    session = repository.create_session(
        session_id=session_id,
        client_id=client_id,
        title="尚未命名的对话",
    )
    print("创建会话：")
    print(asdict(session))
    print("返回类型：", type(session))

    first_turn = repository.append_turn(
        session_id=session_id,
        user_content="我叫小林，正在学习 Repository。",
        assistant_content="你好小林，Repository 用于集中管理数据访问。",
    )
    second_turn = repository.append_turn(
        session_id=session_id,
        user_content="它为什么比到处写 SQL 更好？",
        assistant_content="因为它隔离了业务逻辑和数据库细节，更容易测试和维护。",
    )

    print("\n第一次 append_turn() 返回：")
    for message in first_turn:
        print(asdict(message))

    print("\n第二次 append_turn() 返回：")
    for message in second_turn:
        print(asdict(message))

    renamed = repository.rename_session(
        session_id,
        "学习 Repository 数据访问层",
    )
    print("\n修改标题是否成功：", renamed)

    loaded_session = repository.get_session(session_id)
    print("重新查询的会话：")
    print(asdict(loaded_session) if loaded_session else None)

    messages = repository.load_messages(session_id, limit=20)
    print("\n按时间顺序加载消息：")
    for index, message in enumerate(messages, start=1):
        print(
            f"{index}. id={message.message_id} "
            f"role={message.role} content={message.content}"
        )

    sessions = repository.list_sessions(client_id)
    print("\n当前 client_id 的会话列表：")
    for item in sessions:
        print(asdict(item))

    print("\n统计：")
    print("会话数量：", len(sessions))
    print("消息数量：", repository.count_messages(session_id))

    print("\n" + "=" * 70)
    print("调用关系：")
    print("业务代码 → ChatRepository方法 → 参数化SQL → SQLite")
    print("SQLite Row → ChatSession/ChatMessage → 业务代码")


if __name__ == "__main__":
    main()
