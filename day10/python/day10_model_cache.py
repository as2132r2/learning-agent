"""Day10 第六步：使用SQLite持久化缓存复用模型回答。"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.caches import BaseCache, RETURN_VAL_TYPE
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 一个只服务于聊天模型的简单SQLite缓存
# ============================================================

class SQLiteChatCache(BaseCache):
    """用prompt和模型配置作为键，保存ChatGeneration。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.lookup_count = 0
        self.hit_count = 0
        self.update_count = 0

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    generations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @staticmethod
    def make_key(prompt: str, llm_string: str) -> str:
        """相同提示词和相同模型配置一定生成相同键。"""

        raw_key = f"{prompt}\n---MODEL---\n{llm_string}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def lookup(
        self,
        prompt: str,
        llm_string: str
    ) -> RETURN_VAL_TYPE | None:
        self.lookup_count += 1
        cache_key = self.make_key(prompt, llm_string)

        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT generations_json
                FROM llm_cache
                WHERE cache_key = ?
                """,
                (cache_key,)
            ).fetchone()

        if row is None:
            print("[缓存] MISS：没有找到相同请求")
            return None

        self.hit_count += 1
        print("[缓存] HIT：直接读取SQLite，不请求模型")
        saved_items = json.loads(row[0])

        generations: list[ChatGeneration] = []

        for item in saved_items:
            message = AIMessage.model_validate(item["message"])
            generations.append(ChatGeneration(
                message=message,
                generation_info=item.get("generation_info")
            ))

        return generations

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: RETURN_VAL_TYPE
    ) -> None:
        self.update_count += 1
        cache_key = self.make_key(prompt, llm_string)
        saved_items: list[dict[str, Any]] = []

        for generation in return_val:
            if not isinstance(generation, ChatGeneration):
                raise TypeError(
                    "SQLiteChatCache只支持ChatGeneration。"
                )

            saved_items.append({
                "message": generation.message.model_dump(
                    mode="json"
                ),
                "generation_info": generation.generation_info
            })

        generations_json = json.dumps(
            saved_items,
            ensure_ascii=False
        )

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_cache (
                    cache_key,
                    generations_json
                )
                VALUES (?, ?)
                """,
                (cache_key, generations_json)
            )

        print("[缓存] WRITE：模型回答已经写入SQLite")

    def clear(self, **kwargs: Any) -> None:
        del kwargs

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM llm_cache")

        print("[缓存] CLEAR：缓存已经清空")

    def entry_count(self) -> int:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM llm_cache"
            ).fetchone()

        return int(row[0]) if row is not None else 0


# ============================================================
# 2. 创建带持久化缓存的真实模型
# ============================================================

load_dotenv()

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent
cache_database = (
    project_directory
    / "day10"
    / "database"
    / "day10_llm_cache.db"
)

model_cache = SQLiteChatCache(cache_database)

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=20,
    max_retries=0,

    # ChatOpenAI会自动执行lookup()和update()。
    cache=model_cache
)


# ============================================================
# 3. 调用并记录耗时
# ============================================================

def ask(question: str) -> None:
    print("\n" + "-" * 60)
    print("问题：", question)

    started_at = time.perf_counter()
    response = llm.invoke(question)
    elapsed = time.perf_counter() - started_at

    print("回答：", response.content)
    print(f"本次耗时：{elapsed:.4f} 秒")


print("SQLite缓存：", cache_database)
print("启动时已有缓存数量：", model_cache.entry_count())

question = "请用一句话解释什么是模型缓存。"

# 第一次运行通常MISS并调用模型；随后写入SQLite。
ask(question)

# 提示词和模型配置完全相同，应该直接HIT。
ask(question)

# 只增加一个问号，也会成为不同的精确缓存键。
ask("请用一句话解释什么是模型缓存？")


# ============================================================
# 4. 显示本次运行的缓存统计
# ============================================================

print("\n" + "=" * 60)
print("本次缓存查询次数：", model_cache.lookup_count)
print("本次缓存命中次数：", model_cache.hit_count)
print("本次缓存写入次数：", model_cache.update_count)
print("数据库缓存总数：", model_cache.entry_count())

if model_cache.lookup_count:
    hit_rate = (
        model_cache.hit_count
        / model_cache.lookup_count
        * 100
    )
    print(f"本次缓存命中率：{hit_rate:.1f}%")


# ============================================================
# 数据流
# ============================================================
#
# question + llm_string
#          ↓ SHA-256
#       cache_key
#          ↓
# SQLiteChatCache.lookup()
#    ├── HIT → ChatGeneration → AIMessage（不请求模型）
#    └── MISS
#          ↓
#       llm.invoke()
#          ↓
#       AIMessage
#          ↓
# SQLiteChatCache.update()
#          ↓
# 下次相同请求可以直接读取
#
# 缓存键不仅包含问题，也包含模型配置llm_string。
# 更换模型、温度或其他调用参数后，通常不会命中旧缓存。
