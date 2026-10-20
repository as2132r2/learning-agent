"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Route(str, Enum):
    """限定毕业项目支持的四条路线。"""

    CHAT = "chat"
    CALCULATE = "calculate"
    KNOWLEDGE = "knowledge"
    SAVE_NOTE = "save_note"


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not all(character.isascii() and (character.isalnum() or character in "-_") for character in value):
            raise ValueError("会话编号只允许英文字母、数字、短横线和下划线。")
        return value


class Evidence(BaseModel):
    """检索资料的结构；引用以chunk_id定位。"""

    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content: str = Field(min_length=1)

