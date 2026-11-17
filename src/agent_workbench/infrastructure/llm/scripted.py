"""确定性替身聊天模型：固定话术，返回可断言。"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

__all__ = ["DEFAULT_SCRIPTED_RESPONSE", "ScriptedChatModel"]

DEFAULT_SCRIPTED_RESPONSE = "（确定性替身固定回复）"


class ScriptedChatModel(BaseChatModel):
    """固定话术的确定性替身聊天模型。

    - 同输入序列 → 同输出序列，保证离线评估可复现；
    - `scripted_responses` 非空时按序循环返回，空则返回默认话术。
    """

    scripted_responses: list[str] = Field(
        default_factory=list,
        description="预置回复列表；按调用顺序循环取用",
    )

    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        """LangChain 模型类型标识。"""
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """返回下一条预置回复（确定性）。"""
        if self.scripted_responses:
            content = self.scripted_responses[self._cursor % len(self.scripted_responses)]
            self._cursor += 1
        else:
            content = DEFAULT_SCRIPTED_RESPONSE
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
