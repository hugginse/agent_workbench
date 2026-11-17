"""确定性替身聊天模型：固定话术，返回可断言。"""

from __future__ import annotations

from collections.abc import Sequence
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
    - `scripted_responses` 非空时按序循环取用，空则返回默认话术；
    - 元素可为 `str`（纯文本）或 `AIMessage`（可携带 tool_calls）；
    - `bind_tools` 返回自身，忽略工具绑定（便于单 Agent 图测试）。
    """

    scripted_responses: list[str | AIMessage] = Field(
        default_factory=list,
        description="预置回复列表；按调用顺序循环取用",
    )

    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        """LangChain 模型类型标识。"""
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        """忽略工具绑定，返回自身（替身不真正消费 schema）。"""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """返回下一条预置回复（确定性）。"""
        if self.scripted_responses:
            item = self.scripted_responses[self._cursor % len(self.scripted_responses)]
            self._cursor += 1
        else:
            item = DEFAULT_SCRIPTED_RESPONSE

        if isinstance(item, AIMessage):
            # 每次返回副本并清除 id，避免 add_messages 按同 id 原地替换
            message = item.model_copy(deep=True)
            message.id = None
        else:
            message = AIMessage(content=item)
        return ChatResult(generations=[ChatGeneration(message=message)])
