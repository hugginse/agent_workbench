"""LLM 适配：工厂与确定性替身。"""

from agent_workbench.infrastructure.llm.factory import get_chat_model
from agent_workbench.infrastructure.llm.scripted import DEFAULT_SCRIPTED_RESPONSE, ScriptedChatModel

__all__ = ["DEFAULT_SCRIPTED_RESPONSE", "ScriptedChatModel", "get_chat_model"]
