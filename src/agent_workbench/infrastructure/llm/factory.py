"""聊天模型集中工厂：真实模型 / 确定性替身双路径。

LLM 只经本工厂或依赖注入创建；模块 import 时不构造模型、不建网络连接。
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from agent_workbench.infrastructure.llm.scripted import DEFAULT_SCRIPTED_RESPONSE, ScriptedChatModel
from config.settings import Settings, get_settings

__all__ = ["get_chat_model"]


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """获取聊天模型实例。

    - 替身路径：`APP_ENV=test` 或未配置 `CHAT_MODEL` → `ScriptedChatModel`；
    - 真实路径：已配置 `CHAT_MODEL` 且非 test 环境 → 按需构造真实客户端。

    Args:
        settings: 显式配置；为 None 时从环境与 `.env` 读取。

    Returns:
        可用的 `BaseChatModel` 实例。
    """
    if settings is None:
        settings = get_settings()

    if settings.use_scripted_chat_model:
        return ScriptedChatModel(
            scripted_responses=[DEFAULT_SCRIPTED_RESPONSE],
        )

    return _create_real_chat_model(settings)


def _create_real_chat_model(settings: Settings) -> BaseChatModel:
    """构造真实聊天模型（延迟 import，避免加载期昂贵初始化）。

    Args:
        settings: 含 `chat_model` 的应用配置。

    Returns:
        真实 `BaseChatModel` 实例。

    Raises:
        ValueError: 模型名为空（正常情况下替身分支已拦截）。
    """
    model_name = settings.chat_model.strip()
    if not model_name:
        msg = "chat_model 为空，无法构造真实模型"
        raise ValueError(msg)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=0)
