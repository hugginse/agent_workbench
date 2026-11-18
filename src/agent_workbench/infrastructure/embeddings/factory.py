"""Embedding 集中工厂：Fake 替身 / OpenAI 兼容真实路径。"""

from __future__ import annotations

from typing import Any

from agent_workbench.infrastructure.embeddings.fake import FakeEmbeddings
from config.settings import Settings, get_settings

__all__ = ["get_embeddings"]


def get_embeddings(settings: Settings | None = None) -> Any:
    """获取 Embeddings 实例。

    - 替身：`APP_ENV=test` 或未配置 `EMBEDDING_MODEL` → `FakeEmbeddings`；
    - 真实：`OpenAIEmbeddings`（支持 api_key / base_url，embedding 专用字段可覆盖）。

    Args:
        settings: 显式配置；None 时从环境读取。

    Returns:
        具有 embed_documents / embed_query 的对象。
    """
    if settings is None:
        settings = get_settings()

    if settings.use_fake_embeddings:
        return FakeEmbeddings()

    return _create_real_embeddings(settings)


def _create_real_embeddings(settings: Settings) -> Any:
    """构造真实 Embedding（延迟 import）。

    Args:
        settings: 应用配置。

    Returns:
        OpenAIEmbeddings 实例。

    Raises:
        ValueError: embedding_model 为空。
    """
    model_name = settings.embedding_model.strip()
    if not model_name:
        msg = "embedding_model 为空，无法构造真实 Embedding"
        raise ValueError(msg)

    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {"model": model_name}
    api_key = settings.resolved_embedding_api_key()
    if api_key:
        kwargs["api_key"] = api_key
    base_url = settings.resolved_embedding_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)
