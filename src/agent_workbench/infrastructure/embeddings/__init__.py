"""Embedding 替身与工厂。"""

from agent_workbench.infrastructure.embeddings.factory import get_embeddings
from agent_workbench.infrastructure.embeddings.fake import DEFAULT_DIM, FakeEmbeddings

__all__ = ["DEFAULT_DIM", "FakeEmbeddings", "get_embeddings"]
