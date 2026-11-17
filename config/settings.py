"""应用配置入口。

唯一读取 `.env` / 环境变量的地方；业务代码禁止直接调用 `os.getenv()`，
一律经本模块的 `Settings` 访问配置。
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings"]


class Settings(BaseSettings):
    """应用配置模型。

    字段与 `.env.example` 对齐；未填写的项走默认值。
    替身触发条件：`APP_ENV=test` 或未配置 `CHAT_MODEL`。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = Field(default="dev", description="运行环境：dev | test | prod")
    chat_model: str = Field(default="", description="聊天模型名；留空则使用替身")
    embedding_model: str = Field(default="", description="Embedding 模型名；留空则使用替身")
    host: str = Field(default="127.0.0.1", description="服务监听地址")
    port: int = Field(default=8000, description="服务监听端口")
    log_level: str = Field(default="INFO", description="日志级别")
    langsmith_tracing: bool = Field(default=False, description="LangSmith 追踪开关，默认关闭")
    langsmith_api_key: str = Field(default="", description="LangSmith API Key（可空）")

    @property
    def use_scripted_chat_model(self) -> bool:
        """是否启用确定性替身聊天模型。

        APP_ENV=test 或未配置 CHAT_MODEL 时为 True（见 CLAUDE.md「测试替身与离线可复现」）。
        """
        return self.app_env == "test" or not self.chat_model.strip()

    @property
    def use_fake_embeddings(self) -> bool:
        """是否启用确定性替身 Embedding（条件与聊天模型一致）。"""
        return self.app_env == "test" or not self.embedding_model.strip()


def get_settings() -> Settings:
    """从环境变量与 `.env` 构建配置。

    每次调用返回新实例，便于测试覆盖与依赖注入；不做全局可变缓存。
    """
    return Settings()
