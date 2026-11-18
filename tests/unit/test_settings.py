"""Settings 与聊天模型工厂（替身 / 真实双路径）单测。"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from agent_workbench.infrastructure.llm import (
    DEFAULT_SCRIPTED_RESPONSE,
    ScriptedChatModel,
    get_chat_model,
)
from config.settings import Settings, get_settings


def _settings(**overrides: object) -> Settings:
    """构建隔离于 .env 的测试配置；默认 dev + 空凭据，避免读到进程环境。"""
    base: dict[str, object] = {
        "app_env": "dev",
        "api_key": "",
        "base_url": "",
        "embedding_api_key": "",
        "embedding_base_url": "",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


class TestSettings:
    """Settings 字段默认值与替身触发条件。"""

    def test_defaults(self) -> None:
        """无环境覆盖时使用契约默认值。"""
        s = _settings()
        assert s.app_env == "dev"
        assert s.chat_model == ""
        assert s.embedding_model == ""
        assert s.port == 8000
        assert s.langsmith_tracing is False

    def test_env_style_field_names(self) -> None:
        """字段名与 .env 中 APP_ENV / CHAT_MODEL 等键对应（不区分大小写）。"""
        s = Settings(_env_file=None, APP_ENV="test", CHAT_MODEL="gpt-4o-mini")
        assert s.app_env == "test"
        assert s.chat_model == "gpt-4o-mini"

    def test_scripted_when_app_env_test(self) -> None:
        """APP_ENV=test 即使配置了 CHAT_MODEL 也强制替身。"""
        s = _settings(app_env="test", chat_model="gpt-4o-mini")
        assert s.use_scripted_chat_model is True

    def test_scripted_when_chat_model_empty(self) -> None:
        """dev 环境未配置 CHAT_MODEL 时使用替身。"""
        s = _settings(app_env="dev", chat_model="")
        assert s.use_scripted_chat_model is True

    def test_scripted_when_chat_model_blank(self) -> None:
        """仅空白字符的 CHAT_MODEL 视为未配置。"""
        s = _settings(app_env="dev", chat_model="   ")
        assert s.use_scripted_chat_model is True

    def test_real_when_chat_model_configured(self) -> None:
        """dev + 非空 CHAT_MODEL 走真实路径。"""
        s = _settings(app_env="dev", chat_model="gpt-4o-mini")
        assert s.use_scripted_chat_model is False

    def test_get_settings_returns_new_instance(self) -> None:
        """get_settings 每次返回新实例，避免全局可变单例。"""
        a = get_settings()
        b = get_settings()
        assert a is not b
        assert type(a) is Settings


class TestGetChatModel:
    """get_chat_model 双路径。"""

    def test_scripted_in_test_env(self) -> None:
        """APP_ENV=test → ScriptedChatModel。"""
        model = get_chat_model(_settings(app_env="test", chat_model="gpt-4o-mini"))
        assert isinstance(model, ScriptedChatModel)

    def test_scripted_when_model_unconfigured(self) -> None:
        """未配置 CHAT_MODEL → ScriptedChatModel。"""
        model = get_chat_model(_settings(app_env="dev", chat_model=""))
        assert isinstance(model, ScriptedChatModel)

    def test_scripted_is_deterministic(self) -> None:
        """替身模型同输入序列产生同输出序列。"""
        model = get_chat_model(_settings(app_env="test"))
        assert isinstance(model, ScriptedChatModel)
        from langchain_core.messages import HumanMessage

        out1 = model.invoke([HumanMessage(content="任意问题")])
        out2 = model.invoke([HumanMessage(content="任意问题")])
        assert out1.content == DEFAULT_SCRIPTED_RESPONSE
        assert out2.content == DEFAULT_SCRIPTED_RESPONSE

    def test_scripted_cycles_scripted_responses(self) -> None:
        """预置多条回复时按序循环，保证可断言。"""
        model = ScriptedChatModel(scripted_responses=["甲", "乙"])
        from langchain_core.messages import HumanMessage

        msgs = [HumanMessage(content="q")]
        assert model.invoke(msgs).content == "甲"
        assert model.invoke(msgs).content == "乙"
        assert model.invoke(msgs).content == "甲"

    def test_real_path_returns_chat_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """dev + CHAT_MODEL → ChatOpenAI（构造期不发起网络调用）。"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        model = get_chat_model(_settings(app_env="dev", chat_model="gpt-4o-mini"))
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"

    def test_real_path_passes_api_key_and_base_url(self) -> None:
        """API_KEY / BASE_URL 写入 ChatOpenAI 客户端配置。"""
        model = get_chat_model(
            _settings(
                app_env="dev",
                chat_model="gpt-4o-mini",
                api_key="sk-unit-test",
                base_url="https://gateway.example.com/v1",
            )
        )
        assert isinstance(model, ChatOpenAI)
        # SecretStr 不在日志/断言里展开明文比较方式：取 secret 值确认注入
        secret = model.openai_api_key
        assert secret is not None
        assert secret.get_secret_value() == "sk-unit-test"
        assert str(model.openai_api_base).startswith("https://gateway.example.com")


class TestApiKeyBaseUrlSettings:
    """api_key / base_url 配置与 Embedding 回退。"""

    def test_defaults_empty(self) -> None:
        s = _settings()
        assert s.api_key == ""
        assert s.base_url == ""
        assert s.embedding_api_key == ""
        assert s.embedding_base_url == ""

    def test_env_style_keys(self) -> None:
        s = Settings(
            _env_file=None,
            API_KEY="sk-env",
            BASE_URL="https://gw.example/v1",
            EMBEDDING_API_KEY="emb-key",
            EMBEDDING_BASE_URL="https://emb.example/v1",
        )
        assert s.api_key == "sk-env"
        assert s.base_url == "https://gw.example/v1"
        assert s.resolved_embedding_api_key() == "emb-key"
        assert s.resolved_embedding_base_url() == "https://emb.example/v1"

    def test_embedding_fallback_to_common(self) -> None:
        s = _settings(api_key="sk-common", base_url="https://common/v1")
        assert s.resolved_embedding_api_key() == "sk-common"
        assert s.resolved_embedding_base_url() == "https://common/v1"

    def test_embeddings_factory_fake(self) -> None:
        from agent_workbench.infrastructure.embeddings import FakeEmbeddings, get_embeddings

        emb = get_embeddings(_settings(app_env="test", embedding_model="text-embedding-3-small"))
        assert isinstance(emb, FakeEmbeddings)

    def test_embeddings_factory_real_with_key_url(self) -> None:
        from agent_workbench.infrastructure.embeddings import get_embeddings

        emb = get_embeddings(
            _settings(
                app_env="dev",
                embedding_model="text-embedding-3-small",
                api_key="sk-emb",
                base_url="https://gw.example/v1",
            )
        )
        # 构造期不联网
        assert emb is not None
        assert not isinstance(emb, type(get_embeddings(_settings(app_env="test"))))
