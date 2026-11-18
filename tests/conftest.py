"""pytest 全局夹具：强制离线替身，避免 import main 时构造真实 LLM。"""

from __future__ import annotations

import os

# 必须在任何业务模块 import 之前生效（conftest 最先加载）
os.environ["APP_ENV"] = "test"
# 清掉可能干扰的模型名，确保 use_scripted_chat_model 为 True
os.environ.pop("CHAT_MODEL", None)
os.environ.pop("EMBEDDING_MODEL", None)
