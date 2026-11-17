"""上下文裁剪与 token 估算（确定性、离线、无外部依赖）。"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

__all__ = [
    "DEFAULT_MAX_CONTEXT_MESSAGES",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_tokens",
    "merge_token_usage",
    "prune_messages",
]

# 单次送入模型的最大消息条数（含 system）
DEFAULT_MAX_CONTEXT_MESSAGES = 24


def estimate_tokens(text: str) -> int:
    """近似 token 数（确定性启发式，非真实 tokenizer）。

    - CJK 统一表意文字：约 1 字 = 1 token
    - 其他字符：约 4 字符 = 1 token

    Args:
        text: 任意文本。

    Returns:
        >= 0 的近似 token 数；空串为 0。
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        code = ord(ch)
        # CJK 及全角标点大致区间
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3000 <= code <= 0x303F
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk += 1
        else:
            other += 1
    return cjk + math.ceil(other / 4)


def estimate_message_tokens(message: BaseMessage) -> int:
    """单条消息近似 token（正文 + tool_calls 序列化）。"""
    content = message.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    total = estimate_tokens(content)
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        total += estimate_tokens(json.dumps(call, ensure_ascii=False, default=str))
    name = getattr(message, "name", None)
    if name:
        total += estimate_tokens(str(name))
    return total


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    """消息列表近似 token 总和。"""
    return sum(estimate_message_tokens(m) for m in messages)


def prune_messages(
    messages: Sequence[BaseMessage],
    *,
    max_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
    keep_system: bool = True,
) -> list[BaseMessage]:
    """按节点裁剪历史消息：保留头部 system + 最近若干条。

    Args:
        messages: 原始消息序列（顺序为时间序）。
        max_messages: 裁剪后最大条数（含 system），必须 >= 2。
        keep_system: 是否固定保留首条 SystemMessage。

    Returns:
        新列表（不修改入参）；不超过 max_messages。
    """
    if max_messages < 2:
        msg = f"max_messages 必须 >= 2，当前 {max_messages}"
        raise ValueError(msg)
    seq = list(messages)
    if len(seq) <= max_messages:
        return seq

    system: list[BaseMessage] = []
    rest = seq
    if keep_system and seq and isinstance(seq[0], SystemMessage):
        system = [seq[0]]
        rest = seq[1:]

    budget = max_messages - len(system)
    if budget <= 0:
        # 极端：只留 system
        return system[:1]
    kept = rest[-budget:]
    return system + kept


def merge_token_usage(
    current: dict[str, int] | None,
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, int]:
    """合并节点 token 增量，返回新的 usage 字典（不修改入参）。

    Args:
        current: 已有 usage（input/output/total）；None 视为全 0。
        input_tokens: 本节点输入增量。
        output_tokens: 本节点输出增量。

    Returns:
        合并后的 `{"input", "output", "total"}`。
    """
    base = current or {}
    prev_in = int(base.get("input", 0))
    prev_out = int(base.get("output", 0))
    new_in = prev_in + max(0, input_tokens)
    new_out = prev_out + max(0, output_tokens)
    return {"input": new_in, "output": new_out, "total": new_in + new_out}


def estimate_any_tokens(value: Any) -> int:
    """任意可转文本对象的近似 token。"""
    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_tokens(value)
    if isinstance(value, BaseMessage):
        return estimate_message_tokens(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(estimate_any_tokens(v) for v in value)
    return estimate_tokens(json.dumps(value, ensure_ascii=False, default=str))
