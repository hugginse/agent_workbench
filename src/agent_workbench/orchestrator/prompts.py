"""编排 Prompt（与业务代码分离，可单独测试）。"""

from __future__ import annotations

__all__ = ["build_plan_prompt"]

_PLAN_TEMPLATE = """你是一名任务规划专家。请将用户任务拆解为可独立执行的子任务。

要求：
- 子任务数量 1~8 个，覆盖任务全部必要环节
- 每个子任务 id 为短标识（如 s1、s2），互不重复
- description 具体、可执行，避免空话
- 若任务本身已足够小，只返回 1 个子任务

用户任务：
{task_description}
"""


def build_plan_prompt(task_description: str) -> str:
    """构造 plan 节点提示词。

    Args:
        task_description: 原始任务描述。

    Returns:
        送入模型的完整提示词。
    """
    return _PLAN_TEMPLATE.format(task_description=task_description.strip())
