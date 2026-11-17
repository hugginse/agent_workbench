"""编排 Prompt（与业务代码分离，可单独测试）。"""

from __future__ import annotations

from agent_workbench.orchestrator.state import SubTask

__all__ = ["build_execute_prompt", "build_plan_prompt"]

_PLAN_TEMPLATE = """你是一名任务规划专家。请将用户任务拆解为可独立执行的子任务。

要求：
- 子任务数量 1~8 个，覆盖任务全部必要环节
- 每个子任务 id 为短标识（如 s1、s2），互不重复
- description 具体、可执行，避免空话
- 若任务本身已足够小，只返回 1 个子任务

用户任务：
{task_description}
"""

_EXECUTE_TEMPLATE = """## 总任务
{task_description}

## 当前子任务
- id: {subtask_id}
- 描述: {subtask_description}
{feedback_section}

请只针对「当前子任务」给出可交付的结果正文。
"""

_REPAIR_SECTION = """
## 回修反馈（必须处理）
{feedback}
"""


def build_plan_prompt(task_description: str) -> str:
    """构造 plan 节点提示词。

    Args:
        task_description: 原始任务描述。

    Returns:
        送入模型的完整提示词。
    """
    return _PLAN_TEMPLATE.format(task_description=task_description.strip())


def build_execute_prompt(
    task_description: str,
    subtask: SubTask,
    feedback: str | None = None,
) -> str:
    """构造 execute 节点中单个 SubAgent 的用户提示词。

    Args:
        task_description: 总任务描述。
        subtask: 当前子任务。
        feedback: 回修反馈；首次执行为 None。

    Returns:
        用户消息正文。
    """
    if feedback and feedback.strip():
        feedback_section = _REPAIR_SECTION.format(feedback=feedback.strip())
    else:
        feedback_section = ""
    return _EXECUTE_TEMPLATE.format(
        task_description=task_description.strip(),
        subtask_id=subtask.id,
        subtask_description=subtask.description.strip(),
        feedback_section=feedback_section,
    )
