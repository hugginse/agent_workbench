"""编排 Prompt（与业务代码分离，可单独测试）。"""

from __future__ import annotations

from agent_workbench.orchestrator.state import SubTask

__all__ = [
    "build_aggregate_prompt",
    "build_execute_prompt",
    "build_plan_prompt",
    "build_review_prompt",
]

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


_REVIEW_TEMPLATE = """你是评审员。请根据总任务与各子任务产出给出评审结论。

## 总任务
{task_description}

## 子任务与产出
{results_block}

## 评审要求
- 全部子任务产出满足要求才 passed=true
- passed=false 时必须给出 feedback，并指定需要回修的 target_subtask_id（须为上面出现过的 id）
- 不得编造不存在的子任务 id
"""


def build_review_prompt(task_description: str, results: dict[str, str]) -> str:
    """构造 review 节点提示词。

    Args:
        task_description: 总任务描述。
        results: 子任务 id → 执行结果。

    Returns:
        送入模型的完整提示词。
    """
    if results:
        lines = [f"### {sid}\n{content}" for sid, content in results.items()]
        results_block = "\n\n".join(lines)
    else:
        results_block = "（尚无产出）"
    return _REVIEW_TEMPLATE.format(
        task_description=task_description.strip(),
        results_block=results_block,
    )


_AGGREGATE_TEMPLATE = """你是汇总助手。请将各子任务产出整合为一份完整、连贯的最终交付。

## 总任务
{task_description}

## 子任务产出
{results_block}

要求：
- 保留各子任务的有效结论，消除重复
- 以总任务为目标输出可直接使用的最终正文
- 仅依据给定产出，不编造新事实
"""


def build_aggregate_prompt(task_description: str, results: dict[str, str]) -> str:
    """构造 aggregate 节点汇总提示词。

    Args:
        task_description: 总任务描述。
        results: 子任务 id → 执行结果。

    Returns:
        送入模型的完整提示词。
    """
    if results:
        lines = [f"### {sid}\n{content}" for sid, content in results.items()]
        results_block = "\n\n".join(lines)
    else:
        results_block = "（尚无产出）"
    return _AGGREGATE_TEMPLATE.format(
        task_description=task_description.strip(),
        results_block=results_block,
    )
