"""SubAgent 声明：prompt + 工具子集 + 输出 schema（组合式配置）。"""

from agent_workbench.agents.specs import (
    EXECUTOR,
    RESEARCHER,
    REVIEWER,
    SUBAGENTS,
    SubAgentSpec,
    SubTaskResult,
    get_subagent,
)

__all__ = [
    "EXECUTOR",
    "RESEARCHER",
    "REVIEWER",
    "SUBAGENTS",
    "SubAgentSpec",
    "SubTaskResult",
    "get_subagent",
]
