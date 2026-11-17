#!/usr/bin/env python3
"""评测 CLI：`uv run python evals/run_evaluation.py [--mode single|multi]`。

默认使用确定性替身模型（离线可复现）；`--json-out` 写出完整结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接以脚本运行：把仓库根与 src 加入 path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数。"""
    p = argparse.ArgumentParser(description="agent_workbench 评测运行器")
    p.add_argument(
        "--mode",
        choices=("single", "multi"),
        default="multi",
        help="single=单 Agent 基线；multi=Supervisor（默认）",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="JSONL 数据集路径（默认 evals/datasets/tasks_v1.jsonl）",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="将结果 JSON 写入该路径",
    )
    p.add_argument("--max-steps", type=int, default=8, help="工具轮次上限")
    p.add_argument(
        "--max-repair-attempts",
        type=int,
        default=2,
        help="Supervisor 回修上限",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """入口：跑数据集并打印/写出指标。

    Returns:
        进程退出码（0 正常）。
    """
    args = _build_parser().parse_args(argv)

    from agent_workbench.harness import load_dataset, run_dataset
    from agent_workbench.infrastructure.llm import ScriptedChatModel
    from agent_workbench.main import build_default_registry

    tasks = load_dataset(args.dataset)
    # 确定性替身：空脚本 → 默认话术，全程可复现
    llm = ScriptedChatModel(scripted_responses=[])
    registry = build_default_registry()

    runs, metrics = run_dataset(
        tasks,
        llm,
        registry,
        mode=args.mode,
        max_steps=args.max_steps,
        max_repair_attempts=args.max_repair_attempts,
    )

    payload = {
        "mode": args.mode,
        "metrics": metrics.model_dump(),
        "runs": [r.model_dump() for r in runs],
    }
    print(json.dumps(metrics.model_dump(), ensure_ascii=False, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已写入 {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
