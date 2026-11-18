#!/usr/bin/env python3
"""评测 CLI：`uv run python evals/run_evaluation.py [--mode single|multi|compare]`。

- 默认按 `.env` 构造真实模型（未配置则自动替身）；
- `--scripted` 强制确定性替身（离线可复现）；
- `--json-out` 写出完整结果；`compare` 输出 A/B Markdown 表。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
        choices=("single", "multi", "compare"),
        default="multi",
        help="single=单 Agent；multi=Supervisor；compare=A/B 对比（默认 multi）",
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
    p.add_argument(
        "--scripted",
        action="store_true",
        help="强制使用确定性替身（忽略 .env 中的真实模型）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """入口：跑数据集并打印/写出指标。

    Returns:
        进程退出码（0 正常）。
    """
    args = _build_parser().parse_args(argv)

    from agent_workbench.harness import load_dataset, run_dataset
    from agent_workbench.harness.compare import (
        compare_modes,
        format_comparison_markdown,
    )
    from agent_workbench.infrastructure.llm import ScriptedChatModel, get_chat_model
    from agent_workbench.main import build_default_registry
    from config.settings import get_settings

    tasks = load_dataset(args.dataset)
    registry = build_default_registry()

    def make_llm() -> Any:
        """每次新建模型实例，保证 single/multi 起点一致。"""
        if args.scripted:
            return ScriptedChatModel(scripted_responses=[])
        return get_chat_model(get_settings())

    if args.mode == "compare":
        report = compare_modes(
            tasks,
            make_llm,
            registry,
            max_steps=args.max_steps,
            max_repair_attempts=args.max_repair_attempts,
        )
        payload = {
            "mode": "compare",
            "scripted": bool(args.scripted),
            "single": report.single.model_dump(),
            "multi": report.multi.model_dump(),
            "deltas": report.deltas,
            "single_runs": [r.model_dump() for r in report.single_runs],
            "multi_runs": [r.model_dump() for r in report.multi_runs],
        }
        print(format_comparison_markdown(report))
        print()
        print(
            json.dumps(
                {
                    "single": report.single.model_dump(),
                    "multi": report.multi.model_dump(),
                    "deltas": report.deltas,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        llm = make_llm()
        runs, metrics = run_dataset(
            tasks,
            llm,
            registry,
            mode=args.mode,  # type: ignore[arg-type]
            max_steps=args.max_steps,
            max_repair_attempts=args.max_repair_attempts,
        )
        payload = {
            "mode": args.mode,
            "scripted": bool(args.scripted),
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
