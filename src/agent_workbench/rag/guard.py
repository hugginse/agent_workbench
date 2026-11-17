"""引用铁律与无证据拒答（Phase 4 强制执行层）。

规则（契约）：
1. 引用 id / 编号只能来自本次检索命中，模型不得生成；
2. 无证据必须拒答；
3. confidence 不得直接复制相似度分数。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from agent_workbench.rag.citation import CitationSet

__all__ = [
    "ABSTAIN_ANSWER",
    "AnswerCheck",
    "check_answer",
    "confidence_from_support",
]

# 统一拒答话术（可被上层覆盖）
ABSTAIN_ANSWER = "知识库中没有足以支持该问题的证据，无法给出可靠回答（拒答）。"

# [1] / [12] 等编号引用
_REF_PATTERN = re.compile(r"\[(\d+)\]")


class AnswerCheck(BaseModel):
    """对模型答案的引用合规检查结果。

    Attributes:
        ok: 是否通过（无非法引用且非应拒答而未拒答）。
        must_abstain: 是否必须拒答（无证据）。
        used_refs: 答案中出现且合法的引用标记。
        unsupported_refs: 答案中出现但不在命中集合的引用标记。
        fabricated_ids: 答案中出现但不在允许列表的 chunk_id 字面量。
        reason: 人类可读原因。
        confidence: 置信度（由支撑比例导出，不是 max(score)）。
    """

    ok: bool
    must_abstain: bool
    used_refs: list[str] = Field(default_factory=list)
    unsupported_refs: list[str] = Field(default_factory=list)
    fabricated_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


def confidence_from_support(
    citation_set: CitationSet,
    *,
    used_ref_count: int,
    max_score: float | None = None,
) -> float:
    """由「有效引用支撑」导出 confidence。

    **不**直接复制相似度：以合法引用覆盖率为主，相似度仅作轻微平滑，
    且封顶 0.95（避免把检索分数当成答案置信度）。

    Args:
        citation_set: 本次检索引用集合。
        used_ref_count: 答案实际使用的合法引用条数。
        max_score: 可选最高相似度（仅用于平滑，不直接输出）。

    Returns:
        [0, 1] 的 confidence。
    """
    if not citation_set.citations or used_ref_count <= 0:
        return 0.0
    coverage = min(1.0, used_ref_count / len(citation_set.citations))
    # 覆盖率为主；相似度最多贡献 0.05 的加成，且不透明传导
    smooth = 0.0
    if max_score is not None:
        smooth = max(0.0, min(float(max_score), 1.0)) * 0.05
    conf = min(0.95, coverage * 0.9 + smooth)
    return round(conf, 4)


def check_answer(
    answer: str,
    citation_set: CitationSet,
    *,
    max_score: float | None = None,
    require_citation: bool = True,
) -> AnswerCheck:
    """校验答案是否满足引用铁律与拒答规则。

    Args:
        answer: 模型（或管道）产出的答案正文。
        citation_set: 与该答案对应的检索引用集合。
        max_score: 可选最高相似度，仅用于 confidence 平滑。
        require_citation: 有命中时是否要求至少引用一条。

    Returns:
        AnswerCheck；`ok=False` 时 reason 说明违规原因。
    """
    text = answer or ""
    stripped = text.strip()

    # —— 无证据：必须拒答 ——
    if not citation_set.citations:
        # 若答案本身已是拒答话术，则合规
        abstained = (
            stripped == ABSTAIN_ANSWER
            or stripped.startswith("知识库中没有")
            or "拒答" in stripped[:40]
        )
        if abstained:
            return AnswerCheck(
                ok=True,
                must_abstain=True,
                reason="无证据且已拒答",
                confidence=0.0,
            )
        return AnswerCheck(
            ok=False,
            must_abstain=True,
            reason="无检索命中，必须拒答，不得给出实体答案",
            confidence=0.0,
        )

    allowed_ids = set(citation_set.ids)
    allowed_refs = {c.ref for c in citation_set.citations}

    used_refs: list[str] = []
    unsupported_refs: list[str] = []
    for match in _REF_PATTERN.finditer(text):
        ref = f"[{match.group(1)}]"
        if ref in allowed_refs:
            if ref not in used_refs:
                used_refs.append(ref)
        else:
            if ref not in unsupported_refs:
                unsupported_refs.append(ref)

    # 也允许直接写 chunk_id；字面出现但不在白名单 → 视为编造风险
    fabricated: list[str] = []
    # 抽取形如 uuid / 常见 id 的长 token 成本高；仅检查：文本中若出现
    # 「id=xxx」或「chunk_id」字段值且不在白名单
    for match in re.finditer(r"(?:chunk_id|id)\s*[:=]\s*([A-Za-z0-9._-]{6,})", text):
        candidate = match.group(1)
        if candidate not in allowed_ids:
            fabricated.append(candidate)

    # 先判定非法引用（越界编号 / 编造 id）
    if unsupported_refs or fabricated:
        return AnswerCheck(
            ok=False,
            must_abstain=False,
            used_refs=used_refs,
            unsupported_refs=unsupported_refs,
            fabricated_ids=fabricated,
            reason=(
                "引用越界：仅允许使用本次检索命中的引用；"
                f"非法编号={unsupported_refs or '无'}，"
                f"可疑 id={fabricated or '无'}"
            ),
            confidence=0.0,
        )

    # 未引用编号却给出实体答案（要求引用时）
    if require_citation and not used_refs:
        # 允许答案里只有 chunk_id 直接引用
        direct = [cid for cid in allowed_ids if cid in text]
        if not direct:
            return AnswerCheck(
                ok=False,
                must_abstain=False,
                reason="有检索命中但答案未引用任何合法引用标记",
                confidence=0.0,
            )

    conf = confidence_from_support(
        citation_set,
        used_ref_count=len(used_refs) or len([cid for cid in allowed_ids if cid in text]),
        max_score=max_score,
    )
    if conf <= 0.0:
        return AnswerCheck(
            ok=False,
            must_abstain=False,
            used_refs=used_refs,
            reason="未能建立有效引用支撑",
            confidence=0.0,
        )

    return AnswerCheck(
        ok=True,
        must_abstain=False,
        used_refs=used_refs,
        reason="引用均来自检索命中",
        confidence=conf,
    )


def enforce_answer(
    answer: str,
    citation_set: CitationSet,
    *,
    max_score: float | None = None,
) -> tuple[str, AnswerCheck]:
    """强制执行铁律：不合规则改写为拒答。

    Args:
        answer: 原始答案。
        citation_set: 引用集合。
        max_score: 可选相似度。

    Returns:
        (最终答案, AnswerCheck)。
    """
    check = check_answer(answer, citation_set, max_score=max_score)
    if check.ok:
        return answer, check
    # 无证据或引用违规 → 拒答（不输出带非法引用的正文）
    return ABSTAIN_ANSWER, check


def summarize_check(check: AnswerCheck) -> dict[str, Any]:
    """序列化检查结果（便于日志/指标，不含隐藏推理）。"""
    return {
        "ok": check.ok,
        "must_abstain": check.must_abstain,
        "used_refs": check.used_refs,
        "unsupported_refs": check.unsupported_refs,
        "fabricated_ids": check.fabricated_ids,
        "reason": check.reason,
        "confidence": check.confidence,
    }
