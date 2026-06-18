"""
将编排后 DAG 评估（结构 + 任务 LLM）的失败信息写回 understanding 落盘，
供下一轮三阶段编排在前两阶段显式读取，避免无反馈地重复 orchestrate。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HISTORY_MAX = 8
_ISSUE_LINE_MAX = 380
_PILOT_FEEDBACK_HISTORY_MAX = 24


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _issue_to_line(item: Any) -> str:
    if isinstance(item, dict):
        s = json.dumps(item, ensure_ascii=False)
    else:
        s = str(item)
    if len(s) > _ISSUE_LINE_MAX:
        return s[: _ISSUE_LINE_MAX - 1] + "…"
    return s


def _compact_for_history(latest: dict[str, Any]) -> dict[str, Any]:
    return {
        "assessed_at": latest.get("assessed_at"),
        "structural_issue_count": latest.get("structural_issue_count"),
        "task_fit_reasoning": (str(latest.get("task_fit_reasoning") or "")[:240] + "…")
        if len(str(latest.get("task_fit_reasoning") or "")) > 240
        else latest.get("task_fit_reasoning"),
        "recommended_fixes_head": (latest.get("recommended_fixes") or [])[:3],
        "next_steps_head": (latest.get("next_steps_for_user") or [])[:3],
    }


def persist_assessment_feedback_to_understanding(
    root: Path,
    pipeline_id: str,
    *,
    combined_ok: bool,
    assessed_at: str | None,
    structural_issue_count: int,
    structural_issues: list[Any],
    task_ok: bool,
    task_fit_skipped: bool,
    reasoning: str,
    recommend_new_operators: bool,
    recommended_fixes: list[str],
    optimization_suggestions: list[str],
    next_steps_for_user: list[str],
) -> bool:
    """
    更新 `data/understanding_results/{id}.json` 中的 `orchestration_assessment_feedback`。
    - 评估通过：标记 last_assessment_passed，清空 history（避免陈旧负反馈干扰）。
    - 评估未通过：写入 latest，并将上一轮失败 latest 压入 history（限长）。
    若无 understanding 文件则跳过并打日志。
    """
    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    if not u_path.is_file():
        logger.warning("无 understanding 文件，跳过写入编排评估反馈: %s", u_path)
        return False
    try:
        u_raw = json.loads(u_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.warning("读取 understanding 失败，跳过反馈合并: %s", e)
        return False
    if not isinstance(u_raw, dict):
        return False

    prev_fb = u_raw.get("orchestration_assessment_feedback")
    prev_fb = prev_fb if isinstance(prev_fb, dict) else {}
    now = datetime.now(timezone.utc).isoformat()

    if combined_ok:
        u_raw["orchestration_assessment_feedback"] = {
            "updated_at": now,
            "last_assessment_passed": True,
            "iteration": int(prev_fb.get("iteration") or 0),
            "latest": {
                "passed": True,
                "is_valid": True,
                "assessed_at": assessed_at or now,
            },
            "history": [],
        }
        _atomic_write_json(u_path, u_raw)
        return True

    issue_lines = [_issue_to_line(x) for x in (structural_issues or [])[:25]]
    latest: dict[str, Any] = {
        "passed": False,
        "is_valid": False,
        "assessed_at": assessed_at or now,
        "structural_issue_count": int(structural_issue_count),
        "structural_issues_summary": issue_lines,
        "task_fit_satisfied": task_ok,
        "task_fit_skipped": task_fit_skipped,
        "task_fit_reasoning": (reasoning or "")[:4000],
        "recommend_new_operators": recommend_new_operators,
        "recommended_fixes": list(recommended_fixes or [])[:16],
        "optimization_suggestions": list(optimization_suggestions or [])[:16],
        "next_steps_for_user": list(next_steps_for_user or [])[:16],
    }

    hist: list[Any] = list(prev_fb.get("history") or [])
    if isinstance(hist, list):
        old_latest = prev_fb.get("latest")
        if isinstance(old_latest, dict) and old_latest.get("passed") is False and old_latest.get("is_valid") is False:
            hist.append(_compact_for_history(old_latest))
        hist = hist[-_HISTORY_MAX:]
    else:
        hist = []

    iteration = int(prev_fb.get("iteration") or 0) + 1

    u_raw["orchestration_assessment_feedback"] = {
        "updated_at": now,
        "last_assessment_passed": False,
        "iteration": iteration,
        "latest": latest,
        "history": hist,
    }
    _atomic_write_json(u_path, u_raw)
    return True


def format_prior_orchestration_feedback_for_prompt(understanding: dict[str, Any]) -> str:
    """供 Free Fitting / Template Combination 用户提示拼接。"""
    block = understanding.get("orchestration_assessment_feedback")
    if not isinstance(block, dict):
        return (
            "（无编排评估反馈：尚未在理解结果中记录过失败评估，或尚未运行过带评估的编排。）"
        )

    if block.get("last_assessment_passed") is True:
        return "（上一轮 DAG 评估**已通过**。按数据集目标设计蓝图即可，无需针对旧失败项做额外规避。）"

    latest = block.get("latest")
    if not isinstance(latest, dict) or latest.get("passed") is True:
        return "（无待处理的失败评估条目。）"

    lines: list[str] = [
        "以下条目来自**最近一次**编排后的自动评估（结构校验 + 任务/语义 LLM）。"
        "**你必须**在蓝图与模板选择中显式规避并修正这些问题，禁止重复相同错误（例如：DAG 自环、键不衔接、遗漏输出键、与任务目标不一致的流程）。",
        "",
        f"- 结构问题条数: {latest.get('structural_issue_count', 0)}",
        f"- 任务语义是否达标: {latest.get('task_fit_satisfied')}（模型跳过={latest.get('task_fit_skipped')}）",
        f"- 是否建议新增注册算子: {latest.get('recommend_new_operators')}",
    ]
    reasoning = (latest.get("task_fit_reasoning") or "").strip()
    if reasoning:
        lines.append("")
        lines.append("**模型结论（摘要）**:")
        lines.append(reasoning[:1200] + ("…" if len(reasoning) > 1200 else ""))

    sis = latest.get("structural_issues_summary") or []
    if isinstance(sis, list) and sis:
        lines.append("")
        lines.append("**结构/数据流问题（摘录）**:")
        for i, s in enumerate(sis[:12], 1):
            lines.append(f"  {i}. {s}")

    def _bullets(title: str, key: str, max_n: int) -> None:
        xs = latest.get(key) or []
        if not isinstance(xs, list) or not xs:
            return
        lines.append("")
        lines.append(f"**{title}**:")
        for x in xs[:max_n]:
            lines.append(f"  • {str(x)[:500]}")

    _bullets("建议修复", "recommended_fixes", 8)
    _bullets("可优化方向", "optimization_suggestions", 6)
    _bullets("建议你下一步", "next_steps_for_user", 8)

    hist = block.get("history")
    if isinstance(hist, list) and len(hist) >= 1:
        lines.append("")
        lines.append("**更早轮次（压缩摘要，供避免重复踩坑）**:")
        for h in hist[-3:]:
            if not isinstance(h, dict):
                continue
            ra = h.get("assessed_at") or ""
            cnt = h.get("structural_issue_count")
            rr = h.get("task_fit_reasoning") or ""
            rr_s = str(rr)[:160] + ("…" if len(str(rr)) > 160 else "")
            lines.append(f"  — {ra}  结构问题数={cnt}  {rr_s}")

    return "\n".join(lines)


def _compact_pilot_for_history(block: dict[str, Any]) -> dict[str, Any]:
    js = block.get("judge_summary") if isinstance(block.get("judge_summary"), dict) else {}
    ci = js.get("critical_insights") if isinstance(js.get("critical_insights"), list) else []
    return {
        "updated_at": block.get("updated_at"),
        "workflow_snapshot": block.get("workflow_snapshot")
        if isinstance(block.get("workflow_snapshot"), dict)
        else {},
        "execution_ok": block.get("execution_ok"),
        "present": block.get("present"),
        "overall_score": block.get("overall_score"),
        "recommendation": block.get("recommendation"),
        "recommendation_rationale": (str(block.get("recommendation_rationale") or ""))[:500],
        "dimension_scores": block.get("dimension_scores")
        if isinstance(block.get("dimension_scores"), dict)
        else {},
        "experience_bullets": [str(x) for x in (block.get("experience_bullets") or []) if x][:8],
        "critical_insights": [str(x) for x in ci if x][:6],
        "overall_assessment_head": (str(js.get("overall_assessment") or ""))[:600],
    }


def persist_pilot_evaluation_to_understanding(
    root: Path,
    pipeline_id: str,
    trial_result: dict[str, Any],
    workflow_meta: dict[str, Any] | None = None,
) -> bool:
    """
    将试运行 / Pilot LLM 评估摘要写入 `understanding_results`，供下一轮 `understand` 在提示中携带，
    引导模型把改进点融入结构化 profile（随后应重新 orchestrate → …）。

    - `pilot_run_feedback`：最近一次 trial 的完整摘要。
    - `pilot_run_feedback_history`：历次 trial 的压缩条目（追加，有上限），便于 LLM 看到多轮经验而非覆盖。
    """
    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    if not u_path.is_file():
        logger.warning("无 understanding 文件，跳过 Pilot 反馈写入: %s", u_path)
        return False
    try:
        u_raw = json.loads(u_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.warning("读取 understanding 失败，跳过 Pilot 反馈: %s", e)
        return False
    if not isinstance(u_raw, dict):
        return False

    pilot = trial_result.get("llm_pilot_evaluation")
    pilot = pilot if isinstance(pilot, dict) else {}
    reflux = trial_result.get("reflux_recommendation")
    reflux = reflux if isinstance(reflux, dict) else {}
    now = datetime.now(timezone.utc).isoformat()

    jr = pilot.get("judge_result") if isinstance(pilot.get("judge_result"), dict) else {}
    wm = workflow_meta if isinstance(workflow_meta, dict) else {}
    old_latest = u_raw.get("pilot_run_feedback")
    hist: list[Any] = list(u_raw.get("pilot_run_feedback_history") or [])
    if not isinstance(hist, list):
        hist = []
    if isinstance(old_latest, dict) and (
        old_latest.get("updated_at") or old_latest.get("overall_score") is not None
    ):
        hist.append(_compact_pilot_for_history(old_latest))
    hist = hist[-_PILOT_FEEDBACK_HISTORY_MAX:]

    workflow_snap = {
        "understanding_revision": int(wm.get("understanding_revision", 0)),
        "orchestration_revision": int(wm.get("orchestration_revision", 0)),
        "dag_evolution_cycles": int(wm.get("dag_evolution_cycles", 0)),
    }
    u_raw["pilot_run_feedback_history"] = hist
    u_raw["pilot_run_feedback"] = {
        "updated_at": now,
        "trial_artifact": f"data/trial_runs/{pipeline_id}/trial_result.json",
        "workflow_snapshot": workflow_snap,
        "execution_ok": trial_result.get("execution_ok"),
        "present": bool(pilot.get("present")),
        "skipped_reason": pilot.get("skipped_reason"),
        "overall_score": pilot.get("overall_score"),
        "recommendation": pilot.get("recommendation"),
        "recommendation_rationale": (str(pilot.get("recommendation_rationale") or ""))[:1200],
        "experience_bullets": [str(x) for x in (pilot.get("experience_bullets") or []) if x][:12],
        "dimension_scores": pilot.get("dimension_scores")
        if isinstance(pilot.get("dimension_scores"), dict)
        else {},
        "dimension_notes": pilot.get("dimension_notes")
        if isinstance(pilot.get("dimension_notes"), dict)
        else {},
        "scores": pilot.get("scores") if isinstance(pilot.get("scores"), dict) else {},
        "judge_summary": {
            "overall_assessment": (str(jr.get("overall_assessment") or ""))[:1500],
            "critical_insights": [str(x) for x in (jr.get("critical_insights") or []) if x][:10],
        },
        "reflux_targets": reflux.get("targets"),
        "reflux_reasons": [str(x) for x in (reflux.get("reasons") or []) if x][:8],
    }
    _atomic_write_json(u_path, u_raw)
    return True


def _format_pilot_history_section(understanding: dict[str, Any]) -> str | None:
    hist = understanding.get("pilot_run_feedback_history")
    if not isinstance(hist, list) or not hist:
        return None
    tail = hist[-12:]
    lines: list[str] = [
        "### 历史试运行 / Pilot 反馈（按时间顺序；与「最近一次」一并阅读，形成连续改进叙事）",
        "",
    ]
    for ent in tail:
        if not isinstance(ent, dict):
            continue
        ws = ent.get("workflow_snapshot") if isinstance(ent.get("workflow_snapshot"), dict) else {}
        ur = ws.get("understanding_revision", "?")
        orv = ws.get("orchestration_revision", "?")
        dec = ws.get("dag_evolution_cycles", "?")
        lines.append(
            f"- **版次** 理解≈{ur} · 编排≈{orv} · 编排↔进化环≈{dec} · 记录于 {ent.get('updated_at', '')}"
        )
        lines.append(
            f"  · 执行成功: {ent.get('execution_ok')} · Pilot 分: {ent.get('overall_score')} · 建议: {ent.get('recommendation')}"
        )
        rat = str(ent.get("recommendation_rationale") or "").strip()
        if rat:
            lines.append(f"  · 理由: {rat[:320]}{'…' if len(rat) > 320 else ''}")
        ds = ent.get("dimension_scores") if isinstance(ent.get("dimension_scores"), dict) else {}
        if ds:
            lines.append(f"  · 维度(0-100): {ds}")
        eb = ent.get("experience_bullets") or []
        if isinstance(eb, list) and eb:
            lines.append("  · 经验要点:")
            for x in eb[:5]:
                lines.append(f"    — {str(x)[:280]}")
        oa = str(ent.get("overall_assessment_head") or "").strip()
        if oa:
            lines.append(f"  · 评价摘要: {oa[:400]}{'…' if len(oa) > 400 else ''}")
        lines.append("")
    return "\n".join(lines).strip()


def format_pilot_feedback_for_understanding_prompt(understanding: dict[str, Any]) -> str | None:
    """
    拼入 unified profile：`pilot_run_feedback_history`（多轮压缩）+ `pilot_run_feedback`（最近一次详情）。
    无内容时返回 None。
    """
    parts: list[str] = []
    hsec = _format_pilot_history_section(understanding)
    if hsec:
        parts.append(hsec)
    latest = format_pilot_run_feedback_for_understanding_prompt(understanding)
    if latest:
        parts.append(latest)
    if not parts:
        return None
    return "\n\n---\n\n".join(parts)


def format_pilot_run_feedback_for_understanding_prompt(understanding: dict[str, Any]) -> str | None:
    """仅格式化最近一次 `pilot_run_feedback`（供组合函数与单测复用）。"""
    block = understanding.get("pilot_run_feedback")
    if not isinstance(block, dict):
        return None
    if not block.get("present"):
        if block.get("skipped_reason"):
            return (
                f"（上一轮试运行 Pilot 未评分：{block.get('skipped_reason')}。"
                "可忽略或补全 API 后重跑 trial。）"
            )
        if block.get("execution_ok") is False:
            return (
                "### 上一轮试运行采样执行失败\n"
                "请结合 trial 落盘中的错误信息，在 `dataset_level_delta` / `schema_analysis` 中反映可操作的改进假设。"
            )
        return None

    lines: list[str] = [
        "### 上一轮试运行 / Pilot LLM 反馈（必须在本次 profile 中吸收、对齐或显式反驳）",
        f"- 采样执行成功: {block.get('execution_ok')}",
        f"- Pilot 总分(0-100): {block.get('overall_score')}",
        f"- 建议动作: {block.get('recommendation')}",
    ]
    ds = block.get("dimension_scores") if isinstance(block.get("dimension_scores"), dict) else {}
    dn = block.get("dimension_notes") if isinstance(block.get("dimension_notes"), dict) else {}
    dim_order = ("semantic", "format", "diversity", "info", "noise", "logic")
    if ds:
        lines.append("- 多维度评分(0-100):")
        for k in dim_order:
            if k not in ds:
                continue
            note = str(dn.get(k) or "").strip()
            tail = f" — {note[:160]}{'…' if len(note) > 160 else ''}" if note else ""
            lines.append(f"  • {k}={ds.get(k)}{tail}")

    rat = str(block.get("recommendation_rationale") or "").strip()
    if rat:
        lines.append(f"- 理由摘要: {rat[:800]}{'…' if len(rat) > 800 else ''}")

    js = block.get("judge_summary") if isinstance(block.get("judge_summary"), dict) else {}
    oa = str(js.get("overall_assessment") or "").strip()
    if oa:
        lines.append("")
        lines.append("**整体评价**:")
        lines.append(oa[:1200] + ("…" if len(oa) > 1200 else ""))
    ci = js.get("critical_insights") or []
    if isinstance(ci, list) and ci:
        lines.append("")
        lines.append("**要点**:")
        for x in ci[:8]:
            lines.append(f"  • {str(x)[:400]}")

    eb = block.get("experience_bullets") or []
    if isinstance(eb, list) and eb:
        lines.append("")
        lines.append("**管线级经验（供 dataset_level_delta / schema 改进）**:")
        for x in eb[:8]:
            lines.append(f"  • {str(x)[:400]}")

    rr = block.get("reflux_reasons") or []
    if isinstance(rr, list) and rr:
        lines.append("")
        lines.append("**规则侧回流提示**:")
        for x in rr[:6]:
            lines.append(f"  • {str(x)[:400]}")

    # ── 算子级诊断注入（来自experience_snapshot_v2）──
    try:
        import json as _json
        _root_guess = Path(__file__).parent.parent.parent
        _pid = (understanding.get("pipeline_id") or
                understanding.get("meta", {}).get("pipeline_id") or "")
        if _pid:
            _exp_path = _root_guess / "data" / "experiences" / f"{_pid}.json"
            if _exp_path.is_file():
                _exp = _json.loads(_exp_path.read_text(encoding="utf-8"))
                _diagnoses = _exp.get("operator_diagnoses") or []
                _priority = str(_exp.get("priority_fix") or "").strip()
                _instruction = str(_exp.get("next_round_instruction") or "").strip()
                if _diagnoses:
                    lines.append("")
                    lines.append("**【算子级诊断】必须在下一轮编排中针对性修复：**")
                    for d in _diagnoses[:5]:
                        sev = d.get("severity", "medium")
                        op = d.get("operator", "unknown")
                        prob = str(d.get("problem") or "")[:200]
                        fix = str(d.get("fix_suggestion") or "")[:200]
                        lines.append(f"  ▸ [{sev.upper()}] {op}: {prob}")
                        if fix:
                            lines.append(f"    → 建议: {fix}")
                if _priority:
                    lines.append("")
                    lines.append(f"**【最优先修复】**: {_priority[:300]}")
                if _instruction:
                    lines.append("")
                    lines.append(f"**【下轮编排指令】**: {_instruction[:300]}")
    except Exception:
        pass

    lines.append("")
    lines.append(
        "请把上述反馈融入 `dataset_level_delta`、`schema_analysis` 等字段；"
        "若你认为反馈不适用，在 delta.summary 中简要说明原因。"
    )
    return "\n".join(lines)
