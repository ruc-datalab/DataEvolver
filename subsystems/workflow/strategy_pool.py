"""
Strategy Pool：记录每轮数据准备策略及其trial分数，供下一轮理解和编排参考。

策略来源：
  1. 用户选择某条候选策略后触发trial（模式A）
  2. 用户授权并行trial多条策略（模式B）
两种模式的结果都写入strategy_pool。
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pool_path(root: Path, pipeline_id: str) -> Path:
    return root / "data" / "strategy_pool" / f"{pipeline_id}.json"


def load_strategy_pool(root: Path, pipeline_id: str) -> dict[str, Any]:
    p = _pool_path(root, pipeline_id)
    if not p.is_file():
        return {"pipeline_id": pipeline_id, "strategies": [], "best_strategy_id": None}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"pipeline_id": pipeline_id, "strategies": [], "best_strategy_id": None}
    except (json.JSONDecodeError, OSError):
        return {"pipeline_id": pipeline_id, "strategies": [], "best_strategy_id": None}


def save_strategy_pool(root: Path, pipeline_id: str, pool: dict[str, Any]) -> None:
    p = _pool_path(root, pipeline_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def add_strategy(
    root: Path,
    pipeline_id: str,
    *,
    description: str,
    operator_sequence: list[str],
    key_changes: list[str],
    source: str = "llm_generated",
    round_id: int = 1,
) -> str:
    """添加一条候选策略，返回strategy_id。"""
    pool = load_strategy_pool(root, pipeline_id)
    strategies = pool.get("strategies") or []
    sid = f"s{len(strategies) + 1:03d}"
    strategies.append({
        "strategy_id": sid,
        "description": description,
        "operator_sequence": operator_sequence,
        "key_changes": key_changes,
        "source": source,
        "round": round_id,
        "score": None,
        "pilot_dimension_scores": None,
        "trial_path": None,
        "status": "pending",
        "created_at": _iso(),
        "executed_at": None,
        "selected_by_user": False,
        "parallel_trial": False,
    })
    pool["strategies"] = strategies
    save_strategy_pool(root, pipeline_id, pool)
    return sid


def update_strategy_score(
    root: Path,
    pipeline_id: str,
    strategy_id: str,
    *,
    score: int,
    dimension_scores: dict[str, int] | None = None,
    trial_path: str | None = None,
    selected_by_user: bool = False,
    parallel_trial: bool = False,
) -> None:
    """trial执行完后把分数写回strategy_pool，并更新best_strategy_id。"""
    pool = load_strategy_pool(root, pipeline_id)
    strategies = pool.get("strategies") or []
    for s in strategies:
        if s.get("strategy_id") == strategy_id:
            s["score"] = score
            s["pilot_dimension_scores"] = dimension_scores
            s["trial_path"] = trial_path
            s["status"] = "scored"
            s["executed_at"] = _iso()
            s["selected_by_user"] = selected_by_user
            s["parallel_trial"] = parallel_trial
            break

    # 更新best
    scored = [s for s in strategies if s.get("score") is not None]
    if scored:
        best = max(scored, key=lambda x: x["score"])
        pool["best_strategy_id"] = best["strategy_id"]

    pool["strategies"] = strategies
    save_strategy_pool(root, pipeline_id, pool)


def get_best_strategy(root: Path, pipeline_id: str) -> dict[str, Any] | None:
    pool = load_strategy_pool(root, pipeline_id)
    best_id = pool.get("best_strategy_id")
    if not best_id:
        return None
    for s in (pool.get("strategies") or []):
        if s.get("strategy_id") == best_id:
            return s
    return None


def format_strategy_pool_for_prompt(root: Path, pipeline_id: str) -> str | None:
    """格式化strategy_pool内容，注入到下一轮理解prompt里。"""
    pool = load_strategy_pool(root, pipeline_id)
    strategies = pool.get("strategies") or []
    if not strategies:
        return None

    lines = ["【策略池历史 - 下一轮编排应参考最优策略并在其基础上改进】"]
    best_id = pool.get("best_strategy_id")

    scored = [s for s in strategies if s.get("score") is not None]
    pending = [s for s in strategies if s.get("score") is None]

    if scored:
        lines.append(f"\n已评分策略（共{len(scored)}条）：")
        for s in sorted(scored, key=lambda x: x["score"], reverse=True)[:5]:
            sid = s["strategy_id"]
            is_best = "★最优" if sid == best_id else ""
            lines.append(f"  {sid} {is_best} | 得分: {s['score']}/100 | {s['description'][:100]}")
            if s.get("key_changes"):
                lines.append(f"    关键改动: {', '.join(s['key_changes'][:3])}")
            if s.get("pilot_dimension_scores"):
                ds = s["pilot_dimension_scores"]
                lines.append(f"    维度: {ds}")
            src = "用户选择" if s.get("selected_by_user") else ("并行trial" if s.get("parallel_trial") else "自动生成")
            lines.append(f"    来源: {src} | 轮次: Round {s.get('round', '?')}")

    if pending:
        lines.append(f"\n待执行策略（共{len(pending)}条）：")
        for s in pending[:3]:
            lines.append(f"  {s['strategy_id']} | {s['description'][:100]}")

    best = get_best_strategy(root, pipeline_id)
    if best:
        lines.append(f"\n【当前最优策略 {best['strategy_id']} 得分{best['score']}分，下轮编排建议在此基础上改进】")
        lines.append(f"算子序列: {' → '.join(best.get('operator_sequence', []))}")

    return "\n".join(lines)


def generate_candidate_strategies(
    root: Path,
    pipeline_id: str,
    *,
    operator_diagnoses: list[dict],
    current_pipeline: list[dict],
    pilot_score: int | None,
    cfg: dict,
    round_id: int = 1,
) -> list[str]:
    """
    基于算子诊断，用LLM生成候选策略并写入strategy_pool。
    返回生成的strategy_id列表。
    用户可以从中选一条执行（模式A），或者授权并行执行所有（模式B）。
    """
    base_url = str(cfg.get("base_url", "https://api.openai.com/v1")).rstrip("/")
    api_key = str(cfg.get("api_key", ""))
    model = str(cfg.get("model", "gpt-4o"))

    op_list = [s.get("operator_name") or s.get("operator", "") for s in (current_pipeline or []) if isinstance(s, dict)]
    op_list = [o for o in op_list if o]
    pipeline_text = " → ".join(op_list) if op_list else "（未知）"

    diag_text = "\n".join([
        f"  [{d.get('severity','?').upper()}] {d.get('operator','?')}: {d.get('problem','')[:150]} → 建议: {d.get('fix_suggestion','')[:150]}"
        for d in (operator_diagnoses or [])[:5]
    ]) or "  （无算子诊断）"

    # 读取策略池历史
    pool = load_strategy_pool(root, pipeline_id)
    history_text = "（无历史策略）"
    scored = [s for s in (pool.get("strategies") or []) if s.get("score") is not None]
    if scored:
        history_text = "\n".join([
            f"  {s['strategy_id']} 得分{s['score']}: {s['description'][:80]}"
            for s in sorted(scored, key=lambda x: x["score"], reverse=True)[:3]
        ])

    prompt = f"""你是DataEvolver数据准备系统的策略设计专家。基于算子诊断结果，生成2条不同方向的候选数据准备策略。

## 当前Pipeline
{pipeline_text}

## Pilot评分
{pilot_score}/100

## 算子级诊断
{diag_text}

## 历史策略（避免重复）
{history_text}

请生成2条差异明显的候选策略，每条策略代表一个不同的改进方向（比如：换用不同工具、调整处理深度、修改算子参数等）。
严格输出JSON，不要markdown：
{{
  "strategies": [
    {{
      "description": "策略描述（50字以内，说明核心改进思路）",
      "operator_sequence": ["算子1", "算子2", "算子3"],
      "key_changes": ["具体改动1", "具体改动2"]
    }},
    {{
      "description": "策略描述",
      "operator_sequence": ["算子1", "算子2"],
      "key_changes": ["具体改动1"]
    }}
  ]
}}"""

    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.5,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        raw = result["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        strategies = parsed.get("strategies") or []
    except Exception as e:
        # LLM失败时生成一条默认策略
        strategies = [{
            "description": f"基于算子诊断的改进策略（LLM生成失败: {str(e)[:50]}）",
            "operator_sequence": op_list,
            "key_changes": [d.get("fix_suggestion", "")[:100] for d in (operator_diagnoses or [])[:2]],
        }]

    sids = []
    for s in strategies[:2]:
        sid = add_strategy(
            root, pipeline_id,
            description=str(s.get("description", ""))[:200],
            operator_sequence=s.get("operator_sequence") or op_list,
            key_changes=[str(c) for c in (s.get("key_changes") or [])[:5]],
            source="llm_generated",
            round_id=round_id,
        )
        sids.append(sid)

    return sids
