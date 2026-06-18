"""
Workflow 后半段落盘：质量快照 + 经验摘要（含 LLM 算子级诊断，读取step快照数据）。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _load_llm_cfg(root: Path) -> dict[str, Any]:
    """从config/api_config.json读取LLM配置，包含中转站地址。"""
    for p in [
        root / "config" / "api_config.json",
        root / "config" / "api_keys.json",
    ]:
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    return {}


def _read_step_snapshots(root: Path, step_trace: list[dict], max_records_per_step: int = 2) -> list[dict]:
    """读取每步执行后的数据快照，供LLM诊断用。"""
    snapshots = []
    for step in (step_trace or [])[:10]:
        op = step.get("operator", "unknown")
        dump_path = step.get("records_dump", "")
        if not dump_path:
            continue
        full_path = root / dump_path
        if not full_path.is_file():
            continue
        records = []
        try:
            with open(full_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_records_per_step:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        slim = {}
                        for k, v in obj.items():
                            if k.startswith("_"):
                                continue
                            if isinstance(v, str) and len(v) > 150:
                                slim[k] = v[:150] + "…"
                            else:
                                slim[k] = v
                        records.append(slim)
        except Exception:
            pass
        snapshots.append({
            "operator": op,
            "step_index": step.get("step_index", 0),
            "ok": step.get("ok", True),
            "n_records": step.get("n_records", 0),
            "sample_records": records,
        })
    return snapshots


def _call_llm_diagnosis(
    step_trace: list[dict],
    pilot_score: int | None,
    pilot_dim: dict,
    pilot_notes: dict,
    experience_bullets: list[str],
    current_pipeline: list[dict],
    cfg: dict,
    root: Path | None = None,
) -> dict[str, Any]:
    """用LLM分析step快照数据，生成算子级诊断。"""
    base_url = str(cfg.get("base_url", "https://api.openai.com/v1")).rstrip("/")
    api_key = str(cfg.get("api_key", ""))
    model = str(cfg.get("model", "gpt-4o"))

    # 读取每步数据快照
    step_snapshots = []
    if root is not None:
        step_snapshots = _read_step_snapshots(root, step_trace)

    # 构造step IO摘要（含实际数据样本）
    step_lines = []
    if step_snapshots:
        for snap in step_snapshots:
            op = snap.get("operator", "unknown")
            n = snap.get("n_records", "?")
            ok = snap.get("ok", True)
            step_lines.append(f"  算子: {op} | 输出{n}条 | {'✓' if ok else '✗失败'}")
            for i, rec in enumerate(snap.get("sample_records", [])[:2]):
                step_lines.append(f"    样本{i+1}: {json.dumps(rec, ensure_ascii=False)[:250]}")
    else:
        for step in (step_trace or [])[:12]:
            op = step.get("operator", "unknown")
            n = step.get("n_records", "?")
            ok = step.get("ok", True)
            dur = step.get("duration_ms", 0)
            step_lines.append(f"  - {op}: {n}条记录 {'✓' if ok else '✗失败'} ({dur:.0f}ms)")
    step_text = "\n".join(step_lines) if step_lines else "  （无step信息）"

    op_list = [s.get("operator", "") for s in (current_pipeline or []) if s.get("operator")]
    pipeline_text = " → ".join(op_list) if op_list else "（未知）"

    dim_text = "\n".join(
        [f"  - {k}: {v}/100 — {pilot_notes.get(k, '')}" for k, v in (pilot_dim or {}).items()]
    ) or "  （无维度评分）"

    bullets_text = "\n".join(
        [f"  - {b}" for b in (experience_bullets or [])[:6]]
    ) or "  （无经验要点）"

    prompt = f"""你是DataEvolver数据准备系统的专家。分析以下trial run的每步算子执行情况，精确定位哪个算子出了问题。

## 当前Pipeline
{pipeline_text}

## 每步算子执行情况（含实际数据样本）
{step_text}

## Pilot Judge评分（总分{pilot_score}/100）
{dim_text}

## 已有经验要点
{bullets_text}

请严格输出JSON，不要有任何markdown：
{{
  "operator_diagnoses": [
    {{
      "operator": "具体算子名",
      "problem": "基于实际数据样本发现的具体问题",
      "severity": "high/medium/low",
      "fix_suggestion": "具体修复方案"
    }}
  ],
  "priority_fix": "最优先修复的一个具体问题，指向具体算子",
  "next_round_instruction": "给下一轮理解和编排阶段的具体指令（100字以内）"
}}"""

    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.3,
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
        raw = raw.strip()
        parsed = json.loads(raw)
        logger.info("LLM算子诊断成功，诊断了%d个算子", len(parsed.get("operator_diagnoses", [])))
        return parsed
    except Exception as e:
        logger.warning("LLM算子诊断失败: %s", e)
        return {
            "operator_diagnoses": [],
            "priority_fix": "",
            "next_round_instruction": "",
            "_error": str(e),
        }


def build_quality_check_snapshot(root: Path, pipeline_id: str) -> dict[str, Any]:
    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    o_path = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    g_path = root / "data" / "generated_pipelines" / f"{pipeline_id}.json"
    trial_path = root / "data" / "trial_runs" / pipeline_id / "trial_result.json"
    run_latest_path = root / "data" / "run_pipeline_results" / pipeline_id / "latest.json"

    u = _read_json(u_path)
    if u is None:
        raise FileNotFoundError(f"缺少理解结果: {u_path}")

    orch = _read_json(o_path)
    gen = _read_json(g_path)
    trial = _read_json(trial_path)
    run_latest = _read_json(run_latest_path)

    schema = u.get("schema_analysis") if isinstance(u.get("schema_analysis"), dict) else {}
    delta = u.get("dataset_level_delta") if isinstance(u.get("dataset_level_delta"), dict) else {}
    new_fields = [str(x) for x in (schema.get("new_fields") or []) if x is not None]
    strategies = [str(x) for x in (delta.get("transformation_strategies") or []) if x]
    key_improvements = [str(x) for x in (delta.get("key_improvements") or []) if x]

    vr: dict[str, Any] = {}
    if isinstance(orch, dict):
        cs = orch.get("constrained_search")
        if isinstance(cs, dict) and isinstance(cs.get("validation_result"), dict):
            vr = cs["validation_result"]

    insights: list[str] = []
    if new_fields:
        preview = ", ".join(new_fields[:12])
        more = "…" if len(new_fields) > 12 else ""
        insights.append(f"理解阶段标记需对齐的字段（new_fields）: {preview}{more}")
    for s in strategies[:6]:
        insights.append(f"转化策略: {s[:200]}")
    for k in key_improvements[:4]:
        insights.append(f"关键改进: {k[:200]}")
    if not insights:
        insights.append("理解结果中未抽取到额外 insight，可检查 schema_analysis / dataset_level_delta。")

    assessment = (
        str(delta.get("summary") or "").strip()
        or str(schema.get("schema_constraint") or "").strip()
        or "基于理解结果的静态快照；尚未接入采样与 judge LLM。"
    )

    gaps: list[str] = []
    god = str(delta.get("global_optimization_direction") or "").strip()
    if god:
        gaps.append(god[:500])
    qf = delta.get("quality_focus") or []
    if isinstance(qf, list):
        gaps.extend(str(x) for x in qf[:8] if x)

    trial_schema: dict[str, Any] = {}
    pilot_eval: dict[str, Any] = {}
    if isinstance(trial, dict):
        ts = trial.get("schema_check")
        if isinstance(ts, dict):
            trial_schema = ts
        pe = trial.get("llm_pilot_evaluation")
        if isinstance(pe, dict) and pe.get("present"):
            pilot_eval = pe
    miss_seed = list(trial_schema.get("missing_for_seed_top_keys") or [])
    trial_failed = bool(trial) and not trial.get("execution_ok", False)
    trial_schema_drift = bool(miss_seed)

    if trial_failed:
        insights.insert(0, f"试运行失败: {trial.get('last_error', 'unknown')}"[:300])
    elif isinstance(trial, dict) and trial.get("execution_ok") and trial_schema_drift:
        insights.insert(0, f"试运行通过但相对 seed 缺键: {', '.join(str(x) for x in miss_seed[:10])}")

    judge_for_ui: dict[str, Any] | None = None
    sample_metrics_01: dict[str, float] | None = None
    pilot_score: int | None = None
    pilot_rec: str | None = None
    pilot_dim: dict[str, int] | None = None
    if pilot_eval:
        jr = pilot_eval.get("judge_result")
        if isinstance(jr, dict):
            judge_for_ui = {
                "has_differences": jr.get("has_differences"),
                "overall_assessment": jr.get("overall_assessment"),
                "critical_insights": jr.get("critical_insights"),
                "implicit_quality_requirements": jr.get("implicit_quality_requirements"),
            }
            for x in jr.get("critical_insights") or []:
                if isinstance(x, str) and x.strip():
                    insights.insert(0, x.strip()[:400])
        sc = pilot_eval.get("scores")
        if isinstance(sc, dict):
            sample_metrics_01 = {}
            for k, v in sc.items():
                try:
                    sample_metrics_01[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
        try:
            pilot_score = int(pilot_eval.get("overall_score"))
        except (TypeError, ValueError):
            pilot_score = None
        pilot_rec = str(pilot_eval.get("recommendation") or "") or None
        pds = pilot_eval.get("dimension_scores")
        if isinstance(pds, dict):
            pilot_dim = {}
            for k, v in pds.items():
                try:
                    pilot_dim[str(k)] = max(0, min(100, int(round(float(v)))))
                except (TypeError, ValueError):
                    pass
            if not pilot_dim:
                pilot_dim = None
        if pilot_eval.get("recommendation_rationale"):
            insights.insert(0, str(pilot_eval["recommendation_rationale"])[:400])
        if judge_for_ui and str(judge_for_ui.get("overall_assessment") or "").strip():
            assessment = str(judge_for_ui["overall_assessment"]).strip()[:2000]

    resp_gaps: list[str] = gaps[:15] or ["待接入 rubric / judge 后填充"]
    fmt_gaps: list[str] = []
    if judge_for_ui and isinstance(judge_for_ui.get("implicit_quality_requirements"), dict):
        ir = judge_for_ui["implicit_quality_requirements"]
        jresp = [str(x) for x in (ir.get("response_quality_gaps") or []) if x]
        if jresp:
            resp_gaps = jresp[:15]
        fmt_gaps = [str(x) for x in (ir.get("format_rigor_gaps") or []) if x][:15]

    return {
        "pipeline_id": pipeline_id,
        "source": "quality_snapshot_v1",
        "has_differences": bool(
            new_fields or strategies or not vr.get("is_valid", True) or trial_failed or trial_schema_drift
        ),
        "overall_assessment": assessment[:2000],
        "critical_insights": insights[:20],
        "implicit_quality_requirements": {
            "response_quality_gaps": resp_gaps,
            "format_rigor_gaps": fmt_gaps,
        },
        "provenance": {
            "understanding_path": f"data/understanding_results/{pipeline_id}.json",
            "orchestration_path": f"data/orchestration_results/{pipeline_id}.json",
            "generated_pipeline_path": f"data/generated_pipelines/{pipeline_id}.json",
            "has_generated_pipeline": gen is not None,
            "orchestration_validation_ok": bool(vr.get("is_valid")) if vr else None,
            "trial_run_path": f"data/trial_runs/{pipeline_id}/trial_result.json",
            "trial_run_found": trial is not None,
            "trial_execution_ok": trial.get("execution_ok") if trial else None,
            "pipeline_run_latest_path": f"data/run_pipeline_results/{pipeline_id}/latest.json",
            "pipeline_run_latest": run_latest,
        },
        "trial_run": {
            "present": trial is not None,
            "execution_ok": trial.get("execution_ok") if trial else None,
            "schema_check": trial.get("schema_check") if trial else None,
            "reflux_recommendation": trial.get("reflux_recommendation") if trial else None,
            "step_trace": trial.get("step_trace") if trial else None,
            "data_samples": trial.get("data_samples") if trial else None,
        },
        "llm_pilot_evaluation": pilot_eval if pilot_eval else None,
        "judge_result": judge_for_ui,
        "pilot_overall_score": pilot_score,
        "pilot_recommendation": pilot_rec,
        "pilot_dimension_scores": pilot_dim,
        "sample_metrics_0_1": sample_metrics_01,
        "meta": {"created_at": _iso(), "note": "含试运行与可选 Pilot LLM 评估"},
    }


def build_experience_snapshot(root: Path, pipeline_id: str) -> dict[str, Any]:
    qc_path = root / "data" / "quality_check_results" / f"{pipeline_id}.json"
    o_path = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    trial_path = root / "data" / "trial_runs" / pipeline_id / "trial_result.json"
    gen_path = root / "data" / "generated_pipelines" / f"{pipeline_id}.json"

    qc = _read_json(qc_path)
    orch = _read_json(o_path)
    u = _read_json(u_path)
    trial = _read_json(trial_path)
    gen = _read_json(gen_path)

    vr: dict[str, Any] = {}
    if isinstance(orch, dict):
        cs = orch.get("constrained_search")
        if isinstance(cs, dict) and isinstance(cs.get("validation_result"), dict):
            vr = cs["validation_result"]
    valid = bool(vr.get("is_valid")) if vr else True

    step_trace: list[dict] = []
    pilot_eval: dict[str, Any] = {}
    if isinstance(trial, dict):
        st = trial.get("step_trace")
        if isinstance(st, list):
            step_trace = st
        pe = trial.get("llm_pilot_evaluation")
        if isinstance(pe, dict) and pe.get("present"):
            pilot_eval = pe

    pilot_score: int | None = None
    pilot_dim: dict[str, int] = {}
    pilot_notes: dict[str, str] = {}
    experience_bullets: list[str] = []
    if pilot_eval:
        try:
            pilot_score = int(pilot_eval.get("overall_score"))
        except (TypeError, ValueError):
            pilot_score = None
        pds = pilot_eval.get("dimension_scores") or {}
        if isinstance(pds, dict):
            for k, v in pds.items():
                try:
                    pilot_dim[str(k)] = max(0, min(100, int(round(float(v)))))
                except (TypeError, ValueError):
                    pass
        pdn = pilot_eval.get("dimension_notes") or {}
        if isinstance(pdn, dict):
            pilot_notes = {str(k): str(v) for k, v in pdn.items()}
        experience_bullets = [
            str(b) for b in (pilot_eval.get("experience_bullets") or []) if b
        ][:8]

    current_pipeline: list[dict] = []
    if isinstance(gen, dict):
        fp = gen.get("final_pipeline") or gen.get("pipeline_plan") or []
        if isinstance(fp, list):
            current_pipeline = fp

    # ── LLM算子级诊断（读取step快照，从config读LLM配置）──
    llm_diagnosis: dict[str, Any] = {}
    should_diagnose = (
        pilot_score is not None and pilot_score < 90
        and (step_trace or pilot_eval)
    )
    candidate_strategy_ids: list[str] = []
    if should_diagnose:
        cfg = _load_llm_cfg(root)
        if cfg.get("api_key") or cfg.get("base_url"):
            llm_diagnosis = _call_llm_diagnosis(
                step_trace=step_trace,
                pilot_score=pilot_score,
                pilot_dim=pilot_dim,
                pilot_notes=pilot_notes,
                experience_bullets=experience_bullets,
                current_pipeline=current_pipeline,
                cfg=cfg,
                root=root,
            )
            # ── 生成候选策略写入strategy_pool ──
            try:
                from subsystems.workflow.strategy_pool import generate_candidate_strategies
                round_id = 1
                if isinstance(trial, dict):
                    round_id = int(trial.get("meta", {}).get("round", 1) or 1)
                candidate_strategy_ids = generate_candidate_strategies(
                    root, pipeline_id,
                    operator_diagnoses=llm_diagnosis.get("operator_diagnoses") or [],
                    current_pipeline=current_pipeline,
                    pilot_score=pilot_score,
                    cfg=cfg,
                    round_id=round_id,
                )
                logger.info("生成候选策略: %s", candidate_strategy_ids)
            except Exception as e:
                logger.warning("生成候选策略失败: %s", e)

    parts: list[str] = []
    parts.append("编排 DAG 校验" + ("通过。" if valid else "未通过，下一轮应优先修复数据流或算子选择。"))
    if isinstance(trial, dict):
        if trial.get("execution_ok"):
            parts.append("试运行：采样数据已沿实例化桩执行完成。")
        else:
            parts.append("试运行失败: " + str(trial.get("last_error") or "unknown")[:220])
        if pilot_eval.get("present"):
            for b in experience_bullets:
                parts.append("Pilot 经验: " + b[:280])
            if pilot_eval.get("recommendation"):
                parts.append(
                    "Pilot 建议下一步: "
                    + str(pilot_eval.get("recommendation"))
                    + (" — " + str(pilot_eval.get("recommendation_rationale"))[:160]
                       if pilot_eval.get("recommendation_rationale") else "")
                )
        rr = trial.get("reflux_recommendation") if isinstance(trial.get("reflux_recommendation"), dict) else {}
        for r in rr.get("reasons") or []:
            if isinstance(r, str) and r.strip():
                parts.append(r[:200])

    if llm_diagnosis.get("priority_fix"):
        parts.append("【优先修复】" + str(llm_diagnosis["priority_fix"])[:300])
    if llm_diagnosis.get("next_round_instruction"):
        parts.append("【下轮指令】" + str(llm_diagnosis["next_round_instruction"])[:300])

    if isinstance(qc, dict) and qc.get("overall_assessment"):
        parts.append(str(qc["overall_assessment"])[:500])
    if isinstance(u, dict):
        delta = u.get("dataset_level_delta")
        if isinstance(delta, dict) and delta.get("summary"):
            parts.append("数据集级摘要: " + str(delta["summary"])[:400])

    exp_text = " ".join(parts).strip()
    if len(exp_text) > 2000:
        exp_text = exp_text[:1997] + "…"

    reflux: list[str] = ["orchestration", "operator_evolution"] if not valid else ["orchestration"]
    if isinstance(qc, dict) and qc.get("has_differences"):
        reflux.append("understanding")
    if isinstance(trial, dict):
        rt = trial.get("reflux_recommendation") if isinstance(trial.get("reflux_recommendation"), dict) else {}
        for t in rt.get("targets") or []:
            if isinstance(t, str) and t.strip():
                reflux.append(t.strip())
    reflux = list(dict.fromkeys(reflux))

    return {
        "pipeline_id": pipeline_id,
        "experience_text": exp_text or "（空经验：请检查上游产物是否完整）",
        "reflux_targets": list(dict.fromkeys(reflux)),
        "operator_diagnoses": llm_diagnosis.get("operator_diagnoses") or [],
        "priority_fix": llm_diagnosis.get("priority_fix") or "",
        "next_round_instruction": llm_diagnosis.get("next_round_instruction") or "",
        "candidate_strategy_ids": candidate_strategy_ids,
        "source": "experience_snapshot_v2",
        "meta": {
            "created_at": _iso(),
            "llm_diagnosis_triggered": should_diagnose,
            "pilot_score": pilot_score,
            "inputs": {
                "quality_check_path": f"data/quality_check_results/{pipeline_id}.json",
                "quality_check_found": qc is not None,
                "trial_run_path": f"data/trial_runs/{pipeline_id}/trial_result.json",
                "trial_run_found": trial is not None,
            },
        },
    }
