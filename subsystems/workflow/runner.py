"""
将「推进一步」映射为可持久化的后端步骤，落盘路径与旧版 DataEvolver 对齐。

- **理解**：单次 LLM 完整 profile（`full_analyzer`）。
- **编排**：三阶段 LLM（`pipeline_orchestration.orchestrator_open`，对齐旧版 `PipelineOrchestratorSimple`），写入 `data/orchestration_results/`。
- **编排**：三阶段 LLM 出 DAG；落盘后 **自动**做结构/注册表检查 + **LLM 综合评估**（理解 + 已有算子 + DAG），写入 `dag_validation` / `validation_result` / `data/dag_assessment_results/`。
- **算子进化**：仅当评估 **recommend_new_operators=true** 时需执行 **evolve-operators**（LLM 生成粗粒度算子）。若评估通过且未建议新算子，workflow **跳过**该步，下一步直接 **instantiate**。评估未通过且未建议新算子时下一步为重新 **orchestrate**。
- **实例化 / 试运行**：`pipeline_runtime`（`instantiation` 生成 `run_pipeline.py` 入口；`trial` 采样 + 默认 Pilot LLM 评估）生成分步桩、采样验证。
- **全量执行**：由独立 `run-full` API 触发，不作为 workflow 中间推进步。
- **质检 / 经验**：`workflow.snapshots` 聚合各阶段落盘结果。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from subsystems.operator_management.registry_store import OperatorRegistryStore
from subsystems.structured_understanding.orchestration_feedback import (
    persist_assessment_feedback_to_understanding,
    persist_pilot_evaluation_to_understanding,
)
from subsystems.pipeline_orchestration import run_open_orchestration
from subsystems.pipeline_orchestration.dag_semantic_llm import assess_pipeline_task_fit, evolve_operators_via_llm
from subsystems.pipeline_orchestration.dag_validator import merge_dag_validation
from subsystems.pipeline_runtime import (
    execute_generated_pipeline,
    patch_orchestration_pipeline_run_summary,
    patch_orchestration_trial_summary,
    run_instantiation,
    write_trial_artifacts,
)
from subsystems.pipeline_runtime.pilot.flow import run_trial_with_optional_pilot_judge
from subsystems.pipeline_session.manifest_io_paths import apply_manifest_file_paths_to_pipeline
from subsystems.pipeline_session.manifest_store import get_latest_manifest_record
from subsystems.structured_understanding import run_understanding
from subsystems.observability.token_usage_ledger import append_token_event
from subsystems.workflow.artifact_history import (
    archive_orchestration_before_overwrite,
    archive_understanding_before_overwrite,
    snapshot_iteration_artifacts,
    snapshot_round_artifacts,
)
from subsystems.workflow.snapshots import build_experience_snapshot, build_quality_check_snapshot

MAX_DAG_EVOLUTION_ORCHESTRATION_CYCLES = 20

STEP_ORDER: list[str] = [
    "understanding",
    "orchestration",
    "operator_evolution",
    "instantiation",
    "trial_run",
    "quality_check",
    # 仅在 quality_check 不达标时进入该步，用于生成经验并回流下一轮
    "experience",
]


class WorkflowStepError(RuntimeError):
    """
    当前步执行失败：已在 advance 内捕获，**不会**写入下一步的 state（step_index / steps_completed 不变）。
    供 HTTP 层返回结构化 detail，便于前端提示「卡在哪一步、是否可重试」。
    """

    def __init__(self, step_key: str, message: str, *, cause: BaseException | None = None) -> None:
        self.step_key = step_key
        self.cause = cause
        super().__init__(message)


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass
class WorkflowState:
    pipeline_id: str
    step_index: int = 0
    steps_completed: list[str] = field(default_factory=list)
    last_message: str = ""
    updated_at: str = ""
    dag_evolution_cycles: int = 0  # 编排↔校验↔进化闭环；校验通过并跳过进化时清零
    # 成功完成次数（用于归档命名与 Pilot 历史）；显式重跑 understand/orchestrate 前若已有成品会先入 artifact_history
    understanding_revision: int = 0
    orchestration_revision: int = 0
    round: int = 1
    quality_passed: bool = False
    ready_for_full_run: bool = False
    next_action: str = "advance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "step_index": self.step_index,
            "steps_completed": list(self.steps_completed),
            "last_message": self.last_message,
            "updated_at": self.updated_at,
            "dag_evolution_cycles": self.dag_evolution_cycles,
            "understanding_revision": self.understanding_revision,
            "orchestration_revision": self.orchestration_revision,
            "round": self.round,
            "quality_passed": self.quality_passed,
            "ready_for_full_run": self.ready_for_full_run,
            "next_action": self.next_action,
            "step_order": STEP_ORDER,
            "is_complete": self.step_index >= len(STEP_ORDER),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowState:
        return cls(
            pipeline_id=str(d.get("pipeline_id", "")),
            step_index=int(d.get("step_index", 0)),
            steps_completed=list(d.get("steps_completed") or []),
            last_message=str(d.get("last_message", "")),
            updated_at=str(d.get("updated_at", "")),
            dag_evolution_cycles=int(d.get("dag_evolution_cycles", 0)),
            understanding_revision=int(d.get("understanding_revision", 0)),
            orchestration_revision=int(d.get("orchestration_revision", 0)),
            round=max(1, int(d.get("round", 1))),
            quality_passed=bool(d.get("quality_passed", False)),
            ready_for_full_run=bool(d.get("ready_for_full_run", False)),
            next_action=str(d.get("next_action", "advance")),
        )


def _state_path(root: Path, pipeline_id: str) -> Path:
    return root / "data" / "workflow_runs" / pipeline_id / "state.json"


def load_workflow_state(root: Path, pipeline_id: str) -> WorkflowState:
    p = _state_path(root, pipeline_id)
    if not p.exists():
        return WorkflowState(pipeline_id=pipeline_id)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            st = WorkflowState.from_dict(data)
            st.pipeline_id = pipeline_id
            return st
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return WorkflowState(pipeline_id=pipeline_id)


def _save_state(root: Path, state: WorkflowState) -> None:
    state.updated_at = _iso()
    _atomic_write_json(_state_path(root, state.pipeline_id), state.to_dict())


def _assert_step_prerequisites(root: Path, pipeline_id: str, step_key: str) -> None:
    """显式执行某步时仅校验产物/注册前置条件，不强制与 state.step_index 一致。"""
    manifest_path = root / "data" / "manifest.jsonl"
    u = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    o = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    g = root / "data" / "generated_pipelines" / f"{pipeline_id}.json"

    need_manifest = {
        "understanding",
        "orchestration",
        "operator_evolution",
        "instantiation",
        "trial_run",
        "quality_check",
        "experience",
    }
    if step_key in need_manifest:
        if get_latest_manifest_record(manifest_path, pipeline_id) is None:
            raise FileNotFoundError(
                f"manifest 中未找到 pipeline_id={pipeline_id!r}，请先完成会话/管线注册。"
            )

    after_u = {
        "orchestration",
        "operator_evolution",
        "instantiation",
        "trial_run",
        "quality_check",
        "experience",
    }
    if step_key in after_u and not u.is_file():
        raise FileNotFoundError("缺少理解结果 data/understanding_results/，请先运行 understand。")

    after_o = {
        "operator_evolution",
        "instantiation",
        "trial_run",
        "quality_check",
        "experience",
    }
    if step_key in after_o and not o.is_file():
        raise FileNotFoundError("缺少编排结果 data/orchestration_results/，请先运行 orchestrate。")

    if step_key == "trial_run" and not g.is_file():
        raise FileNotFoundError("缺少实例化产物，请先运行 instantiate。")
    if step_key in {"quality_check", "experience"}:
        trial = root / "data" / "trial_runs" / pipeline_id / "trial_result.json"
        if not trial.is_file():
            raise FileNotFoundError("缺少试运行结果，请先运行 trial。")


def _registry_summaries_for_assessment(store: OperatorRegistryStore, limit: int = 160) -> list[dict[str, Any]]:
    raw = store.merged_raw()
    out: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return out
    for name, spec in list(raw.items())[:limit]:
        if not isinstance(spec, dict):
            out.append({"name": str(name), "description": ""})
            continue
        out.append(
            {
                "name": str(name),
                "description": str(spec.get("description") or "")[:280],
                "category": spec.get("category"),
                "input_keys": spec.get("input_keys"),
                "output_keys": spec.get("output_keys"),
            }
        )
    return out


def run_pipeline_assessment_and_persist(
    root: Path,
    pipeline_id: str,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None,
    *,
    usage_operation: str = "orchestration.llm_task_fit",
) -> dict[str, Any]:
    """
    对已有编排 JSON 做结构合并校验 + LLM 任务/算子缺口评估，写回编排文件与 dag_assessment_results。
    供编排步末尾与 CLI「仅刷新评估」调用。
    """
    path = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    if not path.exists():
        raise FileNotFoundError("缺少编排结果")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("编排结果 JSON 格式无效")
    store = OperatorRegistryStore(root, pipeline_id=pipeline_id)
    base = merge_dag_validation(data, merged_registry=store.merged_raw(), checked_at=_iso())
    structural_valid = bool(base.get("is_valid"))
    structural_issues: list[Any] = list(base.get("validation_issues") or [])

    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    understanding: dict[str, Any] = {}
    if u_path.is_file():
        try:
            u_raw = json.loads(u_path.read_text(encoding="utf-8"))
            if isinstance(u_raw, dict):
                understanding = u_raw
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    reg_sum = _registry_summaries_for_assessment(store)
    llm_a = assess_pipeline_task_fit(
        understanding=understanding,
        orchestration=data,
        structural_valid=structural_valid,
        structural_issues=structural_issues,
        llm_config=llm_config,
        on_usage=on_usage,
        registry_operator_summaries=reg_sum,
        usage_operation=usage_operation,
    )
    task_ok = bool(llm_a.get("satisfied", True))
    combined_ok = structural_valid and task_ok

    vr = {
        **base,
        "structural_valid": structural_valid,
        "llm_task_assessment": llm_a,
        "is_valid": combined_ok,
    }
    cs = data.setdefault("constrained_search", {})
    cs["validation_result"] = vr

    assess_rel = f"data/dag_assessment_results/{pipeline_id}.json"
    assess_path = root / assess_rel
    assess_path.parent.mkdir(parents=True, exist_ok=True)
    issues_serializable: list[Any] = []
    for it in structural_issues[:80]:
        if isinstance(it, dict):
            issues_serializable.append({str(k): v for k, v in it.items()})
        else:
            issues_serializable.append({"description": str(it)})

    fixes = list(llm_a.get("recommended_fixes") or [])
    if not isinstance(fixes, list):
        fixes = []
    opts = list(llm_a.get("optimization_suggestions") or [])
    if not isinstance(opts, list):
        opts = []
    rec_ops = bool(llm_a.get("recommend_new_operators"))
    nxt = list(llm_a.get("next_steps_for_user") or [])
    if not isinstance(nxt, list):
        nxt = []

    assessment_doc: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "assessed_at": vr.get("checked_at"),
        "structural_valid": structural_valid,
        "structural_issue_count": len(structural_issues),
        "structural_issues": issues_serializable,
        "task_fit": {
            "satisfied": task_ok,
            "skipped": bool(llm_a.get("skipped")),
            "reasoning": llm_a.get("reasoning"),
            "recommend_new_operators": rec_ops,
            "recommended_fixes": fixes,
            "optimization_suggestions": opts,
            "next_steps_for_user": nxt,
        },
        "is_valid": combined_ok,
        "orchestration_results_path": f"data/orchestration_results/{pipeline_id}.json",
    }
    _atomic_write_json(assess_path, assessment_doc)

    data["dag_validation"] = {
        "checked_at": vr.get("checked_at"),
        "is_valid": combined_ok,
        "issue_count": len(structural_issues),
        "structural_valid": structural_valid,
        "task_fit_satisfied": task_ok,
        "task_fit_skipped": bool(llm_a.get("skipped")),
        "recommend_new_operators": rec_ops,
        "assessment_detail_path": assess_rel,
        "llm": {
            "reasoning": llm_a.get("reasoning"),
            "recommend_new_operators": rec_ops,
            "recommended_fixes": fixes,
            "optimization_suggestions": opts,
            "next_steps_for_user": nxt,
        },
    }
    _atomic_write_json(path, data)
    fb_ok = persist_assessment_feedback_to_understanding(
        root,
        pipeline_id,
        combined_ok=combined_ok,
        assessed_at=str(vr.get("checked_at") or ""),
        structural_issue_count=len(structural_issues),
        structural_issues=structural_issues,
        task_ok=task_ok,
        task_fit_skipped=bool(llm_a.get("skipped")),
        reasoning=str(llm_a.get("reasoning") or ""),
        recommend_new_operators=rec_ops,
        recommended_fixes=fixes,
        optimization_suggestions=opts,
        next_steps_for_user=nxt,
    )
    return {
        "path": f"data/orchestration_results/{pipeline_id}.json",
        "assessment_path": assess_rel,
        "is_valid": combined_ok,
        "structural_valid": structural_valid,
        "structural_issue_count": len(structural_issues),
        "issue_count": len(structural_issues),
        "task_fit_satisfied": task_ok,
        "task_fit_skipped": bool(llm_a.get("skipped")),
        "recommend_new_operators": rec_ops,
        "task_fit_reasoning": (llm_a.get("reasoning") or "")[:2000],
        "recommended_fixes": fixes[:8],
        "optimization_suggestions": opts[:8],
        "next_steps_for_user": nxt[:8],
        "understanding_feedback_persisted": fb_ok,
    }


def _step_understanding(
    root: Path,
    pipeline_id: str,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    out = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    if force and out.exists():
        st0 = load_workflow_state(root, pipeline_id)
        if st0.understanding_revision >= 1:
            archive_understanding_before_overwrite(
                root,
                pipeline_id,
                understanding_revision=st0.understanding_revision,
                dag_evolution_cycles=st0.dag_evolution_cycles,
            )
        cascade_clear_workflow_artifacts(root, pipeline_id, 0)
    if out.exists() and not force:
        return {
            "stage": "understanding",
            "status": "skipped",
            "detail": "understanding 结果已存在，删除该文件可强制重跑",
            "path": f"data/understanding_results/{pipeline_id}.json",
        }
    result = run_understanding(
        root,
        pipeline_id,
        mode="auto",
        llm_config=llm_config,
        on_usage=on_usage,
    )
    return {
        "stage": "understanding",
        "status": "completed",
        "path": result.get("saved_path"),
        "language": result.get("language"),
    }


def _step_orchestration(
    root: Path,
    pipeline_id: str,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    path = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    if force:
        cascade_clear_workflow_artifacts(root, pipeline_id, 3)
        da = root / "data" / "dag_assessment_results" / f"{pipeline_id}.json"
        if da.is_file():
            da.unlink()
    st_orch = load_workflow_state(root, pipeline_id)
    if path.is_file() and st_orch.orchestration_revision >= 1 and force:
        archive_orchestration_before_overwrite(
            root,
            pipeline_id,
            orchestration_revision=st_orch.orchestration_revision,
            understanding_revision=st_orch.understanding_revision,
            dag_evolution_cycles=st_orch.dag_evolution_cycles,
            round=st_orch.round,
        )
    if path.is_file() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            existing = None
        dag_invalid = False
        if isinstance(existing, dict):
            dv = existing.get("dag_validation")
            if isinstance(dv, dict) and dv.get("is_valid") is False:
                dag_invalid = True
            cs = existing.get("constrained_search")
            if isinstance(cs, dict):
                vr = cs.get("validation_result")
                if isinstance(vr, dict) and vr.get("is_valid") is False:
                    dag_invalid = True
        if dag_invalid and st_orch.orchestration_revision >= 1:
            archive_orchestration_before_overwrite(
                root,
                pipeline_id,
                orchestration_revision=st_orch.orchestration_revision,
                understanding_revision=st_orch.understanding_revision,
                dag_evolution_cycles=st_orch.dag_evolution_cycles,
                round=st_orch.round,
            )
            path.unlink()
        else:
            return {
                "stage": "orchestration",
                "status": "skipped",
                "path": f"data/orchestration_results/{pipeline_id}.json",
            }
    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    if not u_path.exists():
        raise FileNotFoundError("缺少理解结果，无法编排")
    understanding = json.loads(u_path.read_text(encoding="utf-8"))
    if not isinstance(understanding, dict):
        raise ValueError("理解结果 JSON 格式无效")

    manifest_path = root / "data" / "manifest.jsonl"
    record = get_latest_manifest_record(manifest_path, pipeline_id)
    if record is None:
        raise FileNotFoundError(f"manifest 中未找到 pipeline_id={pipeline_id!r}，无法解析 read/write 路径")

    orch_out = run_open_orchestration(
        root,
        pipeline_id,
        understanding,
        record,
        llm_config,
        on_usage=on_usage,
    )
    fp = apply_manifest_file_paths_to_pipeline(
        orch_out["final_pipeline"],
        pipeline_id=pipeline_id,
        manifest_record=record if isinstance(record, dict) else {},
    )
    cs = orch_out.get("constrained_search")
    if isinstance(cs, dict):
        cs["final_pipeline"] = fp
    payload: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "dag": orch_out["dag"],
        "pipeline_plan": orch_out["pipeline_plan"],
        "source": "open_orchestrator_v1",
        "meta": {
            "created_at": _iso(),
            "llm_stages": ["free_fitting", "template_combination", "constrained_search"],
        },
        "free_fitting": orch_out["free_fitting"],
        "template_combination": orch_out["template_combination"],
        "constrained_search": orch_out["constrained_search"],
        "final_pipeline": fp,
    }
    _atomic_write_json(path, payload)
    assess_detail = run_pipeline_assessment_and_persist(
        root, pipeline_id, llm_config, on_usage, usage_operation="orchestration.llm_task_fit"
    )
    return {
        "stage": "orchestration",
        "status": "completed",
        "path": f"data/orchestration_results/{pipeline_id}.json",
        **assess_detail,
    }


def _step_operator_evolution(
    root: Path,
    pipeline_id: str,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None,
) -> dict[str, Any]:
    path = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    if not path.exists():
        raise FileNotFoundError("缺少编排结果")
    data = json.loads(path.read_text(encoding="utf-8"))
    cs = data.setdefault("constrained_search", {})
    vr = cs.get("validation_result") if isinstance(cs.get("validation_result"), dict) else {}
    ols = cs.setdefault("operator_level_self_evolving", {})
    suggested = [x for x in (ols.get("suggested_operators") or []) if x]

    structural_valid = vr.get("structural_valid")
    if structural_valid is None:
        structural_valid = len(vr.get("validation_issues") or []) == 0
    llm_a = vr.get("llm_task_assessment") if isinstance(vr.get("llm_task_assessment"), dict) else {}
    task_ok = bool(llm_a.get("satisfied", True))
    want_new = bool(llm_a.get("recommend_new_operators"))

    def _skip(msg: str, **extra: Any) -> dict[str, Any]:
        ols["triggered"] = False
        ols.setdefault("added_operators", [])
        ols["re_orchestrated"] = False
        ols["llm_proposed"] = False
        ols["skip_reason"] = msg
        _atomic_write_json(path, data)
        return {
            "stage": "operator_evolution",
            "status": "skipped",
            "detail": msg,
            "path": f"data/orchestration_results/{pipeline_id}.json",
            **extra,
        }

    if structural_valid and task_ok and not suggested:
        return _skip("编排评估已通过，无需新增注册算子。")

    if not want_new and not suggested:
        return _skip(
            "评估认为当前问题可通过重新编排（调整 DAG / 选用已有算子）解决，未建议新增注册算子。",
            rewind_to_orchestration_soft=True,
        )

    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    understanding: dict[str, Any] = {}
    if u_path.is_file():
        try:
            u_raw = json.loads(u_path.read_text(encoding="utf-8"))
            if isinstance(u_raw, dict):
                understanding = u_raw
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    proposed = evolve_operators_via_llm(
        pipeline_id=pipeline_id,
        understanding=understanding,
        orchestration=data,
        assessment=llm_a,
        structural_issues=list(vr.get("validation_issues") or []),
        llm_config=llm_config,
        on_usage=on_usage,
    )

    store = OperatorRegistryStore(root, pipeline_id=pipeline_id)
    added_names: list[str] = []
    llm_used = bool(proposed)
    memory_update: dict[str, Any] | None = None
    if proposed:
        memory_update = store.assimilate_evolved_operators(
            proposed,
            pipeline_id=pipeline_id,
            assessment=llm_a if isinstance(llm_a, dict) else None,
        )
        added_names = list(memory_update.get("task_added") or [])

    if not proposed:
        ols["triggered"] = False
        ols["llm_proposed"] = False
        ols["skip_reason"] = "模型未返回可用算子定义，未写入注册表"
        _atomic_write_json(path, data)
        return {
            "stage": "operator_evolution",
            "status": "completed",
            "added_operators": [],
            "llm_generated_operators": False,
            "registry_path": store.user_path_relative,
            "memory_update": {
                "task_added": [],
                "promoted_to_domain": [],
                "promoted_to_general": [],
                "domain_key": store.domain_key,
            },
            "detail": ols["skip_reason"],
            "rewind_to_orchestration_soft": True,
        }

    ols["triggered"] = True
    ols["added_operators"] = list(set(ols.get("added_operators") or []) | set(added_names))
    ols["llm_proposed"] = llm_used
    ols["re_orchestrated"] = False
    ols.pop("skip_reason", None)
    _atomic_write_json(path, data)
    return {
        "stage": "operator_evolution",
        "status": "completed",
        "added_operators": added_names,
        "llm_generated_operators": llm_used,
        "registry_path": store.user_path_relative,
        "memory_update": memory_update
        or {
            "task_added": added_names,
            "promoted_to_domain": [],
            "promoted_to_general": [],
            "domain_key": store.domain_key,
        },
        "rewind_to_orchestration": True,
    }


def _step_instantiation(
    root: Path,
    pipeline_id: str,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    path = root / "data" / "generated_pipelines" / f"{pipeline_id}.json"
    d = root / "data" / "generated_pipelines" / pipeline_id
    if force:
        if path.is_file():
            path.unlink()
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
        cascade_clear_workflow_artifacts(root, pipeline_id, 4)
    if path.exists() and not force:
        return {
            "stage": "instantiation",
            "status": "skipped",
            "path": f"data/generated_pipelines/{pipeline_id}.json",
            "detail": "复用已有实例化产物（未调用 LLM）；删除主 JSON 及同名目录可强制重跑",
            "reused": True,
            "llm_codegen": False,
        }
    orch = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    if not orch.is_file():
        raise FileNotFoundError("缺少编排结果，无法实例化")
    payload = run_instantiation(root, pipeline_id, llm_config=llm_config, on_usage=on_usage)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    llm_steps = meta.get("llm_prompt_generated_steps") if isinstance(meta.get("llm_prompt_generated_steps"), list) else []
    return {
        "stage": "instantiation",
        "status": "completed",
        "path": f"data/generated_pipelines/{pipeline_id}.json",
        "total_steps": meta.get("total_steps"),
        "source": payload.get("source"),
        "codegen_warnings": meta.get("codegen_warnings") or [],
        "entry_script": meta.get("entry_script"),
        "llm_codegen": len(llm_steps) > 0,
        "llm_prompt_generated_steps": llm_steps,
        "reused": False,
    }


def _step_trial_run(
    root: Path,
    pipeline_id: str,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    trial_dir = root / "data" / "trial_runs" / pipeline_id
    if force and trial_dir.exists():
        shutil.rmtree(trial_dir, ignore_errors=True)
    trial_path = trial_dir / "trial_result.json"
    if trial_path.is_file() and not force:
        return {
            "stage": "trial_run",
            "status": "skipped",
            "path": f"data/trial_runs/{pipeline_id}/trial_result.json",
            "detail": "删除该文件可强制重跑试运行",
        }
    gen = root / "data" / "generated_pipelines" / f"{pipeline_id}.json"
    if not gen.is_file():
        raise FileNotFoundError("缺少实例化产物，无法试运行")
    manifest_path = root / "data" / "manifest.jsonl"
    record = get_latest_manifest_record(manifest_path, pipeline_id)
    if record is None:
        raise FileNotFoundError(f"manifest 中未找到 pipeline_id={pipeline_id!r}")
    trial = run_trial_with_optional_pilot_judge(
        root,
        pipeline_id,
        record,
        max_records=8,
        llm_config=llm_config,
        on_usage=on_usage,
        with_llm_judge=True,
    )
    write_trial_artifacts(root, pipeline_id, trial)
    st_tr = load_workflow_state(root, pipeline_id)
    pilot_feedback_persisted = persist_pilot_evaluation_to_understanding(
        root,
        pipeline_id,
        trial,
        workflow_meta={
            "understanding_revision": st_tr.understanding_revision,
            "orchestration_revision": st_tr.orchestration_revision,
            "dag_evolution_cycles": st_tr.dag_evolution_cycles,
        },
    )
    patch_orchestration_trial_summary(root, pipeline_id, trial)
    pilot = trial.get("llm_pilot_evaluation") if isinstance(trial.get("llm_pilot_evaluation"), dict) else {}
    out: dict[str, Any] = {
        "stage": "trial_run",
        "status": "completed",
        "path": f"data/trial_runs/{pipeline_id}/trial_result.json",
        "execution_ok": trial.get("execution_ok"),
        "reflux_targets": (trial.get("reflux_recommendation") or {}).get("targets"),
        "pilot_feedback_persisted": pilot_feedback_persisted,
    }
    if pilot.get("present"):
        out["pilot_overall_score"] = pilot.get("overall_score")
        out["pilot_recommendation"] = pilot.get("recommendation")
        if isinstance(pilot.get("dimension_scores"), dict):
            out["pilot_dimension_scores"] = pilot.get("dimension_scores")
    elif pilot.get("skipped_reason"):
        out["pilot_judge_skipped"] = pilot.get("skipped_reason")
    return out


def _step_pipeline_run(
    root: Path,
    pipeline_id: str,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None,
    *,
    execution_mode: Literal["in_process", "subprocess"] = "in_process",
    subprocess_fallback_in_process: bool = True,
    subprocess_timeout_sec: float = 600.0,
    force: bool = False,
) -> dict[str, Any]:
    run_root = root / "data" / "run_pipeline_results" / pipeline_id
    if force and run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    latest = run_root / "latest.json"
    if latest.is_file() and not force:
        try:
            j = json.loads(latest.read_text(encoding="utf-8"))
            if j.get("status") == "success" and j.get("run_dir"):
                rr = root / str(j["run_dir"]) / "run_report.json"
                if rr.is_file():
                    rep = json.loads(rr.read_text(encoding="utf-8"))
                    if rep.get("status") == "success":
                        return {
                            "stage": "pipeline_run",
                            "status": "skipped",
                            "path": str(j.get("run_dir")),
                            "detail": "删除 data/run_pipeline_results/{id}/latest.json 或整目录后可强制重跑",
                        }
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    gen = root / "data" / "generated_pipelines" / f"{pipeline_id}.json"
    if not gen.is_file():
        raise FileNotFoundError("缺少实例化产物，无法全量执行")
    report = execute_generated_pipeline(
        root,
        pipeline_id,
        llm_config=llm_config,
        on_usage=on_usage,
        max_input_records=None,
        llm_max_records_per_step=32,
        execution_mode=execution_mode,
        subprocess_fallback_in_process=subprocess_fallback_in_process,
        subprocess_timeout_sec=subprocess_timeout_sec,
    )
    patch_orchestration_pipeline_run_summary(root, pipeline_id, report)
    report.pop("records", None)
    if report.get("status") != "success":
        raise RuntimeError(
            report.get("error")
            or f"管线执行失败: step={report.get('failed_step')!r}"
        )
    return {
        "stage": "pipeline_run",
        "status": "completed",
        "run_dir": (report.get("meta") or {}).get("run_dir"),
        "output_jsonl": report.get("output_jsonl"),
        "output_record_count": report.get("output_record_count"),
        "pipeline_status": report.get("status"),
    }


def run_full_pipeline(
    root: Path,
    pipeline_id: str,
    *,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None = None,
    pipeline_run_execution_mode: Literal["in_process", "subprocess"] = "in_process",
    pipeline_run_subprocess_fallback_in_process: bool = True,
    pipeline_run_subprocess_timeout_sec: float = 600.0,
    force: bool = False,
) -> dict[str, Any]:
    """独立执行 full run（不依赖 workflow STEP_ORDER，供 CLI/API 直接调用）。"""
    detail = _step_pipeline_run(
        root,
        pipeline_id,
        llm_config,
        on_usage,
        execution_mode=pipeline_run_execution_mode,
        subprocess_fallback_in_process=pipeline_run_subprocess_fallback_in_process,
        subprocess_timeout_sec=pipeline_run_subprocess_timeout_sec,
        force=force,
    )
    st = load_workflow_state(root, pipeline_id)
    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "step": "pipeline_run",
        "detail": detail,
        "state": st.to_dict(),
        "invocation": "explicit",
    }


def _step_quality_check(root: Path, pipeline_id: str, *, force: bool = False) -> dict[str, Any]:
    path = root / "data" / "quality_check_results" / f"{pipeline_id}.json"
    if force and path.is_file():
        path.unlink()
    if path.exists() and not force:
        has_differences: bool | None = None
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                has_differences = bool(existing.get("has_differences"))
        except (OSError, json.JSONDecodeError, TypeError):
            has_differences = None
        return {
            "stage": "quality_check",
            "status": "skipped",
            "path": f"data/quality_check_results/{pipeline_id}.json",
            "has_differences": has_differences,
        }
    payload = build_quality_check_snapshot(root, pipeline_id)
    _atomic_write_json(path, payload)
    return {
        "stage": "quality_check",
        "status": "completed",
        "path": f"data/quality_check_results/{pipeline_id}.json",
        "has_differences": payload.get("has_differences"),
        "source": payload.get("source"),
    }


def _step_experience(root: Path, pipeline_id: str, *, force: bool = False) -> dict[str, Any]:
    exp_dir = root / "data" / "experiences"
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / f"{pipeline_id}.json"
    if force and path.is_file():
        path.unlink()
    payload = build_experience_snapshot(root, pipeline_id)
    _atomic_write_json(path, payload)

    # ── 把本轮Pilot分数回写到strategy_pool ──
    candidate_sids = payload.get("candidate_strategy_ids") or []
    try:
        trial_path = root / "data" / "trial_runs" / pipeline_id / "trial_result.json"
        if trial_path.is_file() and candidate_sids:
            trial_data = json.loads(trial_path.read_text(encoding="utf-8"))
            pilot = trial_data.get("llm_pilot_evaluation") or {}
            score = pilot.get("overall_score")
            dim_scores = pilot.get("dimension_scores")
            if score is not None:
                from subsystems.workflow.strategy_pool import update_strategy_score
                for sid in candidate_sids:
                    update_strategy_score(
                        root, pipeline_id, sid,
                        score=int(score),
                        dimension_scores=dim_scores,
                        trial_path=f"data/trial_runs/{pipeline_id}/trial_result.json",
                    )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("strategy_pool分数回写失败: %s", e)

    return {
        "stage": "experience",
        "status": "completed",
        "path": f"data/experiences/{pipeline_id}.json",
        "source": payload.get("source"),
        "llm_used": payload.get("meta", {}).get("llm_diagnosis_triggered", False),
        "candidate_strategies": candidate_sids,
        "source_kind": "rule_aggregation_with_llm_diagnosis",
        "detail": "经验由质检/试运行/Pilot结果聚合生成，含LLM算子诊断与策略池更新",
    }


def _clear_for_next_round(root: Path, pipeline_id: str) -> list[str]:
    """
    进入下一轮时仅清理本轮中间产物，保留 experience 供下一轮 understanding 注入。
    """
    data = root / "data"
    touched: list[str] = []
    targets = [
        data / "understanding_results" / f"{pipeline_id}.json",
        data / "orchestration_results" / f"{pipeline_id}.json",
        data / "dag_assessment_results" / f"{pipeline_id}.json",
        data / "generated_pipelines" / f"{pipeline_id}.json",
        data / "quality_check_results" / f"{pipeline_id}.json",
    ]
    for p in targets:
        if p.is_file():
            p.unlink()
            touched.append(str(p.relative_to(root)))

    for d in [
        data / "generated_pipelines" / pipeline_id,
        data / "trial_runs" / pipeline_id,
        data / "run_pipeline_results" / pipeline_id,
    ]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            touched.append(str(d.relative_to(root)))
    return touched


def cascade_clear_workflow_artifacts(root: Path, pipeline_id: str, from_step_index: int) -> list[str]:
    """
    从 `STEP_ORDER[from_step_index]` 起「推倒重来」：删/改该步及之后步骤的落盘产物。
    STEP_ORDER: understanding(0), orchestration(1), operator_evolution(2), instantiation(3), trial_run(4), quality_check(5), experience(6)。
    - from_step_index==2：仅重置编排 JSON 内 operator_level_self_evolving（rerun evolve-operators）。
    """
    actions: list[str] = []
    data = root / "data"

    if from_step_index <= 0:
        p = data / "understanding_results" / f"{pipeline_id}.json"
        if p.is_file():
            p.unlink()
            actions.append(str(p.relative_to(root)))

    if from_step_index <= 1:
        p = data / "orchestration_results" / f"{pipeline_id}.json"
        if p.is_file():
            p.unlink()
            actions.append(str(p.relative_to(root)))
        da = data / "dag_assessment_results" / f"{pipeline_id}.json"
        if da.is_file():
            da.unlink()
            actions.append(str(da.relative_to(root)))
    elif from_step_index == 2:
        p = data / "orchestration_results" / f"{pipeline_id}.json"
        if p.is_file():
            try:
                o = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(o, dict):
                    cs = o.setdefault("constrained_search", {})
                    if isinstance(cs, dict):
                        cs["operator_level_self_evolving"] = {}
                    _atomic_write_json(p, o)
                    actions.append(f"reset operator_level_self_evolving in {p.relative_to(root)}")
            except (OSError, json.JSONDecodeError, TypeError):
                pass

    if from_step_index <= 3:
        p = data / "generated_pipelines" / f"{pipeline_id}.json"
        if p.is_file():
            p.unlink()
            actions.append(str(p.relative_to(root)))
        d = data / "generated_pipelines" / pipeline_id
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            actions.append(str(d.relative_to(root)))

    if from_step_index <= 4:
        d = data / "trial_runs" / pipeline_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            actions.append(str(d.relative_to(root)))

    if from_step_index <= 5:
        d = data / "run_pipeline_results" / pipeline_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            actions.append(str(d.relative_to(root)))

    if from_step_index <= 6:
        p = data / "quality_check_results" / f"{pipeline_id}.json"
        if p.is_file():
            p.unlink()
            actions.append(str(p.relative_to(root)))

    if from_step_index <= 6:
        p = data / "experiences" / f"{pipeline_id}.json"
        if p.is_file():
            p.unlink()
            actions.append(str(p.relative_to(root)))

    return actions


def rerun_workflow_from_step(root: Path, pipeline_id: str, step_key: str) -> dict[str, Any]:
    """
    将 workflow 状态重置为「下一步将执行 step_key」，并级联清理该步及之后的产物。
    例：`rerun_workflow_from_step(..., \"understanding\")` 后执行 `advance` 会重新跑理解（并删掉编排及之后所有产物）。
    """
    if step_key not in STEP_ORDER:
        raise ValueError(f"未知步骤 {step_key!r}，允许: {STEP_ORDER}")
    idx = STEP_ORDER.index(step_key)
    prev_state = load_workflow_state(root, pipeline_id)
    ufile = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    ofile = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    if idx <= 0 and ufile.is_file() and prev_state.understanding_revision >= 1:
        archive_understanding_before_overwrite(
            root,
            pipeline_id,
            understanding_revision=prev_state.understanding_revision,
            dag_evolution_cycles=prev_state.dag_evolution_cycles,
        )
    if idx <= 1 and ofile.is_file() and prev_state.orchestration_revision >= 1:
        archive_orchestration_before_overwrite(
            root,
            pipeline_id,
            orchestration_revision=prev_state.orchestration_revision,
            understanding_revision=prev_state.understanding_revision,
            dag_evolution_cycles=prev_state.dag_evolution_cycles,
            round=prev_state.round,
        )
    touched = cascade_clear_workflow_artifacts(root, pipeline_id, idx)
    state = WorkflowState(
        pipeline_id=pipeline_id,
        step_index=idx,
        steps_completed=[],
        last_message=f"rerun_from={step_key}",
        updated_at="",
        dag_evolution_cycles=0,
        understanding_revision=prev_state.understanding_revision,
        orchestration_revision=prev_state.orchestration_revision,
        # 从 understanding 重跑时视为新一轮从头开始，轮次归 1，避免前端出现“第 2 轮 Raw”。
        round=1 if idx == 0 else max(1, prev_state.round),
        quality_passed=False,
        ready_for_full_run=False,
        next_action="advance",
    )
    _save_state(root, state)
    next_sk = STEP_ORDER[idx] if 0 <= idx < len(STEP_ORDER) else "understanding"
    cli_map = {
        "understanding": "understand",
        "orchestration": "orchestrate",
        "operator_evolution": "evolve-operators",
        "instantiation": "instantiate",
        "trial_run": "trial",
        "quality_check": "quality-check",
        "experience": "experience",
    }
    sub = cli_map.get(next_sk, "advance")
    next_cmd = f"dataevolver workflow {sub} {pipeline_id}"
    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "rerun_from": step_key,
        "step_index": idx,
        "artifacts_touched": touched,
        "next_command": next_cmd,
    }


def reset_workflow_for_debug(root: Path, pipeline_id: str) -> dict[str, Any]:
    """
    开发调试入口：强制回到「第一轮起点」。
    - 清理 understanding 及之后全部中间产物
    - 清理 workflow_runs/{id} 与 artifact_history/{id}
    - 重建 state：round=1, step_index=0
    """
    touched = cascade_clear_workflow_artifacts(root, pipeline_id, 0)
    data = root / "data"
    for d in [
        data / "workflow_runs" / pipeline_id,
        data / "artifact_history" / pipeline_id,
    ]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            touched.append(str(d.relative_to(root)))
    state = WorkflowState(
        pipeline_id=pipeline_id,
        step_index=0,
        steps_completed=[],
        last_message="debug_reset_to_round_1_entry",
        updated_at="",
        dag_evolution_cycles=0,
        understanding_revision=0,
        orchestration_revision=0,
        round=1,
        quality_passed=False,
        ready_for_full_run=False,
        next_action="advance",
    )
    _save_state(root, state)
    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "state": state.to_dict(),
        "artifacts_cleared": touched,
    }


def advance_workflow(
    root: Path,
    pipeline_id: str,
    *,
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None = None,
    force: bool = False,
    pipeline_run_execution_mode: Literal["in_process", "subprocess"] = "in_process",
    pipeline_run_subprocess_fallback_in_process: bool = True,
    pipeline_run_subprocess_timeout_sec: float = 600.0,
    requested_step: str | None = None,
    step_force: bool = False,
) -> dict[str, Any]:
    """
    执行一步。

    - 线性推进：不传 ``requested_step``，按 ``state.step_index`` 取下一步；``step_force`` 无效。
    - 显式步骤（如 CLI 子命令）：传 ``requested_step``，只校验前置产物，不要求与 ``step_index`` 一致；
      ``step_force=True`` 时对会「跳过」的步强制重跑（覆盖/删除旧产物）。
    - ``force=True``：删除该 pipeline 的 ``state.json``（慎用）。
    """
    if force:
        sp = _state_path(root, pipeline_id)
        if sp.exists():
            sp.unlink()

    state = load_workflow_state(root, pipeline_id)
    explicit = requested_step is not None
    if state.step_index >= len(STEP_ORDER) and not explicit:
        return {
            "ok": True,
            "pipeline_id": pipeline_id,
            "done": True,
            "message": (
                "质检闭环已完成；若 quality_passed=true，可执行 run-full。"
                "可 POST reset_state=true 或删除 workflow_runs 目录重来"
            ),
            "state": state.to_dict(),
        }

    if explicit:
        if requested_step not in STEP_ORDER:
            raise WorkflowStepError(
                str(requested_step),
                f"未知 workflow 步骤 {requested_step!r}，允许: {STEP_ORDER}",
            )
        key = requested_step
        _assert_step_prerequisites(root, pipeline_id, key)
        apply_force = step_force
    else:
        key = STEP_ORDER[state.step_index]
        apply_force = False

    model_default = str(llm_config.get("model") or "")

    def _usage_cb(
        *,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
        operation: str | None = None,
        duration_ms: float | int | None = None,
        request_id: str | None = None,
        api_host: str | None = None,
        **extra: Any,
    ) -> None:
        append_token_event(
            root,
            pipeline_id,
            workflow_step=key,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or model_default,
            operation=operation,
            duration_ms=duration_ms if duration_ms is not None else None,
            request_id=request_id if request_id else None,
            api_host=api_host if api_host else None,
        )
        if on_usage is not None:
            try:
                on_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    operation=operation,
                    duration_ms=duration_ms,
                    request_id=request_id,
                    api_host=api_host,
                    **{k: v for k, v in extra.items() if v is not None},
                )
            except TypeError:
                on_usage(input_tokens=input_tokens, output_tokens=output_tokens, model=model)

    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "understanding": lambda: _step_understanding(
            root, pipeline_id, llm_config, _usage_cb, force=apply_force
        ),
        "orchestration": lambda: _step_orchestration(
            root, pipeline_id, llm_config, _usage_cb, force=apply_force
        ),
        "operator_evolution": lambda: _step_operator_evolution(root, pipeline_id, llm_config, _usage_cb),
        "instantiation": lambda: _step_instantiation(
            root, pipeline_id, llm_config, _usage_cb, force=apply_force
        ),
        "trial_run": lambda: _step_trial_run(
            root, pipeline_id, llm_config, _usage_cb, force=apply_force
        ),
        "quality_check": lambda: _step_quality_check(root, pipeline_id, force=apply_force),
        "experience": lambda: _step_experience(root, pipeline_id, force=apply_force),
    }
    fn = handlers[key]
    try:
        detail = fn()
    except Exception as e:
        raise WorkflowStepError(key, str(e) or type(e).__name__, cause=e) from e

    if key == "operator_evolution" and bool(detail.get("rewind_to_orchestration")):
        state.dag_evolution_cycles += 1
        state.ready_for_full_run = False
        state.quality_passed = False
        state.next_action = "advance"
        if state.dag_evolution_cycles > MAX_DAG_EVOLUTION_ORCHESTRATION_CYCLES:
            raise WorkflowStepError(
                key,
                f"编排→进化闭环已超过 {MAX_DAG_EVOLUTION_ORCHESTRATION_CYCLES} 轮（当前 {state.dag_evolution_cycles}）。"
                "请检查理解、注册表或手动调整后再 rerun。",
            )
        snapshot_iteration_artifacts(
            root,
            pipeline_id,
            round_no=max(1, state.round),
            dag_evolution_cycles=state.dag_evolution_cycles,
            reason="operator_evolution_rewind",
        )
        cascade_clear_workflow_artifacts(root, pipeline_id, 1)
        state.steps_completed = ["understanding"]
        state.step_index = STEP_ORDER.index("orchestration")
        state.last_message = (
            f"{key}: completed → re-orchestrate (dag_evolution_cycles={state.dag_evolution_cycles})"
        )
    elif key == "operator_evolution" and bool(detail.get("rewind_to_orchestration_soft")):
        state.steps_completed = ["understanding"]
        state.step_index = STEP_ORDER.index("orchestration")
        state.ready_for_full_run = False
        state.quality_passed = False
        state.next_action = "advance"
        state.last_message = f"{key}: {detail.get('status', 'ok')} → 请重新 orchestrate（编排文件仍保留）"
    elif (
        key == "orchestration"
        and detail.get("status") == "skipped"
    ):
        # 当前编排文件被复用时，保持在 orchestration，避免在 skipped/rewind 间循环跳步
        state.steps_completed = ["understanding"]
        state.step_index = STEP_ORDER.index("orchestration")
        state.ready_for_full_run = False
        state.quality_passed = False
        state.next_action = "advance"
        state.last_message = (
            "orchestration: skipped（复用已有编排）；若需刷新请执行 orchestrate --force-reset-state"
        )
    elif (
        key == "orchestration"
        and detail.get("status") == "completed"
        and "is_valid" in detail
        and not bool(detail.get("is_valid"))
        and detail.get("recommend_new_operators") is not True
    ):
        # 评估未通过且未建议新增注册算子：下一步应为重新编排，而非 operator_evolution
        state.steps_completed = ["understanding"]
        state.step_index = STEP_ORDER.index("orchestration")
        state.dag_evolution_cycles = 0
        state.ready_for_full_run = False
        state.quality_passed = False
        state.next_action = "advance"
        state.last_message = (
            f"{key}: 已完成 → 评估未通过且未建议新增注册算子，下一步请重新 orchestrate"
        )
    elif (
        key == "orchestration"
        and detail.get("status") == "completed"
        and "is_valid" in detail
        and bool(detail.get("is_valid"))
        and detail.get("recommend_new_operators") is not True
    ):
        # 评估通过且未建议新增注册算子：无需跑 evolve-operators，直接进入实例化
        idx_evo = STEP_ORDER.index("operator_evolution")
        state.steps_completed = STEP_ORDER[: idx_evo + 1]
        state.step_index = STEP_ORDER.index("instantiation")
        state.dag_evolution_cycles = 0
        state.ready_for_full_run = False
        state.quality_passed = False
        state.next_action = "advance"
        state.last_message = f"{key}: 已完成 → 评估通过，跳过算子进化，下一步 instantiate"
        if isinstance(detail, dict):
            detail["workflow_skip_operator_evolution"] = True
    elif key == "quality_check":
        idx_done = STEP_ORDER.index(key)
        state.steps_completed = STEP_ORDER[: idx_done + 1]
        has_diff = bool(detail.get("has_differences"))
        if has_diff:
            state.quality_passed = False
            state.ready_for_full_run = False
            state.step_index = STEP_ORDER.index("experience")
            state.next_action = "advance"
            state.last_message = (
                "quality_check: not_passed → 下一步生成经验并回流下一轮"
            )
        else:
            snapshot_round_artifacts(
                root,
                pipeline_id,
                round_no=max(1, state.round),
                quality_passed=True,
            )
            state.quality_passed = True
            state.ready_for_full_run = True
            state.step_index = len(STEP_ORDER)
            state.next_action = "run_full"
            state.last_message = "quality_check: passed → 可执行 run-full"
    elif key == "instantiation":
        idx_done = STEP_ORDER.index(key)
        state.steps_completed = STEP_ORDER[: idx_done + 1]
        state.step_index = idx_done + 1
        state.quality_passed = False
        state.ready_for_full_run = False
        state.next_action = "advance"
        if detail.get("reused"):
            state.last_message = (
                "instantiation: skipped — 复用已有产物（未调用 LLM）；删除 generated_pipelines 可强制重跑"
            )
        elif detail.get("llm_codegen"):
            steps = detail.get("llm_prompt_generated_steps") or []
            names = ", ".join(str(s) for s in steps) if steps else "是"
            state.last_message = f"instantiation: completed — LLM 参与步骤: {names}"
        else:
            state.last_message = (
                "instantiation: completed — 内置算子模板委托（本 DAG 无 requires_llm 算子或未触发 LLM 写码）"
            )
    elif key == "experience":
        # 检查 Pilot 评分，达到阈值则直接标记为可全量执行，不进下一轮
        PILOT_SCORE_THRESHOLD = 81
        trial_path = root / "data" / "trial_runs" / pipeline_id / "trial_result.json"
        pilot_score: int | None = None
        pilot_recommendation: str | None = None
        try:
            if trial_path.is_file():
                trial_data = json.loads(trial_path.read_text(encoding="utf-8"))
                pilot_eval = trial_data.get("llm_pilot_evaluation") or {}
                pilot_score = pilot_eval.get("overall_score")
                pilot_recommendation = pilot_eval.get("recommendation")
        except (OSError, json.JSONDecodeError, TypeError):
            pass

        score_ok = (
            pilot_score is not None and int(pilot_score) >= PILOT_SCORE_THRESHOLD
        ) or pilot_recommendation == "proceed_full"

        snapshot_round_artifacts(
            root,
            pipeline_id,
            round_no=max(1, state.round),
            quality_passed=score_ok,
        )

        if score_ok:
            # 分数达标，标记可全量执行，不进下一轮
            state.quality_passed = True
            state.ready_for_full_run = True
            state.step_index = len(STEP_ORDER)
            state.next_action = "run_full"
            state.last_message = (
                f"experience: completed → Pilot 评分 {pilot_score} >= {PILOT_SCORE_THRESHOLD} 或建议 proceed_full，可执行 run-full"
            )
            if isinstance(detail, dict):
                detail["auto_approved"] = True
                detail["pilot_score"] = pilot_score
                detail["pilot_recommendation"] = pilot_recommendation
        else:
            # 分数未达标，回流下一轮
            snapshot_iteration_artifacts(
                root,
                pipeline_id,
                round_no=max(1, state.round),
                dag_evolution_cycles=state.dag_evolution_cycles,
                reason="round_rollover_after_experience",
            )
            touched = _clear_for_next_round(root, pipeline_id)
            state.round = max(1, state.round) + 1
            state.quality_passed = False
            state.ready_for_full_run = False
            state.steps_completed = []
            state.step_index = STEP_ORDER.index("understanding")
            state.next_action = "advance"
            state.last_message = (
                f"experience: completed → Pilot 评分 {pilot_score} < {PILOT_SCORE_THRESHOLD}，回流下一轮（round={state.round}）"
            )
            if isinstance(detail, dict):
                detail["next_round_started"] = True
                detail["round"] = state.round
                detail["artifacts_cleared"] = touched
                detail["pilot_score"] = pilot_score
                detail["pilot_score_threshold"] = PILOT_SCORE_THRESHOLD
    else:
        if key == "operator_evolution" and detail.get("status") == "skipped":
            state.dag_evolution_cycles = 0
        idx_done = STEP_ORDER.index(key)
        state.steps_completed = STEP_ORDER[: idx_done + 1]
        state.step_index = idx_done + 1
        if key == "orchestration" and detail.get("status") == "completed":
            state.dag_evolution_cycles = 0
        if key in {"understanding", "orchestration", "operator_evolution", "instantiation", "trial_run"}:
            state.quality_passed = False
            state.ready_for_full_run = False
            state.next_action = "advance"
        state.last_message = f"{key}: {detail.get('status', 'ok')}"
    if key == "understanding" and detail.get("status") == "completed":
        state.understanding_revision += 1
    if key == "orchestration" and detail.get("status") == "completed":
        state.orchestration_revision += 1
    _save_state(root, state)
    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "done": state.step_index >= len(STEP_ORDER),
        "step": key,
        "detail": detail,
        "state": state.to_dict(),
        "invocation": "explicit" if explicit else "linear",
    }


class WorkflowRunner:
    """薄封装，便于测试与依赖注入。"""

    def __init__(self, root: Path) -> None:
        self._root = root

    def load_state(self, pipeline_id: str) -> WorkflowState:
        return load_workflow_state(self._root, pipeline_id)

    def advance(
        self,
        pipeline_id: str,
        *,
        llm_config: dict[str, Any],
        on_usage: Callable[..., None] | None = None,
        force: bool = False,
        pipeline_run_execution_mode: Literal["in_process", "subprocess"] = "in_process",
        pipeline_run_subprocess_fallback_in_process: bool = True,
        pipeline_run_subprocess_timeout_sec: float = 600.0,
        requested_step: str | None = None,
        step_force: bool = False,
    ) -> dict[str, Any]:
        return advance_workflow(
            self._root,
            pipeline_id,
            llm_config=llm_config,
            on_usage=on_usage,
            force=force,
            pipeline_run_execution_mode=pipeline_run_execution_mode,
            pipeline_run_subprocess_fallback_in_process=pipeline_run_subprocess_fallback_in_process,
            pipeline_run_subprocess_timeout_sec=pipeline_run_subprocess_timeout_sec,
            requested_step=requested_step,
            step_force=step_force,
        )
