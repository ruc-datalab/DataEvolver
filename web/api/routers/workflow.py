"""
主流程分步推进 API，与画布「推进一步」语义对齐（前端可后续对接）。

步骤顺序（发布语义）：理解 → 编排（含 DAG 校验）→ 算子进化 → 实例化 → 试运行 → 质量评估；
若未达标再进入经验回流并重启下一轮。全量执行由独立 run-full API 触发。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from subsystems.observability.token_usage_ledger import summarize_token_ledger
from subsystems.workflow import (
    WorkflowStepError,
    advance_workflow,
    load_workflow_state,
    reset_workflow_for_debug,
    rerun_workflow_from_step,
)

router = APIRouter(prefix="/workflow", tags=["workflow"])

_PIPELINE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _pid(pipeline_id: str) -> str:
    p = pipeline_id.strip()
    if not _PIPELINE_ID_RE.match(p):
        raise HTTPException(status_code=400, detail="pipeline_id 无效")
    return p


def _artifact_flags(root: Path, pipeline_id: str) -> dict[str, bool]:
    r = root
    return {
        "understanding": (r / "data" / "understanding_results" / f"{pipeline_id}.json").is_file(),
        "orchestration": (r / "data" / "orchestration_results" / f"{pipeline_id}.json").is_file(),
        "instantiation": (r / "data" / "generated_pipelines" / f"{pipeline_id}.json").is_file(),
        "trial_run": (r / "data" / "trial_runs" / pipeline_id / "trial_result.json").is_file(),
        "pipeline_run": (r / "data" / "run_pipeline_results" / pipeline_id / "latest.json").is_file(),
        "quality_check": (r / "data" / "quality_check_results" / f"{pipeline_id}.json").is_file(),
        "experience": (r / "data" / "experiences" / f"{pipeline_id}.json").is_file(),
    }


@router.get("/{pipeline_id}/state")
def get_workflow_state(request: Request, pipeline_id: str) -> dict[str, Any]:
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    st = load_workflow_state(root, pid)
    return {
        "ok": True,
        "pipeline_id": pid,
        "state": st.to_dict(),
        "artifacts": _artifact_flags(root, pid),
    }


@router.get("/{pipeline_id}/artifact-history")
def get_artifact_history(
    request: Request,
    pipeline_id: str,
    limit: int = Query(default=80, ge=1, le=500, description="从索引尾部返回的条数"),
) -> dict[str, Any]:
    """
    返回 `data/artifact_history/{id}/index.jsonl` 中的归档记录（覆盖理解/编排前的快照）。
    用于前端按轮次展示历史卡片；与 `state.understanding_revision` / `orchestration_revision` 对照。
    """
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    idx_path = root / "data" / "artifact_history" / pid / "index.jsonl"
    rounds_path = root / "data" / "artifact_history" / pid / "rounds.jsonl"
    if not idx_path.is_file():
        return {
            "ok": True,
            "pipeline_id": pid,
            "index_path": f"data/artifact_history/{pid}/index.jsonl",
            "entries": [],
            "round_snapshots": [],
        }
    lines = idx_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-limit:] if len(lines) > limit else lines
    entries: list[dict[str, Any]] = []
    for ln in tail:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                entries.append(obj)
        except json.JSONDecodeError:
            continue
    round_snapshots: list[dict[str, Any]] = []
    if rounds_path.is_file():
        r_lines = rounds_path.read_text(encoding="utf-8", errors="replace").splitlines()
        r_tail = r_lines[-limit:] if len(r_lines) > limit else r_lines
        for ln in r_tail:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    round_snapshots.append(obj)
            except json.JSONDecodeError:
                continue
    return {
        "ok": True,
        "pipeline_id": pid,
        "index_path": f"data/artifact_history/{pid}/index.jsonl",
        "entries": entries,
        "round_snapshots": round_snapshots,
    }


@router.get("/{pipeline_id}/artifact-history/file")
def get_artifact_history_file(
    request: Request,
    pipeline_id: str,
    path: str = Query(..., description="相对仓库根目录路径，如 data/artifact_history/<id>/rounds/r0001/orchestration.json"),
) -> dict[str, Any]:
    """
    读取 artifact_history 下的某个 JSON 快照文件，供前端按轮次/迭代恢复历史 DAG 与经验。
    仅允许访问当前 pipeline_id 的 `data/artifact_history/{id}/` 目录内文件。
    """
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    rel = path.strip().replace("\\", "/").lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path 不能为空")
    prefix = f"data/artifact_history/{pid}/"
    if not rel.startswith(prefix):
        raise HTTPException(status_code=400, detail="path 超出允许范围")
    p = (root / rel).resolve()
    allowed_root = (root / "data" / "artifact_history" / pid).resolve()
    try:
        p.relative_to(allowed_root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="path 超出允许范围") from e
    if p.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="仅支持读取 .json 快照文件")
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {rel}")
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON 解析失败: {rel}") from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail=f"文件内容不是 JSON object: {rel}")
    return {
        "ok": True,
        "pipeline_id": pid,
        "relative_path": rel,
        "data": data,
    }


class RerunBody(BaseModel):
    """从指定步骤重新执行：级联删除该步及之后产物，并将 state 置为该步。"""

    step: str = Field(
        ...,
        description="STEP_ORDER 中的键，如 understanding / orchestration / quality_check / experience",
        examples=["understanding"],
    )


class AdvanceBody(BaseModel):
    """
    force_reset_state：删除 `workflow_runs/.../state.json` 后从第 1 步重新执行本请求的 advance
    （不删除理解/编排等产物；若要重跑理解请手动删对应 json）。

    pipeline_run_* 为兼容字段：当前 workflow 主流程不依赖该步，全量执行由 run-full 触发。
    """

    force_reset_state: bool = Field(default=False)
    pipeline_run_execution_mode: Literal["in_process", "subprocess"] = Field(
        default="in_process",
        description="全量执行：是否对无内置 handler 的步骤走子进程 stub",
    )
    pipeline_run_subprocess_fallback_in_process: bool = Field(
        default=True,
        description="子进程失败时回退进程内 stub",
    )
    pipeline_run_subprocess_timeout_sec: float = Field(
        default=600.0,
        ge=10.0,
        le=86400.0,
    )


@router.post("/{pipeline_id}/advance")
def post_advance(request: Request, pipeline_id: str, body: AdvanceBody = AdvanceBody()) -> dict[str, Any]:
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    cm = request.app.state.config
    llm_cfg = cm.llm_config()
    tracker = request.app.state.token_usage_tracker

    def on_usage(*, input_tokens: int, output_tokens: int, model: str | None = None, **__: Any) -> None:
        tracker.record(input_tokens=input_tokens, output_tokens=output_tokens, model=model)

    try:
        return advance_workflow(
            root,
            pid,
            llm_config=llm_cfg,
            on_usage=on_usage,
            force=body.force_reset_state,
            pipeline_run_execution_mode=body.pipeline_run_execution_mode,
            pipeline_run_subprocess_fallback_in_process=body.pipeline_run_subprocess_fallback_in_process,
            pipeline_run_subprocess_timeout_sec=body.pipeline_run_subprocess_timeout_sec,
        )
    except WorkflowStepError as e:
        st = load_workflow_state(root, pid)
        raise HTTPException(
            status_code=422,
            detail={
                "ok": False,
                "error": "workflow_step_failed",
                "step": e.step_key,
                "message": str(e),
                "pipeline_id": pid,
                "state": st.to_dict(),
            },
        ) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{pipeline_id}/rerun")
def post_rerun(request: Request, pipeline_id: str, body: RerunBody) -> dict[str, Any]:
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    try:
        return rerun_workflow_from_step(root, pid, body.step.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{pipeline_id}/tokens")
def get_pipeline_tokens(
    request: Request,
    pipeline_id: str,
    include_events: bool = Query(default=False, description="是否附带最近若干条 JSONL 事件"),
    max_events: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    """汇总 `data/workflow_runs/{id}/token_usage.jsonl`（workflow advance 与各 API LLM 调用写入）。"""
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    return summarize_token_ledger(root, pid, include_events=include_events, max_events=max_events)


@router.post("/{pipeline_id}/reset")
def post_reset(request: Request, pipeline_id: str) -> dict[str, Any]:
    """仅清除分步状态（`data/workflow_runs/{id}/state.json`），便于重新从第一步 advance。"""
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    sp = root / "data" / "workflow_runs" / pid / "state.json"
    existed = sp.exists()
    if existed:
        sp.unlink()
    return {"ok": True, "pipeline_id": pid, "state_removed": existed}


@router.post("/{pipeline_id}/reset-for-debug")
def post_reset_for_debug(request: Request, pipeline_id: str) -> dict[str, Any]:
    """开发调试专用：回到 round=1 的入口，并清理历史轮次与中间产物。"""
    root = request.app.state.config.root
    pid = _pid(pipeline_id)
    return reset_workflow_for_debug(root, pid)


# ── Strategy Pool API ──────────────────────────────────────────
from subsystems.workflow.strategy_pool import (
    load_strategy_pool,
    update_strategy_score,
    get_best_strategy,
)


@router.get("/{pipeline_id}/strategies")
async def get_strategies(pipeline_id: str, request: Request) -> dict:
    """返回当前策略池，用户可从pending策略中选一条执行。"""
    root = _root(request)
    pool = load_strategy_pool(root, pipeline_id)
    strategies = pool.get("strategies") or []
    pending = [s for s in strategies if s.get("status") == "pending"]
    scored = [s for s in strategies if s.get("score") is not None]
    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "best_strategy_id": pool.get("best_strategy_id"),
        "pending_strategies": pending,
        "scored_strategies": sorted(scored, key=lambda x: x.get("score", 0), reverse=True),
        "total": len(strategies),
    }


@router.post("/{pipeline_id}/strategies/{strategy_id}/select")
async def select_strategy(pipeline_id: str, strategy_id: str, request: Request) -> dict:
    """
    模式A：用户选定某条策略，标记为selected并触发重新编排+实例化+trial。
    选定后需要重新跑understand→orchestrate→instantiate→trial流程。
    """
    root = _root(request)
    pool = load_strategy_pool(root, pipeline_id)
    strategies = pool.get("strategies") or []
    target = None
    for s in strategies:
        if s.get("strategy_id") == strategy_id:
            target = s
            s["selected_by_user"] = True
            s["status"] = "selected"
            break
    if target is None:
        return {"ok": False, "error": f"strategy {strategy_id} not found"}

    pool["strategies"] = strategies
    from subsystems.workflow.strategy_pool import save_strategy_pool
    save_strategy_pool(root, pipeline_id, pool)

    # 把策略描述写入experience，让下一轮理解阶段能看到用户选了哪条
    exp_path = root / "data" / "experiences" / f"{pipeline_id}.json"
    try:
        if exp_path.is_file():
            exp = json.loads(exp_path.read_text(encoding="utf-8"))
        else:
            exp = {"pipeline_id": pipeline_id}
        exp["user_selected_strategy"] = {
            "strategy_id": strategy_id,
            "description": target.get("description", ""),
            "key_changes": target.get("key_changes", []),
            "operator_sequence": target.get("operator_sequence", []),
        }
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        exp_path.write_text(json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # 重置workflow到understanding阶段，让用户下一步advance时重新跑
    from subsystems.workflow.runner import rerun_workflow_from_step
    rerun_result = rerun_workflow_from_step(root, pipeline_id, "understanding")

    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "selected_strategy_id": strategy_id,
        "strategy": target,
        "message": "策略已选定，workflow已重置到understanding阶段，请继续advance推进",
        "rerun": rerun_result,
    }


@router.post("/{pipeline_id}/strategies/run_parallel")
async def run_parallel_strategies(pipeline_id: str, request: Request) -> dict:
    """
    模式B：用户授权并行执行所有pending策略。
    把所有pending策略标记为parallel_trial=True，
    实际并行执行由用户在advance时触发（当前版本串行执行所有pending）。
    """
    root = _root(request)
    pool = load_strategy_pool(root, pipeline_id)
    strategies = pool.get("strategies") or []
    pending = [s for s in strategies if s.get("status") == "pending"]
    if not pending:
        return {"ok": False, "error": "没有pending策略可以执行"}

    for s in strategies:
        if s.get("status") == "pending":
            s["parallel_trial"] = True
            s["status"] = "parallel_selected"

    pool["strategies"] = strategies
    from subsystems.workflow.strategy_pool import save_strategy_pool
    save_strategy_pool(root, pipeline_id, pool)

    return {
        "ok": True,
        "pipeline_id": pipeline_id,
        "parallel_strategy_ids": [s["strategy_id"] for s in pending],
        "message": f"已标记{len(pending)}条策略为并行执行，请继续advance推进",
        "strategies": pending,
    }
