"""
从编排结果生成 `data/generated_pipelines/{id}.json` 与 `step_XXX/` 目录。

- **内置算子**（与 execute 的 handler 表一致）：生成 **委托模块**，`run()` 直接调用确定性/LLM handler，非透传桩。
- **其余算子**（如 evolved.*）：若配置了 API Key，则 **LLM 生成** `run()`；否则回退透传桩。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from subsystems.operator_management.registry_store import OperatorRegistryStore
from subsystems.pipeline_session.manifest_io_paths import apply_manifest_file_paths_to_pipeline
from subsystems.pipeline_session.manifest_store import get_latest_manifest_record
from subsystems.pipeline_runtime.execution.handlers_deterministic import DETERMINISTIC_REGISTRY
from subsystems.pipeline_runtime.execution.handlers_llm import LLM_REGISTRY
from subsystems.pipeline_runtime.execution.handlers_multimodal import MULTIMODAL_REGISTRY
from subsystems.pipeline_runtime.instantiation.builtin_delegate_codegen import generate_builtin_delegate_module
from subsystems.pipeline_runtime.instantiation.llm_operator_codegen import generate_llm_operator_module
from subsystems.pipeline_runtime.instantiation.llm_prompt_codegen import generate_llm_parameter_overrides
from subsystems.pipeline_runtime.instantiation.entry_script_codegen import write_run_pipeline_entry
from subsystems.pipeline_runtime.instantiation.stub_codegen import generate_stub_module


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _final_pipeline_from_orchestration(orch: dict[str, Any]) -> list[dict[str, Any]]:
    fp = orch.get("final_pipeline")
    if isinstance(fp, list) and fp:
        return [x for x in fp if isinstance(x, dict)]
    cs = orch.get("constrained_search")
    if isinstance(cs, dict):
        inner = cs.get("final_pipeline")
        if isinstance(inner, list):
            return [x for x in inner if isinstance(x, dict)]
    return []


def _build_summary(operator: str, step: dict[str, Any], spec: dict[str, Any] | None) -> str:
    parts: list[str] = [f"步骤算子 `{operator}`"]
    d = str(step.get("description") or "").strip()
    if d:
        tail = "…" if len(d) > 160 else ""
        parts.append(f"编排说明: {d[:160]}{tail}")
    if isinstance(spec, dict) and spec.get("description"):
        sd = str(spec["description"]).strip()
        tail2 = "…" if len(sd) > 120 else ""
        parts.append(f"注册表: {sd[:120]}{tail2}")
    if isinstance(spec, dict) and spec.get("requires_llm"):
        parts.append("需要 LLM（执行期接线）")
    return "；".join(parts)


def _control_outline(pipeline_id: str, steps: list[dict[str, Any]]) -> str:
    lines = [f"# Pipeline {pipeline_id} — 实例化包（operator_stub.py + step_meta.json）", ""]
    for s in steps:
        idx = s.get("step_index")
        op = s.get("operator_name")
        ad = s.get("artifact_dir")
        kind = s.get("impl_kind", "?")
        lines.append(f"# {idx}. {op}  [{kind}]  ->  {ad}")
    lines.append("")
    lines.append(
        "# 内置算子: operator_stub 委托 DETERMINISTIC_REGISTRY / LLM_REGISTRY（与 run-pipeline 一致）。"
    )
    lines.append("# 自定义算子: 可能为 LLM 生成代码或透传桩；trial 需 context['llm_config'] 才能跑通 LLM 步。")
    return "\n".join(lines)


_BUILTIN_OPS: frozenset[str] = frozenset(DETERMINISTIC_REGISTRY.keys()) | frozenset(LLM_REGISTRY.keys()) | frozenset(MULTIMODAL_REGISTRY.keys())


def run_instantiation(
    root: Path,
    pipeline_id: str,
    *,
    llm_config: dict[str, Any] | None = None,
    on_usage: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """
    读取编排落盘结果，生成实例化包。
    调用时会清空 `generated_pipelines/{id}/` 目录后重写。
    """
    orch_path = root / "data" / "orchestration_results" / f"{pipeline_id}.json"
    if not orch_path.is_file():
        raise FileNotFoundError(f"缺少编排结果: {orch_path}")
    raw = json.loads(orch_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("编排结果格式无效")
    fp = _final_pipeline_from_orchestration(raw)
    if not fp:
        raise ValueError("编排结果中 final_pipeline 为空，无法实例化")

    manifest_path = root / "data" / "manifest.jsonl"
    mrec = get_latest_manifest_record(manifest_path, pipeline_id)
    fp = apply_manifest_file_paths_to_pipeline(
        fp, pipeline_id=pipeline_id, manifest_record=mrec if isinstance(mrec, dict) else {}
    )

    store = OperatorRegistryStore(root, pipeline_id=pipeline_id)
    registry = store.merged_raw()
    llm_cfg = llm_config if isinstance(llm_config, dict) else {}
    u_path = root / "data" / "understanding_results" / f"{pipeline_id}.json"
    understanding: dict[str, Any] = {}
    if u_path.is_file():
        try:
            u_raw = json.loads(u_path.read_text(encoding="utf-8"))
            if isinstance(u_raw, dict):
                understanding = u_raw
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    rel_base = f"data/generated_pipelines/{pipeline_id}"
    abs_base = root / rel_base
    if abs_base.is_dir():
        shutil.rmtree(abs_base)
    abs_base.mkdir(parents=True, exist_ok=True)
    write_run_pipeline_entry(abs_base)

    codegen_warnings: list[str] = []
    llm_prompt_generated_steps: list[int] = []
    steps_out: list[dict[str, Any]] = []
    for i, step in enumerate(fp):
        n = i + 1
        step_dir = abs_base / f"step_{n:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        op = str(step.get("operator") or "unknown")
        spec = registry.get(op)
        reg_spec = spec if isinstance(spec, dict) else None
        step_params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}

        impl_kind: str
        if op in _BUILTIN_OPS:
            # 对需要调用 LLM 的内置算子：实例化阶段生成 pipeline-specific 的 prompt / rubric（覆盖参数里的 spec 字段）
            if reg_spec and bool(reg_spec.get("requires_llm")) and op in LLM_REGISTRY:
                overrides, notes = generate_llm_parameter_overrides(
                    pipeline_id=pipeline_id,
                    step=step,
                    registry_spec=reg_spec,
                    understanding=understanding,
                    llm_config=llm_cfg,
                    on_usage=on_usage,
                )
                if overrides:
                    step_params = dict(step_params)
                    step_params.update(overrides)
                    step["parameters"] = step_params
                    step.setdefault("meta", {})
                    if isinstance(step["meta"], dict):
                        step["meta"]["prompt_generated"] = True
                        if notes:
                            step["meta"]["prompt_notes"] = notes[:8]
                    llm_prompt_generated_steps.append(n)
                else:
                    if str(llm_cfg.get("api_key") or "").strip():
                        # 有 key 但生成失败：仍可用 handler 默认 prompt，但应提示用户
                        codegen_warnings.append(
                            f"步骤 {n} ({op}) 需要 LLM，但未能生成 pipeline-specific prompt；将使用内置默认提示词。"
                        )
                    else:
                        codegen_warnings.append(
                            f"步骤 {n} ({op}) 需要 LLM，但未配置 API Key；将使用内置默认提示词。"
                        )
            code = generate_builtin_delegate_module()
            impl_kind = "builtin_delegate"
        else:
            gen = generate_llm_operator_module(
                root=root,
                pipeline_id=pipeline_id,
                step=step,
                registry_spec=reg_spec,
                understanding=understanding,
                llm_config=llm_cfg,
                on_usage=on_usage,
            )
            if gen:
                code = gen
                impl_kind = "llm_generated"
            else:
                code = generate_stub_module(step_index=n, step=step, registry_spec=reg_spec)
                impl_kind = "passthrough_stub"
                if str(llm_cfg.get("api_key") or "").strip():
                    codegen_warnings.append(f"步骤 {n} ({op}) LLM 代码生成失败，已用透传桩。")
                else:
                    codegen_warnings.append(
                        f"步骤 {n} ({op}) 非内置算子且未配置 API Key，已用透传桩；配置 Key 后重新 instantiate 可尝试 LLM 生成。"
                    )

        (step_dir / "operator_stub.py").write_text(code, encoding="utf-8")

        meta = {
            "step_index": n,
            "step_id": step.get("step_id"),
            "operator_name": op,
            "description": step.get("description", ""),
            "input_keys": step.get("input_keys", []),
            "output_keys": step.get("output_keys", []),
            "parameters": step.get("parameters", {}),
            "requires_llm": bool(reg_spec.get("requires_llm")) if reg_spec else False,
            "prompt_generated": bool(isinstance(step.get("meta"), dict) and step["meta"].get("prompt_generated")),
            "relative_dir": f"{rel_base}/step_{n:03d}",
            "orchestration_step": step,
            "impl_kind": impl_kind,
        }
        (step_dir / "step_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary = _build_summary(op, step, reg_spec)
        row = dict(meta)
        row["code"] = code
        row["intermediate_summary"] = summary
        row["artifact_dir"] = meta["relative_dir"]
        steps_out.append(row)

    payload: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "source": "instantiation_v2",
        "meta": {
            "created_at": _iso(),
            "orchestration_path": f"data/orchestration_results/{pipeline_id}.json",
            "understanding_path": f"data/understanding_results/{pipeline_id}.json",
            "total_steps": len(steps_out),
            "codegen_warnings": codegen_warnings,
            "llm_prompt_generated_steps": llm_prompt_generated_steps,
            "entry_script": f"{rel_base}/run_pipeline.py",
            "note": "内置算子使用 handler 委托实现；自定义算子可 LLM 生成或透传桩。推荐运行 entry_script --mode pilot 做采样+LLM 评估后再 full。",
        },
        "dag": raw.get("dag"),
        "final_pipeline_snapshot": fp,
        "steps": steps_out,
        "control_script_outline": _control_outline(pipeline_id, steps_out),
    }

    out_main = root / "data" / "generated_pipelines" / f"{pipeline_id}.json"
    out_main.parent.mkdir(parents=True, exist_ok=True)
    out_main.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
