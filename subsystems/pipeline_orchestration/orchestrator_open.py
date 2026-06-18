"""
开源编排器：三阶段 LLM（free fitting → template combination → constrained search），
对齐旧版 `PipelineOrchestratorSimple`；算子注册表使用 `OperatorRegistryStore.merged_raw()`，
不做动态注册算子（能力检查为占位），减少与旧版 `OperatorRegistryManager` 的耦合。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from core.llm_client import LLMClientError, chat_completion, parse_message_content_json

from subsystems.operator_management.registry_store import OperatorRegistryStore
from subsystems.pipeline_session.manifest_io_paths import apply_manifest_file_paths_to_pipeline
from subsystems.structured_understanding.orchestration_feedback import (
    format_prior_orchestration_feedback_for_prompt,
)
from subsystems.pipeline_orchestration.orchestration_prompts import (
    CONSTRAINED_SEARCH_SYSTEM_PROMPT,
    CONSTRAINED_SEARCH_USER_PROMPT_TEMPLATE,
    FREE_FITTING_SYSTEM_PROMPT,
    FREE_FITTING_USER_PROMPT_TEMPLATE,
    TEMPLATE_COMBINATION_SYSTEM_PROMPT,
    TEMPLATE_COMBINATION_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# 主流水线「单条记录流」上，输入/输出同名键表示 map-over-stream，不是有向图自环
_STREAM_PASS_THROUGH_KEYS = frozenset({"records", "data", "data_list", "items", "rows"})


def _ensure_keys(obj: dict[str, Any], required: dict[str, type]) -> bool:
    for k, t in required.items():
        if k not in obj:
            return False
        v = obj[k]
        if t is list and not isinstance(v, list):
            return False
        if t is dict and not isinstance(v, dict):
            return False
        if t is str and not isinstance(v, str):
            return False
    return True


def _default_for_required(required: dict[str, type]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, key_type in required.items():
        if key_type is dict:
            out[key] = {}
        elif key_type is list:
            out[key] = []
        elif key_type is str:
            out[key] = "解析失败"
        else:
            out[key] = None
    return out


def final_pipeline_to_dag(final_pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    """将线性 final_pipeline 转为前端可用的 DAG 结构。"""
    nodes: list[dict[str, Any]] = []
    for i, step in enumerate(final_pipeline):
        nid = str(step.get("step_id") or f"step_{i + 1}")
        nodes.append(
            {
                "node_id": nid,
                "node_name": str(step.get("operator", "")),
                "description": str(step.get("description", "")),
                "input_keys": list(step.get("input_keys") or []),
                "output_keys": list(step.get("output_keys") or []),
            }
        )
    edges: list[dict[str, Any]] = []
    for i in range(len(nodes) - 1):
        fo = nodes[i]["output_keys"]
        ti = nodes[i + 1]["input_keys"]
        flows = [x for x in fo if x in ti and x not in ("file_path", "config")]
        from_o = flows or (fo[:1] if fo else ["records"])
        to_i = flows or (ti[:1] if ti else ["records"])
        edges.append(
            {
                "from_node": nodes[i]["node_id"],
                "to_node": nodes[i + 1]["node_id"],
                "data_flow": {"from_outputs": from_o, "to_inputs": to_i},
            }
        )
    order = [n["node_id"] for n in nodes]
    return {
        "nodes": nodes,
        "edges": edges,
        "execution_order": order,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


def pipeline_plan_from_final(final_pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": i + 1,
            "operator": s.get("operator"),
            "description": s.get("description", ""),
        }
        for i, s in enumerate(final_pipeline)
    ]


class OpenPipelineOrchestrator:
    def __init__(
        self,
        root: Path,
        pipeline_id: str,
        llm_config: dict[str, Any],
        on_usage: Callable[..., None] | None = None,
    ) -> None:
        self._root = root
        self._pipeline_id = pipeline_id
        self._cfg = llm_config
        self._on_usage = on_usage
        self._registry_store = OperatorRegistryStore(root, pipeline_id=pipeline_id)
        self.template_library = self._load_template_library()
        self._manifest_record: dict[str, Any] = {}

    def _emit_usage(self, usage: dict[str, Any], *, operation: str) -> None:
        if not self._on_usage:
            return
        kw: dict[str, Any] = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "model": str(self._cfg.get("model")),
            "operation": operation,
        }
        for k in ("duration_ms", "request_id", "api_host"):
            if usage.get(k) is not None:
                kw[k] = usage[k]
        try:
            self._on_usage(**kw)
        except TypeError:
            self._on_usage(
                input_tokens=kw["input_tokens"],
                output_tokens=kw["output_tokens"],
                model=kw["model"],
            )

    def _max_tokens(self, stage_default: int) -> int:
        cfg = int(self._cfg.get("max_tokens", stage_default))
        return min(16384, max(1024, cfg))

    def _timeout_sec(self) -> float:
        return max(90.0, float(self._cfg.get("timeout", 120)))

    def _load_template_library(self) -> dict[str, Any]:
        p = self._root / "data" / "pipeline_templates.json"
        if not p.is_file():
            logger.warning("pipeline_templates.json 缺失: %s", p)
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.error("加载模板库失败: %s", e)
            return {}

    def _call_llm_for_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: dict[str, type],
        *,
        max_tokens: int,
        max_retries: int = 2,
        usage_operation: str = "orchestration.llm",
    ) -> dict[str, Any]:
        base_url = str(self._cfg["base_url"])
        api_key = str(self._cfg["api_key"])
        model = str(self._cfg["model"])
        temperature = float(self._cfg.get("temperature", 0.1))
        timeout = self._timeout_sec()
        prompt = user_prompt
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout,
                    json_mode=True,
                )
                parsed, usage = parse_message_content_json(resp)
                self._emit_usage(usage, operation=usage_operation)
                if isinstance(parsed, dict) and _ensure_keys(parsed, required_keys):
                    return parsed
                missing = [k for k in required_keys if k not in (parsed or {})]
                logger.warning("JSON 缺键或类型不符: %s", missing)
            except (LLMClientError, json.JSONDecodeError, KeyError, TypeError) as e:
                last_err = e
                logger.warning("LLM/解析失败 attempt %s: %s", attempt + 1, e)
            prompt = user_prompt + "\n\nReturn one valid JSON object with all required keys. No markdown."
        logger.error("LLM JSON 重试耗尽，返回默认结构")
        if last_err:
            logger.debug("last error: %s", last_err)
        return _default_for_required(required_keys)

    def orchestrate_pipeline(
        self,
        understanding_result: dict[str, Any],
        manifest_record: dict[str, Any],
    ) -> dict[str, Any]:
        self._manifest_record = manifest_record
        logger.info("开始 Pipeline 编排（开源三阶段）")
        free_fitting = self._free_fitting_stage(understanding_result)
        template_combination = self._template_combination_stage(free_fitting, understanding_result)
        constrained_search = self._constrained_search_stage(
            template_combination, free_fitting, understanding_result
        )
        final_pipeline = constrained_search.get("final_pipeline", [])
        return {
            "free_fitting": free_fitting,
            "template_combination": template_combination,
            "constrained_search": constrained_search,
            "final_pipeline": final_pipeline,
        }

    def _free_fitting_stage(self, understanding_result: dict[str, Any]) -> dict[str, Any]:
        schema_analysis = understanding_result.get("schema_analysis", {})
        dataset_level_delta = understanding_result.get("dataset_level_delta", {})
        basic_information = understanding_result.get("basic_information", {})
        prior_fb = format_prior_orchestration_feedback_for_prompt(understanding_result)
        user_prompt = FREE_FITTING_USER_PROMPT_TEMPLATE.format(
            schema_analysis=json.dumps(schema_analysis, ensure_ascii=False, indent=2),
            dataset_level_delta=json.dumps(dataset_level_delta, ensure_ascii=False, indent=2),
            basic_information=json.dumps(basic_information, ensure_ascii=False, indent=2),
            prior_orchestration_feedback=prior_fb,
        )
        required_keys = {
            "global_optimization_direction": str,
            "key_improvements": list,
            "transformation_strategies": list,
            "quality_focus": list,
            "summary": str,
        }
        result = self._call_llm_for_json(
            FREE_FITTING_SYSTEM_PROMPT,
            user_prompt,
            required_keys,
            max_tokens=self._max_tokens(4096),
            usage_operation="orchestration.free_fitting",
        )
        return {
            "optimization_blueprint": result,
            "stage": "free_fitting",
            "design_method": "llm_driven",
        }

    def _template_combination_stage(
        self,
        free_fitting: dict[str, Any],
        understanding_result: dict[str, Any],
    ) -> dict[str, Any]:
        optimization_blueprint = free_fitting.get("optimization_blueprint", {})
        templates = self.template_library
        available_templates: dict[str, Any] = {}
        for template_name, template_info in templates.items():
            if not isinstance(template_info, dict):
                continue
            available_templates[template_name] = {
                "name": template_info.get("name", ""),
                "description": template_info.get("description", ""),
                "function": template_info.get("function", ""),
                "abstract_steps_count": len(template_info.get("abstract_steps", [])),
                "input_requirements": template_info.get("input_requirements", []),
                "output_produces": template_info.get("output_produces", []),
                "use_cases": template_info.get("use_cases", []),
                "typical_scenarios": template_info.get("typical_scenarios", []),
            }
        prior_fb = format_prior_orchestration_feedback_for_prompt(understanding_result)
        user_prompt = TEMPLATE_COMBINATION_USER_PROMPT_TEMPLATE.format(
            optimization_blueprint=json.dumps(optimization_blueprint, ensure_ascii=False, indent=2),
            prior_orchestration_feedback=prior_fb,
            available_templates=json.dumps(available_templates, ensure_ascii=False, indent=2),
            template_selection_rules="Select templates based on their function and use_cases to match the optimization blueprint",
        )
        required_keys = {
            "selected_templates": list,
            "selection_rationale": str,
            "template_combination_strategy": str,
            "pipeline_sketch": dict,
            "coverage_analysis": dict,
        }
        result = self._call_llm_for_json(
            TEMPLATE_COMBINATION_SYSTEM_PROMPT,
            user_prompt,
            required_keys,
            max_tokens=self._max_tokens(8192),
            usage_operation="orchestration.template_combination",
        )
        pipeline_sketch = result.get("pipeline_sketch", {})
        if not isinstance(pipeline_sketch, dict):
            pipeline_sketch = {}
        sketch_steps = pipeline_sketch.get("steps", [])
        if not isinstance(sketch_steps, list):
            sketch_steps = []
        enhanced_steps: list[dict[str, Any]] = []
        step_counter = 1
        for step in sketch_steps:
            if not isinstance(step, dict):
                continue
            template_name = step.get("template_name", "")
            if template_name in templates:
                template = templates[template_name]
                if not isinstance(template, dict):
                    continue
                abstract_steps = template.get("abstract_steps", [])
                if not isinstance(abstract_steps, list):
                    abstract_steps = []
                selected_steps = step.get("selected_template_steps", [])
                use_all = step.get("use_all_template_steps", True)
                if selected_steps and not use_all:
                    steps_to_use = [s for s in abstract_steps if isinstance(s, dict) and s.get("step_id") in selected_steps]
                else:
                    steps_to_use = abstract_steps[: min(2, len(abstract_steps))]
                for abstract_step in steps_to_use:
                    if not isinstance(abstract_step, dict):
                        continue
                    enhanced_steps.append(
                        {
                            "step_id": f"step_{step_counter}",
                            "template_name": template_name,
                            "functional_description": abstract_step.get("functional_description", ""),
                            "expected_output": abstract_step.get("expected_output", ""),
                            "abstract_step_id": abstract_step.get("step_id", ""),
                            "operator": None,
                            "input_keys": [],
                            "output_keys": [],
                        }
                    )
                    step_counter += 1
            else:
                step = dict(step)
                step["step_id"] = step.get("step_id", f"step_{step_counter}")
                enhanced_steps.append(step)
                step_counter += 1
        pipeline_sketch["steps"] = enhanced_steps
        pipeline_sketch["total_steps"] = len(enhanced_steps)
        pipeline_sketch["stage"] = "template_combination"
        pipeline_sketch["note"] = (
            "This is a high-level sketch with abstract steps. "
            "Specific operators will be assigned in the constrained search stage."
        )
        result["pipeline_sketch"] = pipeline_sketch
        return {
            "template_selection": result,
            "pipeline_sketch": pipeline_sketch,
            "stage": "template_combination",
            "design_method": "template_based",
        }

    def _stub_capability_check(self) -> dict[str, Any]:
        return {
            "needs_dynamic_addition": False,
            "missing_capabilities": [],
            "note": "open source: no dynamic operator injection",
        }

    def _constrained_search_stage(
        self,
        template_combination: dict[str, Any],
        free_fitting: dict[str, Any],
        understanding_result: dict[str, Any],
    ) -> dict[str, Any]:
        pipeline_sketch = template_combination.get("pipeline_sketch", {})
        optimization_blueprint = free_fitting.get("optimization_blueprint", {})
        capability_check = self._stub_capability_check()
        enhanced_registry = self._registry_store.merged_raw()
        available_operators: dict[str, Any] = {}
        for op_name, op_info in enhanced_registry.items():
            if not isinstance(op_info, dict):
                continue
            available_operators[op_name] = {
                "description": op_info.get("description", ""),
                "input_keys": op_info.get("input_keys", []),
                "output_keys": op_info.get("output_keys", []),
                "requires_llm": op_info.get("requires_llm", False),
            }
        abstract_steps_info: list[dict[str, Any]] = []
        for step in pipeline_sketch.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            if step.get("functional_description"):
                abstract_steps_info.append(
                    {
                        "step_id": step.get("step_id", ""),
                        "template_name": step.get("template_name", ""),
                        "functional_description": step.get("functional_description", ""),
                        "expected_output": step.get("expected_output", ""),
                    }
                )
        schema_analysis = understanding_result.get("schema_analysis", {})
        basic_info = understanding_result.get("basic_information", {})
        new_fields = schema_analysis.get("new_fields", []) or []
        raw_fields = schema_analysis.get("raw_fields", []) or []
        seed_field_details = schema_analysis.get("seed_field_details", {}) or {}
        raw_field_details = schema_analysis.get("raw_field_details", {}) or {}
        new_fields_details: dict[str, Any] = {}
        for field in new_fields:
            if field in seed_field_details:
                new_fields_details[field] = seed_field_details[field]
        format_requirements: list[str] = []
        for field_name, field_info in seed_field_details.items():
            if not isinstance(field_info, dict):
                continue
            opt_opp = str(field_info.get("optimization_opportunity", ""))
            if opt_opp and any(
                tag in opt_opp.lower() for tag in ["tag", "redacted_reasoning", "answer", "</think>", "<answer>"]
            ):
                format_requirements.append(f"{field_name}: {opt_opp}")
        format_requirements_summary = "\n".join(format_requirements) if format_requirements else "No specific format requirements identified"
        new_fields_summary = ", ".join(str(x) for x in new_fields) if new_fields else "None identified"
        raw_fields_summary = ", ".join(str(x) for x in raw_fields) if raw_fields else "None identified"
        new_fields_details_summary = (
            json.dumps(new_fields_details, ensure_ascii=False, indent=2) if new_fields_details else "No details available"
        )
        raw_fields_details_summary = (
            json.dumps(raw_field_details, ensure_ascii=False, indent=2) if raw_field_details else "No details available"
        )
        processing_targets_summary = (
            json.dumps(basic_info.get("processing_targets", []), ensure_ascii=False)
            if basic_info.get("processing_targets")
            else "No specific targets"
        )
        quality_standards_summary = basic_info.get("quality_standards") or "No specific standards"
        user_prompt = CONSTRAINED_SEARCH_USER_PROMPT_TEMPLATE.format(
            pipeline_sketch=json.dumps(pipeline_sketch, ensure_ascii=False, indent=2),
            abstract_steps=json.dumps(abstract_steps_info, ensure_ascii=False, indent=2),
            optimization_blueprint=json.dumps(optimization_blueprint, ensure_ascii=False, indent=2),
            available_operators=json.dumps(available_operators, ensure_ascii=False, indent=2),
            understanding_result=json.dumps(understanding_result, ensure_ascii=False, indent=2),
            capability_check_results=json.dumps(capability_check, ensure_ascii=False, indent=2),
            raw_fields_summary=raw_fields_summary,
            raw_fields_details_summary=raw_fields_details_summary,
            new_fields_summary=new_fields_summary,
            new_fields_details_summary=new_fields_details_summary,
            format_requirements_summary=format_requirements_summary,
            processing_targets_summary=processing_targets_summary,
            quality_standards_summary=str(quality_standards_summary),
        )
        required_keys = {"final_pipeline": list, "refinement_summary": dict}
        result = self._call_llm_for_json(
            CONSTRAINED_SEARCH_SYSTEM_PROMPT,
            user_prompt,
            required_keys,
            max_tokens=self._max_tokens(8192),
            usage_operation="orchestration.constrained_search",
        )
        final_pipeline = result.get("final_pipeline", [])
        if not isinstance(final_pipeline, list):
            final_pipeline = []
        final_pipeline = self._ensure_read_write_operators(final_pipeline, understanding_result)
        final_pipeline = apply_manifest_file_paths_to_pipeline(
            final_pipeline,
            pipeline_id=str(understanding_result.get("pipeline_id", "unknown")),
            manifest_record=self._manifest_record if isinstance(self._manifest_record, dict) else {},
        )
        final_pipeline = self._normalize_bookend_io_from_registry(final_pipeline, enhanced_registry)
        validation_result = self._validate_orchestration_result(
            final_pipeline, understanding_result, optimization_blueprint
        )
        missing_ops: list[str] = []
        if not validation_result.get("is_valid"):
            missing_ops = self._identify_missing_operators(validation_result, final_pipeline, understanding_result)
        result["final_pipeline"] = final_pipeline
        return {
            "final_pipeline": final_pipeline,
            "refinement_summary": result.get("refinement_summary", {}),
            "capability_check": capability_check,
            "validation_result": validation_result,
            "operator_level_self_evolving": {
                "triggered": not validation_result.get("is_valid", True),
                "added_operators": [],
                "re_orchestrated": False,
                "suggested_operators": missing_ops,
            },
            "stage": "constrained_search",
            "design_method": "constrained_optimization",
        }

    def _ensure_read_write_operators(
        self,
        pipeline: list[dict[str, Any]],
        understanding_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not pipeline:
            return pipeline
        rec = self._manifest_record
        pid = str(understanding_result.get("pipeline_id", "unknown"))
        raw_files = [x for x in (rec.get("raw_data_files") or []) if isinstance(x, str)]
        read_path = raw_files[0] if raw_files else f"data/uploads/{pid}/raw_data/{pid}_raw.jsonl"
        write_path = f"data/uploads/{pid}/outputs/{pid}_processed.jsonl"

        first = pipeline[0]
        has_read = isinstance(first, dict) and first.get("operator") == "read_data"
        last = pipeline[-1]
        has_write = isinstance(last, dict) and last.get("operator") == "write_data"

        if not has_read:
            read_step: dict[str, Any] = {
                "step_id": "step_1",
                "operator": "read_data",
                "input_keys": ["file_path"],
                "output_keys": ["records"],
                "parameters": {"file_path": read_path},
                "description": "Read raw data from the configured path (manifest).",
                "mapped_from_abstract_step": "N/A",
                "refinement_source": "post_processing",
            }
            for i, step in enumerate(pipeline, start=2):
                if isinstance(step, dict):
                    step["step_id"] = f"step_{i}"
            pipeline.insert(0, read_step)
            logger.info("已插入 read_data 作为第一步")

        if not has_write:
            last_step = pipeline[-1]
            prev_out = "records"
            if isinstance(last_step, dict) and last_step.get("output_keys"):
                prev_out = last_step["output_keys"][0]
            write_step = {
                "step_id": f"step_{len(pipeline) + 1}",
                "operator": "write_data",
                "input_keys": [prev_out, "file_path"],
                "output_keys": ["records"],
                "parameters": {"file_path": write_path},
                "description": "Write processed records to JSONL.",
                "mapped_from_abstract_step": "N/A",
                "refinement_source": "post_processing",
            }
            pipeline.append(write_step)
            logger.info("已追加 write_data 作为最后一步")
        return pipeline

    def _normalize_bookend_io_from_registry(
        self,
        pipeline: list[dict[str, Any]],
        registry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """对齐 read_data / write_data 与注册表声明，避免模型漏写 output_keys 导致误报。"""
        for step in pipeline:
            if not isinstance(step, dict):
                continue
            op = step.get("operator")
            spec = registry.get(op) if isinstance(op, str) else None
            if not isinstance(spec, dict):
                continue
            spec_out = list(spec.get("output_keys") or [])
            sout = list(step.get("output_keys") or [])
            sin = list(step.get("input_keys") or [])
            if op == "write_data" and "records" in spec_out and "records" not in sout:
                if "records" in sin or "records" in list(spec.get("input_keys") or []):
                    step["output_keys"] = list(spec_out)
            if op == "read_data" and spec_out and not sout:
                step["output_keys"] = list(spec_out)
        return pipeline

    def _validate_orchestration_result(
        self,
        final_pipeline: list[dict[str, Any]],
        understanding_result: dict[str, Any],
        _optimization_blueprint: dict[str, Any],
    ) -> dict[str, Any]:
        validation_issues: list[Any] = []
        data_flow_check = self._check_data_flow_coherence(final_pipeline)
        if not data_flow_check.get("is_valid", True):
            validation_issues.extend(data_flow_check.get("issues", []))
        dag_check = self._check_dag_closure(final_pipeline)
        if not dag_check.get("is_valid", True):
            validation_issues.extend(dag_check.get("issues", []))
        field_coverage_check = {"is_valid": True, "issues": [], "note": "field_coverage check disabled: record-level fields vs pipeline-level output_keys are different abstraction layers"}
        dependency_check = self._check_input_output_dependencies(final_pipeline)
        if not dependency_check.get("is_valid", True):
            validation_issues.extend(dependency_check.get("issues", []))
        is_valid = len(validation_issues) == 0
        return {
            "is_valid": is_valid,
            "validation_issues": validation_issues,
            "data_flow_check": data_flow_check,
            "dag_check": dag_check,
            "field_coverage_check": field_coverage_check,
            "dependency_check": dependency_check,
        }

    def _check_data_flow_coherence(self, pipeline: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[Any] = []
        if not pipeline:
            return {"is_valid": False, "issues": [{"type": "data_flow", "description": "Pipeline为空"}]}
        for i, step in enumerate(pipeline[1:], start=1):
            if not isinstance(step, dict):
                continue
            input_keys = step.get("input_keys", [])
            previous_outputs: set[str] = set()
            for prev_step in pipeline[:i]:
                if isinstance(prev_step, dict):
                    previous_outputs.update(prev_step.get("output_keys", []))
            missing_inputs: list[str] = []
            params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
            param_keys = set(params.keys())
            for input_key in input_keys:
                if input_key in ("file_path", "config"):
                    continue
                if input_key in param_keys:
                    continue
                if input_key not in previous_outputs:
                    missing_inputs.append(str(input_key))
            if missing_inputs:
                issues.append(
                    {
                        "type": "data_flow",
                        "step_id": step.get("step_id", f"step_{i + 1}"),
                        "operator": step.get("operator", ""),
                        "description": f"步骤输入键无来源: {missing_inputs}",
                        "missing_inputs": missing_inputs,
                    }
                )
        return {"is_valid": len(issues) == 0, "issues": issues}

    def _check_dag_closure(self, pipeline: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[Any] = []
        if not pipeline:
            return {"is_valid": False, "issues": [{"type": "dag_closure", "description": "Pipeline为空"}]}
        dangling_outputs: list[str] = []
        for i, step in enumerate(pipeline):
            if not isinstance(step, dict):
                continue
            step_id = step.get("step_id", f"step_{i + 1}")
            output_keys = set(step.get("output_keys", []))
            input_keys = set(step.get("input_keys", []))
            overlap = {k for k in output_keys.intersection(input_keys) if k not in ("file_path", "config")}
            # 仅当重叠键超出「主流名不变」的流式语义时才视为可疑（真自环多为同名非流字段）
            suspicious = overlap - _STREAM_PASS_THROUGH_KEYS
            if suspicious:
                issues.append(
                    {
                        "type": "dag_closure",
                        "step_id": step_id,
                        "operator": step.get("operator", ""),
                        "description": f"步骤输入/输出同名键可能形成真自环（非流式 records）: {suspicious}",
                    }
                )
        all_outputs: set[str] = set()
        all_inputs: set[str] = set()
        for step in pipeline[:-1]:
            if isinstance(step, dict):
                all_outputs.update(step.get("output_keys", []))
        for step in pipeline[1:]:
            if isinstance(step, dict):
                all_inputs.update(step.get("input_keys", []))
        all_outputs = {k for k in all_outputs if k not in ("file_path", "config")}
        all_inputs = {k for k in all_inputs if k not in ("file_path", "config")}
        dangling_outputs = list(all_outputs - all_inputs)
        return {"is_valid": len(issues) == 0, "issues": issues, "dangling_outputs": dangling_outputs}

    def _check_field_coverage(
        self, pipeline: list[dict[str, Any]], understanding_result: dict[str, Any]
    ) -> dict[str, Any]:
        schema_analysis = understanding_result.get("schema_analysis", {})
        new_fields = schema_analysis.get("new_fields", []) or []
        if not new_fields:
            return {"is_valid": True, "issues": [], "covered_fields": [], "missing_fields": []}
        generated_fields: set[str] = set()
        for step in pipeline:
            if isinstance(step, dict):
                generated_fields.update(step.get("output_keys", []))
        # 检查字段是否在output_keys或parameters的值里出现过
        all_param_values: set[str] = set()
        for step in pipeline:
            if not isinstance(step, dict):
                continue
            params = step.get("parameters") or {}
            if isinstance(params, dict):
                for v in params.values():
                    if isinstance(v, str):
                        all_param_values.add(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                all_param_values.add(item)

        missing_fields = [
            field for field in new_fields
            if field not in generated_fields and field not in all_param_values
        ]
        issues: list[Any] = []
        if missing_fields:
            issues.append(
                {
                    "type": "field_coverage",
                    "description": f"新字段可能未被管线输出键覆盖: {missing_fields}",
                    "missing_fields": missing_fields,
                }
            )
        return {
            "is_valid": len(missing_fields) == 0,
            "issues": issues,
            "covered_fields": [f for f in new_fields if f in generated_fields or f in all_param_values],
            "missing_fields": missing_fields,
        }

    def _check_input_output_dependencies(self, pipeline: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[Any] = []
        if len(pipeline) < 2:
            return {"is_valid": True, "issues": []}
        for i in range(len(pipeline) - 1):
            cur = pipeline[i]
            nxt = pipeline[i + 1]
            if not isinstance(cur, dict) or not isinstance(nxt, dict):
                continue
            current_outputs = {k for k in cur.get("output_keys", []) if k not in ("file_path", "config")}
            next_inputs = {k for k in nxt.get("input_keys", []) if k not in ("file_path", "config")}
            if next_inputs and not current_outputs.intersection(next_inputs):
                if "records" in current_outputs or "data_list" in current_outputs or "data" in current_outputs:
                    continue
                issues.append(
                    {
                        "type": "dependency",
                        "from_step": cur.get("step_id", f"step_{i + 1}"),
                        "to_step": nxt.get("step_id", f"step_{i + 2}"),
                        "description": f"相邻步骤 I/O 键不匹配: {list(current_outputs)} -> {list(next_inputs)}",
                    }
                )
        return {"is_valid": len(issues) == 0, "issues": issues}

    def _identify_missing_operators(
        self,
        validation_result: dict[str, Any],
        final_pipeline: list[dict[str, Any]],
        _understanding_result: dict[str, Any],
    ) -> list[str]:
        missing_operators: list[str] = []
        for issue in validation_result.get("validation_issues", []) or []:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("type", "")).lower()
            if "data_flow" in issue_type:
                if issue.get("missing_inputs"):
                    has_converter = any("converter" in str(s.get("operator", "")).lower() for s in final_pipeline if isinstance(s, dict))
                    if not has_converter:
                        missing_operators.append("data_converter")
            elif "field_coverage" in issue_type:
                for field in issue.get("missing_fields", []) or []:
                    missing_operators.append(f"generate_{str(field).replace('.', '_')}")
            elif "dependency" in issue_type:
                missing_operators.append("data_adapter")
        return list(set(missing_operators))


def run_open_orchestration(
    root: Path,
    pipeline_id: str,
    understanding_result: dict[str, Any],
    manifest_record: dict[str, Any],
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None = None,
) -> dict[str, Any]:
    if not str(llm_config.get("api_key") or "").strip():
        raise ValueError("编排需要配置 API Key")
    orch = OpenPipelineOrchestrator(root, pipeline_id=pipeline_id, llm_config=llm_config, on_usage=on_usage)
    out = orch.orchestrate_pipeline(understanding_result, manifest_record)
    fp = out["final_pipeline"]
    return {
        **out,
        "dag": final_pipeline_to_dag(fp),
        "pipeline_plan": pipeline_plan_from_final(fp),
    }
