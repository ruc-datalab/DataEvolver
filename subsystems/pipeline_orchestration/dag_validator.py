"""
编排产物校验：算子注册表对齐 + DAG 与 final_pipeline 一致性。
在 workflow「dag_validation」步写入 `constrained_search.validation_result`，
并与编排阶段已有的校验问题合并（不重复执行 LLM）。
"""

from __future__ import annotations

from typing import Any

from subsystems.pipeline_orchestration.dag_graph_analysis import validate_dag_graph_topology


def _norm_issue(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    return {"type": "generic", "description": str(x)}


def validate_steps_against_registry(
    final_pipeline: list[Any],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    """检查每步 operator 存在于注册表，且 input/output 与注册表声明基本一致。"""
    issues: list[dict[str, Any]] = []
    if not isinstance(final_pipeline, list):
        return [
            {
                "type": "registry",
                "description": "final_pipeline 不是列表",
            }
        ]
    for idx, step in enumerate(final_pipeline):
        if not isinstance(step, dict):
            issues.append(
                {
                    "type": "registry",
                    "step_index": idx,
                    "description": f"步骤 {idx} 不是对象",
                }
            )
            continue
        op = step.get("operator")
        sid = step.get("step_id", f"step_{idx + 1}")
        if not op or not isinstance(op, str):
            issues.append(
                {
                    "type": "registry",
                    "step_id": sid,
                    "description": "缺少 operator 字段",
                }
            )
            continue
        spec = registry.get(op)
        if not isinstance(spec, dict):
            issues.append(
                {
                    "type": "unknown_operator",
                    "step_id": sid,
                    "operator": op,
                    "description": f"算子未在注册表中定义: {op}",
                }
            )
            continue
        step_in = set(step.get("input_keys") or [])
        step_out = set(step.get("output_keys") or [])
        spec_in = list(spec.get("input_keys") or [])
        spec_out = list(spec.get("output_keys") or [])
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        param_keys = set(params.keys())
        requires_llm = bool(spec.get("requires_llm", False))
        for ink in spec_in:
            if ink in step_in or ink in param_keys:
                continue
            # 语义类算子的大量配置键在「实例化」阶段写入 parameters；DAG 步只强约束数据流主键
            if requires_llm and ink not in ("file_path", "records"):
                continue
            # 非LLM算子的配置参数也允许在parameters里提供，只强约束数据流主键
            if not requires_llm and ink not in ("records", "file_path"):
                continue
            issues.append(
                {
                    "type": "registry_io",
                    "step_id": sid,
                    "operator": op,
                    "description": (
                        f"步骤声明的 input_keys / parameters 未覆盖注册表要求的输入键 `{ink}`"
                    ),
                    "missing_input_key": ink,
                }
            )
        for outk in spec_out:
            if outk not in step_out:
                # write_data 常规模型漏写透传 records；有 records 输入即可视为与注册表语义一致
                if op == "write_data" and outk == "records" and "records" in step_in:
                    continue
                issues.append(
                    {
                        "type": "registry_io",
                        "step_id": sid,
                        "operator": op,
                        "description": (
                            f"步骤 output_keys 缺少注册表声明的输出键 `{outk}`"
                        ),
                        "missing_output_key": outk,
                    }
                )
    return issues


def validate_dag_matches_pipeline(
    dag: Any,
    final_pipeline: list[Any],
) -> list[dict[str, Any]]:
    """检查 dag.nodes 与 final_pipeline 一一对应（顺序、算子名、边与 execution_order）。"""
    issues: list[dict[str, Any]] = []
    if not isinstance(dag, dict):
        return [{"type": "dag", "description": "缺少 dag 对象或类型无效"}]
    nodes = dag.get("nodes")
    if not isinstance(nodes, list):
        return [{"type": "dag", "description": "dag.nodes 不是列表"}]
    if not isinstance(final_pipeline, list):
        return [{"type": "dag", "description": "final_pipeline 不是列表"}]
    if len(nodes) != len(final_pipeline):
        issues.append(
            {
                "type": "dag_count",
                "description": (
                    f"dag.nodes 数量 ({len(nodes)}) 与 final_pipeline ({len(final_pipeline)}) 不一致"
                ),
            }
        )
    n = min(len(nodes), len(final_pipeline))
    for i in range(n):
        node = nodes[i]
        step = final_pipeline[i]
        if not isinstance(node, dict) or not isinstance(step, dict):
            continue
        nn = node.get("node_name")
        op = step.get("operator")
        if nn != op:
            issues.append(
                {
                    "type": "dag_operator_mismatch",
                    "index": i,
                    "description": f"位置 {i}: dag.node_name={nn!r} 与 pipeline.operator={op!r} 不一致",
                }
            )
    order = dag.get("execution_order")
    if isinstance(order, list) and order:
        if len(order) != len(nodes):
            issues.append(
                {
                    "type": "dag_order",
                    "description": (
                        f"execution_order 长度 ({len(order)}) 与 nodes ({len(nodes)}) 不一致"
                    ),
                }
            )
        else:
            for i, nid in enumerate(order):
                node = nodes[i] if i < len(nodes) else {}
                if isinstance(node, dict) and node.get("node_id") != nid:
                    issues.append(
                        {
                            "type": "dag_order",
                            "index": i,
                            "description": (
                                f"execution_order[{i}]={nid!r} 与 nodes[{i}].node_id={node.get('node_id')!r} 不一致"
                            ),
                        }
                    )
    edges = dag.get("edges")
    if isinstance(edges, list) and len(nodes) >= 2:
        if len(edges) < len(nodes) - 1:
            issues.append(
                {
                    "type": "dag_edges",
                    "description": (
                        f"线性管线期望至少 {len(nodes) - 1} 条边，当前 {len(edges)}"
                    ),
                }
            )
        order_ids = order if isinstance(order, list) and len(order) == len(nodes) else [n.get("node_id") for n in nodes if isinstance(n, dict)]
        if len(order_ids) >= 2:
            edge_set = {(e.get("from_node"), e.get("to_node")) for e in edges if isinstance(e, dict)}
            for i in range(len(order_ids) - 1):
                a, b = order_ids[i], order_ids[i + 1]
                if (a, b) not in edge_set:
                    issues.append(
                        {
                            "type": "dag_edges",
                            "description": f"缺少边 {a!r} -> {b!r}（与 execution_order 相邻步）",
                        }
                    )
    return issues


def validate_read_write_bookends(final_pipeline: list[Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not final_pipeline or not isinstance(final_pipeline, list):
        return issues
    first = final_pipeline[0]
    last = final_pipeline[-1]
    if isinstance(first, dict) and first.get("operator") != "read_data":
        issues.append(
            {
                "type": "pipeline_structure",
                "description": "第一步应为 read_data",
            }
        )
    if isinstance(last, dict) and last.get("operator") != "write_data":
        issues.append(
            {
                "type": "pipeline_structure",
                "description": "最后一步应为 write_data",
            }
        )
    return issues


# 本模块写入的 issue.type，合并时先从旧结果中剥掉再追加，避免重复跑 dag_validation 时叠加
# dag_closure：编排器内启发式曾误报「records 入出同名」；合并时丢弃旧条，由当前规则与图论重算
_DAG_VALIDATOR_ISSUE_TYPES = frozenset(
    {
        "registry",
        "unknown_operator",
        "registry_io",
        "dag",
        "dag_count",
        "dag_operator_mismatch",
        "dag_order",
        "dag_edges",
        "pipeline_structure",
        "dag_graph",
        "dag_graph_bad_node",
        "dag_graph_duplicate_id",
        "dag_graph_unknown_node",
        "dag_graph_multi_source",
        "dag_graph_multi_sink",
        "dag_cycle_or_disconnected",
        "dag_topo_order_conflict",
        "dag_disconnected",
        "dag_graph_order",
        "dag_closure",
    }
)


def merge_dag_validation(
    orchestration: dict[str, Any],
    *,
    merged_registry: dict[str, Any],
    checked_at: str,
) -> dict[str, Any]:
    """
    读取编排 JSON，合并原有 validation_result.validation_issues 与新增校验项，
    返回应写入的 `validation_result` 字典。
    """
    cs = orchestration.get("constrained_search")
    if not isinstance(cs, dict):
        cs = {}
    prev = cs.get("validation_result")
    prev_issues: list[Any] = []
    if isinstance(prev, dict):
        prev_issues = list(prev.get("validation_issues") or [])

    preserved: list[dict[str, Any]] = []
    for x in prev_issues:
        d = _norm_issue(x)
        if d.get("type") not in _DAG_VALIDATOR_ISSUE_TYPES:
            preserved.append(d)

    fp = orchestration.get("final_pipeline")
    if not isinstance(fp, list):
        fp = cs.get("final_pipeline") if isinstance(cs.get("final_pipeline"), list) else []

    dag = orchestration.get("dag")
    new_issues: list[dict[str, Any]] = []
    new_issues.extend(validate_read_write_bookends(fp))
    new_issues.extend(validate_steps_against_registry(fp, merged_registry))
    new_issues.extend(validate_dag_matches_pipeline(dag, fp))
    new_issues.extend(validate_dag_graph_topology(dag))

    combined = preserved + new_issues
    is_valid = len(combined) == 0

    return {
        "is_valid": is_valid,
        "validation_issues": combined,
        "checked_at": checked_at,
        "dag_validation_meta": {
            "preserved_issue_count": len(preserved),
            "added_issue_count": len(new_issues),
            "sources": [
                "orchestration",
                "registry",
                "dag_structure",
                "io_contract",
                "dag_graph_topology",
            ],
        },
        "note": "workflow dag_validation：合并编排阶段校验 + 注册表/DAG 对齐 + 图论检查",
    }
