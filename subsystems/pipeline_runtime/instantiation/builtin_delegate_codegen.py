# -*- coding: utf-8 -*-
"""内置算子：生成委托到 `handlers_deterministic` / `handlers_llm` 的 `operator_stub.py`（与全量执行同一逻辑）。"""

from __future__ import annotations

BUILTIN_DELEGATE_MODULE = '''# -*- coding: utf-8 -*-
# DataEvolver: built-in operator — delegates to the same Python handlers as `execute_generated_pipeline`.
# Not a passthrough stub; trial/subprocess 与本文件一致。
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _step_dict() -> dict[str, Any]:
    meta_p = Path(__file__).resolve().parent / "step_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    s = meta.get("orchestration_step")
    if not isinstance(s, dict):
        raise RuntimeError("step_meta.json missing orchestration_step")
    return s


def run(records: list[dict[str, Any]] | None, context: dict[str, Any] | None) -> list[dict[str, Any]]:
    from subsystems.pipeline_runtime.execution.handlers_deterministic import DETERMINISTIC_REGISTRY
    from subsystems.pipeline_runtime.execution.handlers_llm import LLM_REGISTRY
    from subsystems.pipeline_runtime.execution.handlers_multimodal import MULTIMODAL_REGISTRY

    handlers: dict[str, Any] = {**DETERMINISTIC_REGISTRY, **LLM_REGISTRY, **MULTIMODAL_REGISTRY}
    step = _step_dict()
    op = str(step.get("operator") or "")
    fn = handlers.get(op)
    if fn is None:
        raise RuntimeError(f"no built-in handler for operator {op!r}")
    return fn(records or [], step, context or {})


def describe() -> str:
    s = _step_dict()
    return str(s.get("operator", "?")) + ": " + str(s.get("description", ""))[:200]
'''


def generate_builtin_delegate_module() -> str:
    return BUILTIN_DELEGATE_MODULE
