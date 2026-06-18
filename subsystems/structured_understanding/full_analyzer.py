"""
完整结构化理解：单次 LLM 调用，由浅入深输出完整 profile。

对齐业务字段：`basic_information`、`schema_analysis`、`dataset_level_delta`；
使用 `core.llm_client` + JSON mode；数据从 manifest 路径经仓库根解析。
"""

from __future__ import annotations

import base64
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

from core.llm_client import LLMClientError, chat_completion, parse_message_content_json

from subsystems.structured_understanding.orchestration_feedback import (
    format_pilot_feedback_for_understanding_prompt,
)
from subsystems.structured_understanding.prompts_full import (
    UNIFIED_PROFILE_SYSTEM_PROMPT,
    UNIFIED_PROFILE_USER_TEMPLATE,
    MULTIMODAL_SYSTEM_HINT,
    DOCUMENT_SYSTEM_HINT,
)

logger = logging.getLogger(__name__)

MAX_JSONL_ROWS = 2000
# 单次合并分析，适当提高上限；仍受配置 max_tokens 约束
MAX_LLM_TOKENS_UNIFIED_CAP = 16384


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
        if t is bool and not isinstance(v, bool):
            return False
    return True


def _default_for_type(t: type) -> Any:
    if t is dict:
        return {}
    if t is list:
        return []
    if t is str:
        return "解析失败"
    if t is bool:
        return False
    return None


def _ensure_basic_information(bi: Any) -> dict[str, Any]:
    if not isinstance(bi, dict):
        bi = {}
    bi.setdefault("language", "unknown")
    ff = bi.setdefault("file_format_analysis", {})
    if isinstance(ff, dict):
        ff.setdefault("raw_data_format", "")
        ff.setdefault("seed_data_format", "")
        ff.setdefault("format_differences", "")
        ff.setdefault("nested_structure_notes", "")
    sq = bi.setdefault("seed_vs_raw_quality", {})
    if isinstance(sq, dict):
        sq.setdefault("quality_improvements", [])
        sq.setdefault("content_differences", "")
        sq.setdefault("style_differences", "")
        sq.setdefault("format_improvements", "")
        sq.setdefault("detail_level_comparison", "")
        sq.setdefault("nested_field_improvements", "")
    bi.setdefault("processing_targets", [])
    bi.setdefault("domain_characteristics", "")
    bi.setdefault("data_types", [])
    bi.setdefault("transformation_direction", "")
    bi.setdefault("quality_standards", "")
    return bi


def _ensure_schema_analysis(sa: Any) -> dict[str, Any]:
    if not isinstance(sa, dict):
        sa = {}
    sa.setdefault("schema_stable", True)
    sa.setdefault("raw_fields", [])
    sa.setdefault("seed_fields", [])
    sa.setdefault("raw_field_details", {})
    sa.setdefault("seed_field_details", {})
    sa.setdefault("new_fields", [])
    sa.setdefault("missing_fields", [])
    sa.setdefault("nested_structure_changes", "")
    sa.setdefault("schema_constraint", "")
    sa.setdefault("field_usage_guidance", {})
    return sa


def _ensure_dataset_level_delta(dl: Any) -> dict[str, Any]:
    if not isinstance(dl, dict):
        dl = {}
    dl.setdefault("global_optimization_direction", "")
    dl.setdefault("key_improvements", [])
    dl.setdefault("transformation_strategies", [])
    dl.setdefault("quality_focus", [])
    dl.setdefault("summary", "")
    dl.setdefault("concrete_field_and_schema_deltas", [])
    dl.setdefault("value_pattern_contrasts", [])
    dl.setdefault("seed_higher_bar_signals", [])
    dl.setdefault("risks_if_ignored", [])
    return dl


def normalize_language_code_from_label(language_str: str) -> str:
    if not language_str or str(language_str).lower() == "unknown":
        return "unknown"
    s = str(language_str).strip().lower()
    language_map = {
        "english": "en",
        "chinese": "zh",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "japanese": "ja",
        "korean": "ko",
        "russian": "ru",
        "portuguese": "pt",
        "italian": "it",
        "arabic": "ar",
        "hindi": "hi",
    }
    if s in language_map:
        return language_map[s]
    for key, code in language_map.items():
        if key in s or s in key:
            return code
    if any(c in language_str for c in ("中文", "Chinese", "chinese")):
        return "zh"
    if any(c in language_str for c in ("English", "english", "英文")):
        return "en"
    return s[:8] if s else "unknown"


def load_raw_data(root: Path, record: dict[str, Any]) -> dict[str, list[Any]]:
    raw_data: dict[str, list[Any]] = {}
    for rel in record.get("raw_data_files") or []:
        if not isinstance(rel, str):
            continue
        path = (root / rel).resolve()
        if not str(path).startswith(str(root.resolve())) or not path.is_file():
            logger.warning("skip raw path: %s", rel)
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                rows: list[Any] = []
                with open(path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= MAX_JSONL_ROWS:
                            break
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                raw_data[path.name] = rows
            elif path.suffix.lower() == ".json":
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                raw_data[path.name] = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, OSError) as e:
            logger.error("load raw %s: %s", path, e)
    return raw_data


def load_seed_data(root: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    files = [x for x in (record.get("seed_data_files") or []) if isinstance(x, str)]
    if not files:
        raise FileNotFoundError("manifest 中无 seed_data_files")
    path = (root / files[0]).resolve()
    if not str(path).startswith(str(root.resolve())) or not path.is_file():
        raise FileNotFoundError(f"seed 文件不存在: {files[0]}")
    if path.suffix.lower() == ".jsonl":
        out: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= MAX_JSONL_ROWS:
                    break
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
        return out
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    raise ValueError(f"不支持的 seed 格式: {path.suffix}")


def load_user_requirements(root: Path, record: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for rel in record.get("description_data_files") or []:
        if not isinstance(rel, str):
            continue
        path = (root / rel).resolve()
        if str(path).startswith(str(root.resolve())) and path.is_file():
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace").strip())
            except OSError:
                pass
    text = "\n\n".join(p for p in parts if p)
    return text if text else None


def load_experience(root: Path, pipeline_id: str) -> str | None:
    p = root / "data" / "experiences" / f"{pipeline_id}.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                parts: list[str] = []

                # 基础经验文本
                exp = data.get("experience_text")
                if isinstance(exp, str) and exp.strip():
                    parts.append(exp.strip())

                # 算子级诊断（experience_snapshot_v2新增）
                diagnoses = data.get("operator_diagnoses") or []
                if isinstance(diagnoses, list) and diagnoses:
                    parts.append("【算子级诊断 - 下一轮编排必须针对性修复】")
                    for d in diagnoses[:5]:
                        if not isinstance(d, dict):
                            continue
                        sev = d.get("severity", "medium")
                        op = d.get("operator", "unknown")
                        prob = str(d.get("problem") or "")[:200]
                        fix = str(d.get("fix_suggestion") or "")[:200]
                        parts.append(f"  [{sev.upper()}] {op}: {prob}")
                        if fix:
                            parts.append(f"    建议修复: {fix}")

                # 最优先修复项
                priority = str(data.get("priority_fix") or "").strip()
                if priority:
                    parts.append(f"【最优先修复】{priority}")

                # 下轮编排指令
                instruction = str(data.get("next_round_instruction") or "").strip()
                if instruction:
                    parts.append(f"【下轮编排指令】{instruction}")

                # strategy_pool历史
                try:
                    from subsystems.workflow.strategy_pool import format_strategy_pool_for_prompt
                    sp_text = format_strategy_pool_for_prompt(root, pipeline_id)
                    if sp_text:
                        parts.append(sp_text)
                except Exception:
                    pass

                if parts:
                    return "\n".join(parts)
        except (json.JSONDecodeError, OSError):
            pass
    legacy = root / "datasets" / pipeline_id / "experience.txt"
    if legacy.is_file():
        try:
            t = legacy.read_text(encoding="utf-8", errors="replace").strip()
            return t or None
        except OSError:
            pass
    return None


def sample_raw_data(raw_data: dict[str, list[Any]], target_count: int) -> dict[str, list[Any]]:
    sampled: dict[str, list[Any]] = {}
    for file_name, data in raw_data.items():
        if isinstance(data, list):
            if len(data) > target_count:
                sampled[file_name] = random.sample(data, target_count)
            else:
                sampled[file_name] = data
        else:
            sampled[file_name] = data if isinstance(data, list) else [data]
    return sampled


def extract_all_fields(obj: Any, prefix: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            field_path = f"{prefix}.{key}" if prefix else key
            fields.append(field_path)
            if isinstance(value, dict):
                fields.extend(extract_all_fields(value, field_path))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                fields.append(f"{field_path}[]")
                fields.extend(extract_all_fields(value[0], f"{field_path}[]"))
    return fields



def _encode_images_from_records(
    records: list[dict[str, Any]],
    root: Path,
    max_images: int = 2,
) -> list[dict]:
    """从records里提取image_path字段，编码成base64供LLM vision使用。"""
    result = []
    image_fields = ["image_path", "image", "img_path"]
    seen = 0
    for r in records:
        if not isinstance(r, dict):
            continue
        for field in image_fields:
            val = r.get(field)
            if not isinstance(val, str) or not val:
                continue
            p = Path(val)
            if not p.is_absolute():
                p = root / val
            if not p.is_file() or p.stat().st_size == 0:
                continue
            try:
                with open(p, "rb") as f:
                    data = f.read()
                ext = p.suffix.lower().lstrip(".")
                mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
                b64 = base64.b64encode(data).decode("utf-8")
                result.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    "_source_field": field,
                    "_path": str(val),
                })
                seen += 1
                break
            except Exception as e:
                logger.warning("encode image failed: %s", e)
        if seen >= max_images:
            break
    return result


class FullUnderstandingAnalyzer:
    def __init__(
        self,
        root: Path,
        llm_config: dict[str, Any],
        on_usage: Callable[..., None] | None = None,
    ) -> None:
        self._root = root
        self._cfg = llm_config
        self._on_usage = on_usage

    def _emit_usage(self, usage: dict[str, Any]) -> None:
        if not self._on_usage:
            return
        kw: dict[str, Any] = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "model": str(self._cfg.get("model")),
            "operation": "understanding.unified_profile",
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

    def _max_tokens_unified(self) -> int:
        cfg_mt = int(self._cfg.get("max_tokens", 8192))
        return min(MAX_LLM_TOKENS_UNIFIED_CAP, max(2048, cfg_mt))

    def _timeout_sec(self) -> float:
        return max(120.0, float(self._cfg.get("timeout", 120)))

    def _call_llm_json(
        self,
        user_prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        max_retries: int = 2,
        images: list[dict] | None = None,
    ) -> dict[str, Any]:
        base_url = str(self._cfg["base_url"])
        api_key = str(self._cfg["api_key"])
        model = str(self._cfg["model"])
        temperature = float(self._cfg.get("temperature", 0.1))
        timeout = self._timeout_sec()

        last_err: Exception | None = None
        prompt = user_prompt
        for attempt in range(max_retries + 1):
            try:
                # 构建user message content，支持图片
                if images and attempt == 0:
                    user_content: Any = [{"type": "text", "text": prompt}]
                    for img in images:
                        safe_img = {k: v for k, v in img.items() if not k.startswith("_")}
                        user_content.append(safe_img)
                else:
                    user_content = prompt

                resp = chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout,
                    json_mode=True,
                )
                parsed, usage = parse_message_content_json(resp)
                self._emit_usage(usage)
                return parsed
            except LLMClientError as e:
                last_err = e
                logger.warning("LLM attempt %s failed: %s", attempt + 1, e)
            except (json.JSONDecodeError, KeyError) as e:
                last_err = e
                logger.warning("parse attempt %s failed: %s", attempt + 1, e)
            prompt = user_prompt + "\n\nReturn a single valid JSON object only. No markdown."
        assert last_err is not None
        raise last_err

    def _call_llm_json_with_keys(
        self,
        user_prompt: str,
        required_keys: dict[str, type],
        *,
        system_prompt: str,
        max_tokens: int,
        images: list[dict] | None = None,
    ) -> dict[str, Any]:
        parsed = self._call_llm_json(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens, images=images)
        if _ensure_keys(parsed, required_keys):
            return parsed
        missing = [k for k in required_keys if k not in parsed]
        logger.warning("missing keys %s, got %s", missing, list(parsed.keys()))
        retry_prompt = (
            user_prompt
            + f"\n\nYour previous JSON was missing required keys: {missing}. "
            + f"Required keys and types: { {k: t.__name__ for k, t in required_keys.items()} }."
        )
        parsed2 = self._call_llm_json(retry_prompt, system_prompt=system_prompt, max_tokens=max_tokens)
        if _ensure_keys(parsed2, required_keys):
            return parsed2

        def _fits(v: Any, t: type) -> bool:
            if t is str:
                return isinstance(v, str)
            if t is dict:
                return isinstance(v, dict)
            if t is list:
                return isinstance(v, list)
            if t is bool:
                return isinstance(v, bool)
            return False

        out: dict[str, Any] = {}
        for k, t in required_keys.items():
            if k in parsed2 and _fits(parsed2[k], t):
                out[k] = parsed2[k]
            elif k in parsed and _fits(parsed[k], t):
                out[k] = parsed[k]
            else:
                out[k] = _default_for_type(t)
        return out

    def _build_unified_user_prompt(
        self,
        raw_sampled: dict[str, list[Any]],
        seed_data: list[dict[str, Any]],
        pipeline_config: dict[str, Any],
        user_requirements: str | None,
        experience: str | None,
    ) -> str:
        raw_samples: list[dict[str, Any]] = []
        for file_name, data in list(raw_sampled.items())[:2]:
            if isinstance(data, list) and data:
                raw_samples.append({"file": file_name, "sample": data[0]})
        seed_samples = seed_data[:3]
        ur = user_requirements if user_requirements else "No user requirements provided."
        exp_section = "(none)"
        if experience:
            exp_section = experience

        raw_fields: list[str] = []
        if raw_sampled:
            first = list(raw_sampled.values())[0]
            if isinstance(first, list) and first:
                raw_fields = extract_all_fields(first[0])
        seed_fields: list[str] = []
        if seed_data:
            seed_fields = extract_all_fields(seed_data[0])
        raw_sample: dict[str, Any] = {}
        if raw_sampled:
            fd = list(raw_sampled.values())[0]
            if isinstance(fd, list) and fd and isinstance(fd[0], dict):
                raw_sample = fd[0]
        seed_sample = seed_data[0] if seed_data else {}

        seed_examples: list[dict[str, Any]] = []
        for item in seed_data[:3]:
            example = {
                k: (str(v)[:100] if not isinstance(v, (dict, list)) else type(v).__name__)
                for k, v in list(item.items())[:5]
            }
            seed_examples.append(example)

        return UNIFIED_PROFILE_USER_TEMPLATE.format(
            pipeline_config=json.dumps(pipeline_config, ensure_ascii=False, indent=2),
            user_requirements=ur,
            experience_section=exp_section,
            raw_data_preview=json.dumps(raw_samples, ensure_ascii=False, indent=2),
            seed_data_preview=json.dumps(seed_samples, ensure_ascii=False, indent=2),
            raw_schema=json.dumps(raw_fields, ensure_ascii=False, indent=2),
            seed_schema=json.dumps(seed_fields, ensure_ascii=False, indent=2),
            raw_sample=json.dumps(raw_sample, ensure_ascii=False, indent=2),
            seed_sample=json.dumps(seed_sample, ensure_ascii=False, indent=2),
            seed_examples=json.dumps(seed_examples, ensure_ascii=False, indent=2),
        )

    def run(self, pipeline_id: str, record: dict[str, Any]) -> dict[str, Any]:
        raw_data = load_raw_data(self._root, record)
        if not raw_data:
            raise ValueError("未能加载任何 raw 数据")
        seed_data = load_seed_data(self._root, record)
        if not seed_data:
            raise ValueError("未能加载 seed 数据")
        seed_count = len(seed_data)
        raw_sampled = sample_raw_data(raw_data, max(1, min(seed_count, 20)))
        pipeline_config = {
            "pipeline_id": pipeline_id,
            "raw_data_files": record.get("raw_data_files"),
            "seed_data_files": record.get("seed_data_files"),
            "description_data_files": record.get("description_data_files"),
            "domain": record.get("domain"),
            "task_type": record.get("task_type"),
            "language": record.get("language"),
        }
        user_req = load_user_requirements(self._root, record)
        experience = load_experience(self._root, pipeline_id)

        pilot_prompt: str | None = None
        u_path = self._root / "data" / "understanding_results" / f"{pipeline_id}.json"
        if u_path.is_file():
            try:
                prev_u = json.loads(u_path.read_text(encoding="utf-8"))
                if isinstance(prev_u, dict):
                    pilot_prompt = format_pilot_feedback_for_understanding_prompt(prev_u)
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        exp_parts: list[str] = []
        if experience:
            exp_parts.append(experience)
        if pilot_prompt:
            exp_parts.append(pilot_prompt)
        merged_experience = "\n\n".join(exp_parts) if exp_parts else None

        user_prompt = self._build_unified_user_prompt(
            raw_sampled, seed_data, pipeline_config, user_req, merged_experience
        )
        required_top = {
            "basic_information": dict,
            "schema_analysis": dict,
            "dataset_level_delta": dict,
        }
        # 提取raw和seed的图片供vision理解
        raw_records: list[dict] = []
        for rows in raw_sampled.values():
            if isinstance(rows, list):
                raw_records.extend(rows[:2])
        seed_records = seed_data[:2]

        raw_images = _encode_images_from_records(raw_records, self._root, max_images=2)
        seed_images = _encode_images_from_records(seed_records, self._root, max_images=2)

        all_images: list[dict] = []
        if raw_images:
            all_images.append({"type": "text", "text": "=== RAW DATA IMAGES (before processing) ==="})
            all_images.extend(raw_images)
        if seed_images:
            all_images.append({"type": "text", "text": "=== SEED DATA IMAGES (target quality, faces should be blurred) ==="})
            all_images.extend(seed_images)

        unified = self._call_llm_json_with_keys(
            user_prompt,
            required_top,
            system_prompt=UNIFIED_PROFILE_SYSTEM_PROMPT + "\n\n" + MULTIMODAL_SYSTEM_HINT + "\n\n" + DOCUMENT_SYSTEM_HINT,
            max_tokens=self._max_tokens_unified(),
            images=all_images if all_images else None,
        )
        basic = _ensure_basic_information(unified.get("basic_information"))
        schema = _ensure_schema_analysis(unified.get("schema_analysis"))
        delta = _ensure_dataset_level_delta(unified.get("dataset_level_delta"))

        lang_label = str(basic.get("language", "unknown"))
        language_code = normalize_language_code_from_label(lang_label)

        return {
            "pipeline_id": pipeline_id,
            "domain": basic.get("domain_characteristics", "unknown"),
            "language": language_code,
            "language_label": lang_label,
            "language_constraint": language_code,
            "language_consistency": True,
            "user_declared_language": record.get("language") if isinstance(record.get("language"), str) else None,
            "task_type": record.get("task_type") if isinstance(record.get("task_type"), str) else None,
            "basic_information": basic,
            "schema_analysis": schema,
            "dataset_level_delta": delta,
            "content_slot_library": {},
            "quality_rubrics": {},
        }


def run_full_understanding(
    root: Path,
    pipeline_id: str,
    record: dict[str, Any],
    llm_config: dict[str, Any],
    on_usage: Callable[..., None] | None = None,
) -> dict[str, Any]:
    if not str(llm_config.get("api_key") or "").strip():
        raise ValueError("完整理解需要配置 API Key")
    analyzer = FullUnderstandingAnalyzer(root, llm_config, on_usage=on_usage)
    return analyzer.run(pipeline_id, record)
