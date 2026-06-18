"""多模态算子：图像处理、图文对齐、VLM调用（需要vision模型）。"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

OpFn = Callable[[list[dict[str, Any]], dict[str, Any], dict[str, Any]], list[dict[str, Any]]]


def _params(step: dict[str, Any]) -> dict[str, Any]:
    p = step.get("parameters")
    return p if isinstance(p, dict) else {}


def _stub_not_implemented(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """算子已注册但尚未实现，原样返回并记录警告。"""
    op = step.get("operator", "unknown")
    ctx.setdefault("execution_warnings", []).append(
        f"算子 {op!r} 尚未完整实现，已跳过（原样透传）"
    )
    return records


# ── 图像处理算子 ──────────────────────────────────────────────

def op_image_face_blur(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """检测图片中的人脸并打马赛克，输出新图片路径。"""
    try:
        import cv2
    except ImportError:
        ctx.setdefault("execution_warnings", []).append(
            "image_face_blur: 缺少opencv-python-headless，已跳过"
        )
        return records

    p = _params(step)
    image_field = str(p.get("image_field", "image_path"))
    blur_strength = int(p.get("blur_strength", 50))
    suffix = str(p.get("output_suffix", "_blurred"))

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    root = str(ctx.get("root", ""))
    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        c = dict(r)
        img_path = c.get(image_field)
        if not img_path:
            out.append(c)
            continue
        # 支持相对路径：相对于项目根目录
        p = Path(str(img_path))
        if not p.is_absolute() and root:
            p = Path(root) / p
        img_path = str(p)
        if not os.path.exists(img_path):
            ctx.setdefault("execution_warnings", []).append(
                f"image_face_blur: 找不到图片 {img_path!r}，跳过"
            )
            out.append(c)
            continue
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                out.append(c)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            face_count = 0
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi = img[y : y + h, x : x + w]
                    # 马赛克效果，视觉上更明显
                    small = cv2.resize(roi, (max(1, w//10), max(1, h//10)))
                    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    img[y : y + h, x : x + w] = mosaic
                    face_count += 1
            p_obj = Path(str(img_path))
            new_path = str(p_obj.parent / (p_obj.stem + suffix + p_obj.suffix))
            cv2.imwrite(new_path, img)
            c[image_field] = new_path
            c["_face_blur_meta"] = {"faces_detected": face_count, "original_path": str(img_path)}
        except Exception as e:
            ctx.setdefault("execution_warnings", []).append(
                f"image_face_blur: 处理 {img_path!r} 失败: {e}"
            )
        out.append(c)
    return out


def op_image_quality_filter(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """过滤分辨率过低或文件损坏的图片。"""
    try:
        from PIL import Image
    except ImportError:
        ctx.setdefault("execution_warnings", []).append(
            "image_quality_filter: 缺少pillow，已跳过"
        )
        return records

    p = _params(step)
    image_field = str(p.get("image_field", "image_path"))
    min_width = int(p.get("min_width", 64))
    min_height = int(p.get("min_height", 64))
    max_file_size_mb = float(p.get("max_file_size_mb", 50.0))

    root = str(ctx.get("root", ""))
    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        img_path = r.get(image_field)
        if not img_path:
            out.append(r)
            continue
        p = Path(str(img_path))
        if not p.is_absolute() and root:
            p = Path(root) / p
        img_path = str(p)
        if not os.path.exists(img_path):
            out.append(r)
            continue
        try:
            file_size_mb = os.path.getsize(str(img_path)) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                ctx.setdefault("execution_warnings", []).append(
                    f"image_quality_filter: {img_path} 文件过大({file_size_mb:.1f}MB)，已过滤"
                )
                continue
            with Image.open(str(img_path)) as im:
                w, h = im.size
                if w < min_width or h < min_height:
                    ctx.setdefault("execution_warnings", []).append(
                        f"image_quality_filter: {img_path} 分辨率过低({w}x{h})，已过滤"
                    )
                    continue
            out.append(r)
        except Exception as e:
            ctx.setdefault("execution_warnings", []).append(
                f"image_quality_filter: 无法读取 {img_path}: {e}，已过滤"
            )
    return out


def op_image_deduplicator(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """基于MD5哈希对图片去重。"""
    import hashlib

    p = _params(step)
    image_field = str(p.get("image_field", "image_path"))

    seen: set[str] = set()
    root = str(ctx.get("root", ""))
    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        img_path = r.get(image_field)
        if not img_path:
            out.append(r)
            continue
        p = Path(str(img_path))
        if not p.is_absolute() and root:
            p = Path(root) / p
        img_path = str(p)
        if not os.path.exists(img_path):
            out.append(r)
            continue
        try:
            with open(str(img_path), "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append(r)
        except Exception:
            out.append(r)
    return out


def op_image_resize_normalizer(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """统一图片尺寸。"""
    try:
        from PIL import Image
    except ImportError:
        ctx.setdefault("execution_warnings", []).append("image_resize_normalizer: 缺少pillow")
        return records

    p = _params(step)
    image_field = str(p.get("image_field", "image_path"))
    max_size = int(p.get("max_size", 1024))

    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        c = dict(r)
        img_path = c.get(image_field)
        if not img_path or not os.path.exists(str(img_path)):
            out.append(c)
            continue
        try:
            with Image.open(str(img_path)) as im:
                w, h = im.size
                if max(w, h) > max_size:
                    ratio = max_size / max(w, h)
                    new_w, new_h = int(w * ratio), int(h * ratio)
                    resized = im.resize((new_w, new_h), Image.LANCZOS)
                    p_obj = Path(str(img_path))
                    new_path = str(p_obj.parent / (p_obj.stem + "_resized" + p_obj.suffix))
                    resized.save(new_path)
                    c[image_field] = new_path
        except Exception as e:
            ctx.setdefault("execution_warnings", []).append(f"image_resize_normalizer: {e}")
        out.append(c)
    return out


# ── VLM / 图文多模态算子 ──────────────────────────────────────

def _call_vlm(ctx: dict[str, Any], image_path: str, prompt: str) -> str:
    """调用支持vision的LLM API（GPT-4o等），返回文本结果。"""
    cfg = ctx.get("llm_config", {})
    base_url = str(cfg.get("base_url", ""))
    api_key = str(cfg.get("api_key", ""))
    model = str(cfg.get("model", "gpt-4o-mini"))

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(
        ext.lstrip("."), "jpeg"
    )

    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 1024,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def op_vlm_generate_qa(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """用VLM根据图片生成高质量QA对，替换粗糙的question/answer字段。"""
    p = _params(step)
    image_field = str(p.get("image_field", "image_path"))
    question_field = str(p.get("question_field", "question"))
    answer_field = str(p.get("answer_field", "answer"))
    prompt_spec = str(
        p.get(
            "prompt_spec",
            "请根据图片内容生成一个有意义的问题和详细答案。注意：如果图片中有人脸被打码，请不要询问人物身份，而是描述场景、活动或其他可见内容。"
            '请以JSON格式返回：{"question": "问题", "answer": "详细答案"}',
        )
    )

    from subsystems.pipeline_runtime.execution.handlers_deterministic import _params as _p
    head = records[: ctx.get("llm_max_records_per_step") or len(records)]
    tail = records[len(head) :]

    out = []
    for r in head:
        if not isinstance(r, dict):
            continue
        c = dict(r)
        img_path = c.get(image_field)
        if not img_path or not os.path.exists(str(img_path)):
            out.append(c)
            continue
        try:
            raw_response = _call_vlm(ctx, str(img_path), prompt_spec)
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            # 先尝试JSON解析，失败则直接用文本作为answer
            try:
                parsed = json.loads(clean)
                if isinstance(parsed, dict):
                    if "question" in parsed:
                        c[question_field] = parsed["question"]
                    if "answer" in parsed:
                        c[answer_field] = parsed["answer"]
                else:
                    c[answer_field] = clean
            except json.JSONDecodeError:
                # VLM返回的是自然语言文本，直接作为answer
                c[answer_field] = clean
        except Exception as e:
            import traceback
            ctx.setdefault("execution_warnings", []).append(
                f"vlm_generate_qa: 处理 {img_path!r} 失败: {e}\n{traceback.format_exc()[-500:]}"
            )
        out.append(c)
    return out + tail


def op_vlm_image_caption(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """用VLM为图片生成caption描述。"""
    p = _params(step)
    image_field = str(p.get("image_field", "image_path"))
    caption_field = str(p.get("caption_field", "caption"))
    prompt = str(p.get("prompt", "请用一两句话描述这张图片的内容。"))

    head = records[: ctx.get("llm_max_records_per_step") or len(records)]
    tail = records[len(head) :]

    out = []
    for r in head:
        if not isinstance(r, dict):
            continue
        c = dict(r)
        img_path = c.get(image_field)
        if not img_path or not os.path.exists(str(img_path)):
            out.append(c)
            continue
        try:
            caption = _call_vlm(ctx, str(img_path), prompt)
            c[caption_field] = caption.strip()
        except Exception as e:
            ctx.setdefault("execution_warnings", []).append(
                f"vlm_image_caption: {img_path!r} 失败: {e}"
            )
        out.append(c)
    return out + tail


def op_image_text_alignment_filter(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """用VLM过滤图文不一致的样本（stub，需要vision API）。"""
    return _stub_not_implemented(records, step, ctx)


def op_image_watermark_filter(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """过滤有水印的图片（stub）。"""
    return _stub_not_implemented(records, step, ctx)


def op_image_nsfw_filter(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """过滤NSFW内容（stub，需要专用模型）。"""
    return _stub_not_implemented(records, step, ctx)


# ── 注册表 ────────────────────────────────────────────────────

MULTIMODAL_REGISTRY: dict[str, OpFn] = {
    # 图像处理
    "image_face_blur": op_image_face_blur,
    "image_quality_filter": op_image_quality_filter,
    "image_deduplicator": op_image_deduplicator,
    "image_resize_normalizer": op_image_resize_normalizer,
    # VLM图文
    "vlm_generate_qa": op_vlm_generate_qa,
    "vlm_image_caption": op_vlm_image_caption,
    # stub（已注册未实现）
    "image_text_alignment_filter": op_image_text_alignment_filter,
    "image_watermark_filter": op_image_watermark_filter,
    "image_nsfw_filter": op_image_nsfw_filter,
}

MULTIMODAL_OPERATOR_NAMES = frozenset(MULTIMODAL_REGISTRY.keys())


# ── 文档解析算子 ──────────────────────────────────────────────

def op_parse_pdf_to_chunks(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """用pypdfium2把PDF解析成文本chunks，每个chunk一条record。"""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        ctx.setdefault("execution_warnings", []).append(
            "parse_pdf_to_chunks: 缺少pypdfium2，已跳过"
        )
        return records

    p = _params(step)
    pdf_field = str(p.get("pdf_field", "pdf_path"))
    chunk_size = int(p.get("chunk_size", 500))
    root = str(ctx.get("root", ""))

    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        pdf_path = r.get(pdf_field)
        if not pdf_path:
            out.append(r)
            continue
        p_obj = Path(str(pdf_path))
        if not p_obj.is_absolute() and root:
            p_obj = Path(root) / p_obj
        if not p_obj.exists():
            ctx.setdefault("execution_warnings", []).append(
                f"parse_pdf_to_chunks: 找不到PDF {pdf_path!r}，跳过"
            )
            out.append(r)
            continue
        try:
            pdf = pdfium.PdfDocument(str(p_obj))
            # 按页提取文字，合并成段落
            all_text = ""
            for i in range(len(pdf)):
                page = pdf[i]
                textpage = page.get_textpage()
                page_text = textpage.get_text_range().strip()
                if page_text:
                    all_text += page_text + "\n\n"

            # 按段落切块
            paragraphs = [p.strip() for p in all_text.split("\n\n") if len(p.strip()) > 20]

            # 合并短段落成chunks
            chunks = []
            current = ""
            for para in paragraphs:
                if len(current) + len(para) < chunk_size:
                    current = current + "\n" + para if current else para
                else:
                    if current:
                        chunks.append(current.strip())
                    current = para
            if current:
                chunks.append(current.strip())

            # 每个chunk生成一条record
            for i, chunk in enumerate(chunks):
                new_record = {
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_type": "text",
                    "source_pdf": str(pdf_path),
                    "question": "",
                    "answer": "",
                }
                for k, v in r.items():
                    if k not in new_record:
                        new_record[k] = v
                out.append(new_record)

            ctx.setdefault("execution_warnings", []).append(
                f"parse_pdf_to_chunks: {pdf_path!r} 解析成功，共{len(chunks)}个chunks"
            )
        except Exception as e:
            ctx.setdefault("execution_warnings", []).append(
                f"parse_pdf_to_chunks: 处理 {pdf_path!r} 失败: {e}"
            )
            out.append(r)
    return out


def op_ocr_noise_clean(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """清洗chunk_text中的OCR噪声：去除乱码字符、多余空白、页眉页脚等。"""
    import re
    p = _params(step)
    text_field = str(p.get("text_field", "chunk_text"))
    min_length = int(p.get("min_length", 20))

    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        c = dict(r)
        text = c.get(text_field, "")
        if not isinstance(text, str):
            out.append(c)
            continue
        # 去除乱码（连续的特殊字符）
        text = re.sub(r'[^\w\s\u4e00-\u9fff\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\\+\=\*\&\%\$\#\@]', ' ', text)
        # 规范化空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 过滤太短的chunk
        if len(text) < min_length:
            ctx.setdefault("execution_warnings", []).append(
                f"ocr_noise_clean: chunk过短({len(text)}字符)，已过滤"
            )
            continue
        c[text_field] = text
        out.append(c)
    return out


def op_document_qa_generator(
    records: list[dict[str, Any]], step: dict[str, Any], ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """根据chunk_text用LLM生成DocQA训练数据，填充question和answer字段。"""
    import urllib.request

    p = _params(step)
    text_field = str(p.get("text_field", "chunk_text"))
    question_field = str(p.get("question_field", "question"))
    answer_field = str(p.get("answer_field", "answer"))
    qa_style = str(p.get("qa_style", "extractive"))
    prompt_spec = str(p.get("prompt_spec", ""))

    cfg = ctx.get("llm_config", {})
    base_url = str(cfg.get("base_url", "")).rstrip("/")
    api_key = str(cfg.get("api_key", ""))
    model = str(cfg.get("model", "gpt-4o-mini"))

    head = records[: ctx.get("llm_max_records_per_step") or len(records)]
    tail = records[len(head):]

    default_prompt = (
        "你是一个文档问答生成助手。根据下面的文档片段，生成恰好一个问题和对应的答案。"
        "严格要求：\n"
        "1. 只输出纯JSON，不要有任何markdown、代码块、解释或其他文字\n"
        "2. 格式必须完全是：{\"question\": \"问题内容\", \"answer\": \"答案内容\"}\n"
        "3. 问题和答案都用中文\n"
        "4. 答案100字以内，直接回答问题\n"
        "5. 不要生成多个QA，只要一个\n"
        "文档片段如下：\n"
    )
    prompt_template = prompt_spec if prompt_spec else default_prompt

    out = []
    for r in head:
        if not isinstance(r, dict):
            continue
        c = dict(r)
        text = c.get(text_field, "")
        if not text or not isinstance(text, str) or len(text) < 20:
            out.append(c)
            continue
        # 跳过已有高质量QA的记录
        if c.get(question_field) and c.get(answer_field):
            out.append(c)
            continue
        try:
            def _call_llm(prompt_text):
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 200,
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
                return result["choices"][0]["message"]["content"].strip()

            chunk = text[:1500]
            # 第一步：只生成一个问题
            q_prompt = f"根据以下文档片段，用中文写一个可以从文档中找到答案的问题。只输出问题本身，不要任何其他内容。\n\n{chunk}"
            question = _call_llm(q_prompt)
            # 第二步：根据问题和文档生成答案
            a_prompt = f"根据以下文档片段回答问题，用中文，100字以内，直接给出答案。\n\n文档：{chunk}\n\n问题：{question}"
            answer = _call_llm(a_prompt)
            c[question_field] = question
            c[answer_field] = answer
        except Exception as e:
            ctx.setdefault("execution_warnings", []).append(
                f"document_qa_generator: chunk处理失败: {e}"
            )
        out.append(c)
    return out + tail


# 注册到MULTIMODAL_REGISTRY
MULTIMODAL_REGISTRY.update({
    "parse_pdf_to_chunks": op_parse_pdf_to_chunks,
    "ocr_noise_clean": op_ocr_noise_clean,
    "document_qa_generator": op_document_qa_generator,
})
MULTIMODAL_OPERATOR_NAMES = frozenset(MULTIMODAL_REGISTRY.keys())
