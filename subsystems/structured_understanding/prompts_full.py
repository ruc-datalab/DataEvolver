# 自旧版 DataEvolver `prompts.py` 迁入；示例 JSON 内补全逗号，避免误导模型。
# 单次 UNIFIED_PROFILE 在旧版「三步理解」之上强化了 delta 可证伪性；下方块保留旧版多轮调试的 schema 深度要求。

STRUCTURED_UNDERSTANDING_SYSTEM_PROMPT = """You are an expert in data analysis and understanding. Your task is to analyze raw data and seed data to produce a comprehensive structured understanding.

Key requirements:
1. Output **ONLY valid JSON**, no additional text or explanations
2. Be dataset-agnostic: focus on general patterns, not specific dataset names
3. Provide detailed, actionable insights
4. Ensure all required fields are present in the output
5. **Pay special attention to nested structures**: analyze all fields including nested objects and arrays, identify their types, meanings, and relationships

Your analysis should help understand:
- The format and structure of raw data vs seed data (including nested structures)
- Field meanings and purposes (especially for nested fields)
- Quality differences and improvement directions
- Processing targets and transformation needs
- Domain characteristics and data types
- Schema differences and stability (including nested schema)
- Content patterns and quality standards
- Optimization opportunities for nested fields"""

STRUCTURED_UNDERSTANDING_USER_PROMPT_TEMPLATE = """Analyze the following raw data and seed data to produce a structured understanding. **Pay special attention to nested structures** - analyze all fields including nested objects and arrays.

## Pipeline Configuration
{pipeline_config}

## User Requirements / Task Description (Optional)
{user_requirements}

**Note**: If user requirements are provided, you should incorporate them into your analysis. The user requirements may specify:
- Specific processing goals or constraints
- Desired output format or style
- Quality expectations
- Domain-specific requirements
- Any other task-specific instructions

When user requirements are provided, prioritize aligning your analysis with these requirements while still considering the differences between raw and seed data.

## Raw Data Preview
{raw_data_preview}

## Seed Data Preview
{seed_data_preview}

## Analysis Requirements

**IMPORTANT**: 
- If user requirements are provided, incorporate them into your analysis and ensure the processing targets align with user requirements
- Analyze ALL fields including nested structures (objects, arrays, nested arrays)
- Identify the type, meaning, and purpose of each field (including nested fields)
- Note any nested objects or arrays and their structure
- Identify optimization opportunities for nested fields
- When user requirements are provided, prioritize understanding how the transformation from raw to seed data should align with these requirements

Please provide a comprehensive analysis in JSON format with the following structure:

{{
  "language": "primary language of the data (e.g., 'English', 'Chinese', 'Spanish', etc.). Analyze the text content in both raw and seed data to determine the language.",
  "file_format_analysis": {{
    "raw_data_format": "description of raw data format (JSONL/JSON/TXT/CSV/etc)",
    "seed_data_format": "description of seed data format",
    "format_differences": "key differences between formats",
    "nested_structure_notes": "description of any nested structures (objects/arrays) in the data"
  }},
  "seed_vs_raw_quality": {{
    "quality_improvements": ["list of quality improvements in seed data"],
    "content_differences": "description of content differences",
    "style_differences": "description of style differences (including any special tags, markers, or formatting structures in seed data)",
    "format_improvements": "description of format improvements in seed data (e.g., special tags, structured sections, etc.)",
    "detail_level_comparison": "comparison of detail levels",
    "nested_field_improvements": "description of improvements in nested fields/structures"
  }},
  "processing_targets": [
    "list of processing targets/goals (including nested field processing)"
  ],
  "domain_characteristics": "description of domain characteristics and task types",
  "data_types": [
    "list of data types present (including nested types like 'list of objects', 'nested dict', etc.)"
  ],
  "transformation_direction": "description of transformation direction from raw to seed (including nested structure transformations)",
  "quality_standards": "description of quality standards expected in seed data (including nested field quality)"
}}

Please analyze the data carefully, including all nested structures, and provide a complete JSON response."""

SCHEMA_ANALYSIS_PROMPT_TEMPLATE = """Analyze the schema differences between raw data and seed data. **Pay special attention to nested structures** - identify all fields including nested objects and arrays.

## Raw Data Schema
{raw_schema}

## Seed Data Schema
{seed_schema}

## Raw Data Sample
{raw_sample}

## Seed Data Sample
{seed_sample}

**IMPORTANT**: 
- Analyze ALL fields including nested structures
- For each field, identify its type (string, int, bool, list, dict, nested list, nested dict, etc.)
- **Pay special attention to format differences**: If seed data contains special tags, markers, or structured formats that are not in raw data, identify these as critical format requirements
- For nested structures, describe the nested schema in detail
- Identify the meaning and purpose of each field, especially nested ones
- **For format differences**: In optimization_opportunity, explicitly mention if special tags, markers, or structured formats need to be added to match seed data format
- In seed_field_details, include entries for nested structures with their complete nested schema

Please provide a schema analysis in JSON format:

{{
  "schema_stable": true,
  "raw_fields": ["list of ALL fields in raw data (including nested paths like 'answers[].text')"],
  "seed_fields": ["list of ALL fields in seed data (including nested paths)"],
  "raw_field_details": {{}},
  "seed_field_details": {{}},
  "new_fields": [],
  "missing_fields": [],
  "nested_structure_changes": "description of changes in nested structures between raw and seed data",
  "schema_constraint": "description of schema constraints (including nested structure constraints)",
  "field_usage_guidance": {{}}
}}"""

DATASET_LEVEL_DELTA_PROMPT_TEMPLATE = """Analyze the dataset-level differences and optimization directions at a high level.

## Raw Data Characteristics
{raw_characteristics}

## Seed Data Characteristics
{seed_characteristics}

## Seed Data Examples
{seed_examples}

Please provide a high-level dataset analysis in JSON format (keep it concise and general):

{{
  "global_optimization_direction": "high-level description of overall optimization direction",
  "key_improvements": [
    "list of key improvements in seed data compared to raw data (as many as relevant, no fixed number)"
  ],
  "transformation_strategies": [
    "list of high-level transformation strategies needed (as many as relevant, no fixed number)"
  ],
  "quality_focus": [
    "list of key quality focus areas (as many as relevant, no fixed number)"
  ],
  "summary": "A summary of the structured understanding results and the main tasks/strategies needed for the next steps in the data processing pipeline. This should guide the system on what transformations and operations are required to convert raw data to seed data quality."
}}"""

# ---------------------------------------------------------------------------
# 单次 LLM：由浅入深输出完整 profile（减少调用次数与总 token）
# ---------------------------------------------------------------------------

UNIFIED_PROFILE_SYSTEM_PROMPT = """You are a senior data/ML engineer for the DataEvolver system. Your job is **structured, evidence-based** comparison of **raw** training inputs vs **seed** target examples—not generic “quality” commentary.

Hard rules:
1. Output **ONLY one JSON object** with exactly three top-level keys: `basic_information`, `schema_analysis`, `dataset_level_delta`.
2. No markdown, no code fences, no text outside JSON.
3. **Ground every non-trivial claim** in observable differences: field paths (from helpers or samples), value shapes, lengths, tags, nesting, or missing vs present keys. If you cannot point to evidence, do not invent it—say what is unknown in `schema_analysis` or `basic_information` instead.
4. **Forbidden** in `dataset_level_delta`: vague phrases alone such as “improve quality”, “better structure”, “more detail”, “enhance clarity”, “consistent formatting” **unless** each is immediately tied to a **named field or pattern** (e.g. “seed adds field `x` that raw lacks”).
5. `dataset_level_delta` must read like a **delta spec** for downstream pipeline design: what exactly changes from raw → seed, why it matters, and what would break if ignored.
6. Use the dominant natural language of the **content** for narrative strings inside JSON (e.g. Chinese if the samples are Chinese); keep JSON keys in English as given in the user template.
7. Cover nested JSON explicitly where the samples differ."""

UNIFIED_PROFILE_USER_TEMPLATE = """Analyze the following in **one pass**, from concrete samples → schema → dataset-level delta.

## Pipeline configuration
{pipeline_config}

## User requirements / task description
{user_requirements}

## Previous experience (optional, from last round)
{experience_section}

## Raw samples (up to 2 files, first record each)
{raw_data_preview}

## Seed samples (first records)
{seed_data_preview}

## Precomputed raw field paths (helper)
{raw_schema}

## Precomputed seed field paths (helper)
{seed_schema}

## Full raw record (one object, for nested inspection)
{raw_sample}

## Full seed record (one object)
{seed_sample}

## Seed field short examples (first keys only)
{seed_examples}

## Critical analytical task
- **Primary goal**: characterize the **measurable / inspectable gap** between raw and seed (schema, values, format envelopes, reasoning structure, metadata, tags, list vs scalar, depth of nesting).
- **`basic_information` + `schema_analysis`**: be specific—list **new_fields** / **missing_fields** with real paths; describe **nested_structure_changes** with examples from `raw_sample` vs `seed_sample`.
- **`dataset_level_delta`** (most important): must **not** restate generic goals. It must synthesize **concrete deltas** that a pipeline could implement. Every bullet in the list fields below should be **falsifiable** (a human could check it against the two full records).

## Legacy schema depth requirements (from production `STRUCTURED_UNDERSTANDING` / `SCHEMA_ANALYSIS` prompts — keep all that apply)
- Analyze **ALL** fields including nested objects/arrays; use paths like `answers[].text` or `qa_metadata.wh_type` where relevant.
- For each important field in `raw_field_details` / `seed_field_details`, include **type**, **meaning**, and when nested a **nested_structure** sentence (e.g. "list of objects, each with keys: text, answer_start").
- In **seed_field_details**, **optimization_opportunity** must call out format gaps: special tags/markers/sections seed uses that raw does not (e.g. reasoning tags, `<answer>...</answer>`, markdown sections). This drives downstream orchestration format rules.
- **field_usage_guidance** should give per-key hints (instruction/input/output and nested paths) when samples show constraints.
- **style_differences** / **format_improvements** in `seed_vs_raw_quality` should mention any systematic tag or wrapper pattern visible in `seed_sample`.

Return **one** JSON object of this exact shape:

{{
  "basic_information": {{
    "language": "natural language name of dominant data language e.g. Chinese or English",
    "file_format_analysis": {{
      "raw_data_format": "string",
      "seed_data_format": "string",
      "format_differences": "string — cite concrete aspects (e.g. JSONL vs JSON, extra wrapper keys, record shape)",
      "nested_structure_notes": "string"
    }},
    "seed_vs_raw_quality": {{
      "quality_improvements": ["each item: what is better in seed with a concrete anchor (field or pattern)"],
      "content_differences": "string — compare substance, not adjectives only",
      "style_differences": "string — tags, markers, sections, delimiters, tone, language mix",
      "format_improvements": "string — structural formatting seed enforces that raw does not",
      "detail_level_comparison": "string — e.g. typical length, list cardinality, depth",
      "nested_field_improvements": "string"
    }},
    "processing_targets": ["actionable targets tied to named fields or transformations"],
    "domain_characteristics": "string",
    "data_types": [],
    "transformation_direction": "string — one paragraph: raw → seed as a transformation story",
    "quality_standards": "string — what seed encodes as 'done' (observable criteria)"
  }},
  "schema_analysis": {{
    "schema_stable": true,
    "raw_fields": [],
    "seed_fields": [],
    "raw_field_details": {{}},
    "seed_field_details": {{}},
    "new_fields": [],
    "missing_fields": [],
    "nested_structure_changes": "string — explicit comparison",
    "schema_constraint": "string",
    "field_usage_guidance": {{}}
  }},
  "dataset_level_delta": {{
    "global_optimization_direction": "string — ONE sentence naming the dominant shift (e.g. 'unstructured answer → tagged solution + auxiliary metadata'), not 'improve quality'",
    "key_improvements": ["each bullet MUST mention at least one field path OR a concrete pattern contrast visible in samples"],
    "transformation_strategies": ["pipeline-level strategies; each must map to a real delta (field add/remove/reshape/tag/format), not generic 'refine'"],
    "quality_focus": ["observable bar raised by seed: cite what to check in outputs"],
    "summary": "string — 3–6 sentences tying schema + value patterns together; no empty platitudes",
    "concrete_field_and_schema_deltas": [
      "each: a single delta, e.g. 'seed adds path a.b that raw lacks' or 'field x changes from string to list<object> with keys …'"
    ],
    "value_pattern_contrasts": [
      "each: contrast raw_sample vs seed_sample (length, delimiters, tags, missing sections, numeric vs text, etc.)"
    ],
    "seed_higher_bar_signals": [
      "observable signals in seed that impose stricter expectations than raw (structure, verification, metadata, grounding, etc.)"
    ],
    "risks_if_ignored": [
      "what downstream orchestration or operators would get wrong if these deltas were not captured"
    ]
  }}
}}

Use the helpers above; expand with your own judgment. All three top-level objects must be non-null and complete. Prefer **specificity over breadth**."""


MULTIMODAL_SYSTEM_HINT = """
MULTIMODAL DATA DETECTION:
If records contain image_path, image, or similar image reference fields, this is a MULTIMODAL dataset.
For multimodal data, your analysis MUST include:
1. image_processing_needs: whether images need privacy protection (face blurring), quality filtering, deduplication, or resizing
2. visual_qa_quality: whether the question/answer pairs are grounded in image content
3. recommended_image_operators: suggest from [image_face_blur, image_quality_filter, image_deduplicator, image_resize_normalizer, vlm_generate_qa, vlm_image_caption]

KEY INSIGHT for multimodal pipelines:
- If seed images have faces blurred but raw images do not → use image_face_blur
- If seed QA is detailed and image-grounded but raw QA is vague → use vlm_generate_qa
- Always put image processing operators (image_face_blur, image_quality_filter) BEFORE QA generation operators (vlm_generate_qa)
- image_face_blur and image_quality_filter do NOT require LLM
- vlm_generate_qa and vlm_image_caption DO require LLM (vision model)
"""


DOCUMENT_SYSTEM_HINT = """
DOCUMENT DATA DETECTION:
If records contain pdf_path, doc_path, or chunk_text fields, this is a DOCUMENT dataset.
For document data, your analysis MUST include:
1. document_parsing_needs: whether PDFs need to be parsed into chunks first
2. text_quality: whether chunk text has OCR noise, formatting issues
3. qa_generation_needs: whether QA pairs need to be generated from chunks
4. recommended_document_operators: suggest from [parse_pdf_to_chunks, ocr_noise_clean, document_qa_generator]

KEY INSIGHT for document pipelines:
- If raw data has pdf_path field → use parse_pdf_to_chunks FIRST before any text processing
- If chunk_text has garbled characters or very short chunks → use ocr_noise_clean
- If question/answer fields are empty or low quality → use document_qa_generator
- Always: parse_pdf_to_chunks → ocr_noise_clean → document_qa_generator → write_data
- parse_pdf_to_chunks and ocr_noise_clean do NOT require LLM
- document_qa_generator DOES require LLM
"""
