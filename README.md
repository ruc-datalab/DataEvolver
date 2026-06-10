<p align="center" width="100%">
<img src="assets/DataEvolver.png" alt="DataEvolver" style="width: 65%; min-width: 300px; display: block; margin: auto;">
</p>

# DataEvolver: Automatic Data Preparation for LLMs via Multi-Level Self-Evolving

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/dataevolver?color=306998)](https://pypi.org/project/dataevolver/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader&logoColor=white)](assets/DataEvolver.pdf)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[Paper](assets/DataEvolver.pdf)** · **[Demo](#-demo)** · **[Quick Start](#-quick-start)** · **[Usage](#usage)** · **[Results](#-results)**

**DataEvolver** is a seed-driven, multi-level self-evolving system for **LLM training data preparation**. Starting from noisy raw data and only a handful of seed examples, it automatically understands target data characteristics, builds and repairs executable operator DAGs, trial-runs on samples, and iteratively refines the pipeline until outputs align with seeds — then runs full preparation.

DataEvolver supports:

- 🌱 **Seed-guided understanding** — distill schema, format, style, and quality constraints from seed data (not just task descriptions)
- 🔧 **Operator-level self-evolving** — orchestrate DAGs, detect structural gaps, and synthesize bridging / task-specific operators when needed
- 🔄 **Pipeline-level self-evolving** — trial runs, Pilot LLM judging, experience reflow, and next-round understanding + orchestration updates
- 🖥 **Three aligned interfaces** — Web UI (evolution canvas), CLI, and HTTP API share the same workflow semantics
- 📦 **Fully open & deployable** — git clone for full stack, or `pip install dataevolver` for CLI/API; all stage artifacts are inspectable

<p align="center" width="100%">
<img src="assets/ill.png" alt="DataEvolver framework" style="width: 75%; min-width: 300px; display: block; margin: auto;">
</p>

## 🖥 Demo

Upload raw data and seed examples — DataEvolver automatically performs structured understanding, DAG orchestration, instantiation, trial evaluation, and experience reflow across rounds.

<p align="center">
  <img src="assets/demo_preview.gif" alt="DataEvolver evolution canvas demo" width="92%"/>
</p>

<p align="center" width="100%">
▶ <b><a href="https://github.com/ruc-datalab/DataEvolver/raw/main/assets/demo.mp4">Watch full demo</a></b> (2m50s)
· <a href="https://github.com/ruc-datalab/DataEvolver/releases/download/demo-2026-04-18/DataEvolver_Demo_small.mov">Download HD .mov</a>
</p>

> [!TIP]
>
> Clone this repository and configure an OpenAI-compatible LLM API to run DataEvolver locally as your data-prep agent — no closed-source workflow engine required. The evolution canvas lets you inspect every stage artifact, DAG retry, and trial score.

## ✨ Highlights

- **Executable + seed-aligned** — jointly optimizes *can the pipeline run end-to-end?* and *does output match seed supervision style?*
- **DAG-native orchestration** — three-stage LLM planning with automatic structural check + LLM assessment after each orchestration
- **Evolvable operator registry** — auto-evolve operators on demand; manually add custom operators via CLI/API and re-orchestrate
- **Observable workflow** — per-step artifacts, orchestration retry tabs, token ledger, and round/iteration archives
- **Automation with control** — `advance-all` for hands-off runs, or step-by-step `understand` / `orchestrate` / `trial` with `rerun` and `--force`
- **Cross-platform** — Linux / macOS / Windows setup scripts; PyPI package for portable CLI workspaces

## 🚀 Quick Start

> Full deployment guide: **[docs/INSTALL.md](docs/INSTALL.md)**

### Requirements

| Component | Version |
|:--|:--|
| Python | **3.10+** |
| Node.js | **18+** LTS (Web UI only) |
| LLM API | OpenAI-compatible endpoint + key |

### 1. Install

**From source (Web UI + CLI + API)**

```bash
git clone https://github.com/ruc-datalab/DataEvolver.git
cd DataEvolver
python scripts/setup_env.py    # or: bash setup_env.sh / setup_env.ps1
```

**From PyPI (CLI + API only)**

```bash
pip install dataevolver
mkdir my_project && cd my_project
dataevolver init
dataevolver --help
```

### 2. Configure LLM

```text
config/api_config.json   # provider, base URL, model
config/api_keys.json     # API key (gitignored — do not commit)
```

### 3. Start services

| Service | Command | URL |
|:--|:--|:--|
| Backend | `python scripts/dev.py backend` | http://127.0.0.1:8000 |
| Frontend | `python scripts/dev.py frontend` | http://127.0.0.1:5173 |
| API docs | — | http://127.0.0.1:8000/docs |

### 4. Run your first pipeline

```bash
dataevolver session-start my_pipeline \
  --raw tmp/samples/finance_raw.jsonl \
  --seed tmp/samples/finance_seed.jsonl \
  --description tmp/samples/finance_description.txt

dataevolver workflow advance-all my_pipeline --max-steps 32
dataevolver state my_pipeline
```

Open **http://127.0.0.1:5173** for the evolution canvas, or continue in the terminal with `dataevolver advance my_pipeline`.

## Usage

DataEvolver exposes the **same workflow** through three interfaces.

| Interface | Best for | Entry |
|:--|:--|:--|
| **Web UI** | Visual DAG, trial scores, evolution history | http://127.0.0.1:5173 |
| **CLI** | Reproducible runs, scripting, CI | `dataevolver --help` · `dataevolver wf --help` |
| **HTTP API** | Integration & automation | http://127.0.0.1:8000/docs |

### Web UI

1. Create or select a pipeline session
2. Upload raw data, seed data, and optional task description
3. Advance step-by-step or run continuously
4. Inspect DAG tabs, instantiation code, trial scores, and experience
5. Trigger **full run** only after quality gates pass

### CLI

```bash
dataevolver state my_pipeline
dataevolver advance my_pipeline
dataevolver workflow advance-all my_pipeline --max-steps 32
dataevolver lang en    # CLI output: zh / en
```

| Stage | Command |
|:--|:--|
| Understanding | `dataevolver understand my_pipeline` |
| Orchestration | `dataevolver orchestrate my_pipeline` |
| Operator evolution | `dataevolver evolve-operators my_pipeline` |
| Instantiation | `dataevolver instantiate my_pipeline` |
| Trial run | `dataevolver trial my_pipeline` |
| Quality check | `dataevolver quality-check my_pipeline` |
| Experience | `dataevolver experience my_pipeline` |
| Full run | `dataevolver run my_pipeline` |

```bash
dataevolver rerun my_pipeline orchestration
dataevolver tokens my_pipeline
dataevolver op add my_op -p my_pipeline -d "Clean records" -c structure
```

### HTTP API

| Endpoint | Purpose |
|:--|:--|
| `POST /api/sessions/start` | Create session & register manifest |
| `GET /api/workflow/{pipeline_id}/state` | Read workflow state |
| `POST /api/workflow/{pipeline_id}/advance` | Advance one step |
| `POST /api/pipeline/{pipeline_id}/run-full` | Full dataset execution |
| `GET /api/operators/?pipeline_id=` | List merged operator pool |

<details>
<summary><b>Operator pool, project layout, configuration & FAQ</b></summary>

**Operator pool** — add custom operators, then re-orchestrate:

```bash
dataevolver op list -p my_pipeline
dataevolver op add -p my_pipeline --from-file examples/operator_template.json
dataevolver workflow orchestrate my_pipeline
```

**Project layout**

```text
DataEvolver/
├── subsystems/     # understanding, orchestration, workflow, …
├── web/            # FastAPI
├── frontend/       # React evolution canvas
├── cli/            # Typer CLI (dataevolver)
├── config/         # api_config, api_keys, operator registry
└── data/           # runtime artifacts (gitignored)
```

**Tips:** use `rerun` / `--force` to regenerate stages · `dataevolver tokens` for LLM usage · artifacts under `data/artifact_history/`

**FAQ**

- *Instantiation finishes instantly?* — artifacts may be reused (`skipped`); built-in operators use templates.
- *Experience finishes quickly?* — rule-based aggregation for deterministic reflow, not LLM rewriting.
- *Multiple orchestration tabs?* — each tab is a distinct attempt (failed validation → repaired DAG).

</details>

## 📊 Results

Empirical evaluation from our paper — DataEvolver improves downstream training data quality and model performance across diverse task types.

<table align="center">
<tr>
<td align="center"><b>~12%</b><br/><sub>avg relative gain vs.<br/>weaker prep settings</sub></td>
<td align="center"><b>7</b> benchmarks<br/><sub>instruction · MC-QA · math · SQL</sub></td>
<td align="center"><b>~40%</b><br/><sub>lower amortized token cost<br/>on average</sub></td>
</tr>
</table>

<p align="center">
  <img src="assets/main_exp.png" width="92%" alt="Main experiment results"/>
</p>
<p align="center"><i>Downstream performance across 7 benchmarks from 4 task categories.</i></p>

<p align="center">
  <img src="assets/compare.png" width="88%" alt="Comparison against baselines"/>
</p>
<p align="center"><i>vs. vanilla SFT on raw data and strong data-prep baselines — fewer, better-prepared samples can match larger weakly-prepared sets.</i></p>

<details>
<summary><b>Ablation, case study & how it works</b></summary>

<p align="center"><img src="assets/ablation.png" width="80%" alt="Ablation study"/></p>

- Without **operator-level** evolution → less executable, less coherent pipelines
- Without **pipeline-level** evolution → less seed-aligned outputs

<p align="center"><img src="assets/case.png" width="92%" alt="Case study"/></p>

**Workflow loop**

```text
understanding → orchestration → operator_evolution → instantiation
             → trial_run → quality_check → experience → (refine or full run)
```

| Layer | What happens |
|:--|:--|
| **Understanding** | Learn target profile from seeds + raw samples |
| **Operator evolution** | Repair DAG; synthesize missing operators |
| **Pipeline evolution** | Trial feedback → experience → next-round refinement |

Compared to predefined recipes (stable but rigid) or one-shot pipeline synthesis (flexible but fragile), DataEvolver treats data preparation as an **iterative, self-improving closed loop**.

</details>

## 🤝 Community

| Channel | Link |
|:--|:--|
| Issues & features | [GitHub Issues](https://github.com/ruc-datalab/DataEvolver/issues) |
| Questions | [GitHub Discussions](https://github.com/ruc-datalab/DataEvolver/discussions) |

Contributions welcome — operators, dataset adapters, UI polish, docs, and benchmark scripts.

## 📖 Citation

If you use DataEvolver in research, please cite our paper:

```bibtex
@misc{deng2026dataevolverautomaticdatapreparation,
      title={DataEvolver: Automatic Data Preparation for Large Language Models through Multi-Level Self-Evolving}, 
      author={Chao Deng and Shaolei Zhang and Ju Fan and Xiaoyong Du},
      year={2026},
      eprint={2606.07001},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2606.07001}, 
}
```

📄 [assets/DataEvolver.pdf](assets/DataEvolver.pdf)

<p align="center">
<sub>If DataEvolver helps your data-prep workflow, give us a ⭐</sub>
</p>
