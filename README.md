# FamilyBotBench

FamilyBotBench is an offline research framework for constructing LLM-driven social bot datasets and evaluating detector generalization across language families, social interaction graphs, and successive generations of behavioral drift.

It combines persona-driven agents, a local Reddit environment, controllable family policies, generation traces, and text/graph detection protocols. Researchers can generate new accounts, vary generation strategies, run multi-generation experiments, and evaluate detectors on the included formal datasets.

**Research paper:** [大模型 Agent 驱动的社交机器人动态演化与泛化测评框架](research%20report.pdf), by muum.

## Research foundation

The framework builds on **[BotSim](https://github.com/QQQQQQBY/BotSim)** and the **[BotSim-24 dataset](https://github.com/QQQQQQBY/BotSim/tree/main/BotSim-24-Dataset)** from Qiao et al., *BotSim: LLM-Powered Malicious Social Botnet Simulation* ([paper](https://arxiv.org/abs/2412.13420)). It extends the original Reddit simulation with language families, human-anchor behavior, shared-environment graph generation, drift control, and generalization evaluation.

## Tasks and datasets

| Task | Included formal data | Main directory |
| --- | --- | --- |
| Language-family generalization | F1–F5; 250 generated accounts and 738 comments | `static_language/` |
| Graph-family generalization | G1–G3; 150 generated accounts, 258 posts and 790 comments; separate supplementary G4 data | `graph_network/` |
| Continuous drift | A/C/E; three generations and eight accounts per family-generation cell, totaling 72 target accounts | `dynamic_drift/` |
| Unsupervised account discovery | Original-human/original-bot multi-view and density discovery | `unsupervised/` |

The original BotSim-24 evaluation data contains 1,907 human accounts and 1,000 bot accounts. The smaller seed tables in `RedditBotSim/Data/` initialize the generation environment. These are distinct inputs. See [the data dictionary](docs/DATASETS.md) for file schemas, graph structure, embedding alignment, and family definitions.

The primary graph has 3,057 users and 54,067 user edges. Its name, `partial_1202actions_stopped_live_20260604`, identifies the stopped-run snapshot used in the reported analysis. The drift data preserves the selected `best_rejected` C-family cells at generations 1 and 3.

## Installation

Use Python 3.12. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the components needed for your experiment:

```bash
python -m pip install -r requirements-bge.txt    # fresh BGE embeddings
python -m pip install -r requirements-graph.txt  # graph model backends
python -m pip install -r requirements-llm.txt    # online LLM generation
```

The base environment includes PyTorch for shared graph utilities. Select a [PyTorch installation](https://pytorch.org/get-started/locally/) appropriate for your hardware. Existing BGE embeddings are included; generating new embeddings requires the `BAAI/bge-small-en-v1.5` model weights. UMAP is optional; density visualization can use PCA.

## Configure data and models

`configs/release.yaml` defines the original evaluation data, static language data, embedding caches, simulator configuration, and output directories. Paths in YAML files resolve relative to the file containing them. Explicit CLI input/output paths resolve relative to the working directory.

`RedditBotSim/config.yml` controls simulation seed tables, background knowledge and the simulated time range. The graph pipeline uses `graph_network/configs/graph_static_dataset_v1.yaml` for its input data, family policies, time windows and model mix.

Set credentials and model names in your shell before online generation:

- OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`.
- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`.
- Optional compatible endpoints: `OPENAI_BASE_URL`, `GEMINI_BASE_URL`.

The supported variables are listed in `.env.example`. Credentials are read from the process environment; files named `.env` or `.env.local` are not automatically loaded. Each provider uses its own key. Default endpoints are the public provider APIs, and `*_BASE_URL` can override them.

Model names stored with the formal data describe the original experiment. For new experiments, choose models available through your own provider account. Provider/model choice is written into the generated manifests and traces.

## Construct a language-family dataset

F1–F4 use the configured OpenAI model; F5 uses the configured Gemini model. With both providers configured:

```bash
python static_language/code/run_static_language_family_benchmark.py \
  --bots-per-family 50 --rounds 3 \
  --openai-model "$OPENAI_MODEL" --gemini-model "$GEMINI_MODEL" \
  --output-dir outputs/language_run
```

Use `--families` to select a subset, such as `F1,F2,F3,F4`. The program writes account profiles, generated content, environment states, traces and detector evaluations. F3 combines human language anchors with detector-feedback revision.

For a small offline example:

```bash
python static_language/code/run_static_language_family_benchmark.py \
  --bots-per-family 2 --rounds 2 --dry-run --skip-bge \
  --f3-revision-mode local --output-dir outputs/language_example
```

Offline template generation illustrates the workflow; its text is not a substitute for live LLM-generated research data.

## Construct a graph-family dataset

Edit the graph YAML to select available models and the desired family/time settings, then run:

```bash
python graph_network/code/run_static_graph_pipeline.py \
  --bots-per-family 50 --time-windows 24 --live-llm --no-dry-run \
  --output-dir outputs/graph_run --result-dir outputs/graph_results
```

Agents from G1–G3 share an environment and act in interleaved time windows. The pipeline exports events, account/content tables, graphs and splits, then runs the selected detection models. For an offline example, replace `--live-llm --no-dry-run` with `--dry-run` and reduce account/window counts. G4 construction has a separate entry point in `graph_network/code/build_g4_slot_conditioned_dataset.py`.

## Run continuous drift experiments

A targets language drift, C targets graph/interaction drift, and E combines several behavior axes. Each generation mutates the selected parent personas, generates interactions, measures drift, applies alignment gates, and passes detector feedback to the next generation.

For example, with your own OpenAI key and `OPENAI_MODEL` configured:

```bash
python dynamic_drift/run_multiround_generalization_benchmark.py \
  --providers openai \
  --families A_language_drift,C_graph_drift,E_mixed_emergent_drift \
  --generations 5 --bots-per-family 8 --rounds 4 --max-resample 3 \
  --output-dir outputs/drift_run
```

Use `--providers gemini` or `--providers openai,gemini` for other provider settings. `OPENAI_MODEL` and `GEMINI_MODEL` select the models used in new drift manifests. `--generations`, `--bots-per-family`, `--rounds`, and `--seed` control experiment size and randomization. See [the experiment guide](docs/REPRODUCIBILITY.md) for entry points and evaluation settings.

Start new experiments in a new output directory. The current generator starts from generation zero; it does not append generation four to an archived three-generation experiment. It rejects an existing experiment directory unless `--clean` explicitly requests replacement. Evaluate an existing lineage with the separate command below.

## Evaluate existing datasets

```bash
# Static language LOFO using included account documents and BGE vectors.
python text_evaluation/code/run_lofo_formal.py --output-dir outputs/text_evaluation

# All nine selected drift cells, without generating new accounts.
python tools/evaluate_saved_drift.py --output-dir outputs/drift_evaluation

# GraphStats on the primary shared-environment graph.
python graph_network/code/run_graphstats_baselines.py \
  --run-dir graph_network/outputs/partial_1202actions_stopped_live_20260604 \
  --result-dir outputs/graph_evaluation

# Label-free density discovery, followed by revealed-label evaluation.
python unsupervised/code/run_density_account_discovery.py \
  --output-dir outputs/density_discovery
```

`text_evaluation/code/` also provides topic controls, human-anchor similarity, family separability and static OOD evaluation. `order_evaluation/` implements curriculum-order comparisons. GraphSAGE and RGCN have separate entry points in `graph_network/code/`. Use `--help` to inspect the arguments for each program.

## Project structure

- `RedditBotSim/`: Reddit environment, persona policies, family and drift modules, generation seed data.
- `BotSim-24-Dataset/`, `BotSim-24-mini-sample/`: original evaluation data and graph tensors.
- `baselines/`: shared detection utilities and original-account embedding caches.
- `static_language/`, `graph_network/`, `dynamic_drift/`, `unsupervised/`: study programs, datasets, traces and results.
- `text_evaluation/`, `order_evaluation/`: extended generalization protocols and their results.
- `configs/`: data and output configuration.
- `outputs/`: new experiment results.

## Contributors

- **muum** — research direction, experimental design, analysis, and manuscript.
- **Codex (OpenAI)** — AI assistance with software implementation, reproducibility tooling, and documentation.

## Citation and license

Please cite BotSim when using its environment or data and acknowledge FamilyBotBench when using these extensions. Citation metadata is in [CITATION.cff](CITATION.cff) and [references.bib](references.bib).

Original FamilyBotBench contributions use the MIT license. Upstream BotSim code, original/derived datasets and copied research artifacts retain their own attribution and permission status. See [license scope](docs/UPSTREAM_AND_LICENSES.md).
