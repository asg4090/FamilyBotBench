# Experiments and reproducibility

## Three distinct reproduction tasks

1. **Inspect and reaggregate:** use the included predictions, metrics, manifests, embeddings and graph tensors to inspect the reported generalization matrices and recovery curves.
2. **Retrain detectors on saved data:** run the formal text, graph, drift, or discovery evaluation entry points. Use new output directories. BGE linear classifiers can reuse the included vectors. Hardware and library differences may change retrained models, particularly GNNs.
3. **Generate new accounts:** run the generators with configured provider credentials. Live LLM generation is stochastic and depends on model availability. A new live run is not guaranteed to recreate the archived responses.

The formal graph records a stopped-run snapshot. The drift evaluation includes two selected `best_rejected` C-family cells. These are observed limitations of the archived experiments, not data-cleaning errors. RGCN's observed/transductive setting and GraphSAGE's inductive setting must remain distinct when comparing results.

## Entry points

| Task | Entry point |
| --- | --- |
| Static generation and original protocols | `static_language/code/run_static_language_family_benchmark.py` |
| Static protocol figures / curriculum curves | `static_language/code/plot_static_language_results.py`, `build_curriculum_prepost.py` |
| Formal LOFO | `text_evaluation/code/run_lofo_formal.py` |
| Topic controls / anchors / family separability / OOD | Corresponding `run_*.py` files in `text_evaluation/code/` |
| Curriculum order comparison | `order_evaluation/run_meaningful_order_experiment.py` |
| Graph generation, export, splits and baselines | `graph_network/code/run_static_graph_pipeline.py` |
| Supplementary G4 generation | `graph_network/code/build_g4_slot_conditioned_dataset.py` |
| GraphStats / GraphSAGE / RGCN | Separate `run_*.py` files in `graph_network/code/` |
| Drift generation and evaluation | `dynamic_drift/run_multiround_generalization_benchmark.py` |
| Saved drift evaluation only | `tools/evaluate_saved_drift.py` |
| Account discovery / density discovery | Two primary `run_*.py` files in `unsupervised/code/` |

For GNN retraining, specify `--run-dir`, `--result-dir`, `--epochs`, and `--seed` explicitly. Formal model settings and seeds are recorded in the archived configurations and result summaries. Use the same epoch count, seed, split and setting when comparing results.

## Configuration

The release YAML separates `original_root` (evaluation data), `embedding_root` (original-account vector caches), `static_run` (formal generated language data), and `runtime_config` (simulation seed/configuration). Paths are relative to the YAML file. Copy the YAML to another location only if its relative paths are adjusted accordingly. CLI input/output paths are relative to the invoking working directory.

Graph generation uses its own YAML. Its `paths` fields resolve relative to that file; explicit configured original data is preferred over the bundled fallback. New pipeline outputs default to the root `outputs/` folder, leaving archived formal runs separate.

## New continuous drift experiments

Set `OPENAI_API_KEY` / `GEMINI_API_KEY` for each selected provider and use `OPENAI_MODEL` / `GEMINI_MODEL` to choose the models for new personas. The `--providers` option accepts `openai`, `gemini`, or both. The default family set A/C/E controls language, graph/interaction, and combined drift. Additional drift-family definitions are in `RedditBotSim/persona_social/drift_controller.py`.

At each generation, the selected parent personas are mutated according to the family plan and detector feedback. Candidate attempts are audited against both the immediate parent and generation-zero reference. The first passing candidate is selected; if every attempt fails the gate, the best rejected candidate is retained with an explicit status. Manifests, action/memory/LLM traces, local/cumulative drift scores and detector predictions make this lineage inspectable.

`--generations` controls the number of target generations in a new run. `--max-resample` limits generation attempts per cell. `--skip-quick-feedback` removes detector feedback from the generation loop, so results from this option should be reported separately from feedback-enabled experiments. `--skip-bge` omits BGE evaluation; fresh embedding generation otherwise needs the model weights.

## Existing lineages and output directories

The current generator starts from generation zero and has no append/resume interface for an existing archived lineage. Use a new output directory for a new experiment. To replace a previous experiment deliberately, `--clean` removes that output directory. To evaluate the existing A/C/E three-generation data without generating new content, use `tools/evaluate_saved_drift.py` with a separate output directory.

Live LLM generation is stochastic, and historical model names may no longer be available to a given provider account. New runs should record the actual model, provider and configuration used. All interaction occurs in the local Reddit simulation.
