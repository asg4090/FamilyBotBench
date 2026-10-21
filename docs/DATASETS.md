# Dataset dictionary

## Original inputs

`BotSim-24-Dataset/Users.csv` contains 2,907 accounts. Following the upstream dataset definition, the first 1,907 are human accounts and the remaining 1,000 are generated bot accounts. This ordering determines labels; do not reorder the file independently of graph tensors or embedding rows. `user_post_comment.json` records posts, first-level comments, and second-level replies by user id.

`BotSim-24-mini-sample/` retains the upstream directory name. Its edge tensors are used with the full account ordering in this research; the folder name should not be interpreted as a substitute small dataset. `edge_index.pt`, `edge_type.pt`, and `edge_weight.pt` describe source/target indices, interaction types, and weights.

`RedditBotSim/Data/` contains the **generation seed**, including 56 posts, 37 first-level comments, four second-level comments, seven users, six subreddits, background-knowledge CSVs, and initial profile material. These small seed tables are distinct from the full account evaluation source above. `graph_network/inputs/original_botsim_snapshot/` retains the graph pipeline's full original-input snapshot.

## Static language families

The archived run is `static_language/outputs/full_5family_50bot_3round_gpt54_20260601/`.

- `generated_profiles_all.csv`: 250 generated accounts, 50 each for F1–F5; persona, family, provider/model, temperature, and behavior traits.
- `generated_comments_all.csv`: 738 comments with generated user references and family metadata.
- `combined_sample_index.csv`: 3,157 account documents, comprising 1,907 original humans, 1,000 original bots, and 250 new accounts. `row` aligns with the embedding row.
- `combined_bge_embeddings.npy`: float32 array of shape `(3157, 384)`.
- `families/`: per-family manifests, profiles, content, environment tables, action/LLM/memory traces, and anchor/rewrite records.
- `*_metrics.csv`, `*_predictions.csv`, and protocol matrices: original-only, single-family transfer, LOFO, and incremental/curriculum evaluations.

F1 uses the original architecture; F2 introduces detector-informed language; F3 uses human anchors and detector-feedback revision; F4 uses trait prompting; F5 changes the language model provider. Family labels and provider fields are audit/grouping metadata, not detector inputs.

`text_evaluation/results/` contains the final extended LOFO, topic controls, anchor similarity, family separability, and OOD results. `order_evaluation/results/` contains the selected curriculum-order comparisons. Formal supplementary fingerprint, representation, and rewrite results remain in the static run.

## Graph datasets

The primary archive is `graph_network/outputs/partial_1202actions_stopped_live_20260604/`:

- `users.csv` and `labels.csv`: 3,057 users, including 150 generated G1/G2/G3 accounts.
- `posts.csv` and `comments.csv`: original context plus 258 generated posts and 790 generated comments. Use `is_generated` to distinguish generated rows.
- `events.jsonl`, `action_trace.jsonl`, `llm_trace.jsonl`, `memory_trace.jsonl`: simulation decisions and provenance.
- `graph/`: exported node/edge tables and a homogeneous user graph tensor with 54,067 edges.
- `splits/`: named train/validation/test node-id memberships and setting metadata.

G1 calibrates population-level behavior; G2 uses human-anchor behavior; G3 adds contextual interaction and revisits. Human replay and original context are preserved with their recorded source labels. G4 slot-conditioned generation is a separate supplementary archive under `partial_plus_g4_slot_conditioned_20260606/`; it has 3,107 users including 50 additional G4 accounts. Do not combine these archives without updating node/embedding mappings.

Matching predictions, metrics, curriculum steps, label-scarcity summaries, behavior-mixing audits, and final G4 verification results are under `graph_network/results/`.

## Continuous drift

The archive is `dynamic_drift/continuous_generalization_formal_v1/`, with A (language), C (graph), and E (mixed) lineages. Each has three target generations of eight accounts, totaling 72 target-generation accounts. Parents and individual attempts are additional provenance, not additional independent target examples.

- `lineage_summary.csv`: nine selected lineage cells, their selection status, alignment gate and drift scores.
- `generated_doc_table.csv` and `generated_doc_embeddings.npy`: 72 aligned documents and `(72,384)` vectors.
- `generation_000_parent/`, `generation_NNN/attempt_*/`, and selected-generation directories: full parent/attempt/selected manifests, environment states and traces.
- `quick_feedback_*`: detector feedback used during generation.
- `text_generalization_*`, `graph_generalization_*`: per-target protocol predictions and metrics.

The selected C-family cells at generations 1 and 3 have `selection_status=best_rejected` and `gate_passed=False`. They remain part of the recorded formal evaluation. Preserve these flags when reporting results.

## Unsupervised discovery

`unsupervised/inputs/` preserves raw-original and static-family provenance. The primary programs select `original_human` and `original_bot` rows from the combined account index. Labels are used for revealed-label evaluation after discovery, not to fit representations or clusters. Stage-one multi-view discovery and stage-two HDBSCAN/dense-component discovery outputs are retained in their named formal output directories.

## Alignment and reuse

Account identifiers, labels, generation status, embedding row order, graph node indices, and split memberships are part of the dataset schema. Keep them aligned when filtering accounts, combining graphs or constructing new train/test sets. Provider and family fields describe generation provenance and evaluation groups; they must not be used as detector features.

Data attribution and permission status are described in [UPSTREAM_AND_LICENSES.md](UPSTREAM_AND_LICENSES.md).
