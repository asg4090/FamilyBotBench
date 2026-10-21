# 连续漂移泛化性 benchmark 报告

## 图表阅读说明

- 本实验同时对三个 family 做了连续漂移：`A_language_drift`、`C_graph_drift`、`E_mixed_emergent_drift`，每条 family 都各自从第 1 代漂移到第 3 代。
- 三个 family 都同时参加文本检测和图检测。`A/C/E` 表示漂移设计重点不同，不表示只在某一种检测模态中评估。
- 折线图中的横轴 `target_generation=1/2/3` 表示目标测试样本属于第几代；折线图每个点都是对三个 target family 的 recall 取平均后的结果，不是单独某一个 family 的结果。
- `incremental_generation` 协议：测试第 `g` 代时，训练集中包含所有 family 的更早代 bot（`generation < g`）。
- `heldout_family_same_generation` 协议：测试第 `g` 代某个 target family 时，训练集中只包含同代 `g` 的其他 family，不包含目标 family；它是同代跨 family 泛化，不是 zero-shot。

## 漂移 lineage

| family                 |   generation | run_dir                                                                                                                                                                   | plan_id      | child_family_version   | selection_status   | gate_passed   | target_axes                              |   target_pass_rate |   cumulative_target_pass_rate |   severe_non_target_count |
|:-----------------------|-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------|:-----------------------|:-------------------|:--------------|:-----------------------------------------|-------------------:|------------------------------:|--------------------------:|
| A_language_drift       |            1 | dynamic_drift/continuous_generalization_formal_v1/A_language_drift/generation_001_qualified       | 101201933616 | g001-a01-10120193      | qualified          | True          | language,topic_content                   |                  1 |                        1      |                         0 |
| A_language_drift       |            2 | dynamic_drift/continuous_generalization_formal_v1/A_language_drift/generation_002_qualified       | 7960cd0eacc0 | g002-a01-7960cd0e      | qualified          | True          | language,topic_content                   |                  1 |                        1      |                         0 |
| A_language_drift       |            3 | dynamic_drift/continuous_generalization_formal_v1/A_language_drift/generation_003_qualified       | 4229e2850ada | g003-a01-4229e285      | qualified          | True          | language,topic_content                   |                  1 |                        1      |                         0 |
| C_graph_drift          |            1 | dynamic_drift/continuous_generalization_formal_v1/C_graph_drift/generation_001_best_rejected      | 4224604ebd4a | g001-a03-4224604e      | best_rejected      | False         | graph_structure,memory_revisit           |                  1 |                        1      |                         1 |
| C_graph_drift          |            2 | dynamic_drift/continuous_generalization_formal_v1/C_graph_drift/generation_002_qualified          | e67a6eaea8d5 | g002-a03-e67a6eae      | qualified          | True          | graph_structure,memory_revisit           |                  1 |                        1      |                         0 |
| C_graph_drift          |            3 | dynamic_drift/continuous_generalization_formal_v1/C_graph_drift/generation_003_best_rejected      | c70b38c4fc10 | g003-a01-c70b38c4      | best_rejected      | False         | graph_structure,memory_revisit           |                  1 |                        0.5    |                         0 |
| E_mixed_emergent_drift |            1 | dynamic_drift/continuous_generalization_formal_v1/E_mixed_emergent_drift/generation_001_qualified | d3fd5dae6b1f | g001-a02-d3fd5dae      | qualified          | True          | action_behavior,language,graph_structure |                  1 |                        1      |                         0 |
| E_mixed_emergent_drift |            2 | dynamic_drift/continuous_generalization_formal_v1/E_mixed_emergent_drift/generation_002_qualified | b4df58539ced | g002-a03-b4df5853      | qualified          | True          | language,action_behavior,graph_structure |                  1 |                        0.6667 |                         0 |
| E_mixed_emergent_drift |            3 | dynamic_drift/continuous_generalization_formal_v1/E_mixed_emergent_drift/generation_003_qualified | 873a817dc1fb | g003-a02-873a817d      | qualified          | True          | graph_structure,language,action_behavior |                  1 |                        0.6667 |                         0 |
