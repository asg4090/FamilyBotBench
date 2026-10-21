# graph_network检测实验报告

## 数据集概况

- run_name: `full_3family_150bot_24window_live`
- schema_version: `static-graph-network-dataset-v1`
- environment: `mixed_dynamic_reddit_offline`
- source_dataset: `graph_network/inputs/original_botsim_snapshot/BotSim-24-Dataset`
- 节点数: `3057`
- user-user 边数: `53775`
- GraphStats 特征数: `39`

三类生成 family 设计用于区分不同社交拟人机制：G1 偏总体行为分布校准，G2 偏 human anchor 轨迹拟合，G3 偏上下文浏览、通知、回访等交互反馈。报告中的 `family` 仅用于分组解释和泛化矩阵，不作为训练特征。

## Split 设计

| split_name                    | setting                     |   train |   val |   test |   unlabeled_visible | holdout_families                                                      |
|:------------------------------|:----------------------------|--------:|------:|-------:|--------------------:|:----------------------------------------------------------------------|
| incremental_curriculum_order1 | incremental_curriculum      |    2326 |   581 |     50 |                 100 | G1_population_calibrated;G2_anchor_behavior;G3_contextual_interactive |
| incremental_curriculum_order2 | incremental_curriculum      |    2326 |   581 |     50 |                 100 | G2_anchor_behavior;G1_population_calibrated;G3_contextual_interactive |
| incremental_curriculum_order3 | incremental_curriculum      |    2326 |   581 |     50 |                 100 | G3_contextual_interactive;G1_population_calibrated;G2_anchor_behavior |
| label_scarcity_001pct         | transductive_label_scarcity |    2328 |   582 |    147 |                   0 |                                                                       |
| label_scarcity_005pct         | transductive_label_scarcity |    2331 |   582 |    144 |                   0 |                                                                       |
| label_scarcity_010pct         | transductive_label_scarcity |    2338 |   584 |    135 |                   0 |                                                                       |
| label_scarcity_025pct         | transductive_label_scarcity |    2355 |   588 |    114 |                   0 |                                                                       |
| label_scarcity_050pct         | transductive_label_scarcity |    2386 |   596 |     75 |                   0 |                                                                       |
| label_scarcity_100pct         | transductive_label_scarcity |    2446 |   611 |      0 |                   0 |                                                                       |
| lofo_g1                       | inductive_family_holdout    |    2406 |   601 |     50 |                   0 | G1_population_calibrated                                              |
| lofo_g2                       | inductive_family_holdout    |    2406 |   601 |     50 |                   0 | G2_anchor_behavior                                                    |
| lofo_g3                       | inductive_family_holdout    |    2406 |   601 |     50 |                   0 | G3_contextual_interactive                                             |
| original_only_zero_shot       | inductive_family_holdout    |    2326 |   581 |    150 |                   0 | G1_population_calibrated;G2_anchor_behavior;G3_contextual_interactive |
| time_forward                  | temporal_transductive       |     187 |    47 |      0 |                2823 |                                                                       |

协议要求关注 zero-shot、LOFO、label scarcity、time-forward 与 incremental curriculum。当前脚本按 `splits/*.json` 逐个执行；若 smoke 数据只提供少量 split，本报告只展示已运行 split。

## GraphStats 结果

### GraphStats 汇总

| split_name                    | setting                     | model          | backend   | eval_group   | family   |   size |   bot_count |   bot_recall |   bot_precision |   bot_f1 |   macro_f1 |   bot_mean_probability | status   |
|:------------------------------|:----------------------------|:---------------|:----------|:-------------|:---------|-------:|------------:|-------------:|----------------:|---------:|-----------:|-----------------------:|:---------|
| incremental_curriculum_order1 | incremental_curriculum      | GraphStats_LR  | sklearn   | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0.0002 | ok       |
| incremental_curriculum_order1 | incremental_curriculum      | GraphStats_RF  | sklearn   | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order1 | incremental_curriculum      | GraphStats_EBM | interpret | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order2 | incremental_curriculum      | GraphStats_LR  | sklearn   | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0.0001 | ok       |
| incremental_curriculum_order2 | incremental_curriculum      | GraphStats_RF  | sklearn   | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order2 | incremental_curriculum      | GraphStats_EBM | interpret | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order3 | incremental_curriculum      | GraphStats_LR  | sklearn   | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0.0001 | ok       |
| incremental_curriculum_order3 | incremental_curriculum      | GraphStats_RF  | sklearn   | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0.0001 | ok       |
| incremental_curriculum_order3 | incremental_curriculum      | GraphStats_EBM | interpret | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| label_scarcity_001pct         | transductive_label_scarcity | GraphStats_LR  | sklearn   | test         | ALL      |    147 |         147 |       0.5238 |               1 |   0.6875 |     0.3438 |                 0.5121 | ok       |
| label_scarcity_001pct         | transductive_label_scarcity | GraphStats_RF  | sklearn   | test         | ALL      |    147 |         147 |       0.0612 |               1 |   0.1154 |     0.0577 |                 0.2667 | ok       |
| label_scarcity_001pct         | transductive_label_scarcity | GraphStats_EBM | interpret | test         | ALL      |    147 |         147 |       0.3469 |               1 |   0.5152 |     0.2576 |                 0.3452 | ok       |
| label_scarcity_005pct         | transductive_label_scarcity | GraphStats_LR  | sklearn   | test         | ALL      |    144 |         144 |       0.6806 |               1 |   0.8099 |     0.405  |                 0.6582 | ok       |
| label_scarcity_005pct         | transductive_label_scarcity | GraphStats_RF  | sklearn   | test         | ALL      |    144 |         144 |       0.5208 |               1 |   0.6849 |     0.3425 |                 0.4917 | ok       |
| label_scarcity_005pct         | transductive_label_scarcity | GraphStats_EBM | interpret | test         | ALL      |    144 |         144 |       0.8611 |               1 |   0.9254 |     0.4627 |                 0.8424 | ok       |
| label_scarcity_010pct         | transductive_label_scarcity | GraphStats_LR  | sklearn   | test         | ALL      |    135 |         135 |       0.8074 |               1 |   0.8934 |     0.4467 |                 0.8181 | ok       |
| label_scarcity_010pct         | transductive_label_scarcity | GraphStats_RF  | sklearn   | test         | ALL      |    135 |         135 |       0.8963 |               1 |   0.9453 |     0.4727 |                 0.7544 | ok       |
| label_scarcity_010pct         | transductive_label_scarcity | GraphStats_EBM | interpret | test         | ALL      |    135 |         135 |       0.9407 |               1 |   0.9695 |     0.4847 |                 0.9407 | ok       |
| label_scarcity_025pct         | transductive_label_scarcity | GraphStats_LR  | sklearn   | test         | ALL      |    114 |         114 |       0.8333 |               1 |   0.9091 |     0.4545 |                 0.8251 | ok       |
| label_scarcity_025pct         | transductive_label_scarcity | GraphStats_RF  | sklearn   | test         | ALL      |    114 |         114 |       0.8947 |               1 |   0.9444 |     0.4722 |                 0.8194 | ok       |
| label_scarcity_025pct         | transductive_label_scarcity | GraphStats_EBM | interpret | test         | ALL      |    114 |         114 |       0.9123 |               1 |   0.9541 |     0.4771 |                 0.9108 | ok       |
| label_scarcity_050pct         | transductive_label_scarcity | GraphStats_LR  | sklearn   | test         | ALL      |     75 |          75 |       0.92   |               1 |   0.9583 |     0.4792 |                 0.8633 | ok       |
| label_scarcity_050pct         | transductive_label_scarcity | GraphStats_RF  | sklearn   | test         | ALL      |     75 |          75 |       0.9467 |               1 |   0.9726 |     0.4863 |                 0.8616 | ok       |
| label_scarcity_050pct         | transductive_label_scarcity | GraphStats_EBM | interpret | test         | ALL      |     75 |          75 |       0.9467 |               1 |   0.9726 |     0.4863 |                 0.9467 | ok       |
| lofo_g1                       | inductive_family_holdout    | GraphStats_LR  | sklearn   | test         | ALL      |     50 |          50 |       0.88   |               1 |   0.9362 |     0.4681 |                 0.8498 | ok       |
| lofo_g1                       | inductive_family_holdout    | GraphStats_RF  | sklearn   | test         | ALL      |     50 |          50 |       0.92   |               1 |   0.9583 |     0.4792 |                 0.8542 | ok       |
| lofo_g1                       | inductive_family_holdout    | GraphStats_EBM | interpret | test         | ALL      |     50 |          50 |       0.94   |               1 |   0.9691 |     0.4845 |                 0.9352 | ok       |
| lofo_g2                       | inductive_family_holdout    | GraphStats_LR  | sklearn   | test         | ALL      |     50 |          50 |       0.92   |               1 |   0.9583 |     0.4792 |                 0.9186 | ok       |
| lofo_g2                       | inductive_family_holdout    | GraphStats_RF  | sklearn   | test         | ALL      |     50 |          50 |       0.98   |               1 |   0.9899 |     0.4949 |                 0.9327 | ok       |
| lofo_g2                       | inductive_family_holdout    | GraphStats_EBM | interpret | test         | ALL      |     50 |          50 |       0.98   |               1 |   0.9899 |     0.4949 |                 0.9718 | ok       |
| lofo_g3                       | inductive_family_holdout    | GraphStats_LR  | sklearn   | test         | ALL      |     50 |          50 |       0.98   |               1 |   0.9899 |     0.4949 |                 0.9383 | ok       |
| lofo_g3                       | inductive_family_holdout    | GraphStats_RF  | sklearn   | test         | ALL      |     50 |          50 |       0.98   |               1 |   0.9899 |     0.4949 |                 0.91   | ok       |
| lofo_g3                       | inductive_family_holdout    | GraphStats_EBM | interpret | test         | ALL      |     50 |          50 |       0.98   |               1 |   0.9899 |     0.4949 |                 0.9803 | ok       |
| original_only_zero_shot       | inductive_family_holdout    | GraphStats_LR  | sklearn   | test         | ALL      |    150 |         150 |       0      |               0 |   0      |     0      |                 0.0001 | ok       |
| original_only_zero_shot       | inductive_family_holdout    | GraphStats_RF  | sklearn   | test         | ALL      |    150 |         150 |       0      |               0 |   0      |     0      |                 0.0004 | ok       |
| original_only_zero_shot       | inductive_family_holdout    | GraphStats_EBM | interpret | test         | ALL      |    150 |         150 |       0      |               0 |   0      |     0      |                 0      | ok       |

### GraphStats family recall matrix

| split_name                    | model          | backend   | family                    |   bot_count |   bot_recall |   bot_mean_probability |
|:------------------------------|:---------------|:----------|:--------------------------|------------:|-------------:|-----------------------:|
| incremental_curriculum_order1 | GraphStats_LR  | sklearn   | G1_population_calibrated  |          50 |       0      |                 0.0002 |
| incremental_curriculum_order1 | GraphStats_RF  | sklearn   | G1_population_calibrated  |          50 |       0      |                 0      |
| incremental_curriculum_order1 | GraphStats_EBM | interpret | G1_population_calibrated  |          50 |       0      |                 0      |
| incremental_curriculum_order2 | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          50 |       0      |                 0.0001 |
| incremental_curriculum_order2 | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          50 |       0      |                 0      |
| incremental_curriculum_order2 | GraphStats_EBM | interpret | G2_anchor_behavior        |          50 |       0      |                 0      |
| incremental_curriculum_order3 | GraphStats_LR  | sklearn   | G3_contextual_interactive |          50 |       0      |                 0.0001 |
| incremental_curriculum_order3 | GraphStats_RF  | sklearn   | G3_contextual_interactive |          50 |       0      |                 0.0001 |
| incremental_curriculum_order3 | GraphStats_EBM | interpret | G3_contextual_interactive |          50 |       0      |                 0      |
| label_scarcity_001pct         | GraphStats_LR  | sklearn   | G1_population_calibrated  |          49 |       0.4286 |                 0.4305 |
| label_scarcity_001pct         | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          49 |       0.551  |                 0.5357 |
| label_scarcity_001pct         | GraphStats_LR  | sklearn   | G3_contextual_interactive |          49 |       0.5918 |                 0.5703 |
| label_scarcity_001pct         | GraphStats_RF  | sklearn   | G1_population_calibrated  |          49 |       0.0204 |                 0.2208 |
| label_scarcity_001pct         | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          49 |       0.0612 |                 0.2834 |
| label_scarcity_001pct         | GraphStats_RF  | sklearn   | G3_contextual_interactive |          49 |       0.102  |                 0.2958 |
| label_scarcity_001pct         | GraphStats_EBM | interpret | G1_population_calibrated  |          49 |       0.2041 |                 0.2132 |
| label_scarcity_001pct         | GraphStats_EBM | interpret | G2_anchor_behavior        |          49 |       0.3878 |                 0.3832 |
| label_scarcity_001pct         | GraphStats_EBM | interpret | G3_contextual_interactive |          49 |       0.449  |                 0.4392 |
| label_scarcity_005pct         | GraphStats_LR  | sklearn   | G1_population_calibrated  |          48 |       0.6042 |                 0.5804 |
| label_scarcity_005pct         | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          48 |       0.75   |                 0.7118 |
| label_scarcity_005pct         | GraphStats_LR  | sklearn   | G3_contextual_interactive |          48 |       0.6875 |                 0.6825 |
| label_scarcity_005pct         | GraphStats_RF  | sklearn   | G1_population_calibrated  |          48 |       0.4375 |                 0.4589 |
| label_scarcity_005pct         | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          48 |       0.625  |                 0.5319 |
| label_scarcity_005pct         | GraphStats_RF  | sklearn   | G3_contextual_interactive |          48 |       0.5    |                 0.4843 |
| label_scarcity_005pct         | GraphStats_EBM | interpret | G1_population_calibrated  |          48 |       0.8958 |                 0.8554 |
| label_scarcity_005pct         | GraphStats_EBM | interpret | G2_anchor_behavior        |          48 |       0.875  |                 0.8724 |
| label_scarcity_005pct         | GraphStats_EBM | interpret | G3_contextual_interactive |          48 |       0.8125 |                 0.7994 |
| label_scarcity_010pct         | GraphStats_LR  | sklearn   | G1_population_calibrated  |          45 |       0.7778 |                 0.7775 |
| label_scarcity_010pct         | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          45 |       0.8222 |                 0.8336 |
| label_scarcity_010pct         | GraphStats_LR  | sklearn   | G3_contextual_interactive |          45 |       0.8222 |                 0.8433 |
| label_scarcity_010pct         | GraphStats_RF  | sklearn   | G1_population_calibrated  |          45 |       0.8889 |                 0.719  |
| label_scarcity_010pct         | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          45 |       0.9111 |                 0.7918 |
| label_scarcity_010pct         | GraphStats_RF  | sklearn   | G3_contextual_interactive |          45 |       0.8889 |                 0.7524 |
| label_scarcity_010pct         | GraphStats_EBM | interpret | G1_population_calibrated  |          45 |       0.9111 |                 0.9111 |
| label_scarcity_010pct         | GraphStats_EBM | interpret | G2_anchor_behavior        |          45 |       0.9556 |                 0.9555 |
| label_scarcity_010pct         | GraphStats_EBM | interpret | G3_contextual_interactive |          45 |       0.9556 |                 0.9555 |
| label_scarcity_025pct         | GraphStats_LR  | sklearn   | G1_population_calibrated  |          38 |       0.8158 |                 0.7991 |
| label_scarcity_025pct         | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          38 |       0.8158 |                 0.8245 |
| label_scarcity_025pct         | GraphStats_LR  | sklearn   | G3_contextual_interactive |          38 |       0.8684 |                 0.8518 |
| label_scarcity_025pct         | GraphStats_RF  | sklearn   | G1_population_calibrated  |          38 |       0.8684 |                 0.7879 |
| label_scarcity_025pct         | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          38 |       0.8684 |                 0.8272 |
| label_scarcity_025pct         | GraphStats_RF  | sklearn   | G3_contextual_interactive |          38 |       0.9474 |                 0.8432 |
| label_scarcity_025pct         | GraphStats_EBM | interpret | G1_population_calibrated  |          38 |       0.8947 |                 0.8943 |
| label_scarcity_025pct         | GraphStats_EBM | interpret | G2_anchor_behavior        |          38 |       0.8947 |                 0.8909 |
| label_scarcity_025pct         | GraphStats_EBM | interpret | G3_contextual_interactive |          38 |       0.9474 |                 0.9472 |
| label_scarcity_050pct         | GraphStats_LR  | sklearn   | G1_population_calibrated  |          25 |       0.88   |                 0.7952 |
| label_scarcity_050pct         | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          25 |       0.92   |                 0.912  |
| label_scarcity_050pct         | GraphStats_LR  | sklearn   | G3_contextual_interactive |          25 |       0.96   |                 0.8826 |
| label_scarcity_050pct         | GraphStats_RF  | sklearn   | G1_population_calibrated  |          25 |       0.92   |                 0.821  |
| label_scarcity_050pct         | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          25 |       0.96   |                 0.8883 |
| label_scarcity_050pct         | GraphStats_RF  | sklearn   | G3_contextual_interactive |          25 |       0.96   |                 0.8756 |
| label_scarcity_050pct         | GraphStats_EBM | interpret | G1_population_calibrated  |          25 |       0.92   |                 0.9111 |
| label_scarcity_050pct         | GraphStats_EBM | interpret | G2_anchor_behavior        |          25 |       0.96   |                 0.9705 |
| label_scarcity_050pct         | GraphStats_EBM | interpret | G3_contextual_interactive |          25 |       0.96   |                 0.9585 |
| lofo_g1                       | GraphStats_LR  | sklearn   | G1_population_calibrated  |          50 |       0.88   |                 0.8498 |
| lofo_g1                       | GraphStats_RF  | sklearn   | G1_population_calibrated  |          50 |       0.92   |                 0.8542 |
| lofo_g1                       | GraphStats_EBM | interpret | G1_population_calibrated  |          50 |       0.94   |                 0.9352 |
| lofo_g2                       | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          50 |       0.92   |                 0.9186 |
| lofo_g2                       | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          50 |       0.98   |                 0.9327 |
| lofo_g2                       | GraphStats_EBM | interpret | G2_anchor_behavior        |          50 |       0.98   |                 0.9718 |
| lofo_g3                       | GraphStats_LR  | sklearn   | G3_contextual_interactive |          50 |       0.98   |                 0.9383 |
| lofo_g3                       | GraphStats_RF  | sklearn   | G3_contextual_interactive |          50 |       0.98   |                 0.91   |
| lofo_g3                       | GraphStats_EBM | interpret | G3_contextual_interactive |          50 |       0.98   |                 0.9803 |
| original_only_zero_shot       | GraphStats_LR  | sklearn   | G1_population_calibrated  |          50 |       0      |                 0.0001 |
| original_only_zero_shot       | GraphStats_LR  | sklearn   | G2_anchor_behavior        |          50 |       0      |                 0.0001 |
| original_only_zero_shot       | GraphStats_LR  | sklearn   | G3_contextual_interactive |          50 |       0      |                 0.0001 |
| original_only_zero_shot       | GraphStats_RF  | sklearn   | G1_population_calibrated  |          50 |       0      |                 0.0001 |
| original_only_zero_shot       | GraphStats_RF  | sklearn   | G2_anchor_behavior        |          50 |       0      |                 0.0001 |
| original_only_zero_shot       | GraphStats_RF  | sklearn   | G3_contextual_interactive |          50 |       0      |                 0.0011 |
| original_only_zero_shot       | GraphStats_EBM | interpret | G1_population_calibrated  |          50 |       0      |                 0      |
| original_only_zero_shot       | GraphStats_EBM | interpret | G2_anchor_behavior        |          50 |       0      |                 0      |
| original_only_zero_shot       | GraphStats_EBM | interpret | G3_contextual_interactive |          50 |       0      |                 0      |


## GNN 结果

### R-GCN transductive 汇总

| split_name                    | setting                     | model             | backend      | eval_group   | family   |   size |   bot_count |   bot_recall |   bot_precision |   bot_f1 |   macro_f1 |   bot_mean_probability | status   |
|:------------------------------|:----------------------------|:------------------|:-------------|:-------------|:---------|-------:|------------:|-------------:|----------------:|---------:|-----------:|-----------------------:|:---------|
| incremental_curriculum_order1 | incremental_curriculum      | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order2 | incremental_curriculum      | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order3 | incremental_curriculum      | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| label_scarcity_001pct         | transductive_label_scarcity | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |    147 |         147 |       0.7415 |               1 |   0.8516 |     0.4258 |                 0.7481 | ok       |
| label_scarcity_005pct         | transductive_label_scarcity | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |    144 |         144 |       0.7361 |               1 |   0.848  |     0.424  |                 0.7448 | ok       |
| label_scarcity_010pct         | transductive_label_scarcity | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |    135 |         135 |       0.9407 |               1 |   0.9695 |     0.4847 |                 0.9394 | ok       |
| label_scarcity_025pct         | transductive_label_scarcity | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |    114 |         114 |       0.8947 |               1 |   0.9444 |     0.4722 |                 0.8979 | ok       |
| label_scarcity_050pct         | transductive_label_scarcity | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |     75 |          75 |       0.8933 |               1 |   0.9437 |     0.4718 |                 0.9005 | ok       |
| lofo_g1                       | inductive_family_holdout    | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |     50 |          50 |       0.96   |               1 |   0.9796 |     0.4898 |                 0.96   | ok       |
| lofo_g2                       | inductive_family_holdout    | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |     50 |          50 |       0.96   |               1 |   0.9796 |     0.4898 |                 0.9482 | ok       |
| lofo_g3                       | inductive_family_holdout    | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |     50 |          50 |       0.98   |               1 |   0.9899 |     0.4949 |                 0.9752 | ok       |
| original_only_zero_shot       | inductive_family_holdout    | RGCN_Transductive | pyg_rgcnconv | test         | ALL      |    150 |         150 |       0      |               0 |   0      |     0      |                 0      | ok       |

### GraphSAGE inductive 汇总

| split_name                    | setting                     | model               | backend      | eval_group   | family   |   size |   bot_count |   bot_recall |   bot_precision |   bot_f1 |   macro_f1 |   bot_mean_probability | status   |
|:------------------------------|:----------------------------|:--------------------|:-------------|:-------------|:---------|-------:|------------:|-------------:|----------------:|---------:|-----------:|-----------------------:|:---------|
| incremental_curriculum_order1 | incremental_curriculum      | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order2 | incremental_curriculum      | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| incremental_curriculum_order3 | incremental_curriculum      | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |     50 |          50 |       0      |               0 |   0      |     0      |                 0      | ok       |
| label_scarcity_001pct         | transductive_label_scarcity | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |    147 |         147 |       0.0272 |               1 |   0.053  |     0.0265 |                 0.0789 | ok       |
| label_scarcity_005pct         | transductive_label_scarcity | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |    144 |         144 |       0.0069 |               1 |   0.0138 |     0.0069 |                 0.0435 | ok       |
| label_scarcity_010pct         | transductive_label_scarcity | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |    135 |         135 |       0      |               0 |   0      |     0      |                 0.0033 | ok       |
| label_scarcity_025pct         | transductive_label_scarcity | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |    114 |         114 |       0.9649 |               1 |   0.9821 |     0.4911 |                 0.9535 | ok       |
| label_scarcity_050pct         | transductive_label_scarcity | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |     75 |          75 |       0.9867 |               1 |   0.9933 |     0.4966 |                 0.9743 | ok       |
| lofo_g1                       | inductive_family_holdout    | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |     50 |          50 |       0.98   |               1 |   0.9899 |     0.4949 |                 0.9582 | ok       |
| lofo_g2                       | inductive_family_holdout    | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |     50 |          50 |       0.96   |               1 |   0.9796 |     0.4898 |                 0.9724 | ok       |
| lofo_g3                       | inductive_family_holdout    | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |     50 |          50 |       1      |               1 |   1      |     1      |                 0.9955 | ok       |
| original_only_zero_shot       | inductive_family_holdout    | GraphSAGE_Inductive | pyg_sageconv | test         | ALL      |    150 |         150 |       0      |               0 |   0      |     0      |                 0      | ok       |

### GNN family recall matrix

| split_name                    | model               | backend      | family                    |   bot_count |   bot_recall |   bot_mean_probability |
|:------------------------------|:--------------------|:-------------|:--------------------------|------------:|-------------:|-----------------------:|
| incremental_curriculum_order1 | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          50 |       0      |                 0      |
| incremental_curriculum_order2 | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          50 |       0      |                 0      |
| incremental_curriculum_order3 | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          50 |       0      |                 0      |
| label_scarcity_001pct         | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          49 |       0.7347 |                 0.7383 |
| label_scarcity_001pct         | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          49 |       0.7755 |                 0.7801 |
| label_scarcity_001pct         | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          49 |       0.7143 |                 0.7258 |
| label_scarcity_005pct         | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          48 |       0.7083 |                 0.7252 |
| label_scarcity_005pct         | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          48 |       0.7292 |                 0.747  |
| label_scarcity_005pct         | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          48 |       0.7708 |                 0.7622 |
| label_scarcity_010pct         | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          45 |       0.9556 |                 0.9515 |
| label_scarcity_010pct         | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          45 |       0.9556 |                 0.9555 |
| label_scarcity_010pct         | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          45 |       0.9111 |                 0.9111 |
| label_scarcity_025pct         | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          38 |       0.8947 |                 0.8979 |
| label_scarcity_025pct         | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          38 |       0.9211 |                 0.9184 |
| label_scarcity_025pct         | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          38 |       0.8684 |                 0.8774 |
| label_scarcity_050pct         | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          25 |       0.92   |                 0.937  |
| label_scarcity_050pct         | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          25 |       0.92   |                 0.9275 |
| label_scarcity_050pct         | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          25 |       0.84   |                 0.837  |
| lofo_g1                       | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          50 |       0.96   |                 0.96   |
| lofo_g2                       | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          50 |       0.96   |                 0.9482 |
| lofo_g3                       | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          50 |       0.98   |                 0.9752 |
| original_only_zero_shot       | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          50 |       0      |                 0      |
| original_only_zero_shot       | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          50 |       0      |                 0      |
| original_only_zero_shot       | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          50 |       0      |                 0      |
| incremental_curriculum_order1 | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          50 |       0      |                 0      |
| incremental_curriculum_order2 | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          50 |       0      |                 0      |
| incremental_curriculum_order3 | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          50 |       0      |                 0      |
| label_scarcity_001pct         | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          49 |       0.0204 |                 0.0456 |
| label_scarcity_001pct         | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          49 |       0.0408 |                 0.0925 |
| label_scarcity_001pct         | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          49 |       0.0204 |                 0.0987 |
| label_scarcity_005pct         | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          48 |       0.0208 |                 0.0313 |
| label_scarcity_005pct         | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          48 |       0      |                 0.0456 |
| label_scarcity_005pct         | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          48 |       0      |                 0.0537 |
| label_scarcity_010pct         | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          45 |       0      |                 0.0042 |
| label_scarcity_010pct         | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          45 |       0      |                 0.0032 |
| label_scarcity_010pct         | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          45 |       0      |                 0.0026 |
| label_scarcity_025pct         | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          38 |       0.9474 |                 0.932  |
| label_scarcity_025pct         | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          38 |       0.9474 |                 0.9494 |
| label_scarcity_025pct         | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          38 |       1      |                 0.979  |
| label_scarcity_050pct         | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          25 |       0.96   |                 0.9488 |
| label_scarcity_050pct         | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          25 |       1      |                 0.988  |
| label_scarcity_050pct         | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          25 |       1      |                 0.9861 |
| lofo_g1                       | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          50 |       0.98   |                 0.9582 |
| lofo_g2                       | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          50 |       0.96   |                 0.9724 |
| lofo_g3                       | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          50 |       1      |                 0.9955 |
| original_only_zero_shot       | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          50 |       0      |                 0      |
| original_only_zero_shot       | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          50 |       0      |                 0      |
| original_only_zero_shot       | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          50 |       0      |                 0      |


## Text-only vs Graph-only vs Fusion

本轮 Worker E 只实现 Graph-only 检测骨架；Text-only 与 Fusion 结果文件缺失时不做数值比较。后续可在相同 split 下接入 TF-IDF/BGE 与文本图融合结果，再判断各 family 是文本更容易检出、图结构更容易检出，还是 Fusion 继承了单一路径 shortcut。

## Shortcut 风险审计

训练特征仅由 user-user 图结构派生；`family`、`user_id/user_name`、`provider/model` 只保留在分组、预测输出和审计字段中，不进入特征矩阵。

| split_name                    | setting                     | family   |   size |   bot_count |   zero_inbound_share |   over_directional_share |   low_reciprocity_share |   isolated_share |   bot_only_component_share | forbidden_metadata_columns_present   | forbidden_metadata_used_as_features   |
|:------------------------------|:----------------------------|:---------|-------:|------------:|---------------------:|-------------------------:|------------------------:|-----------------:|---------------------------:|:-------------------------------------|:--------------------------------------|
| incremental_curriculum_order1 | incremental_curriculum      | ALL      |     50 |          50 |               0.02   |                   0.02   |                  0.02   |                0 |                          0 | family;user_id;user_name             | False                                 |
| incremental_curriculum_order2 | incremental_curriculum      | ALL      |     50 |          50 |               0      |                   0      |                  0      |                0 |                          0 | family;user_id;user_name             | False                                 |
| incremental_curriculum_order3 | incremental_curriculum      | ALL      |     50 |          50 |               0      |                   0      |                  0      |                0 |                          0 | family;user_id;user_name             | False                                 |
| label_scarcity_001pct         | transductive_label_scarcity | ALL      |    147 |         147 |               0.0068 |                   0.0068 |                  0.0068 |                0 |                          0 | family;user_id;user_name             | False                                 |
| label_scarcity_005pct         | transductive_label_scarcity | ALL      |    144 |         144 |               0.0069 |                   0.0069 |                  0.0069 |                0 |                          0 | family;user_id;user_name             | False                                 |
| label_scarcity_010pct         | transductive_label_scarcity | ALL      |    135 |         135 |               0.0074 |                   0.0074 |                  0.0074 |                0 |                          0 | family;user_id;user_name             | False                                 |
| label_scarcity_025pct         | transductive_label_scarcity | ALL      |    114 |         114 |               0.0088 |                   0.0088 |                  0.0088 |                0 |                          0 | family;user_id;user_name             | False                                 |
| label_scarcity_050pct         | transductive_label_scarcity | ALL      |     75 |          75 |               0.0133 |                   0.0133 |                  0.0133 |                0 |                          0 | family;user_id;user_name             | False                                 |
| lofo_g1                       | inductive_family_holdout    | ALL      |     50 |          50 |               0.02   |                   0.02   |                  0.02   |                0 |                          0 | family;user_id;user_name             | False                                 |
| lofo_g2                       | inductive_family_holdout    | ALL      |     50 |          50 |               0      |                   0      |                  0      |                0 |                          0 | family;user_id;user_name             | False                                 |
| lofo_g3                       | inductive_family_holdout    | ALL      |     50 |          50 |               0      |                   0      |                  0      |                0 |                          0 | family;user_id;user_name             | False                                 |
| original_only_zero_shot       | inductive_family_holdout    | ALL      |    150 |         150 |               0.0067 |                   0.0067 |                  0.0067 |                0 |                          0 | family;user_id;user_name             | False                                 |


## 可解释特征

| split_name                    | model          |   rank | feature                                         |   importance | importance_kind   |
|:------------------------------|:---------------|-------:|:------------------------------------------------|-------------:|:------------------|
| incremental_curriculum_order1 | GraphStats_EBM |      1 | avg_in_weight                                   |      1.23696 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      2 | avg_out_weight                                  |      1.21944 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      3 | component_size                                  |      1.2052  | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      4 | has_any_edge                                    |      1.20498 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      5 | is_isolated                                     |      1.20318 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      6 | log_component_size                              |      1.20219 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      7 | is_isolated & pagerank                          |      0.98426 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      8 | log_component_size & pagerank                   |      0.98383 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |      9 | is_isolated & log_pagerank                      |      0.98075 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_EBM |     10 | is_isolated & clustering                        |      0.98075 | ebm_term          |
| incremental_curriculum_order1 | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      0.54891 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      0.54891 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      3 | has_any_edge                                    |      0.54071 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      4 | is_isolated                                     |      0.54071 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      5 | etype_user_receives_reply_weight_share          |      0.53757 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      6 | etype_user_receives_reply_degree_share          |      0.53757 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      7 | log_component_size                              |      0.50938 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      8 | component_size                                  |      0.50578 | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |      9 | avg_in_weight                                   |      0.2721  | feature           |
| incremental_curriculum_order1 | GraphStats_LR  |     10 | log_total_degree                                |      0.23607 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      1 | is_isolated                                     |      0.13002 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      2 | log_total_degree                                |      0.12075 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      3 | total_weight                                    |      0.10478 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      4 | log_component_size                              |      0.09668 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      5 | etype_user_receives_reply_degree_share          |      0.08801 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      6 | etype_user_receives_reply_weight_share          |      0.08734 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      7 | has_any_edge                                    |      0.07869 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      8 | log_total_weight                                |      0.07736 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |      9 | total_degree                                    |      0.07199 | feature           |
| incremental_curriculum_order1 | GraphStats_RF  |     10 | component_size                                  |      0.05667 | feature           |
| incremental_curriculum_order2 | GraphStats_EBM |      1 | avg_in_weight                                   |      1.23457 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      2 | avg_out_weight                                  |      1.22263 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      3 | has_any_edge                                    |      1.20899 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      4 | component_size                                  |      1.20835 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      5 | is_isolated                                     |      1.20708 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      6 | log_component_size                              |      1.20562 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      7 | is_isolated & log_pagerank                      |      1.02265 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      8 | is_isolated & pagerank                          |      1.02265 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |      9 | is_isolated & clustering                        |      1.01912 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_EBM |     10 | log_component_size & pagerank                   |      1.01866 | ebm_term          |
| incremental_curriculum_order2 | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      0.51171 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      0.51171 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      3 | has_any_edge                                    |      0.50411 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      4 | is_isolated                                     |      0.50411 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      5 | etype_user_receives_reply_weight_share          |      0.50128 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      6 | etype_user_receives_reply_degree_share          |      0.50128 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      7 | log_component_size                              |      0.4772  | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      8 | component_size                                  |      0.47409 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |      9 | avg_in_weight                                   |      0.27042 | feature           |
| incremental_curriculum_order2 | GraphStats_LR  |     10 | log_total_weight                                |      0.25165 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      1 | log_total_degree                                |      0.11547 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      2 | total_weight                                    |      0.11    | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      3 | is_isolated                                     |      0.10667 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      4 | log_component_size                              |      0.10067 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      5 | etype_user_receives_reply_degree_share          |      0.09468 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      6 | etype_user_receives_reply_weight_share          |      0.08281 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      7 | has_any_edge                                    |      0.07729 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      8 | log_total_weight                                |      0.074   | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |      9 | total_degree                                    |      0.06741 | feature           |
| incremental_curriculum_order2 | GraphStats_RF  |     10 | component_size                                  |      0.0573  | feature           |
| incremental_curriculum_order3 | GraphStats_EBM |      1 | avg_in_weight                                   |      1.21359 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      2 | avg_out_weight                                  |      1.20568 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      3 | component_size                                  |      1.19233 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      4 | has_any_edge                                    |      1.18904 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      5 | is_isolated                                     |      1.18709 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      6 | log_component_size                              |      1.18558 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      7 | comment1_rate                                   |      0.95638 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      8 | etype_user_receives_reply_degree_share          |      0.95628 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |      9 | etype_user_receives_reply_weight_share          |      0.94423 | ebm_term          |
| incremental_curriculum_order3 | GraphStats_EBM |     10 | post_rate                                       |      0.9337  | ebm_term          |
| incremental_curriculum_order3 | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      0.50638 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      0.50638 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      3 | is_isolated                                     |      0.50393 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      4 | has_any_edge                                    |      0.50393 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      5 | etype_user_receives_reply_weight_share          |      0.50128 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      6 | etype_user_receives_reply_degree_share          |      0.50128 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      7 | log_component_size                              |      0.47773 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      8 | component_size                                  |      0.4747  | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |      9 | avg_in_weight                                   |      0.29656 | feature           |
| incremental_curriculum_order3 | GraphStats_LR  |     10 | log_total_degree                                |      0.27512 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      1 | log_total_degree                                |      0.12    | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      2 | total_weight                                    |      0.11073 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      3 | is_isolated                                     |      0.10667 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      4 | log_component_size                              |      0.10472 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      5 | has_any_edge                                    |      0.08454 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      6 | etype_user_receives_reply_weight_share          |      0.08125 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      7 | etype_user_receives_reply_degree_share          |      0.07667 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      8 | total_degree                                    |      0.07667 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |      9 | log_total_weight                                |      0.07407 | feature           |
| incremental_curriculum_order3 | GraphStats_RF  |     10 | component_size                                  |      0.05805 | feature           |
| label_scarcity_001pct         | GraphStats_EBM |      1 | etype_user_comments_on_user_post_degree_share   |      2.50402 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      2 | etype_user_comments_on_user_post_weight_share   |      2.43952 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      3 | etype_user_receives_reply_in_degree             |      0.95638 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      4 | post_rate                                       |      0.71313 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      5 | etype_user_comments_on_user_post_in_degree      |      0.70409 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      6 | comment1_rate                                   |      0.70229 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      7 | post_rate & log_pagerank                        |      0.58254 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      8 | post_rate & pagerank                            |      0.55856 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |      9 | is_isolated                                     |      0.44583 | ebm_term          |
| label_scarcity_001pct         | GraphStats_EBM |     10 | log_component_size                              |      0.44504 | ebm_term          |
| label_scarcity_001pct         | GraphStats_LR  |      1 | reciprocal_share                                |      0.90498 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      2 | etype_user_replies_to_user_comment_degree_share |      0.60432 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      3 | etype_user_replies_to_user_comment_weight_share |      0.60432 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      4 | etype_user_comments_on_user_post_degree_share   |      0.58109 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      5 | etype_user_comments_on_user_post_weight_share   |      0.58109 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      6 | has_any_edge                                    |      0.54313 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      7 | is_isolated                                     |      0.54313 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      8 | log_component_size                              |      0.52659 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |      9 | component_size                                  |      0.52466 | feature           |
| label_scarcity_001pct         | GraphStats_LR  |     10 | comment2_rate                                   |      0.47964 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      1 | log_total_degree                                |      0.1036  | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      2 | total_weight                                    |      0.09955 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      3 | etype_user_comments_on_user_post_weight_share   |      0.08742 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      4 | etype_user_comments_on_user_post_degree_share   |      0.08414 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      5 | log_component_size                              |      0.08382 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      6 | is_isolated                                     |      0.08308 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      7 | etype_user_receives_reply_degree_share          |      0.08234 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      8 | etype_user_receives_reply_weight_share          |      0.07533 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |      9 | has_any_edge                                    |      0.07355 | feature           |
| label_scarcity_001pct         | GraphStats_RF  |     10 | total_degree                                    |      0.05714 | feature           |
| label_scarcity_005pct         | GraphStats_EBM |      1 | etype_user_comments_on_user_post_degree_share   |      1.41131 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      2 | etype_user_comments_on_user_post_weight_share   |      1.41044 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      3 | comment1_rate                                   |      0.97212 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      4 | post_rate                                       |      0.88738 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      5 | reciprocal_share & clustering                   |      0.79161 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      6 | receive_rate & reciprocal_share                 |      0.71403 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      7 | reciprocal_share & pagerank                     |      0.60692 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      8 | comment2_rate & receive_rate                    |      0.6056  | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |      9 | reciprocal_share & log_pagerank                 |      0.59843 | ebm_term          |
| label_scarcity_005pct         | GraphStats_EBM |     10 | component_size                                  |      0.4948  | ebm_term          |
| label_scarcity_005pct         | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      0.92975 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      0.92975 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      3 | reciprocal_share                                |      0.83559 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      4 | etype_user_replies_to_user_comment_weight_share |      0.64223 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      5 | etype_user_replies_to_user_comment_degree_share |      0.64223 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      6 | has_any_edge                                    |      0.52686 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      7 | is_isolated                                     |      0.52686 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      8 | log_component_size                              |      0.48684 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |      9 | component_size                                  |      0.48232 | feature           |
| label_scarcity_005pct         | GraphStats_LR  |     10 | direction_balance                               |      0.40593 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.10081 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      2 | log_total_degree                                |      0.10049 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      3 | total_weight                                    |      0.09973 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      4 | is_isolated                                     |      0.09893 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      5 | log_component_size                              |      0.08283 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      6 | etype_user_comments_on_user_post_degree_share   |      0.08011 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      7 | etype_user_receives_reply_weight_share          |      0.07479 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      8 | has_any_edge                                    |      0.07379 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |      9 | etype_user_receives_reply_degree_share          |      0.06873 | feature           |
| label_scarcity_005pct         | GraphStats_RF  |     10 | total_degree                                    |      0.05806 | feature           |
| label_scarcity_010pct         | GraphStats_EBM |      1 | etype_user_comments_on_user_post_degree_share   |      1.4063  | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      2 | etype_user_comments_on_user_post_weight_share   |      1.28427 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      3 | etype_user_comments_on_user_post_in_degree      |      0.59197 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      4 | etype_user_replies_to_user_comment_degree_share |      0.57667 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      5 | has_any_edge                                    |      0.56734 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      6 | log_component_size                              |      0.56076 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      7 | is_isolated                                     |      0.55393 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      8 | component_size                                  |      0.54914 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |      9 | etype_user_replies_to_user_comment_weight_share |      0.53033 | ebm_term          |
| label_scarcity_010pct         | GraphStats_EBM |     10 | clustering                                      |      0.46886 | ebm_term          |
| label_scarcity_010pct         | GraphStats_LR  |      1 | reciprocal_share                                |      1.45086 | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      2 | etype_user_replies_to_user_comment_weight_share |      1.31809 | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      3 | etype_user_replies_to_user_comment_degree_share |      1.31809 | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      4 | etype_user_comments_on_user_post_weight_share   |      1.22638 | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      5 | etype_user_comments_on_user_post_degree_share   |      1.22638 | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      6 | direction_balance                               |      1.13013 | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      7 | avg_in_weight                                   |      0.63093 | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      8 | is_isolated                                     |      0.61    | feature           |
| label_scarcity_010pct         | GraphStats_LR  |      9 | has_any_edge                                    |      0.61    | feature           |
| label_scarcity_010pct         | GraphStats_LR  |     10 | receive_rate                                    |      0.48637 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.13605 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      2 | etype_user_comments_on_user_post_degree_share   |      0.11652 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      3 | log_total_degree                                |      0.09408 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      4 | is_isolated                                     |      0.09014 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      5 | etype_user_receives_reply_weight_share          |      0.07627 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      6 | has_any_edge                                    |      0.07263 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      7 | total_weight                                    |      0.07227 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      8 | log_component_size                              |      0.07031 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |      9 | etype_user_receives_reply_degree_share          |      0.06584 | feature           |
| label_scarcity_010pct         | GraphStats_RF  |     10 | total_degree                                    |      0.05271 | feature           |
| label_scarcity_025pct         | GraphStats_EBM |      1 | etype_user_comments_on_user_post_degree_share   |      2.0206  | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      2 | etype_user_comments_on_user_post_weight_share   |      2.00197 | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      3 | comment1_rate                                   |      0.62518 | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      4 | post_rate                                       |      0.61913 | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      5 | direction_balance                               |      0.4876  | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      6 | is_isolated                                     |      0.48719 | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      7 | has_any_edge                                    |      0.48637 | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      8 | log_component_size                              |      0.48479 | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |      9 | etype_user_comments_on_user_post_in_degree      |      0.48445 | ebm_term          |
| label_scarcity_025pct         | GraphStats_EBM |     10 | component_size                                  |      0.48331 | ebm_term          |
| label_scarcity_025pct         | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      1.75838 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      1.75838 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      3 | reciprocal_share                                |      1.65422 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      4 | etype_user_replies_to_user_comment_weight_share |      1.19261 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      5 | etype_user_replies_to_user_comment_degree_share |      1.19261 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      6 | direction_balance                               |      0.93899 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      7 | etype_user_receives_reply_degree_share          |      0.80965 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      8 | etype_user_receives_reply_weight_share          |      0.80965 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |      9 | avg_in_weight                                   |      0.68952 | feature           |
| label_scarcity_025pct         | GraphStats_LR  |     10 | has_any_edge                                    |      0.59011 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.14817 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      2 | etype_user_comments_on_user_post_degree_share   |      0.12476 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      3 | total_weight                                    |      0.08284 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      4 | log_total_degree                                |      0.07696 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      5 | etype_user_receives_reply_weight_share          |      0.07639 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      6 | log_component_size                              |      0.07514 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      7 | has_any_edge                                    |      0.07195 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      8 | is_isolated                                     |      0.07095 | feature           |
| label_scarcity_025pct         | GraphStats_RF  |      9 | etype_user_receives_reply_degree_share          |      0.0697  | feature           |
| label_scarcity_025pct         | GraphStats_RF  |     10 | total_degree                                    |      0.05047 | feature           |
| label_scarcity_050pct         | GraphStats_EBM |      1 | etype_user_comments_on_user_post_weight_share   |      0.97823 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      2 | total_in_degree                                 |      0.96836 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      3 | etype_user_comments_on_user_post_degree_share   |      0.94601 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      4 | etype_user_comments_on_user_post_in_degree      |      0.94167 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      5 | reciprocal_share & clustering                   |      0.9254  | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      6 | total_in_weight                                 |      0.90483 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      7 | receive_rate                                    |      0.64503 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      8 | post_rate                                       |      0.61111 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |      9 | comment1_rate                                   |      0.59311 | ebm_term          |
| label_scarcity_050pct         | GraphStats_EBM |     10 | direction_balance                               |      0.42944 | ebm_term          |
| label_scarcity_050pct         | GraphStats_LR  |      1 | direction_balance                               |      2.07072 | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      2 | etype_user_comments_on_user_post_degree_share   |      1.95933 | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      3 | etype_user_comments_on_user_post_weight_share   |      1.95933 | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      4 | reciprocal_share                                |      1.34364 | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      5 | avg_in_weight                                   |      1.07999 | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      6 | etype_user_receives_reply_degree_share          |      0.8306  | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      7 | etype_user_receives_reply_weight_share          |      0.8306  | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      8 | etype_user_replies_to_user_comment_in_degree    |      0.77977 | feature           |
| label_scarcity_050pct         | GraphStats_LR  |      9 | etype_user_replies_to_user_comment_out_degree   |      0.747   | feature           |
| label_scarcity_050pct         | GraphStats_LR  |     10 | etype_user_comments_on_user_post_out_degree     |      0.63653 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.15346 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      2 | etype_user_comments_on_user_post_degree_share   |      0.12824 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      3 | etype_user_receives_reply_weight_share          |      0.08249 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      4 | etype_user_receives_reply_degree_share          |      0.07818 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      5 | total_weight                                    |      0.07476 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      6 | log_total_degree                                |      0.07333 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      7 | is_isolated                                     |      0.07256 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      8 | log_component_size                              |      0.06413 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |      9 | has_any_edge                                    |      0.06118 | feature           |
| label_scarcity_050pct         | GraphStats_RF  |     10 | total_degree                                    |      0.04737 | feature           |
| label_scarcity_100pct         | GraphStats_EBM |      1 | reciprocal_share & clustering                   |      1.29919 | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      2 | etype_user_comments_on_user_post_in_degree      |      0.8354  | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      3 | etype_user_comments_on_user_post_out_degree     |      0.72079 | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      4 | total_in_degree                                 |      0.69339 | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      5 | etype_user_comments_on_user_post_weight_share   |      0.68529 | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      6 | total_in_weight                                 |      0.6808  | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      7 | etype_user_comments_on_user_post_degree_share   |      0.67768 | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      8 | receive_rate                                    |      0.64403 | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |      9 | direction_balance                               |      0.48937 | ebm_term          |
| label_scarcity_100pct         | GraphStats_EBM |     10 | etype_user_receives_reply_degree_share          |      0.48642 | ebm_term          |
| label_scarcity_100pct         | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      2.31444 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      2.31444 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      3 | direction_balance                               |      2.2195  | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      4 | reciprocal_share                                |      1.75645 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      5 | etype_user_receives_reply_degree_share          |      1.28701 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      6 | etype_user_receives_reply_weight_share          |      1.28701 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      7 | avg_in_weight                                   |      0.94685 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      8 | etype_user_comments_on_user_post_out_degree     |      0.93992 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |      9 | etype_user_replies_to_user_comment_out_degree   |      0.88425 | feature           |
| label_scarcity_100pct         | GraphStats_LR  |     10 | receive_rate                                    |      0.819   | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.15192 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      2 | etype_user_comments_on_user_post_degree_share   |      0.13618 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      3 | etype_user_receives_reply_weight_share          |      0.09328 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      4 | etype_user_receives_reply_degree_share          |      0.09186 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      5 | log_total_degree                                |      0.06423 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      6 | total_weight                                    |      0.06401 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      7 | is_isolated                                     |      0.05889 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      8 | log_component_size                              |      0.05631 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |      9 | has_any_edge                                    |      0.05445 | feature           |
| label_scarcity_100pct         | GraphStats_RF  |     10 | total_degree                                    |      0.04293 | feature           |
| lofo_g1                       | GraphStats_EBM |      1 | receive_rate                                    |      1.28632 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      2 | etype_user_comments_on_user_post_degree_share   |      1.13223 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      3 | total_in_weight                                 |      1.12804 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      4 | etype_user_comments_on_user_post_weight_share   |      1.11587 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      5 | total_in_degree                                 |      1.05733 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      6 | etype_user_comments_on_user_post_in_degree      |      0.85842 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      7 | etype_user_receives_reply_degree_share          |      0.52086 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      8 | etype_user_receives_reply_weight_share          |      0.49692 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |      9 | total_in_weight & receive_rate                  |      0.49478 | ebm_term          |
| lofo_g1                       | GraphStats_EBM |     10 | comment1_rate                                   |      0.46208 | ebm_term          |
| lofo_g1                       | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      2.18212 | feature           |
| lofo_g1                       | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      2.18212 | feature           |
| lofo_g1                       | GraphStats_LR  |      3 | direction_balance                               |      1.87731 | feature           |
| lofo_g1                       | GraphStats_LR  |      4 | reciprocal_share                                |      1.61224 | feature           |
| lofo_g1                       | GraphStats_LR  |      5 | avg_in_weight                                   |      1.19483 | feature           |
| lofo_g1                       | GraphStats_LR  |      6 | etype_user_receives_reply_degree_share          |      1.05848 | feature           |
| lofo_g1                       | GraphStats_LR  |      7 | etype_user_receives_reply_weight_share          |      1.05848 | feature           |
| lofo_g1                       | GraphStats_LR  |      8 | etype_user_replies_to_user_comment_out_degree   |      0.98689 | feature           |
| lofo_g1                       | GraphStats_LR  |      9 | etype_user_replies_to_user_comment_weight_share |      0.75449 | feature           |
| lofo_g1                       | GraphStats_LR  |     10 | etype_user_replies_to_user_comment_degree_share |      0.75449 | feature           |
| lofo_g1                       | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.15293 | feature           |
| lofo_g1                       | GraphStats_RF  |      2 | etype_user_comments_on_user_post_degree_share   |      0.13302 | feature           |
| lofo_g1                       | GraphStats_RF  |      3 | etype_user_receives_reply_degree_share          |      0.08847 | feature           |
| lofo_g1                       | GraphStats_RF  |      4 | etype_user_receives_reply_weight_share          |      0.07936 | feature           |
| lofo_g1                       | GraphStats_RF  |      5 | total_weight                                    |      0.07886 | feature           |
| lofo_g1                       | GraphStats_RF  |      6 | log_total_degree                                |      0.07509 | feature           |
| lofo_g1                       | GraphStats_RF  |      7 | log_component_size                              |      0.06782 | feature           |
| lofo_g1                       | GraphStats_RF  |      8 | has_any_edge                                    |      0.06323 | feature           |
| lofo_g1                       | GraphStats_RF  |      9 | is_isolated                                     |      0.05404 | feature           |
| lofo_g1                       | GraphStats_RF  |     10 | total_degree                                    |      0.04372 | feature           |
| lofo_g2                       | GraphStats_EBM |      1 | etype_user_comments_on_user_post_degree_share   |      0.67185 | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      2 | etype_user_comments_on_user_post_weight_share   |      0.65791 | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      3 | direction_balance                               |      0.6497  | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      4 | total_in_weight                                 |      0.61194 | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      5 | total_in_degree                                 |      0.6032  | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      6 | receive_rate                                    |      0.55211 | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      7 | etype_user_replies_to_user_comment_in_degree    |      0.52418 | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      8 | etype_user_comments_on_user_post_out_degree     |      0.50587 | ebm_term          |
| lofo_g2                       | GraphStats_EBM |      9 | reciprocal_share & clustering                   |      0.50252 | ebm_term          |
| lofo_g2                       | GraphStats_EBM |     10 | etype_user_comments_on_user_post_in_degree      |      0.42465 | ebm_term          |
| lofo_g2                       | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      1.97431 | feature           |
| lofo_g2                       | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      1.97431 | feature           |
| lofo_g2                       | GraphStats_LR  |      3 | direction_balance                               |      1.96672 | feature           |
| lofo_g2                       | GraphStats_LR  |      4 | reciprocal_share                                |      1.58241 | feature           |
| lofo_g2                       | GraphStats_LR  |      5 | etype_user_receives_reply_degree_share          |      0.95844 | feature           |
| lofo_g2                       | GraphStats_LR  |      6 | etype_user_receives_reply_weight_share          |      0.95844 | feature           |
| lofo_g2                       | GraphStats_LR  |      7 | receive_rate                                    |      0.8062  | feature           |
| lofo_g2                       | GraphStats_LR  |      8 | etype_user_comments_on_user_post_out_degree     |      0.79984 | feature           |
| lofo_g2                       | GraphStats_LR  |      9 | avg_in_weight                                   |      0.79168 | feature           |
| lofo_g2                       | GraphStats_LR  |     10 | etype_user_replies_to_user_comment_weight_share |      0.74003 | feature           |
| lofo_g2                       | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.14956 | feature           |
| lofo_g2                       | GraphStats_RF  |      2 | etype_user_comments_on_user_post_degree_share   |      0.13532 | feature           |
| lofo_g2                       | GraphStats_RF  |      3 | etype_user_receives_reply_weight_share          |      0.08871 | feature           |
| lofo_g2                       | GraphStats_RF  |      4 | etype_user_receives_reply_degree_share          |      0.0779  | feature           |
| lofo_g2                       | GraphStats_RF  |      5 | log_total_degree                                |      0.07037 | feature           |
| lofo_g2                       | GraphStats_RF  |      6 | total_weight                                    |      0.0679  | feature           |
| lofo_g2                       | GraphStats_RF  |      7 | is_isolated                                     |      0.06755 | feature           |
| lofo_g2                       | GraphStats_RF  |      8 | has_any_edge                                    |      0.0605  | feature           |
| lofo_g2                       | GraphStats_RF  |      9 | log_component_size                              |      0.05996 | feature           |
| lofo_g2                       | GraphStats_RF  |     10 | total_degree                                    |      0.04753 | feature           |
| lofo_g3                       | GraphStats_EBM |      1 | etype_user_comments_on_user_post_degree_share   |      0.65602 | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      2 | etype_user_comments_on_user_post_weight_share   |      0.64451 | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      3 | direction_balance                               |      0.6119  | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      4 | receive_rate                                    |      0.58811 | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      5 | total_in_degree                                 |      0.56073 | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      6 | etype_user_comments_on_user_post_in_degree      |      0.5485  | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      7 | reciprocal_share & clustering                   |      0.46869 | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      8 | total_in_weight                                 |      0.45218 | ebm_term          |
| lofo_g3                       | GraphStats_EBM |      9 | comment1_rate                                   |      0.43772 | ebm_term          |
| lofo_g3                       | GraphStats_EBM |     10 | post_rate                                       |      0.4282  | ebm_term          |
| lofo_g3                       | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      2.05158 | feature           |
| lofo_g3                       | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      2.05158 | feature           |
| lofo_g3                       | GraphStats_LR  |      3 | direction_balance                               |      1.90777 | feature           |
| lofo_g3                       | GraphStats_LR  |      4 | reciprocal_share                                |      1.40319 | feature           |
| lofo_g3                       | GraphStats_LR  |      5 | etype_user_receives_reply_degree_share          |      0.99491 | feature           |
| lofo_g3                       | GraphStats_LR  |      6 | etype_user_receives_reply_weight_share          |      0.99491 | feature           |
| lofo_g3                       | GraphStats_LR  |      7 | etype_user_comments_on_user_post_out_degree     |      0.95817 | feature           |
| lofo_g3                       | GraphStats_LR  |      8 | receive_rate                                    |      0.85025 | feature           |
| lofo_g3                       | GraphStats_LR  |      9 | avg_in_weight                                   |      0.82145 | feature           |
| lofo_g3                       | GraphStats_LR  |     10 | etype_user_replies_to_user_comment_out_degree   |      0.75046 | feature           |
| lofo_g3                       | GraphStats_RF  |      1 | etype_user_comments_on_user_post_weight_share   |      0.14689 | feature           |
| lofo_g3                       | GraphStats_RF  |      2 | etype_user_comments_on_user_post_degree_share   |      0.13494 | feature           |
| lofo_g3                       | GraphStats_RF  |      3 | etype_user_receives_reply_weight_share          |      0.07967 | feature           |
| lofo_g3                       | GraphStats_RF  |      4 | etype_user_receives_reply_degree_share          |      0.07799 | feature           |
| lofo_g3                       | GraphStats_RF  |      5 | total_weight                                    |      0.07462 | feature           |
| lofo_g3                       | GraphStats_RF  |      6 | log_total_degree                                |      0.07098 | feature           |
| lofo_g3                       | GraphStats_RF  |      7 | is_isolated                                     |      0.06876 | feature           |
| lofo_g3                       | GraphStats_RF  |      8 | log_component_size                              |      0.06678 | feature           |
| lofo_g3                       | GraphStats_RF  |      9 | has_any_edge                                    |      0.05866 | feature           |
| lofo_g3                       | GraphStats_RF  |     10 | total_degree                                    |      0.05022 | feature           |
| original_only_zero_shot       | GraphStats_EBM |      1 | avg_in_weight                                   |      1.23107 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      2 | avg_out_weight                                  |      1.22061 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      3 | has_any_edge                                    |      1.20565 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      4 | component_size                                  |      1.20549 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      5 | is_isolated                                     |      1.20342 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      6 | log_component_size                              |      1.20217 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      7 | is_isolated & log_pagerank                      |      1.02173 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      8 | is_isolated & pagerank                          |      1.02173 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |      9 | log_component_size & log_pagerank               |      1.02125 | ebm_term          |
| original_only_zero_shot       | GraphStats_EBM |     10 | is_isolated & clustering                        |      1.01821 | ebm_term          |
| original_only_zero_shot       | GraphStats_LR  |      1 | etype_user_comments_on_user_post_degree_share   |      0.50778 | feature           |
| original_only_zero_shot       | GraphStats_LR  |      2 | etype_user_comments_on_user_post_weight_share   |      0.50778 | feature           |
| original_only_zero_shot       | GraphStats_LR  |      3 | has_any_edge                                    |      0.50062 | feature           |
| original_only_zero_shot       | GraphStats_LR  |      4 | is_isolated                                     |      0.50062 | feature           |
| original_only_zero_shot       | GraphStats_LR  |      5 | etype_user_receives_reply_degree_share          |      0.49946 | feature           |
| original_only_zero_shot       | GraphStats_LR  |      6 | etype_user_receives_reply_weight_share          |      0.49946 | feature           |
| original_only_zero_shot       | GraphStats_LR  |      7 | log_component_size                              |      0.4726  | feature           |
| original_only_zero_shot       | GraphStats_LR  |      8 | component_size                                  |      0.46937 | feature           |
| original_only_zero_shot       | GraphStats_LR  |      9 | avg_in_weight                                   |      0.27635 | feature           |
| original_only_zero_shot       | GraphStats_LR  |     10 | log_total_weight                                |      0.26006 | feature           |
| original_only_zero_shot       | GraphStats_RF  |      1 | is_isolated                                     |      0.11335 | feature           |
| original_only_zero_shot       | GraphStats_RF  |      2 | log_total_degree                                |      0.1114  | feature           |
| original_only_zero_shot       | GraphStats_RF  |      3 | total_weight                                    |      0.11001 | feature           |
| original_only_zero_shot       | GraphStats_RF  |      4 | log_component_size                              |      0.09333 | feature           |
| original_only_zero_shot       | GraphStats_RF  |      5 | log_total_weight                                |      0.08744 | feature           |
| original_only_zero_shot       | GraphStats_RF  |      6 | has_any_edge                                    |      0.08478 | feature           |
| original_only_zero_shot       | GraphStats_RF  |      7 | etype_user_receives_reply_weight_share          |      0.0846  | feature           |
| original_only_zero_shot       | GraphStats_RF  |      8 | total_degree                                    |      0.07801 | feature           |
| original_only_zero_shot       | GraphStats_RF  |      9 | etype_user_receives_reply_degree_share          |      0.07738 | feature           |
| original_only_zero_shot       | GraphStats_RF  |     10 | component_size                                  |      0.05667 | feature           |
| time_forward                  | GraphStats_EBM |      1 | etype_user_receives_reply_weight_share          |      3.14569 | ebm_term          |
| time_forward                  | GraphStats_EBM |      2 | etype_user_receives_reply_degree_share          |      3.00794 | ebm_term          |
| time_forward                  | GraphStats_EBM |      3 | receive_rate                                    |      1.32076 | ebm_term          |
| time_forward                  | GraphStats_EBM |      4 | etype_user_replies_to_user_comment_out_degree   |      1.07051 | ebm_term          |
| time_forward                  | GraphStats_EBM |      5 | clustering                                      |      0.80057 | ebm_term          |
| time_forward                  | GraphStats_EBM |      6 | direction_balance                               |      0.75761 | ebm_term          |
| time_forward                  | GraphStats_EBM |      7 | reciprocal_share                                |      0.74082 | ebm_term          |
| time_forward                  | GraphStats_EBM |      8 | etype_user_comments_on_user_post_degree_share   |      0.72371 | ebm_term          |
| time_forward                  | GraphStats_EBM |      9 | avg_out_weight                                  |      0.70835 | ebm_term          |
| time_forward                  | GraphStats_EBM |     10 | etype_user_comments_on_user_post_weight_share   |      0.70312 | ebm_term          |
| time_forward                  | GraphStats_LR  |      1 | direction_balance                               |      1.23236 | feature           |
| time_forward                  | GraphStats_LR  |      2 | etype_user_receives_reply_degree_share          |      0.99499 | feature           |
| time_forward                  | GraphStats_LR  |      3 | etype_user_receives_reply_weight_share          |      0.99499 | feature           |
| time_forward                  | GraphStats_LR  |      4 | etype_user_comments_on_user_post_out_degree     |      0.96117 | feature           |
| time_forward                  | GraphStats_LR  |      5 | etype_user_comments_on_user_post_weight_share   |      0.93727 | feature           |
| time_forward                  | GraphStats_LR  |      6 | etype_user_comments_on_user_post_degree_share   |      0.93727 | feature           |
| time_forward                  | GraphStats_LR  |      7 | etype_user_replies_to_user_comment_out_degree   |      0.89775 | feature           |
| time_forward                  | GraphStats_LR  |      8 | etype_user_comments_on_user_post_in_degree      |      0.76582 | feature           |
| time_forward                  | GraphStats_LR  |      9 | reciprocal_share                                |      0.58349 | feature           |
| time_forward                  | GraphStats_LR  |     10 | unique_target_authors                           |      0.58325 | feature           |
| time_forward                  | GraphStats_RF  |      1 | etype_user_receives_reply_weight_share          |      0.18302 | feature           |
| time_forward                  | GraphStats_RF  |      2 | etype_user_receives_reply_degree_share          |      0.15147 | feature           |
| time_forward                  | GraphStats_RF  |      3 | reciprocal_share                                |      0.1224  | feature           |
| time_forward                  | GraphStats_RF  |      4 | comment2_rate                                   |      0.0734  | feature           |
| time_forward                  | GraphStats_RF  |      5 | total_out_weight                                |      0.05906 | feature           |
| time_forward                  | GraphStats_RF  |      6 | etype_user_receives_reply_out_degree            |      0.05452 | feature           |
| time_forward                  | GraphStats_RF  |      7 | total_out_degree                                |      0.04501 | feature           |
| time_forward                  | GraphStats_RF  |      8 | pagerank                                        |      0.03839 | feature           |
| time_forward                  | GraphStats_RF  |      9 | log_total_degree                                |      0.03445 | feature           |
| time_forward                  | GraphStats_RF  |     10 | log_pagerank                                    |      0.03009 | feature           |


## 结论占位

本报告骨架已经能汇总 smoke dataset 的 GraphStats、R-GCN 与 GraphSAGE 输出。正式实验应补齐全量 family matrix、label scarcity 曲线、time-forward 切分和跨模态比较，并在 shortcut 审计中重点检查零入边、高方向性、低互惠和 bot-only component。
