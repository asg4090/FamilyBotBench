# 图数据泛化性研究报告

## 研究口径

本报告只使用图结构特征，不使用评论文本。输入数据是 partial run，因此结论应表述为 `partial_1202actions_stopped_live` 上的图结构泛化实验。

- 数据目录：`graph_network/outputs/partial_plus_g4_slot_conditioned_20260606`
- run_name：`partial_plus_g4_slot_conditioned_20260606`
- generated bots：`200`
- generated posts/comments：`339` / `1068`
- user nodes / user-user edges：`3107` / `55816`
- GraphStats feature count：`39`

## 核心结论

1. **原始 BotSim 图检测器不能 zero-shot 泛化到新 G-family。** GraphStats_LR 在 `original_only_zero_shot` 上的 bot recall 为 `0.00`；GNN 参考结果为 GraphSAGE=0.00, RGCN=0.00。
2. **加入其它新 family 后，留一 family 的检出率会发生恢复。** GraphStats_LR 的 LOFO 平均 recall 为 `0.69`，各 family 分别约 `G1 population=0.84, G2 anchor=0.94, G3 interactive=0.98, G4 slot=0.00`。
3. **逐步课程学习显示恢复曲线。** GraphStats_LR 分步骤平均 recall：`step 1=0.00, step 2=0.82, step 3=0.95, step 4=0.00`。
4. **shortcut 需要逐 family 判断。** 重点看 zero-inbound、isolated、bot-only component、方向不平衡、互惠和一级/二级回复边类型占比。

## Zero-shot：原始图检测器能否检出新 family

| model               | backend      |   bot_count |   bot_recall |   bot_precision |   bot_f1 |   bot_mean_probability |
|:--------------------|:-------------|------------:|-------------:|----------------:|---------:|-----------------------:|
| GraphStats_LR       | sklearn      |         200 |            0 |               0 |        0 |                 0.0001 |
| GraphStats_RF       | sklearn      |         200 |            0 |               0 |        0 |                 0.0001 |
| GraphStats_EBM      | interpret    |         200 |            0 |               0 |        0 |                 0      |
| GraphSAGE_Inductive | pyg_sageconv |         200 |            0 |               0 |        0 |                 0      |
| RGCN_Transductive   | pyg_rgcnconv |         200 |            0 |               0 |        0 |                 0      |

![zero-shot recall](graph_network/results/partial_plus_g4_slot_conditioned_graph_generalization_20260606/figures/zero_shot_recall.png)

解释：这里训练集只有原始 human + 原始 bot，测试集是所有 generated G-family。若 recall 接近 0，说明原始 BotSim 的图结构判别边界不能直接迁移到新构造的社交模拟 family。

## LOFO：留出一个 family 的跨家族泛化

| split_name   | model               | backend      | family                    |   bot_count |   bot_recall |   bot_mean_probability |
|:-------------|:--------------------|:-------------|:--------------------------|------------:|-------------:|-----------------------:|
| lofo_g1      | GraphStats_LR       | sklearn      | G1_population_calibrated  |          50 |         0.84 |                 0.8334 |
| lofo_g2      | GraphStats_LR       | sklearn      | G2_anchor_behavior        |          50 |         0.94 |                 0.9203 |
| lofo_g3      | GraphStats_LR       | sklearn      | G3_contextual_interactive |          50 |         0.98 |                 0.9346 |
| lofo_g4      | GraphStats_LR       | sklearn      | G4_slot_conditioned       |          50 |         0    |                 0.033  |
| lofo_g1      | GraphStats_RF       | sklearn      | G1_population_calibrated  |          50 |         0.92 |                 0.8504 |
| lofo_g2      | GraphStats_RF       | sklearn      | G2_anchor_behavior        |          50 |         0.96 |                 0.9243 |
| lofo_g3      | GraphStats_RF       | sklearn      | G3_contextual_interactive |          50 |         0.98 |                 0.9292 |
| lofo_g4      | GraphStats_RF       | sklearn      | G4_slot_conditioned       |          50 |         0.94 |                 0.6412 |
| lofo_g1      | GraphStats_EBM      | interpret    | G1_population_calibrated  |          50 |         0.9  |                 0.8937 |
| lofo_g2      | GraphStats_EBM      | interpret    | G2_anchor_behavior        |          50 |         0.94 |                 0.9478 |
| lofo_g3      | GraphStats_EBM      | interpret    | G3_contextual_interactive |          50 |         0.98 |                 0.966  |
| lofo_g4      | GraphStats_EBM      | interpret    | G4_slot_conditioned       |          50 |         0.86 |                 0.7331 |
| lofo_g1      | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          50 |         0.94 |                 0.9441 |
| lofo_g2      | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          50 |         0.98 |                 0.9778 |
| lofo_g3      | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          50 |         1    |                 0.9988 |
| lofo_g4      | GraphSAGE_Inductive | pyg_sageconv | G4_slot_conditioned       |          50 |         0    |                 0.0037 |
| lofo_g1      | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          50 |         0.96 |                 0.9538 |
| lofo_g2      | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          50 |         0.96 |                 0.968  |
| lofo_g3      | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          50 |         1    |                 0.9892 |
| lofo_g4      | RGCN_Transductive   | pyg_rgcnconv | G4_slot_conditioned       |          50 |         0.04 |                 0.0386 |

![LOFO recall matrix](graph_network/results/partial_plus_g4_slot_conditioned_graph_generalization_20260606/figures/lofo_recall_matrix.png)

解释：只要训练集中包含其它新 family，检测器对 unseen family 的 recall 变化就反映跨家族迁移能力；如果迅速升高，说明不同 family 间存在共享图结构痕迹。

## Incremental curriculum：逐步加入 family 的恢复曲线

| curriculum_order                                                                          |   curriculum_step | add_family_after_evaluation   |   bot_count |   bot_recall |   bot_mean_probability |
|:------------------------------------------------------------------------------------------|------------------:|:------------------------------|------------:|-------------:|-----------------------:|
| G1_population_calibrated>G2_anchor_behavior>G3_contextual_interactive>G4_slot_conditioned |                 1 | G1_population_calibrated      |          50 |         0    |                 0.0002 |
| G1_population_calibrated>G2_anchor_behavior>G3_contextual_interactive>G4_slot_conditioned |                 2 | G2_anchor_behavior            |          50 |         0.86 |                 0.8703 |
| G1_population_calibrated>G2_anchor_behavior>G3_contextual_interactive>G4_slot_conditioned |                 3 | G3_contextual_interactive     |          50 |         0.98 |                 0.9394 |
| G1_population_calibrated>G2_anchor_behavior>G3_contextual_interactive>G4_slot_conditioned |                 4 | G4_slot_conditioned           |          50 |         0    |                 0.0352 |
| G2_anchor_behavior>G1_population_calibrated>G3_contextual_interactive>G4_slot_conditioned |                 1 | G2_anchor_behavior            |          50 |         0    |                 0.0001 |
| G2_anchor_behavior>G1_population_calibrated>G3_contextual_interactive>G4_slot_conditioned |                 2 | G1_population_calibrated      |          50 |         0.82 |                 0.7825 |
| G2_anchor_behavior>G1_population_calibrated>G3_contextual_interactive>G4_slot_conditioned |                 3 | G3_contextual_interactive     |          50 |         0.98 |                 0.9325 |
| G2_anchor_behavior>G1_population_calibrated>G3_contextual_interactive>G4_slot_conditioned |                 4 | G4_slot_conditioned           |          50 |         0    |                 0.0205 |
| G3_contextual_interactive>G1_population_calibrated>G2_anchor_behavior>G4_slot_conditioned |                 1 | G3_contextual_interactive     |          50 |         0    |                 0.0001 |
| G3_contextual_interactive>G1_population_calibrated>G2_anchor_behavior>G4_slot_conditioned |                 2 | G1_population_calibrated      |          50 |         0.78 |                 0.8052 |
| G3_contextual_interactive>G1_population_calibrated>G2_anchor_behavior>G4_slot_conditioned |                 3 | G2_anchor_behavior            |          50 |         0.9  |                 0.8968 |
| G3_contextual_interactive>G1_population_calibrated>G2_anchor_behavior>G4_slot_conditioned |                 4 | G4_slot_conditioned           |          50 |         0    |                 0.0273 |

![incremental curriculum recall](graph_network/results/partial_plus_g4_slot_conditioned_graph_generalization_20260606/figures/incremental_curriculum_recall.png)

解释：每一步都是先测试当前 family，再把它加入训练集。曲线越快上升，说明少量 family 经验越容易迁移到后续 unseen family。

## Label scarcity：少量 generated label 是否足够恢复

| split_name            | model               | backend      |   bot_count |   bot_recall |   bot_mean_probability |
|:----------------------|:--------------------|:-------------|------------:|-------------:|-----------------------:|
| label_scarcity_001pct | GraphStats_LR       | sklearn      |         196 |       0.5306 |                 0.523  |
| label_scarcity_005pct | GraphStats_LR       | sklearn      |         192 |       0.5    |                 0.4906 |
| label_scarcity_010pct | GraphStats_LR       | sklearn      |         180 |       0.7333 |                 0.7034 |
| label_scarcity_025pct | GraphStats_LR       | sklearn      |         152 |       0.8224 |                 0.7952 |
| label_scarcity_050pct | GraphStats_LR       | sklearn      |         100 |       0.93   |                 0.878  |
| label_scarcity_001pct | GraphStats_EBM      | interpret    |         196 |       0.8724 |                 0.8688 |
| label_scarcity_005pct | GraphStats_EBM      | interpret    |         192 |       0.7448 |                 0.7424 |
| label_scarcity_010pct | GraphStats_EBM      | interpret    |         180 |       0.9    |                 0.8842 |
| label_scarcity_025pct | GraphStats_EBM      | interpret    |         152 |       0.8947 |                 0.8988 |
| label_scarcity_050pct | GraphStats_EBM      | interpret    |         100 |       0.96   |                 0.9473 |
| label_scarcity_001pct | GraphSAGE_Inductive | pyg_sageconv |         196 |       0      |                 0.0002 |
| label_scarcity_005pct | GraphSAGE_Inductive | pyg_sageconv |         192 |       0.5    |                 0.4249 |
| label_scarcity_010pct | GraphSAGE_Inductive | pyg_sageconv |         180 |       0.7222 |                 0.7333 |
| label_scarcity_025pct | GraphSAGE_Inductive | pyg_sageconv |         152 |       0.7039 |                 0.7342 |
| label_scarcity_050pct | GraphSAGE_Inductive | pyg_sageconv |         100 |       0.98   |                 0.9465 |
| label_scarcity_001pct | RGCN_Transductive   | pyg_rgcnconv |         196 |       0      |                 0      |
| label_scarcity_005pct | RGCN_Transductive   | pyg_rgcnconv |         192 |       0      |                 0.0025 |
| label_scarcity_010pct | RGCN_Transductive   | pyg_rgcnconv |         180 |       0.7222 |                 0.7293 |
| label_scarcity_025pct | RGCN_Transductive   | pyg_rgcnconv |         152 |       0.7829 |                 0.8074 |
| label_scarcity_050pct | RGCN_Transductive   | pyg_rgcnconv |         100 |       0.79   |                 0.8043 |

![label scarcity curve](graph_network/results/partial_plus_g4_slot_conditioned_graph_generalization_20260606/figures/label_scarcity_curve.png)

解释：GraphStats_LR 的 label scarcity recall 为：`0.1%=0.53, 0.5%=0.50, 1.0%=0.73, 2.5%=0.82, 5.0%=0.93`。这表示只要少量 generated 标签进入训练，图结构检测就会开始恢复，但恢复速度取决于被采样到的 family。

## Shortcut attribution：模型抓到了什么

### Shortcut 审计

| family                    |   size |   bot_count |   zero_inbound_share |   over_directional_share |   low_reciprocity_share |   isolated_share |   bot_only_component_share |
|:--------------------------|-------:|------------:|---------------------:|-------------------------:|------------------------:|-----------------:|---------------------------:|
| G1_population_calibrated  |     50 |          50 |                 0.02 |                     0.02 |                    0.02 |                0 |                          0 |
| G2_anchor_behavior        |     50 |          50 |                 0    |                     0    |                    0    |                0 |                          0 |
| G3_contextual_interactive |     50 |          50 |                 0    |                     0    |                    0    |                0 |                          0 |
| G4_slot_conditioned       |     50 |          50 |                 0    |                     0    |                    0.74 |                0 |                          0 |

![shortcut audit](graph_network/results/partial_plus_g4_slot_conditioned_graph_generalization_20260606/figures/shortcut_audit.png)

### GraphStats_LR top features

| split_name                    |   rank | feature                                         |   importance |
|:------------------------------|-------:|:------------------------------------------------|-------------:|
| incremental_curriculum_order1 |      1 | etype_user_comments_on_user_post_degree_share   |       0.5164 |
| incremental_curriculum_order1 |      2 | etype_user_comments_on_user_post_weight_share   |       0.5164 |
| incremental_curriculum_order1 |      3 | is_isolated                                     |       0.5078 |
| incremental_curriculum_order1 |      4 | has_any_edge                                    |       0.5078 |
| incremental_curriculum_order1 |      5 | etype_user_receives_reply_weight_share          |       0.5077 |
| incremental_curriculum_order2 |      1 | etype_user_comments_on_user_post_degree_share   |       0.5117 |
| incremental_curriculum_order2 |      2 | etype_user_comments_on_user_post_weight_share   |       0.5117 |
| incremental_curriculum_order2 |      3 | has_any_edge                                    |       0.5064 |
| incremental_curriculum_order2 |      4 | is_isolated                                     |       0.5064 |
| incremental_curriculum_order2 |      5 | etype_user_receives_reply_degree_share          |       0.5048 |
| incremental_curriculum_order3 |      1 | etype_user_comments_on_user_post_degree_share   |       0.5114 |
| incremental_curriculum_order3 |      2 | etype_user_comments_on_user_post_weight_share   |       0.5114 |
| incremental_curriculum_order3 |      3 | is_isolated                                     |       0.505  |
| incremental_curriculum_order3 |      4 | has_any_edge                                    |       0.505  |
| incremental_curriculum_order3 |      5 | etype_user_receives_reply_degree_share          |       0.5032 |
| label_scarcity_001pct         |      1 | etype_user_comments_on_user_post_degree_share   |       1.6164 |
| label_scarcity_001pct         |      2 | etype_user_comments_on_user_post_weight_share   |       1.6164 |
| label_scarcity_001pct         |      3 | etype_user_replies_to_user_comment_in_degree    |       0.6803 |
| label_scarcity_001pct         |      4 | has_any_edge                                    |       0.5517 |
| label_scarcity_001pct         |      5 | is_isolated                                     |       0.5517 |
| label_scarcity_005pct         |      1 | etype_user_comments_on_user_post_weight_share   |       1.0254 |
| label_scarcity_005pct         |      2 | etype_user_comments_on_user_post_degree_share   |       1.0254 |
| label_scarcity_005pct         |      3 | reciprocal_share                                |       0.9102 |
| label_scarcity_005pct         |      4 | comment2_rate                                   |       0.5743 |
| label_scarcity_005pct         |      5 | etype_user_replies_to_user_comment_weight_share |       0.5616 |
| label_scarcity_010pct         |      1 | etype_user_comments_on_user_post_weight_share   |       2.144  |
| label_scarcity_010pct         |      2 | etype_user_comments_on_user_post_degree_share   |       2.144  |
| label_scarcity_010pct         |      3 | etype_user_receives_reply_weight_share          |       0.8202 |
| label_scarcity_010pct         |      4 | etype_user_receives_reply_degree_share          |       0.8202 |
| label_scarcity_010pct         |      5 | log_total_degree                                |       0.7828 |
| label_scarcity_025pct         |      1 | etype_user_comments_on_user_post_weight_share   |       2.8804 |
| label_scarcity_025pct         |      2 | etype_user_comments_on_user_post_degree_share   |       2.8804 |
| label_scarcity_025pct         |      3 | etype_user_receives_reply_degree_share          |       1.5972 |
| label_scarcity_025pct         |      4 | etype_user_receives_reply_weight_share          |       1.5972 |
| label_scarcity_025pct         |      5 | log_total_degree                                |       0.8139 |
| label_scarcity_050pct         |      1 | etype_user_comments_on_user_post_weight_share   |       2.8625 |
| label_scarcity_050pct         |      2 | etype_user_comments_on_user_post_degree_share   |       2.8625 |
| label_scarcity_050pct         |      3 | etype_user_receives_reply_degree_share          |       1.5394 |
| label_scarcity_050pct         |      4 | etype_user_receives_reply_weight_share          |       1.5394 |
| label_scarcity_050pct         |      5 | etype_user_receives_reply_out_degree            |       0.9317 |
| label_scarcity_100pct         |      1 | etype_user_comments_on_user_post_weight_share   |       3.168  |
| label_scarcity_100pct         |      2 | etype_user_comments_on_user_post_degree_share   |       3.168  |
| label_scarcity_100pct         |      3 | etype_user_receives_reply_degree_share          |       2.0113 |
| label_scarcity_100pct         |      4 | etype_user_receives_reply_weight_share          |       2.0113 |
| label_scarcity_100pct         |      5 | reciprocal_share                                |       0.9766 |
| lofo_g1                       |      1 | etype_user_comments_on_user_post_degree_share   |       3.1767 |
| lofo_g1                       |      2 | etype_user_comments_on_user_post_weight_share   |       3.1767 |
| lofo_g1                       |      3 | etype_user_receives_reply_degree_share          |       1.8509 |
| lofo_g1                       |      4 | etype_user_receives_reply_weight_share          |       1.8509 |
| lofo_g1                       |      5 | etype_user_receives_reply_out_degree            |       1.0458 |

解释：高权重特征集中在 `user_comments_on_user_post`、`user_replies_to_user_comment`、`user_receives_reply`、`reciprocal_share`、`direction_balance` 等互动结构上。它们不是文本特征，也不是 family/provider/model 这类元数据。

## GNN 口径说明

当前已安装 `torch_geometric`，GraphSAGE 使用 PyG `SAGEConv`，RGCN 使用 PyG `RGCNConv`。GraphSAGE 是 inductive 参考口径；RGCN 使用完整可见拓扑，属于 observed-transductive 参考结果。GraphStats_LR 仍是本报告主结论，因为它最可解释、最便于定位图结构 shortcut。

## 产物

- `zero_shot_summary.csv`
- `lofo_family_recall_matrix_long.csv`
- `label_scarcity_summary.csv`
- `curriculum_step_summary.csv`
- `top_lr_features_by_split.csv`
- `figures/*.png`

