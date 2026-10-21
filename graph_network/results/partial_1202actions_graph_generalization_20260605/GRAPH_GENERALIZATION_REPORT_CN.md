# 图数据泛化性研究报告

## 研究口径

本报告只使用图结构特征，不使用评论文本。输入数据是 partial run，因此结论应表述为 `partial_1202actions_stopped_live` 上的图结构泛化实验。

- 数据目录：`graph_network/outputs/partial_1202actions_stopped_live_20260604`
- run_name：`full_3family_150bot_24window_live`
- generated bots：`150`
- generated posts/comments：`258` / `790`
- user nodes / user-user edges：`3057` / `54067`
- GraphStats feature count：`39`

## 核心结论

1. **原始 BotSim 图检测器不能 zero-shot 泛化到新 G-family。** GraphStats_LR 在 `original_only_zero_shot` 上的 bot recall 为 `0.00`；GNN 参考结果为 GraphSAGE=0.00, RGCN=0.00。
2. **加入任意两个新 family 后，留一 family 的检出率快速恢复。** GraphStats_LR 的 LOFO 平均 recall 为 `0.93`，G1/G2/G3 分别约 `G1 population=0.88, G2 anchor=0.92, G3 interactive=0.98`。
3. **逐步课程学习显示明显恢复曲线。** GraphStats_LR 在 step 1 平均 recall `0.00`，step 2 平均 `0.85`，step 3 平均 `0.96`。
4. **当前 shortcut 不是最粗糙的孤立/零入边问题。** G1/G2/G3 的 zero-inbound、isolated、bot-only component share 接近 0；模型更多依赖互动方向、互惠、一级/二级回复边类型占比等行为结构。

## Zero-shot：原始图检测器能否检出新 family

| model               | backend      |   bot_count |   bot_recall |   bot_precision |   bot_f1 |   bot_mean_probability |
|:--------------------|:-------------|------------:|-------------:|----------------:|---------:|-----------------------:|
| GraphStats_LR       | sklearn      |         150 |            0 |               0 |        0 |                 0.0001 |
| GraphStats_RF       | sklearn      |         150 |            0 |               0 |        0 |                 0.0004 |
| GraphStats_EBM      | interpret    |         150 |            0 |               0 |        0 |                 0      |
| GraphSAGE_Inductive | pyg_sageconv |         150 |            0 |               0 |        0 |                 0      |
| RGCN_Transductive   | pyg_rgcnconv |         150 |            0 |               0 |        0 |                 0      |

![zero-shot recall](graph_network/results/partial_1202actions_graph_generalization_20260605/figures/zero_shot_recall.png)

解释：这里训练集只有原始 human + 原始 bot，测试集是 G1/G2/G3。若 recall 接近 0，说明原始 BotSim 的图结构判别边界不能直接迁移到我们新构造的三种社交模拟 family。

## LOFO：留出一个 family 的跨家族泛化

| split_name   | model               | backend      | family                    |   bot_count |   bot_recall |   bot_mean_probability |
|:-------------|:--------------------|:-------------|:--------------------------|------------:|-------------:|-----------------------:|
| lofo_g1      | GraphStats_LR       | sklearn      | G1_population_calibrated  |          50 |         0.88 |                 0.8498 |
| lofo_g2      | GraphStats_LR       | sklearn      | G2_anchor_behavior        |          50 |         0.92 |                 0.9186 |
| lofo_g3      | GraphStats_LR       | sklearn      | G3_contextual_interactive |          50 |         0.98 |                 0.9383 |
| lofo_g1      | GraphStats_RF       | sklearn      | G1_population_calibrated  |          50 |         0.92 |                 0.8542 |
| lofo_g2      | GraphStats_RF       | sklearn      | G2_anchor_behavior        |          50 |         0.98 |                 0.9327 |
| lofo_g3      | GraphStats_RF       | sklearn      | G3_contextual_interactive |          50 |         0.98 |                 0.91   |
| lofo_g1      | GraphStats_EBM      | interpret    | G1_population_calibrated  |          50 |         0.94 |                 0.9352 |
| lofo_g2      | GraphStats_EBM      | interpret    | G2_anchor_behavior        |          50 |         0.98 |                 0.9718 |
| lofo_g3      | GraphStats_EBM      | interpret    | G3_contextual_interactive |          50 |         0.98 |                 0.9803 |
| lofo_g1      | GraphSAGE_Inductive | pyg_sageconv | G1_population_calibrated  |          50 |         0.98 |                 0.9582 |
| lofo_g2      | GraphSAGE_Inductive | pyg_sageconv | G2_anchor_behavior        |          50 |         0.96 |                 0.9724 |
| lofo_g3      | GraphSAGE_Inductive | pyg_sageconv | G3_contextual_interactive |          50 |         1    |                 0.9955 |
| lofo_g1      | RGCN_Transductive   | pyg_rgcnconv | G1_population_calibrated  |          50 |         0.96 |                 0.96   |
| lofo_g2      | RGCN_Transductive   | pyg_rgcnconv | G2_anchor_behavior        |          50 |         0.96 |                 0.9482 |
| lofo_g3      | RGCN_Transductive   | pyg_rgcnconv | G3_contextual_interactive |          50 |         0.98 |                 0.9752 |

![LOFO recall matrix](graph_network/results/partial_1202actions_graph_generalization_20260605/figures/lofo_recall_matrix.png)

解释：只要训练集中包含其它新 family，检测器对 unseen family 的 recall 迅速升高。这说明 G1/G2/G3 之间存在可迁移的图结构共性；这既是 family benchmark 的有效信号，也提示还存在共享生成机制痕迹。

## Incremental curriculum：逐步加入 family 的恢复曲线

| curriculum_order                                                      |   curriculum_step | add_family_after_evaluation   |   bot_count |   bot_recall |   bot_mean_probability |
|:----------------------------------------------------------------------|------------------:|:------------------------------|------------:|-------------:|-----------------------:|
| G1_population_calibrated>G2_anchor_behavior>G3_contextual_interactive |                 1 | G1_population_calibrated      |          50 |         0    |                 0.0002 |
| G1_population_calibrated>G2_anchor_behavior>G3_contextual_interactive |                 2 | G2_anchor_behavior            |          50 |         0.88 |                 0.8738 |
| G1_population_calibrated>G2_anchor_behavior>G3_contextual_interactive |                 3 | G3_contextual_interactive     |          50 |         0.98 |                 0.9355 |
| G2_anchor_behavior>G1_population_calibrated>G3_contextual_interactive |                 1 | G2_anchor_behavior            |          50 |         0    |                 0.0001 |
| G2_anchor_behavior>G1_population_calibrated>G3_contextual_interactive |                 2 | G1_population_calibrated      |          50 |         0.88 |                 0.8173 |
| G2_anchor_behavior>G1_population_calibrated>G3_contextual_interactive |                 3 | G3_contextual_interactive     |          50 |         0.98 |                 0.9473 |
| G3_contextual_interactive>G1_population_calibrated>G2_anchor_behavior |                 1 | G3_contextual_interactive     |          50 |         0    |                 0.0003 |
| G3_contextual_interactive>G1_population_calibrated>G2_anchor_behavior |                 2 | G1_population_calibrated      |          50 |         0.8  |                 0.8082 |
| G3_contextual_interactive>G1_population_calibrated>G2_anchor_behavior |                 3 | G2_anchor_behavior            |          50 |         0.92 |                 0.9059 |

![incremental curriculum recall](graph_network/results/partial_1202actions_graph_generalization_20260605/figures/incremental_curriculum_recall.png)

解释：step 1 只用原始数据训练，recall 为 0；step 2 在加入第一个 generated family 后测试第二个 family，recall 已恢复到约 0.80-0.88；step 3 继续加入第二个 family 后测试第三个 family，recall 约 0.92-0.98。

## Label scarcity：少量 generated label 是否足够恢复

| split_name            | model               | backend      |   bot_count |   bot_recall |   bot_mean_probability |
|:----------------------|:--------------------|:-------------|------------:|-------------:|-----------------------:|
| label_scarcity_001pct | GraphStats_LR       | sklearn      |         147 |       0.5238 |                 0.5121 |
| label_scarcity_005pct | GraphStats_LR       | sklearn      |         144 |       0.6806 |                 0.6582 |
| label_scarcity_010pct | GraphStats_LR       | sklearn      |         135 |       0.8074 |                 0.8181 |
| label_scarcity_025pct | GraphStats_LR       | sklearn      |         114 |       0.8333 |                 0.8251 |
| label_scarcity_050pct | GraphStats_LR       | sklearn      |          75 |       0.92   |                 0.8633 |
| label_scarcity_001pct | GraphStats_EBM      | interpret    |         147 |       0.3469 |                 0.3452 |
| label_scarcity_005pct | GraphStats_EBM      | interpret    |         144 |       0.8611 |                 0.8424 |
| label_scarcity_010pct | GraphStats_EBM      | interpret    |         135 |       0.9407 |                 0.9407 |
| label_scarcity_025pct | GraphStats_EBM      | interpret    |         114 |       0.9123 |                 0.9108 |
| label_scarcity_050pct | GraphStats_EBM      | interpret    |          75 |       0.9467 |                 0.9467 |
| label_scarcity_001pct | GraphSAGE_Inductive | pyg_sageconv |         147 |       0.0272 |                 0.0789 |
| label_scarcity_005pct | GraphSAGE_Inductive | pyg_sageconv |         144 |       0.0069 |                 0.0435 |
| label_scarcity_010pct | GraphSAGE_Inductive | pyg_sageconv |         135 |       0      |                 0.0033 |
| label_scarcity_025pct | GraphSAGE_Inductive | pyg_sageconv |         114 |       0.9649 |                 0.9535 |
| label_scarcity_050pct | GraphSAGE_Inductive | pyg_sageconv |          75 |       0.9867 |                 0.9743 |
| label_scarcity_001pct | RGCN_Transductive   | pyg_rgcnconv |         147 |       0.7415 |                 0.7481 |
| label_scarcity_005pct | RGCN_Transductive   | pyg_rgcnconv |         144 |       0.7361 |                 0.7448 |
| label_scarcity_010pct | RGCN_Transductive   | pyg_rgcnconv |         135 |       0.9407 |                 0.9394 |
| label_scarcity_025pct | RGCN_Transductive   | pyg_rgcnconv |         114 |       0.8947 |                 0.8979 |
| label_scarcity_050pct | RGCN_Transductive   | pyg_rgcnconv |          75 |       0.8933 |                 0.9005 |

![label scarcity curve](graph_network/results/partial_1202actions_graph_generalization_20260605/figures/label_scarcity_curve.png)

解释：GraphStats_LR 只加入 1% generated bot 标签时 recall 约 0.52，5% 时约 0.68，10% 时约 0.81。说明图结构检测对少量新家族标注很敏感，恢复速度较快。

## Shortcut attribution：模型抓到了什么

### Shortcut 审计

| family                    |   size |   bot_count |   zero_inbound_share |   over_directional_share |   low_reciprocity_share |   isolated_share |   bot_only_component_share |
|:--------------------------|-------:|------------:|---------------------:|-------------------------:|------------------------:|-----------------:|---------------------------:|
| G1_population_calibrated  |     50 |          50 |                 0.02 |                     0.02 |                    0.02 |                0 |                          0 |
| G2_anchor_behavior        |     50 |          50 |                 0    |                     0    |                    0    |                0 |                          0 |
| G3_contextual_interactive |     50 |          50 |                 0    |                     0    |                    0    |                0 |                          0 |

![shortcut audit](graph_network/results/partial_1202actions_graph_generalization_20260605/figures/shortcut_audit.png)

### GraphStats_LR top features

| split_name                    |   rank | feature                                         |   importance |
|:------------------------------|-------:|:------------------------------------------------|-------------:|
| incremental_curriculum_order1 |      1 | etype_user_comments_on_user_post_degree_share   |       0.5489 |
| incremental_curriculum_order1 |      2 | etype_user_comments_on_user_post_weight_share   |       0.5489 |
| incremental_curriculum_order1 |      3 | has_any_edge                                    |       0.5407 |
| incremental_curriculum_order1 |      4 | is_isolated                                     |       0.5407 |
| incremental_curriculum_order1 |      5 | etype_user_receives_reply_weight_share          |       0.5376 |
| incremental_curriculum_order2 |      1 | etype_user_comments_on_user_post_degree_share   |       0.5117 |
| incremental_curriculum_order2 |      2 | etype_user_comments_on_user_post_weight_share   |       0.5117 |
| incremental_curriculum_order2 |      3 | has_any_edge                                    |       0.5041 |
| incremental_curriculum_order2 |      4 | is_isolated                                     |       0.5041 |
| incremental_curriculum_order2 |      5 | etype_user_receives_reply_weight_share          |       0.5013 |
| incremental_curriculum_order3 |      1 | etype_user_comments_on_user_post_degree_share   |       0.5064 |
| incremental_curriculum_order3 |      2 | etype_user_comments_on_user_post_weight_share   |       0.5064 |
| incremental_curriculum_order3 |      3 | is_isolated                                     |       0.5039 |
| incremental_curriculum_order3 |      4 | has_any_edge                                    |       0.5039 |
| incremental_curriculum_order3 |      5 | etype_user_receives_reply_weight_share          |       0.5013 |
| label_scarcity_001pct         |      1 | reciprocal_share                                |       0.905  |
| label_scarcity_001pct         |      2 | etype_user_replies_to_user_comment_degree_share |       0.6043 |
| label_scarcity_001pct         |      3 | etype_user_replies_to_user_comment_weight_share |       0.6043 |
| label_scarcity_001pct         |      4 | etype_user_comments_on_user_post_degree_share   |       0.5811 |
| label_scarcity_001pct         |      5 | etype_user_comments_on_user_post_weight_share   |       0.5811 |
| label_scarcity_005pct         |      1 | etype_user_comments_on_user_post_degree_share   |       0.9297 |
| label_scarcity_005pct         |      2 | etype_user_comments_on_user_post_weight_share   |       0.9297 |
| label_scarcity_005pct         |      3 | reciprocal_share                                |       0.8356 |
| label_scarcity_005pct         |      4 | etype_user_replies_to_user_comment_weight_share |       0.6422 |
| label_scarcity_005pct         |      5 | etype_user_replies_to_user_comment_degree_share |       0.6422 |
| label_scarcity_010pct         |      1 | reciprocal_share                                |       1.4509 |
| label_scarcity_010pct         |      2 | etype_user_replies_to_user_comment_weight_share |       1.3181 |
| label_scarcity_010pct         |      3 | etype_user_replies_to_user_comment_degree_share |       1.3181 |
| label_scarcity_010pct         |      4 | etype_user_comments_on_user_post_weight_share   |       1.2264 |
| label_scarcity_010pct         |      5 | etype_user_comments_on_user_post_degree_share   |       1.2264 |
| label_scarcity_025pct         |      1 | etype_user_comments_on_user_post_degree_share   |       1.7584 |
| label_scarcity_025pct         |      2 | etype_user_comments_on_user_post_weight_share   |       1.7584 |
| label_scarcity_025pct         |      3 | reciprocal_share                                |       1.6542 |
| label_scarcity_025pct         |      4 | etype_user_replies_to_user_comment_weight_share |       1.1926 |
| label_scarcity_025pct         |      5 | etype_user_replies_to_user_comment_degree_share |       1.1926 |
| label_scarcity_050pct         |      1 | direction_balance                               |       2.0707 |
| label_scarcity_050pct         |      2 | etype_user_comments_on_user_post_degree_share   |       1.9593 |
| label_scarcity_050pct         |      3 | etype_user_comments_on_user_post_weight_share   |       1.9593 |
| label_scarcity_050pct         |      4 | reciprocal_share                                |       1.3436 |
| label_scarcity_050pct         |      5 | avg_in_weight                                   |       1.08   |
| label_scarcity_100pct         |      1 | etype_user_comments_on_user_post_degree_share   |       2.3144 |
| label_scarcity_100pct         |      2 | etype_user_comments_on_user_post_weight_share   |       2.3144 |
| label_scarcity_100pct         |      3 | direction_balance                               |       2.2195 |
| label_scarcity_100pct         |      4 | reciprocal_share                                |       1.7565 |
| label_scarcity_100pct         |      5 | etype_user_receives_reply_degree_share          |       1.287  |
| lofo_g1                       |      1 | etype_user_comments_on_user_post_degree_share   |       2.1821 |
| lofo_g1                       |      2 | etype_user_comments_on_user_post_weight_share   |       2.1821 |
| lofo_g1                       |      3 | direction_balance                               |       1.8773 |
| lofo_g1                       |      4 | reciprocal_share                                |       1.6122 |
| lofo_g1                       |      5 | avg_in_weight                                   |       1.1948 |

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

