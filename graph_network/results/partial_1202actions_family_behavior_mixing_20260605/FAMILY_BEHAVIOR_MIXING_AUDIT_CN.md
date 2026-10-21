# Family Behavior and Mixing Audit

## 研究口径

本报告只审计生成数据质量，不重新生成 bot，不做文本检测，也不训练 bot-vs-human 检测器。核心问题是：G1/G2/G3 是否真的形成不同社交行为，以及是否出现 family 抱团或明显偏离 human 图行为分布。

- 数据目录：`graph_network/outputs/partial_1202actions_stopped_live_20260604`
- 结果目录：`graph_network/results/partial_1202actions_family_behavior_mixing_20260605`
- 数据完整性 passed：`True`
- generated family counts：`{'G1_population_calibrated': 50, 'G2_anchor_behavior': 50, 'G3_contextual_interactive': 50}`

## 核心判断

1. 三个 family 的行为分布存在可见差异，最大差异来自 `post_count`，family 均值跨度约 `1.16`。
2. 未发现同 family 抱团：真实 same-family share=`0.277`，随机基线均值=`0.328`。
3. 存在明确环境方向性 shortcut：generated→human 边=`646`，generated↔generated 边=`1273`，但 human→generated 边=`0`。这说明 replay human 不会真实回到新 bot 内容下互动。
4. 没有严格 bot-only 社区，但存在 bot-heavy 社区：最高 bot_share=`0.955`，该社区 size=`156`、bot_count=`149`；family modularity=`0.029`，generated-only NMI=`0.013`。这表示不是 G1/G2/G3 各自抱团，而是 generated bot 整体容易集中。
5. G1/G2/G3 在图特征空间中不易区分：最佳模型 `Family_RF`，CV accuracy=`0.467`，macro-F1=`0.464`。
6. 存在明显 human 分布偏离：`G3 interactive` 中 `92.00%` 的 bot 至少一个关键图特征落在人类 1%/99% 极端区间。

## 1. Family 行为分布

| family                    |   post_count_mean |   comment1_count_mean |   comment2_count_mean |   revisit_count_mean |   skip_count_mean |   active_window_count_mean |   bot_to_bot_share_out_mean |   bot_to_human_share_out_mean |   received_reply_count_mean |
|:--------------------------|------------------:|----------------------:|----------------------:|---------------------:|------------------:|---------------------------:|----------------------------:|------------------------------:|----------------------------:|
| G1_population_calibrated  |              2.2  |                  2.44 |                  0.96 |                 0.86 |              1.58 |                       8.04 |                      0.6722 |                        0.3278 |                        5.54 |
| G2_anchor_behavior        |              1.92 |                  2.44 |                  1.36 |                 1.24 |              1.04 |                       8    |                      0.6796 |                        0.3204 |                        5.64 |
| G3_contextual_interactive |              1.04 |                  3.44 |                  1.38 |                 1.68 |              0.46 |                       8    |                      0.6325 |                        0.3675 |                        6.16 |

![action distribution by family](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/action_distribution_by_family.png)

![temporal activity by family](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/temporal_activity_by_family.png)

解释：这部分看每个 family 是否真的表现为不同社交行动者，而不是只在 prompt/persona 名称上不同。重点关注 post/comment1/comment2/revisit/skip 的比例和活跃窗口数。

## 2. Family Mixing Matrix 与同类抱团

| src_label   | dst_label   |   edge_weight |
|:------------|:------------|--------------:|
| Human       | Human       |         51856 |
| G1          | Human       |           176 |
| G2          | Human       |           202 |
| G3          | Human       |           268 |
| Human       | G1          |             0 |
| G1          | G1          |           134 |
| G2          | G1          |           147 |
| G3          | G1          |           167 |
| Human       | G2          |             0 |
| G1          | G2          |           127 |
| G2          | G2          |           113 |
| G3          | G2          |           167 |
| Human       | G3          |             0 |
| G1          | G3          |           142 |
| G2          | G3          |           171 |
| G3          | G3          |           105 |

| scope                                |   edge_count |   observed |   random_mean |   random_std |   z_score |   empirical_p |   permutations |
|:-------------------------------------|-------------:|-----------:|--------------:|-------------:|----------:|--------------:|---------------:|
| generated_bot_to_generated_bot_edges |         1273 |     0.2765 |        0.3283 |       0.0318 |   -1.6298 |         0.952 |           1000 |

![family mixing matrix](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/family_mixing_matrix.png)

![same family permutation test](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/same_family_edge_permutation_test.png)

解释：mixing matrix 直接显示谁更常和谁互动；permutation test 用随机 family 标签作为基线。如果 observed same-family share 显著高于随机基线，说明同类 bot 抱团。

## 3. 社区结构与 bot-only 风险

| community_method   |   node_count |   edge_count |   community_count |   largest_community_size |   mean_community_purity |   weighted_community_purity |   bot_only_community_count |   bot_only_node_share |   bot_heavy_community_count_80pct |   max_bot_share_in_community |   max_bot_share_community_size |   max_bot_share_community_bot_count |   family_modularity |   nmi_all_main |   ari_all_main |   nmi_generated_only |   ari_generated_only |
|:-------------------|-------------:|-------------:|------------------:|-------------------------:|------------------------:|----------------------------:|---------------------------:|----------------------:|----------------------------------:|-----------------------------:|-------------------------------:|------------------------------------:|--------------------:|---------------:|---------------:|---------------------:|---------------------:|
| networkx_louvain   |         2057 |        25887 |               101 |                      755 |                  0.9933 |                       0.948 |                          0 |                     0 |                                 1 |                       0.9551 |                            156 |                                 149 |               0.029 |         0.2198 |         0.0789 |               0.0129 |                    0 |

|   community_id | method           |   size |   bot_count |   bot_share | dominant_family           |   dominant_family_share | is_bot_only   |
|---------------:|:-----------------|-------:|------------:|------------:|:--------------------------|------------------------:|:--------------|
|             47 | networkx_louvain |    755 |           0 |      0      | original_human            |                  1      | False         |
|             75 | networkx_louvain |    505 |           1 |      0.002  | original_human            |                  0.998  | False         |
|             71 | networkx_louvain |    269 |           0 |      0      | original_human            |                  1      | False         |
|             12 | networkx_louvain |    203 |           0 |      0      | original_human            |                  1      | False         |
|            100 | networkx_louvain |    156 |         149 |      0.9551 | G3_contextual_interactive |                  0.3205 | False         |
|             48 | networkx_louvain |     70 |           0 |      0      | original_human            |                  1      | False         |
|             72 | networkx_louvain |      3 |           0 |      0      | original_human            |                  1      | False         |
|             86 | networkx_louvain |      3 |           0 |      0      | original_human            |                  1      | False         |
|             66 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             63 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             64 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             65 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             69 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             67 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             68 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             61 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             70 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             62 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             57 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |
|             60 | networkx_louvain |      1 |           0 |      0      | original_human            |                  1      | False         |

![community family purity](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/community_family_purity.png)

解释：社区检测看的是整体图结构是否按 family 分块。family modularity、generated-only NMI/ARI 越高，说明 family 标签越容易从图社区结构中读出来。

## 4. 图特征空间可分性

| model     |   cv_folds |   accuracy_mean |   accuracy_std |   macro_f1_mean |   macro_f1_std |   overall_macro_f1_from_oof |
|:----------|-----------:|----------------:|---------------:|----------------:|---------------:|----------------------------:|
| Family_LR |          5 |          0.4067 |         0.1321 |          0.4023 |         0.1281 |                      0.4041 |
| Family_RF |          5 |          0.4667 |         0.0333 |          0.464  |         0.0341 |                      0.4652 |

![graph feature pca by family](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/graph_feature_pca_by_family.png)

![family classifier confusion matrix](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/family_classifier_confusion_matrix.png)

解释：这里不是检测 bot，而是检测 family。如果仅用图结构特征就能高准确率区分 G1/G2/G3，说明 family 机制确实造成了分布差异；但如果过高，也意味着存在 family signature。

## 5. Human Alignment 分位数

| family                    |   bot_count |   any_feature_extreme_share |   total_degree_median_percentile |   total_degree_mean_percentile |   total_degree_extreme_share |   total_out_degree_median_percentile |   total_out_degree_mean_percentile |   total_out_degree_extreme_share |   total_in_degree_median_percentile |   total_in_degree_mean_percentile |   total_in_degree_extreme_share |   direction_balance_median_percentile |   direction_balance_mean_percentile |   direction_balance_extreme_share |   reciprocal_share_median_percentile |   reciprocal_share_mean_percentile |   reciprocal_share_extreme_share |   comment1_rate_median_percentile |   comment1_rate_mean_percentile |   comment1_rate_extreme_share |   comment2_rate_median_percentile |   comment2_rate_mean_percentile |   comment2_rate_extreme_share |   receive_rate_median_percentile |   receive_rate_mean_percentile |   receive_rate_extreme_share |   component_size_median_percentile |   component_size_mean_percentile |   component_size_extreme_share |   unique_target_authors_median_percentile |   unique_target_authors_mean_percentile |   unique_target_authors_extreme_share |
|:--------------------------|------------:|----------------------------:|---------------------------------:|-------------------------------:|-----------------------------:|-------------------------------------:|-----------------------------------:|---------------------------------:|------------------------------------:|----------------------------------:|--------------------------------:|--------------------------------------:|------------------------------------:|----------------------------------:|-------------------------------------:|-----------------------------------:|---------------------------------:|----------------------------------:|--------------------------------:|------------------------------:|----------------------------------:|--------------------------------:|------------------------------:|---------------------------------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|-------------------------------:|------------------------------------------:|----------------------------------------:|--------------------------------------:|
| G1_population_calibrated  |          50 |                        0.88 |                          42.7635 |                        42.0472 |                            0 |                              48.6104 |                            48.8637 |                                0 |                             54.8374 |                           54.1353 |                               0 |                               45.0708 |                             45.9816 |                                 0 |                                  100 |                            98.2926 |                             0.88 |                           19.5071 |                         22.9491 |                             0 |                           96.3555 |                         92.7666 |                          0.02 |                          48.8726 |                        49.8998 |                            0 |                            52.5433 |                          52.5433 |                              0 |                                   50.4982 |                                 48.6466 |                                     0 |
| G2_anchor_behavior        |          50 |                        0.9  |                          44.0614 |                        43.4232 |                            0 |                              52.8055 |                            51.1626 |                                0 |                             56.2402 |                           54.5595 |                               0 |                               46.0933 |                             46.9056 |                                 0 |                                  100 |                            99.5144 |                             0.9  |                           19.2318 |                         21.8144 |                             0 |                           96.6308 |                         93.742  |                          0.04 |                          48.0073 |                        50.1909 |                            0 |                            52.5433 |                          52.5433 |                              0 |                                   50.4982 |                                 49.2984 |                                     0 |
| G3_contextual_interactive |          50 |                        0.92 |                          46.2769 |                        45.86   |                            0 |                              52.8055 |                            53.9203 |                                0 |                             56.2402 |                           54.9413 |                               0 |                               48.4006 |                             47.8574 |                                 0 |                                  100 |                            99.7326 |                             0.92 |                           20.0839 |                         25.0425 |                             0 |                           95.7525 |                         90.8285 |                          0    |                          50.4982 |                        50.8254 |                            0 |                            52.5433 |                          52.5433 |                              0 |                                   54.0902 |                                 51.2407 |                                     0 |

![human percentile by family](graph_network/results/partial_1202actions_family_behavior_mixing_20260605/figures/human_percentile_by_family.png)

解释：这部分把每个 bot 的图特征映射到 human 用户分布的百分位。大量落在 1% 或 99% 极端区间，说明虽然生成器有互动环境，但仍有不自然行为。

## 结论使用方式

如果本报告显示 family 行为差异明显、但同类抱团和 human 极端偏离不严重，那么当前数据适合作为图泛化 benchmark。反之，如果 family classifier 很强、same-family permutation 显著、或 human extreme share 很高，那么后续应先修生成环境，而不是继续扩大检测模型。
