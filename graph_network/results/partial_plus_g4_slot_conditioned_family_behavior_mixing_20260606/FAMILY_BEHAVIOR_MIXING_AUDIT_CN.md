# Family Behavior and Mixing Audit

## 研究口径

本报告只审计生成数据质量，不重新生成 bot，不做文本检测，也不训练 bot-vs-human 检测器。核心问题是：各 generated family 是否真的形成不同社交行为，以及是否出现 family 抱团或明显偏离 human 图行为分布。

- 数据目录：`graph_network/outputs/partial_plus_g4_slot_conditioned_20260606`
- 结果目录：`graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606`
- 数据完整性 passed：`True`
- generated family counts：`{'G1_population_calibrated': 50, 'G2_anchor_behavior': 50, 'G3_contextual_interactive': 50, 'G4_slot_conditioned': 50}`

## 核心判断

1. 三个 family 的行为分布存在可见差异，最大差异来自 `comment1_count`，family 均值跨度约 `1.28`。
2. 同 family 互动比例高于随机均值，但未达到强显著：真实=`0.273`，随机均值=`0.247`，p=`0.1898`。
3. human/generated 方向性边统计：human→generated=`948`，generated→human=`1094`，generated↔generated=`1334`。
4. 没有严格 bot-only 社区，但存在 bot-heavy 社区：最高 bot_share=`0.955`，该社区 size=`156`、bot_count=`149`；family modularity=`0.028`，generated-only NMI=`0.495`。这表示不一定是某个 family 单独抱团，而可能是 generated bot 整体容易集中。
5. 各 generated family 在图特征空间中中等可分：最佳模型 `Family_RF`，CV accuracy=`0.570`，macro-F1=`0.565`。
6. 存在明显 human 分布偏离：`G3 interactive` 中 `92.00%` 的 bot 至少一个关键图特征落在人类 1%/99% 极端区间。

## 1. Family 行为分布

| family                    |   post_count_mean |   comment1_count_mean |   comment2_count_mean |   revisit_count_mean |   skip_count_mean |   active_window_count_mean |   bot_to_bot_share_out_mean |   bot_to_human_share_out_mean |   received_reply_count_mean |
|:--------------------------|------------------:|----------------------:|----------------------:|---------------------:|------------------:|---------------------------:|----------------------------:|------------------------------:|----------------------------:|
| G1_population_calibrated  |              2.2  |                  2.44 |                  0.96 |                 0.86 |              1.58 |                       8.04 |                      0.6722 |                        0.3278 |                        5.82 |
| G2_anchor_behavior        |              1.92 |                  2.44 |                  1.36 |                 1.24 |              1.04 |                       8    |                      0.6796 |                        0.3204 |                        5.78 |
| G3_contextual_interactive |              1.04 |                  3.44 |                  1.38 |                 1.68 |              0.46 |                       8    |                      0.6325 |                        0.3675 |                        6.22 |
| G4_slot_conditioned       |              1.62 |                  2.16 |                  2    |                 1.4  |              0.82 |                       8    |                      0.1308 |                        0.8692 |                       12.52 |

![action distribution by family](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/action_distribution_by_family.png)

![temporal activity by family](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/temporal_activity_by_family.png)

解释：这部分看每个 family 是否真的表现为不同社交行动者，而不是只在 prompt/persona 名称上不同。重点关注 post/comment1/comment2/revisit/skip 的比例和活跃窗口数。

## 2. Family Mixing Matrix 与同类抱团

| src_label   | dst_label   |   edge_weight |
|:------------|:------------|--------------:|
| Human       | Human       |         52023 |
| G1          | Human       |           176 |
| G2          | Human       |           202 |
| G3          | Human       |           268 |
| G4          | Human       |           448 |
| Human       | G1          |            12 |
| G1          | G1          |           134 |
| G2          | G1          |           147 |
| G3          | G1          |           167 |
| G4          | G1          |            29 |
| Human       | G2          |             6 |
| G1          | G2          |           127 |
| G2          | G2          |           113 |
| G3          | G2          |           167 |
| G4          | G2          |            14 |
| Human       | G3          |             0 |
| G1          | G3          |           142 |
| G2          | G3          |           171 |
| G3          | G3          |           105 |
| G4          | G3          |             6 |

| scope                                |   edge_count |   observed |   random_mean |   random_std |   z_score |   empirical_p |   permutations |
|:-------------------------------------|-------------:|-----------:|--------------:|-------------:|----------:|--------------:|---------------:|
| generated_bot_to_generated_bot_edges |         1334 |     0.2729 |         0.247 |       0.0276 |    0.9354 |        0.1898 |           1000 |

![family mixing matrix](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/family_mixing_matrix.png)

![same family permutation test](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/same_family_edge_permutation_test.png)

解释：mixing matrix 直接显示谁更常和谁互动；permutation test 用随机 family 标签作为基线。如果 observed same-family share 显著高于随机基线，说明同类 bot 抱团。

## 3. 社区结构与 bot-only 风险

| community_method   |   node_count |   edge_count |   community_count |   largest_community_size |   mean_community_purity |   weighted_community_purity |   bot_only_community_count |   bot_only_node_share |   bot_heavy_community_count_80pct |   max_bot_share_in_community |   max_bot_share_community_size |   max_bot_share_community_bot_count |   family_modularity |   nmi_all_main |   ari_all_main |   nmi_generated_only |   ari_generated_only |
|:-------------------|-------------:|-------------:|------------------:|-------------------------:|------------------------:|----------------------------:|---------------------------:|----------------------:|----------------------------------:|-----------------------------:|-------------------------------:|------------------------------------:|--------------------:|---------------:|---------------:|---------------------:|---------------------:|
| networkx_louvain   |         2107 |        26637 |               102 |                      612 |                  0.9919 |                      0.9255 |                          0 |                     0 |                                 1 |                       0.9551 |                            156 |                                 149 |              0.0284 |          0.206 |         0.0586 |               0.4946 |               0.2327 |

|   community_id | method           |   size |   bot_count |   bot_share | dominant_family    |   dominant_family_share | is_bot_only   |
|---------------:|:-----------------|-------:|------------:|------------:|:-------------------|------------------------:|:--------------|
|             59 | networkx_louvain |    612 |          25 |      0.0408 | original_human     |                  0.9592 | False         |
|             77 | networkx_louvain |    527 |           5 |      0.0095 | original_human     |                  0.9905 | False         |
|             28 | networkx_louvain |    279 |           0 |      0      | original_human     |                  1      | False         |
|             48 | networkx_louvain |    217 |          17 |      0.0783 | original_human     |                  0.9217 | False         |
|             12 | networkx_louvain |    211 |           4 |      0.019  | original_human     |                  0.981  | False         |
|            101 | networkx_louvain |    156 |         149 |      0.9551 | G2_anchor_behavior |                  0.3205 | False         |
|             49 | networkx_louvain |      7 |           0 |      0      | original_human     |                  1      | False         |
|             74 | networkx_louvain |      3 |           0 |      0      | original_human     |                  1      | False         |
|             61 | networkx_louvain |      2 |           0 |      0      | original_human     |                  1      | False         |
|             69 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             66 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             67 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             68 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             73 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             70 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             71 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             72 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             64 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             65 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |
|             57 | networkx_louvain |      1 |           0 |      0      | original_human     |                  1      | False         |

![community family purity](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/community_family_purity.png)

解释：社区检测看的是整体图结构是否按 family 分块。family modularity、generated-only NMI/ARI 越高，说明 family 标签越容易从图社区结构中读出来。

## 4. 图特征空间可分性

| model     |   cv_folds |   accuracy_mean |   accuracy_std |   macro_f1_mean |   macro_f1_std |   overall_macro_f1_from_oof |
|:----------|-----------:|----------------:|---------------:|----------------:|---------------:|----------------------------:|
| Family_LR |          5 |            0.56 |         0.0487 |          0.5503 |         0.0411 |                      0.5539 |
| Family_RF |          5 |            0.57 |         0.0447 |          0.5654 |         0.0447 |                      0.5663 |

![graph feature pca by family](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/graph_feature_pca_by_family.png)

![family classifier confusion matrix](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/family_classifier_confusion_matrix.png)

解释：这里不是检测 bot，而是检测 family。如果仅用图结构特征就能高准确率区分各 generated family，说明 family 机制确实造成了分布差异；但如果过高，也意味着存在 family signature。

## 5. Human Alignment 分位数

| family                    |   bot_count |   any_feature_extreme_share |   total_degree_median_percentile |   total_degree_mean_percentile |   total_degree_extreme_share |   total_out_degree_median_percentile |   total_out_degree_mean_percentile |   total_out_degree_extreme_share |   total_in_degree_median_percentile |   total_in_degree_mean_percentile |   total_in_degree_extreme_share |   direction_balance_median_percentile |   direction_balance_mean_percentile |   direction_balance_extreme_share |   reciprocal_share_median_percentile |   reciprocal_share_mean_percentile |   reciprocal_share_extreme_share |   comment1_rate_median_percentile |   comment1_rate_mean_percentile |   comment1_rate_extreme_share |   comment2_rate_median_percentile |   comment2_rate_mean_percentile |   comment2_rate_extreme_share |   receive_rate_median_percentile |   receive_rate_mean_percentile |   receive_rate_extreme_share |   component_size_median_percentile |   component_size_mean_percentile |   component_size_extreme_share |   unique_target_authors_median_percentile |   unique_target_authors_mean_percentile |   unique_target_authors_extreme_share |
|:--------------------------|------------:|----------------------------:|---------------------------------:|-------------------------------:|-----------------------------:|-------------------------------------:|-----------------------------------:|---------------------------------:|------------------------------------:|----------------------------------:|--------------------------------:|--------------------------------------:|------------------------------------:|----------------------------------:|-------------------------------------:|-----------------------------------:|---------------------------------:|----------------------------------:|--------------------------------:|------------------------------:|----------------------------------:|--------------------------------:|------------------------------:|---------------------------------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|-------------------------------:|------------------------------------------:|----------------------------------------:|--------------------------------------:|
| G1_population_calibrated  |          50 |                        0.84 |                          42.4882 |                        42.8123 |                            0 |                              48.3482 |                            48.6151 |                                0 |                             55.8207 |                           55.6607 |                               0 |                               43.7336 |                             44.1605 |                                 0 |                              99.8558 |                            98.2297 |                             0.84 |                           19.5071 |                         23.1715 |                             0 |                           96.3555 |                         92.5475 |                          0.02 |                          46.9061 |                        48.5469 |                            0 |                            52.5433 |                          52.5433 |                              0 |                                   50.2622 |                                 48.42   |                                     0 |
| G2_anchor_behavior        |          50 |                        0.9  |                          43.7467 |                        43.612  |                            0 |                              52.4908 |                            50.8936 |                                0 |                             55.8207 |                           55.0949 |                               0 |                               45.1494 |                             46.3052 |                                 0 |                             100      |                            99.5071 |                             0.9  |                           19.2056 |                         21.9544 |                             0 |                           96.6571 |                         93.6072 |                          0.04 |                          47.8762 |                        49.6219 |                            0 |                            52.5433 |                          52.5433 |                              0 |                                   50.2622 |                                 49.0572 |                                     0 |
| G3_contextual_interactive |          50 |                        0.92 |                          45.936  |                        45.6854 |                            0 |                              52.4908 |                            53.6256 |                                0 |                             55.8207 |                           54.9837 |                               0 |                               47.8762 |                             47.645  |                                 0 |                             100      |                            99.7347 |                             0.92 |                           20.1363 |                         25.3964 |                             0 |                           95.7525 |                         90.4819 |                          0    |                          49.8427 |                        50.5364 |                            0 |                            52.5433 |                          52.5433 |                              0 |                                   53.8018 |                                 50.9869 |                                     0 |
| G4_slot_conditioned       |          50 |                        0.02 |                          52.4908 |                        52.0734 |                            0 |                              46.1982 |                            46.5066 |                                0 |                             72.3388 |                           70.8547 |                               0 |                               29.2737 |                             30.0435 |                                 0 |                              34.3734 |                            50.3707 |                             0.02 |                           26.6649 |                         30.044  |                             0 |                           89.2239 |                         85.8437 |                          0    |                          32.0399 |                        33.3954 |                            0 |                            52.5433 |                          52.5433 |                              0 |                                   50.2622 |                                 48.4662 |                                     0 |

![human percentile by family](graph_network/results/partial_plus_g4_slot_conditioned_family_behavior_mixing_20260606/figures/human_percentile_by_family.png)

解释：这部分把每个 bot 的图特征映射到 human 用户分布的百分位。大量落在 1% 或 99% 极端区间，说明虽然生成器有互动环境，但仍有不自然行为。

## 结论使用方式

如果本报告显示 family 行为差异明显、但同类抱团和 human 极端偏离不严重，那么当前数据适合作为图泛化 benchmark。反之，如果 family classifier 很强、same-family permutation 显著、或 human extreme share 很高，那么后续应先修生成环境，而不是继续扩大检测模型。
