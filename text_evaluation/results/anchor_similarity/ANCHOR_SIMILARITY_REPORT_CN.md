# Anchor Similarity 实验报告

## 研究问题

Anchor 对齐是否真的让 bot 文本更接近人类文本，尤其是 F3？

## Linkage Audit

```text
family_short  generated_users  generated_comments  user_anchor_linkage_coverage  comment_anchor_linkage_coverage                           analysis_mode                                                                                                    audit_note
          F1               50                 145                           0.0                              0.0 nearest_human_pool_fallback_no_manifest   No specified anchor manifest rows for this family; only nearest original human pool metrics are applicable.
          F2               50                 150                           0.0                              0.0 nearest_human_pool_fallback_no_manifest   No specified anchor manifest rows for this family; only nearest original human pool metrics are applicable.
          F3               50                 144                           1.0                              1.0 assigned_anchor_plus_nearest_human_pool All generated users have resolved specified anchors; assigned-anchor and nearest-pool metrics are both valid.
          F4               50                 150                           0.0                              0.0 nearest_human_pool_fallback_no_manifest   No specified anchor manifest rows for this family; only nearest original human pool metrics are applicable.
          F5               50                 149                           0.0                              0.0 nearest_human_pool_fallback_no_manifest   No specified anchor manifest rows for this family; only nearest original human pool metrics are applicable.
```

## Family-Level Similarity

```text
family_short  n_comments  nearest_anchor_tfidf_cosine_mean  assigned_anchor_tfidf_cosine_mean  nearest_human_bge_cosine_mean  human_centroid_style_z_l2_mean
          F1         145                          0.114263                                NaN                       0.756167                        3.654835
          F2         150                          0.132372                                NaN                       0.743129                        2.902752
          F3         144                          0.133240                           0.034196                       0.771129                        2.710402
          F4         150                          0.131615                                NaN                       0.733969                        3.290292
          F5         149                          0.124336                                NaN                       0.710304                        3.438062
```

## Similarity vs Detection

```text
family_short  detector_model                  similarity_metric  mean_similarity  mean_bot_probability  spearman_similarity_vs_probability
          F1 TF-IDF detector bot_to_nearest_anchor_tfidf_cosine         0.114263              0.235591                           -0.015922
          F1 TF-IDF detector    bot_to_nearest_human_bge_cosine         0.756167              0.235591                           -0.427008
          F1 TF-IDF detector   bot_to_human_centroid_style_z_l2         3.654835              0.235591                            0.096632
          F1    BGE detector bot_to_nearest_anchor_tfidf_cosine         0.114263              0.585172                            0.096046
          F1    BGE detector    bot_to_nearest_human_bge_cosine         0.756167              0.585172                           -0.380340
          F1    BGE detector   bot_to_human_centroid_style_z_l2         3.654835              0.585172                            0.093939
          F2 TF-IDF detector bot_to_nearest_anchor_tfidf_cosine         0.132372              0.148297                            0.112254
          F2 TF-IDF detector    bot_to_nearest_human_bge_cosine         0.743129              0.148297                           -0.286819
          F2 TF-IDF detector   bot_to_human_centroid_style_z_l2         2.902752              0.148297                            0.212106
          F2    BGE detector bot_to_nearest_anchor_tfidf_cosine         0.132372              0.119719                           -0.008594
          F2    BGE detector    bot_to_nearest_human_bge_cosine         0.743129              0.119719                            0.101465
          F2    BGE detector   bot_to_human_centroid_style_z_l2         2.902752              0.119719                           -0.079671
          F3 TF-IDF detector bot_to_nearest_anchor_tfidf_cosine         0.133240              0.107199                           -0.003168
          F3 TF-IDF detector    bot_to_nearest_human_bge_cosine         0.771129              0.107199                           -0.445095
          F3 TF-IDF detector   bot_to_human_centroid_style_z_l2         2.710402              0.107199                            0.245242
          F3    BGE detector bot_to_nearest_anchor_tfidf_cosine         0.133240              0.184218                            0.046082
          F3    BGE detector    bot_to_nearest_human_bge_cosine         0.771129              0.184218                           -0.217932
          F3    BGE detector   bot_to_human_centroid_style_z_l2         2.710402              0.184218                           -0.066920
          F4 TF-IDF detector bot_to_nearest_anchor_tfidf_cosine         0.131615              0.193383                            0.079884
          F4 TF-IDF detector    bot_to_nearest_human_bge_cosine         0.733969              0.193383                            0.176567
```

## 解释

- F3 comment anchor linkage coverage = `1.000`。
- F3 nearest-human BGE cosine mean = `0.771`，在五个 family 中最高。
- F3 human centroid style z-L2 mean = `2.710`，在五个 family 中最低。
- 但 F3 assigned-anchor TF-IDF cosine 不高，不能说逐条指定 anchor 的表层词汇高度相似；更准确说法是 F3 更接近 human pool 的语义/风格区域。
