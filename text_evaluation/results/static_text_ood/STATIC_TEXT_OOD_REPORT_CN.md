# Static Text OOD 实验报告

## 研究问题

不用普通分类器，只用距离或异常分数，能否发现 held-out family 是文本分布外样本？

## Method Summary

```text
                        method     method_category                                                                               score_direction  is_supervised_proxy  mean_auroc_ood_vs_id  median_auroc_ood_vs_id  mean_auprc_ood  mean_score_gap_heldout_minus_id
     bge_centroid_min_distance semi_supervised_ood                       larger score = farther from nearest known human/bot centroid = more OOD                False              0.586913                0.623129        0.137066                         0.007792
       bge_centroid_bot_margin semi_supervised_ood larger score = farther from known bot centroid than human centroid = more bot-family OOD-like                False              0.426295                0.470000        0.101625                        -0.000529
           bge_knn_k5_distance    unsupervised_ood               larger score = larger mean distance to 5 nearest known-train samples = more OOD                False              0.646036                0.659588        0.162562                         0.069889
          bge_knn_k10_distance    unsupervised_ood              larger score = larger mean distance to 10 nearest known-train samples = more OOD                False              0.646525                0.656014        0.162347                         0.069940
          bge_knn_k20_distance    unsupervised_ood              larger score = larger mean distance to 20 nearest known-train samples = more OOD                False              0.647959                0.654124        0.162223                         0.070135
          bge_isolation_forest    unsupervised_ood                               larger score = lower IsolationForest normality score = more OOD                False              0.579520                0.562407        0.133397                         0.001534
tfidf_lr_bot_probability_proxy    supervised_proxy              larger score = higher supervised bot probability used as OOD proxy; not pure OOD                 True              0.489031                0.529202        0.111720                        -0.052180
  bge_lr_bot_probability_proxy    supervised_proxy              larger score = higher supervised bot probability used as OOD proxy; not pure OOD                 True              0.578901                0.547663        0.136760                         0.056633
```

## Scenario Mean AUROC

```text
           scenario                         method  auroc_ood_vs_id
         lofo_known   bge_lr_bot_probability_proxy         0.591699
         lofo_known           bge_knn_k20_distance         0.587699
         lofo_known           bge_knn_k10_distance         0.583552
         lofo_known            bge_knn_k5_distance         0.580994
         lofo_known      bge_centroid_min_distance         0.548163
         lofo_known           bge_isolation_forest         0.545857
         lofo_known tfidf_lr_bot_probability_proxy         0.503560
         lofo_known        bge_centroid_bot_margin         0.418363
original_only_known            bge_knn_k5_distance         0.711079
original_only_known           bge_knn_k10_distance         0.709498
original_only_known           bge_knn_k20_distance         0.708220
original_only_known      bge_centroid_min_distance         0.625663
original_only_known           bge_isolation_forest         0.613182
original_only_known   bge_lr_bot_probability_proxy         0.566103
original_only_known tfidf_lr_bot_probability_proxy         0.474502
original_only_known        bge_centroid_bot_margin         0.434227
```

## 解释

- 平均 AUROC 最高的方法是 `bge_knn_k20_distance`，mean AUROC = `0.648`。
- kNN distance 是最稳定的纯 OOD 信号；original-only known 下更有效，lofo-known 下明显变难。
- TF-IDF supervised probability proxy 不是好的纯 OOD 证据；它已经被标注为 supervised proxy。
- centroid bot-margin 低于 0.5 是结果现象，不应强行反转方向；说明该 margin 对 human ID 样本也敏感。
