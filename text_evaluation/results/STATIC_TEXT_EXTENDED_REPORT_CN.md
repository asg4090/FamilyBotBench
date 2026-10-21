# 静态文本扩展实验总报告

## 完成度

整体状态：`全部核心文件已生成`。

本轮由 5 个独立 worker 分别完成五个实验；主线程只负责公共模块、任务边界、验收和汇总。

## 文件审计

```text
                        section                                                                     path  exists          shape
                       LOFO 正式化                              results/lofo_formal/lofo_formal_metrics.csv    True    10 行 x 10 列
                       LOFO 正式化                          results/lofo_formal/lofo_formal_predictions.csv    True  3410 行 x 13 列
                       LOFO 正式化                               results/lofo_formal/lofo_formal_matrix.csv    True     12 行 x 7 列
                       LOFO 正式化                           results/figures/lofo_formal_recall_heatmap.png    True               
                       LOFO 正式化                      results/figures/lofo_formal_probability_boxplot.png    True               
                       LOFO 正式化                             results/lofo_formal/LOFO_FORMAL_REPORT_CN.md    True               
    Semantic-Cluster-Controlled                           results/topic_controlled/topic_assignments.csv    True  3157 行 x 18 列
    Semantic-Cluster-Controlled                       results/topic_controlled/topic_keyword_summary.csv    True    36 行 x 14 列
    Semantic-Cluster-Controlled             results/topic_controlled/topic_matched_detection_metrics.csv    True     6 行 x 47 列
    Semantic-Cluster-Controlled              results/topic_controlled/within_topic_detection_metrics.csv    True    72 行 x 29 列
    Semantic-Cluster-Controlled                 results/topic_controlled/leave_one_topic_out_metrics.csv    True    72 行 x 29 列
    Semantic-Cluster-Controlled                   results/figures/topic_distribution_by_label_family.png    True               
    Semantic-Cluster-Controlled                         results/figures/topic_controlled_recall_drop.png    True               
    Semantic-Cluster-Controlled                                  results/figures/within_topic_recall.png    True               
    Semantic-Cluster-Controlled                           results/figures/leave_one_topic_out_recall.png    True               
    Semantic-Cluster-Controlled                   results/topic_controlled/TOPIC_CONTROLLED_REPORT_CN.md    True               
Thread-Derived Topic-Controlled                       results/thread_topic_controlled/post_topic_map.csv    True  52672 行 x 9 列
Thread-Derived Topic-Controlled                results/thread_topic_controlled/comment_topic_dataset.csv    True 77533 行 x 14 列
Thread-Derived Topic-Controlled                 results/thread_topic_controlled/topic_resource_audit.csv    True     8 行 x 13 列
Thread-Derived Topic-Controlled      results/thread_topic_controlled/topic_matched_detection_metrics.csv    True     2 行 x 16 列
Thread-Derived Topic-Controlled          results/thread_topic_controlled/leave_one_topic_out_metrics.csv    True     6 行 x 17 列
Thread-Derived Topic-Controlled results/thread_topic_controlled/leave_posts_out_within_topic_metrics.csv    True     6 行 x 20 列
Thread-Derived Topic-Controlled   results/thread_topic_controlled/thread_topic_tfidf_recall_by_topic.csv    True      6 行 x 4 列
Thread-Derived Topic-Controlled            results/figures/thread_topic_distribution_by_label_family.png    True               
Thread-Derived Topic-Controlled                          results/figures/thread_topic_resource_audit.png    True               
Thread-Derived Topic-Controlled                     results/figures/thread_topic_matched_recall_drop.png    True               
Thread-Derived Topic-Controlled           results/figures/thread_topic_tfidf_recall_by_topic_overall.png    True               
Thread-Derived Topic-Controlled                    results/figures/thread_leave_one_topic_out_recall.png    True               
Thread-Derived Topic-Controlled           results/figures/thread_leave_posts_out_within_topic_recall.png    True               
Thread-Derived Topic-Controlled     results/thread_topic_controlled/THREAD_TOPIC_CONTROLLED_REPORT_CN.md    True               
              Anchor Similarity                       results/anchor_similarity/anchor_linkage_audit.csv    True     5 行 x 15 列
              Anchor Similarity               results/anchor_similarity/anchor_similarity_by_comment.csv    True   738 行 x 52 列
              Anchor Similarity                results/anchor_similarity/anchor_similarity_by_family.csv    True     5 行 x 18 列
              Anchor Similarity             results/anchor_similarity/anchor_similarity_vs_detection.csv    True    30 行 x 16 列
              Anchor Similarity                  results/figures/anchor_similarity_by_family_boxplot.png    True               
              Anchor Similarity                 results/figures/anchor_similarity_vs_bot_probability.png    True               
              Anchor Similarity                         results/figures/stylistic_distance_by_family.png    True               
              Anchor Similarity                 results/anchor_similarity/ANCHOR_SIMILARITY_REPORT_CN.md    True               
            Family Separability              results/family_separability/family_separability_metrics.csv    True    49 行 x 17 列
            Family Separability          results/family_separability/family_separability_predictions.csv    True   300 行 x 18 列
            Family Separability            results/family_separability/family_confusion_matrix_tfidf.csv    True      5 行 x 6 列
            Family Separability              results/family_separability/family_confusion_matrix_bge.csv    True      5 行 x 6 列
            Family Separability                        results/figures/family_confusion_matrix_tfidf.png    True               
            Family Separability                          results/figures/family_confusion_matrix_bge.png    True               
            Family Separability                    results/figures/family_separability_embedding_pca.png    True               
            Family Separability             results/family_separability/FAMILY_SEPARABILITY_REPORT_CN.md    True               
                Static Text OOD                         results/static_text_ood/ood_scores_by_sample.csv    True 35280 行 x 29 列
                Static Text OOD                        results/static_text_ood/ood_metrics_by_family.csv    True    80 行 x 26 列
                Static Text OOD                           results/static_text_ood/ood_method_summary.csv    True     8 行 x 15 列
                Static Text OOD                     results/figures/ood_score_distribution_by_family.png    True               
                Static Text OOD                           results/figures/ood_auroc_by_method_family.png    True               
                Static Text OOD                          results/figures/bge_centroid_margin_scatter.png    True               
                Static Text OOD                     results/static_text_ood/STATIC_TEXT_OOD_REPORT_CN.md    True               
```

## 五个实验结论

### 1. LOFO 正式化

- TF-IDF 最难 held-out：`F5=0.600`。
- BGE 最难 held-out：`F5=0.760`。
- 这说明跨 family 泛化不是均匀的，F3/F5 对文本检测器更有压力。

### 2. Semantic-Cluster-Controlled Detection

- k=12 TF-IDF topic-control recall drop：`0.017`。
- k=12 BGE topic-control recall drop：`0.090`。
- 旧 BGE 语义簇控制后 recall 有下降，但它不能严格证明 topic shortcut；正式 topic control 应引用 `thread_topic_controlled` 新实验。

### 3. Anchor Similarity

- F3 nearest-human BGE cosine：`0.771`。
- F3 human centroid style z-L2：`2.710`。
- F3 更靠近 human pool，但指定 anchor 的表层 TF-IDF 相似度并不高，不能夸大为逐条模仿成功。

### 4. Family Separability

- TF-IDF family classification accuracy：`1.000`。
- BGE family classification accuracy：`0.950`。
- family 可分性很强，说明不同生成策略留下明显 family 指纹。

### 5. Static Text OOD

- 最强平均 OOD 方法：`bge_knn_k20_distance`，mean AUROC `0.648`。
- kNN distance 比 supervised probability 更像可靠的 OOD 信号。

## 综合判断

这组静态文本扩展实验把问题从“检出率多少”推进到“为什么泛化失败/恢复”。当前证据显示：

- TF-IDF 仍然很强，但容易留下 family 指纹和表层词汇依赖。
- BGE 泛化更稳，但在 topic control、LOFO 和 OOD 中仍有 family/provider shift 压力。
- Anchor 对齐改善了 F3 的 human-pool 接近度，但不是逐条 anchor 复刻。
- 纯静态文本已经足够支持一个较完整的泛化性分析章节；后续动态图/行为实验可以沿用这套 held-out family 思路。

## 独立报告路径

- `results/lofo_formal/LOFO_FORMAL_REPORT_CN.md`
- `results/topic_controlled/TOPIC_CONTROLLED_REPORT_CN.md`
- `results/anchor_similarity/ANCHOR_SIMILARITY_REPORT_CN.md`
- `results/family_separability/FAMILY_SEPARABILITY_REPORT_CN.md`
- `results/static_text_ood/STATIC_TEXT_OOD_REPORT_CN.md`
