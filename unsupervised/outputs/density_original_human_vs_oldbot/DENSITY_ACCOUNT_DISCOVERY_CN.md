# 密度式账号机制簇发现实验报告

## 1. 实验目的

本实验把上一版 KMeans 基线替换为更贴合问题的密度发现方案。目标不是强行把所有账号分成“人/机器人”两类，而是在账号级 TF-IDF 语言空间里寻找异常高密度的 suspicious dense clusters。

流程是：

```text
账号聚合文本 -> word TF-IDF -> HDBSCAN + dense component scan -> suspicious dense clusters -> 最后揭示标签评估 bot purity / recall -> PCA/UMAP 2D 展示
```

标签不参与 TF-IDF、HDBSCAN、相似图、dense component scan 或 suspicious score，只在最后的 revealed evaluation 中使用。

## 2. 数据口径

- 输入目录：`unsupervised/inputs/static_family`
- 输出目录：`unsupervised/outputs/density_original_human_vs_oldbot`
- 筛选 family：`('original_human', 'original_bot')`
- 最小词数：`10`
- 进入实验账号数：`2671`
- human 数：`1674`
- old bot 数：`997`

## 3. 方法实现

### 3.1 主空间：word TF-IDF

- word 1-2gram TF-IDF，最大特征数 `50000`。
- 使用 cosine 相似度作为账号文本分布接近程度。
- dense component scan 在原始 TF-IDF cosine kNN 图上运行。
- HDBSCAN 在 SVD 降维并标准化后的 TF-IDF 空间运行，避免直接在几万维稀疏空间里估计密度。

### 3.2 HDBSCAN

- SVD 维度：`100` 请求，实际维度见 `density_account_features.csv`。
- min_cluster_size：`10,20,30,50,80`
- min_samples：`5`
- 输出非噪声密集簇，过大的簇按 `max_cluster_fraction` 过滤，避免把半个数据集当作一个机制簇。

### 3.3 Dense component scan

- kNN 图：每个账号连接 TF-IDF cosine 最近的 `50` 个账号。
- 自动阈值 quantiles：`0.90,0.95,0.975,0.99`。
- 实际 cosine thresholds：`0.120522, 0.132899, 0.143546, 0.157668`。
- 最小 component size：`10`。
- 最小 component 内部边密度：`0.025`。
- 最大 cluster 占比：`0.25`。
- 每个阈值下保留高相似边，寻找 connected components。若一批账号在高相似阈值下仍连成组件，就说明它们在词汇使用分布上非常接近。

### 3.4 Suspicious score

每个候选簇都在原始 TF-IDF 空间里计算：

- 簇内平均 cosine similarity；
- 簇半径；
- 与同规模随机账号组的 null baseline 对比；
- dense component 的图边密度；
- HDBSCAN 的平均 membership strength；
- 簇规模校正。

评分越高，表示这个簇越像“异常集中的账号语言机制团块”。

## 4. 输出规模

- suspicious clusters：`31`
- HDBSCAN/dense assignments 文件：`density_cluster_assignments.csv`
- suspicious score 文件：`density_suspicious_cluster_scores.csv`
- 标签揭示评估文件：`density_label_revealed_evaluation.csv`
- Top-1 并集结果：`precision=1.0000, recall=0.1846, selected=184`

## 5. Top suspicious dense clusters

| suspicious_rank | method | cluster_key | selected_account_count | human_count | bot_count | bot_rate | bot_recall | suspicious_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | dense_component | dense_component_t0.157668:c=1370 | 184 | 0 | 184 | 1.0000 | 0.1846 | 111.3487 |
| 2.0000 | dense_component | dense_component_t0.157668:c=1381 | 51 | 0 | 51 | 1.0000 | 0.0512 | 62.5308 |
| 3.0000 | hdbscan | hdbscan_mcs20_ms5:c=4 | 175 | 28 | 147 | 0.8400 | 0.1474 | 52.6025 |
| 4.0000 | hdbscan | hdbscan_mcs80_ms5:c=1 | 175 | 28 | 147 | 0.8400 | 0.1474 | 52.6025 |
| 5.0000 | hdbscan | hdbscan_mcs50_ms5:c=1 | 175 | 28 | 147 | 0.8400 | 0.1474 | 52.5966 |
| 6.0000 | hdbscan | hdbscan_mcs30_ms5:c=2 | 175 | 28 | 147 | 0.8400 | 0.1474 | 52.5933 |
| 7.0000 | hdbscan | hdbscan_mcs10_ms5:c=7 | 65 | 3 | 62 | 0.9538 | 0.0622 | 46.4735 |
| 8.0000 | hdbscan | hdbscan_mcs10_ms5:c=6 | 14 | 0 | 14 | 1.0000 | 0.0140 | 37.8343 |
| 9.0000 | hdbscan | hdbscan_mcs10_ms5:c=5 | 110 | 7 | 103 | 0.9364 | 0.1033 | 35.4906 |
| 10.0000 | hdbscan | hdbscan_mcs20_ms5:c=5 | 110 | 7 | 103 | 0.9364 | 0.1033 | 35.4906 |
| 11.0000 | hdbscan | hdbscan_mcs30_ms5:c=3 | 110 | 7 | 103 | 0.9364 | 0.1033 | 35.4906 |
| 12.0000 | hdbscan | hdbscan_mcs50_ms5:c=2 | 110 | 7 | 103 | 0.9364 | 0.1033 | 35.4906 |
| 13.0000 | hdbscan | hdbscan_mcs80_ms5:c=2 | 110 | 7 | 103 | 0.9364 | 0.1033 | 35.4906 |
| 14.0000 | hdbscan | hdbscan_mcs10_ms5:c=8 | 20 | 0 | 20 | 1.0000 | 0.0201 | 33.1942 |
| 15.0000 | hdbscan | hdbscan_mcs20_ms5:c=3 | 20 | 20 | 0 | 0.0000 | 0.0000 | 32.0119 |
| 16.0000 | hdbscan | hdbscan_mcs10_ms5:c=4 | 20 | 20 | 0 | 0.0000 | 0.0000 | 31.9964 |
| 17.0000 | hdbscan | hdbscan_mcs30_ms5:c=0 | 75 | 75 | 0 | 0.0000 | 0.0000 | 31.8617 |
| 18.0000 | hdbscan | hdbscan_mcs10_ms5:c=3 | 51 | 51 | 0 | 0.0000 | 0.0000 | 29.5404 |
| 19.0000 | hdbscan | hdbscan_mcs20_ms5:c=2 | 51 | 51 | 0 | 0.0000 | 0.0000 | 29.5404 |
| 20.0000 | dense_component | dense_component_t0.157668:c=353 | 18 | 18 | 0 | 0.0000 | 0.0000 | 23.1410 |
| 21.0000 | dense_component | dense_component_t0.143546:c=63 | 16 | 14 | 2 | 0.1250 | 0.0020 | 22.8802 |
| 22.0000 | dense_component | dense_component_t0.157668:c=31 | 20 | 20 | 0 | 0.0000 | 0.0000 | 19.3891 |
| 23.0000 | hdbscan | hdbscan_mcs10_ms5:c=0 | 12 | 12 | 0 | 0.0000 | 0.0000 | 17.2039 |
| 24.0000 | hdbscan | hdbscan_mcs30_ms5:c=1 | 53 | 45 | 8 | 0.1509 | 0.0080 | 15.8762 |
| 25.0000 | hdbscan | hdbscan_mcs10_ms5:c=2 | 53 | 45 | 8 | 0.1509 | 0.0080 | 15.8726 |
| 26.0000 | hdbscan | hdbscan_mcs20_ms5:c=1 | 53 | 45 | 8 | 0.1509 | 0.0080 | 15.8726 |
| 27.0000 | dense_component | dense_component_t0.143546:c=234 | 11 | 10 | 1 | 0.0909 | 0.0010 | 12.2938 |
| 28.0000 | hdbscan | hdbscan_mcs50_ms5:c=0 | 193 | 182 | 11 | 0.0570 | 0.0110 | 6.9523 |
| 29.0000 | hdbscan | hdbscan_mcs80_ms5:c=0 | 193 | 182 | 11 | 0.0570 | 0.0110 | 6.9523 |
| 30.0000 | hdbscan | hdbscan_mcs20_ms5:c=0 | 20 | 19 | 1 | 0.0500 | 0.0010 | 5.4762 |

## 6. Top-N 并集评估

| top_cluster_count | selected_account_count | human_count | bot_count | bot_precision | bot_recall |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 184 | 0 | 184 | 1.0000 | 0.1846 |
| 3.0000 | 315 | 28 | 287 | 0.9111 | 0.2879 |
| 5.0000 | 315 | 28 | 287 | 0.9111 | 0.2879 |
| 10.0000 | 404 | 35 | 369 | 0.9134 | 0.3701 |
| 20.0000 | 497 | 128 | 369 | 0.7425 | 0.3701 |
| 31.0000 | 639 | 258 | 381 | 0.5962 | 0.3821 |

## 7. 分方法表现

| method | clusters | high_purity_ge_095 | max_bot_rate | max_bot_recall | best_rank |
| --- | --- | --- | --- | --- | --- |
| dense_component | 6 | 2 | 1.0000 | 0.1846 | 1.0000 |
| hdbscan | 25 | 3 | 1.0000 | 0.1474 | 3.0000 |

## 8. 图表

- kNN 相似度分布：`unsupervised/outputs/density_original_human_vs_oldbot/figures/density_knn_similarity_distribution.png`
- Top cluster purity：`unsupervised/outputs/density_original_human_vs_oldbot/figures/density_top_cluster_purity.png`
- Suspicious score vs purity：`unsupervised/outputs/density_original_human_vs_oldbot/figures/density_score_vs_purity.png`
- 2D 展示图：`unsupervised/outputs/density_original_human_vs_oldbot/figures/density_account_embedding_2d.png`
- 2D 降维方法：`PCA fallback`

## 9. 结论口径

- 这不是二分类实验，不假设世界只有 human cluster 和 bot cluster。
- HDBSCAN 和 dense component scan 直接寻找高密度账号群，更贴合“某些账号集中在某个点上”的假设。
- 标签揭示只用于评估这些高密度簇是否富含 old bot。
- 2D 图只用于展示，不作为判定依据；真实判定来自原始 TF-IDF 空间的密度、半径和 null baseline。
