# 密度式账号机制簇发现：HDBSCAN + Dense Component Scan 实验汇报

生成时间：2026-06-04  
实验目录：`unsupervised`  
输出目录：`outputs/density_original_human_vs_oldbot/`  
实验对象：`original_human` vs `original_bot`  
核心方法：`word TF-IDF -> HDBSCAN + dense component scan -> suspicious dense clusters -> label reveal -> PCA/UMAP 2D 展示`

---

## 1. 为什么要从 KMeans 改成密度发现

我们上一版用了 KMeans。KMeans 的优点是简单、稳定、可复现，但它和当前问题并不完全匹配。

KMeans 回答的是：

```text
如果我必须把所有账号切成 k 个簇，怎样切能让簇内平方距离最小？
```

但我们真正关心的问题不是“所有账号应该被分成几个类”，而是：

```text
在账号文本分布空间里，是否有某些账号异常集中？
```

也就是说，我们不是要强行把世界分成：

```text
cluster 1 = 人
cluster 2 = 机器人
```

而是要找：

```text
某个局部区域里账号特别密，像是同一套生成机制产生的。
```

这正是用户提出的直觉：

> 按照 TF-IDF 的结果看，词汇使用频率就够了。画出分布后，某些账户集中在某个点上，特别高，那么一圈就能知道这是一批机器人。

因此第二阶段改用更贴合这个直觉的方法：

1. 用 `word TF-IDF` 表示账号级词汇使用分布；
2. 用 `HDBSCAN` 找密度簇；
3. 用 `dense component scan` 在 TF-IDF cosine kNN 图里找高相似连通组件；
4. 对每个候选簇计算相对随机账号组的异常密度；
5. 最后才揭示标签，看这些高密度簇是不是 old bot。

---

## 2. 研究口径

本实验仍然坚持无监督边界：

- 标签不参与 TF-IDF；
- 标签不参与 HDBSCAN；
- 标签不参与 kNN 图；
- 标签不参与 dense component scan；
- 标签不参与 suspicious score；
- 标签只在最后用于 `bot purity / precision / recall` 评估。

我们发现的对象不是“单账号一定是 bot”，而是：

```text
suspicious dense cluster
```

更严谨的中文叫法可以是：

```text
可疑高密度账号机制簇
```

如果一个簇在无标签 TF-IDF 空间里异常集中，且最后揭示标签后大部分都是 bot，就说明它捕捉到了机器人生成机制的语言指纹。

---

## 3. 数据口径

输入文件：

```text
inputs/static_family/combined_sample_index.csv
```

筛选条件：

```text
family in {original_human, original_bot}
doc 非空
word_count >= 10
```

进入完整实验的数据规模：

| 类别 | 账号数 |
| --- | ---: |
| original_human | 1674 |
| original_bot | 997 |
| 合计 | 2671 |

输出文件中标签隔离情况：

```text
density_account_features.csv shape = (2671, 34)
label_like_cols = []
```

标签只保存于：

```text
density_label_revealed_account_metadata.csv
density_label_revealed_evaluation.csv
```

---

## 4. 方法总体流程

完整流程如下：

```text
账号聚合文本
  -> word 1-2gram TF-IDF
  -> 路线 A：SVD 降维 -> HDBSCAN
  -> 路线 B：原始 TF-IDF cosine kNN 图 -> 高阈值连通组件扫描
  -> 候选 dense clusters
  -> 在原始 TF-IDF 空间计算密度、半径、随机 null baseline
  -> suspicious score 排名
  -> 最后揭示标签评估 purity / recall
  -> PCA/UMAP 二维展示
```

这套流程里有两个核心思想：

1. `HDBSCAN` 负责从连续向量空间中找密度簇；
2. `dense component scan` 负责从高相似账号图中找“高相似连成团”的局部组件。

二者互补：

- HDBSCAN 更像空间密度估计；
- dense component scan 更像“账号文本相似图上的高密团块搜索”。

---

## 5. TF-IDF 表示

主空间使用 word TF-IDF：

```text
analyzer = word
ngram_range = (1, 2)
max_features = 50000
stop_words = english
norm = l2
```

为什么只用 word TF-IDF？

因为当前用户提出的重点是：

> 词汇使用频率就够了。

这里我们先故意不引入复杂模型，直接检验最朴素的假设：

```text
如果多个机器人账号来自同一机制，它们在词汇使用分布上是否会聚成高密点？
```

TF-IDF 的优点是：

- 可解释；
- 不需要监督标签；
- 可以直接用 cosine similarity；
- 能捕捉账号整体词汇使用习惯；
- 结果比深度 embedding 更容易向老师解释。

---

## 6. HDBSCAN 路线

### 6.1 为什么 HDBSCAN 合适

HDBSCAN 的目标不是预设 `k` 个簇，而是自动寻找不同密度层级中的稳定簇。

它适合回答：

```text
哪些账号在向量空间中形成自然高密度区域？
```

这比 KMeans 更贴近“某些账号集中在某个点上”的直觉。

### 6.2 为什么先 SVD

原始 TF-IDF 是高维稀疏空间，可能有几万维。HDBSCAN 直接在这种空间里估计密度会不稳定，也比较慢。

所以实现中先做：

```text
TF-IDF -> TruncatedSVD(100) -> StandardScaler -> L2 normalize
```

正式参数：

```text
svd_components = 100
hdbscan_min_cluster_sizes = 10,20,30,50,80
hdbscan_min_samples = 5
```

### 6.3 HDBSCAN 输出

HDBSCAN 会输出：

- cluster label；
- noise label `-1`；
- membership strength；
- 不同 `min_cluster_size` 下的多个候选簇。

我们保留：

```text
非 noise
cluster_size <= max_cluster_fraction * 全部账号数
```

正式实验中：

```text
max_cluster_fraction = 0.25
```

也就是不允许一个“可疑簇”大到覆盖太多账号，否则它更像粗粒度群体，而不是局部高密点。

---

## 7. Dense Component Scan 路线

### 7.1 为什么需要 dense component scan

HDBSCAN 是空间密度算法，但我们还想要一种更直观的方法：

```text
如果两个账号 TF-IDF cosine similarity 很高，就连边。
如果一批账号在高相似阈值下还能连成一团，就说明它们非常像。
```

这不是社交网络图，也不是用户互动图，而是：

```text
账号文本相似图
```

节点：

```text
账号
```

边：

```text
两个账号的 TF-IDF cosine similarity 足够高
```

这和 Louvain 那种社交网络社区发现不同。我们这里没有用 Louvain，而是先做更直接的高密组件扫描：

```text
kNN 相似图 -> 高相似阈值过滤 -> connected components -> 内部边密度过滤
```

### 7.2 正式参数

正式实验中：

```text
component_knn_k = 50
component_threshold_quantiles = 0.90,0.95,0.975,0.99
component_min_size = 10
component_min_edge_density = 0.025
max_cluster_fraction = 0.25
```

### 7.3 实际阈值

根据 kNN 边相似度分布，自动得到阈值：

| quantile | cosine threshold |
| ---: | ---: |
| 0.90 | 0.120522 |
| 0.95 | 0.132899 |
| 0.975 | 0.143546 |
| 0.99 | 0.157668 |

这表示：

```text
只保留 TF-IDF kNN 图中相似度最高的一部分边。
```

### 7.4 为什么要加内部边密度门槛

第一次完整运行时，较低阈值会形成很大的“链式连通组件”。这种组件虽然连通，但可能不是一个真正紧密的点团。

例如：

```text
A 像 B
B 像 C
C 像 D
...
```

这样可能形成一个很大的 connected component，但内部并不够稠密。

所以正式版加入：

```text
component_min_edge_density = 0.025
max_cluster_fraction = 0.25
```

这让 dense component scan 更贴近用户说的：

> 某些账户集中在某个点上，特别高，一圈就知道。

也就是说，保留“小而密”的局部组件，而不是低阈值下的大链条。

---

## 8. Suspicious Score

每个 HDBSCAN 或 dense component 候选簇都在原始 TF-IDF 空间里重新评分。

评分项包括：

1. 簇内平均 cosine similarity；
2. 簇半径；
3. 与同规模随机账号组的 null baseline 对比；
4. dense component 的内部图边密度；
5. HDBSCAN 的 membership strength；
6. 簇规模校正。

核心思想：

```text
一个簇只有比随机同规模账号组明显更紧、更相似，才应该排到前面。
```

随机基线：

```text
null_repeats = 300
```

输出文件：

```text
density_suspicious_cluster_scores.csv
```

---

## 9. 输出文件

正式输出目录：

```text
outputs/density_original_human_vs_oldbot/
```

主要输出：

| 文件 | 说明 |
| --- | --- |
| `density_account_features.csv` | 无标签账号特征与 TF-IDF/SVD 元信息 |
| `density_label_revealed_account_metadata.csv` | 标签揭示元数据，只用于最终评估 |
| `density_similarity_edges.csv` | TF-IDF cosine kNN 无向边 |
| `density_similarity_directed_knn_edges.csv` | TF-IDF cosine kNN 有向近邻边 |
| `density_cluster_assignments.csv` | HDBSCAN 与 dense component assignment |
| `density_suspicious_cluster_scores.csv` | 无标签 suspicious score 排名 |
| `density_label_revealed_evaluation.csv` | 最后揭示标签后的 purity/recall |
| `DENSITY_ACCOUNT_DISCOVERY_CN.md` | 自动生成结果报告 |

图表：

| 图 | 说明 |
| --- | --- |
| `density_knn_similarity_distribution.png` | kNN cosine 相似度分布与阈值线 |
| `density_top_cluster_purity.png` | Top suspicious clusters 标签揭示后的 bot purity |
| `density_score_vs_purity.png` | suspicious score 与揭示后 bot purity 的关系 |
| `density_account_embedding_2d.png` | PCA/UMAP 二维展示图 |

---

## 10. Smoke Test

smoke 设置：

```text
sample_human = 100
sample_bot = 100
hdbscan_min_cluster_sizes = 10,20
hdbscan_min_samples = 5
component_knn_k = 20
component_threshold_quantiles = 0.90,0.95,0.98
component_min_size = 5
component_min_edge_density = 0.025
max_cluster_fraction = 0.25
null_repeats = 50
```

smoke 输出：

```text
outputs/density_smoke_original_100_100/
```

用途：

- 验证脚本能完整运行；
- 验证 HDBSCAN、dense component scan、评分、标签揭示、图表生成都可用；
- 验证无标签特征文件没有 label/family/bot/human 泄漏列。

---

## 11. 完整实验结果

完整实验规模：

```text
accounts = 2671
human = 1674
old bot = 997
```

输出候选簇：

| 来源方法 | suspicious clusters |
| --- | ---: |
| dense_component | 6 |
| hdbscan | 25 |
| 合计 | 31 |

assignment 行数：

```text
density_cluster_assignments.csv = 13802 行
```

其中：

| method | assignment 行数 |
| --- | ---: |
| hdbscan | 13355 |
| dense_component | 447 |

---

## 12. Top suspicious dense clusters

前 14 个结果如下：

| rank | method | cluster | size | human | bot | bot_rate | bot_recall |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | dense_component | `dense_component_t0.157668:c=1370` | 184 | 0 | 184 | 1.0000 | 0.1846 |
| 2 | dense_component | `dense_component_t0.157668:c=1381` | 51 | 0 | 51 | 1.0000 | 0.0512 |
| 3 | hdbscan | `hdbscan_mcs20_ms5:c=4` | 175 | 28 | 147 | 0.8400 | 0.1474 |
| 4 | hdbscan | `hdbscan_mcs80_ms5:c=1` | 175 | 28 | 147 | 0.8400 | 0.1474 |
| 5 | hdbscan | `hdbscan_mcs50_ms5:c=1` | 175 | 28 | 147 | 0.8400 | 0.1474 |
| 6 | hdbscan | `hdbscan_mcs30_ms5:c=2` | 175 | 28 | 147 | 0.8400 | 0.1474 |
| 7 | hdbscan | `hdbscan_mcs10_ms5:c=7` | 65 | 3 | 62 | 0.9538 | 0.0622 |
| 8 | hdbscan | `hdbscan_mcs10_ms5:c=6` | 14 | 0 | 14 | 1.0000 | 0.0140 |
| 9 | hdbscan | `hdbscan_mcs10_ms5:c=5` | 110 | 7 | 103 | 0.9364 | 0.1033 |
| 10 | hdbscan | `hdbscan_mcs20_ms5:c=5` | 110 | 7 | 103 | 0.9364 | 0.1033 |
| 11 | hdbscan | `hdbscan_mcs30_ms5:c=3` | 110 | 7 | 103 | 0.9364 | 0.1033 |
| 12 | hdbscan | `hdbscan_mcs50_ms5:c=2` | 110 | 7 | 103 | 0.9364 | 0.1033 |
| 13 | hdbscan | `hdbscan_mcs80_ms5:c=2` | 110 | 7 | 103 | 0.9364 | 0.1033 |
| 14 | hdbscan | `hdbscan_mcs10_ms5:c=8` | 20 | 0 | 20 | 1.0000 | 0.0201 |

最重要的发现：

```text
Top 1 dense component:
size = 184
human = 0
bot = 184
bot_rate = 1.0000
bot_recall = 0.1846
```

这非常符合用户提出的直觉：

> 有一团账号在 TF-IDF 词汇使用空间里高度集中，最后揭示标签发现这一团全是 bot。

Top 2 也一样：

```text
Top 2 dense component:
size = 51
human = 0
bot = 51
bot_rate = 1.0000
```

这说明 dense component scan 找到的最强两个“高相似点团”确实是纯 old bot 机制簇。

---

## 13. Top-N 并集评估

| Top-N clusters | selected accounts | human | bot | bot precision | bot recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 184 | 0 | 184 | 1.0000 | 0.1846 |
| 3 | 315 | 28 | 287 | 0.9111 | 0.2879 |
| 5 | 315 | 28 | 287 | 0.9111 | 0.2879 |
| 10 | 404 | 35 | 369 | 0.9134 | 0.3701 |
| 20 | 497 | 128 | 369 | 0.7425 | 0.3701 |
| 31 | 639 | 258 | 381 | 0.5962 | 0.3821 |

如何理解：

- Top 1：极高置信，precision 100%，但只覆盖 18.46% bot。
- Top 3：precision 91.11%，recall 28.79%。
- Top 10：precision 91.34%，recall 37.01%。
- Top 20：recall 没增加，但混入更多 human，precision 降到 74.25%。
- 全部 31 个候选簇：precision 59.62%，recall 38.21%。

这和 KMeans 基线不一样。

KMeans 基线中 Top 1 style cluster 覆盖更大：

```text
precision = 0.9638
recall = 0.5346
```

而 density 方法更保守：

```text
Top 1 precision = 1.0000
Top 1 recall = 0.1846
```

因此两者定位不同：

| 方法 | 更擅长什么 |
| --- | --- |
| KMeans 基线 | 找一个大范围分布差异簇，recall 更高 |
| Density 方法 | 找局部特别集中的高置信机制簇，precision 更高 |

这正是本实验的意义：density 方法更接近“圈出一个特别密的点团”。

---

## 14. HDBSCAN 与 Dense Component 的差异

### 14.1 Dense Component

结果：

| 指标 | 数值 |
| --- | ---: |
| 候选簇数 | 6 |
| bot_rate >= 0.95 的簇数 | 2 |
| 最高 bot_rate | 1.0000 |
| 单簇最高 recall | 0.1846 |
| 最好排名 | 1 |

特点：

- 排名前两位都是纯 bot；
- 非常符合“高相似边连成团”的直觉；
- 找到的是局部极高置信点团；
- recall 较低。

### 14.2 HDBSCAN

结果：

| 指标 | 数值 |
| --- | ---: |
| 候选簇数 | 25 |
| bot_rate >= 0.95 的簇数 | 3 |
| 最高 bot_rate | 1.0000 |
| 单簇最高 recall | 0.1474 |
| 最好排名 | 3 |

特点：

- 能找到一些高纯 bot 簇；
- 也会找到 human 高密簇；
- 对 SVD 空间和参数比较敏感；
- 比 dense component 更像“空间密度簇发现”。

### 14.3 结论

在当前数据上：

```text
dense component scan 更符合用户提出的“一圈圈出高密 bot 点团”的直觉。
HDBSCAN 可以作为补充，发现其他密度结构。
```

---

## 15. 为什么 2D 图只做展示

用户提出“画出一个分布，一眼就能看出来”，这是很好的展示方式，但不能把 2D 图当作唯一判定依据。

原因是：

```text
TF-IDF 是高维空间。
PCA/UMAP 会压缩和扭曲距离。
二维图上的距离不完全等于原始空间距离。
```

所以本实验的判定依据仍然是：

```text
原始 TF-IDF cosine space 中的密度、半径、kNN 边、null baseline。
```

2D 图只是展示：

```text
density_account_embedding_2d.png
```

当前环境没有安装 UMAP，因此使用：

```text
PCA fallback
```

这张图可以帮助直观检查 top suspicious cluster 是否在二维投影中也比较集中，但不能作为严格证据。

---

## 16. 图表路径

正式图表目录：

```text
outputs/density_original_human_vs_oldbot/figures/
```

图表：

```text
density_knn_similarity_distribution.png
density_top_cluster_purity.png
density_score_vs_purity.png
density_account_embedding_2d.png
```

对应解释：

- `density_knn_similarity_distribution.png`：展示 TF-IDF kNN 边的相似度分布和自动阈值线。
- `density_top_cluster_purity.png`：展示 Top suspicious clusters 揭示标签后的 bot purity。
- `density_score_vs_purity.png`：展示无标签 suspicious score 与最后揭示 purity 的关系。
- `density_account_embedding_2d.png`：PCA/UMAP 二维展示，当前是 PCA fallback。

---

## 17. 方法局限

### 17.1 高密度不必然等于 bot

本实验仍然发现了一些高密 human 簇。例如 HDBSCAN 和 dense component 后段中有若干簇 bot_rate 为 0。

这说明：

```text
高密度 = 机制相似
但机制相似不一定 = 机器人
```

人类账号也可能因为文本长度、话题、表达方式或数据采样机制形成高密度团。

所以结论必须写成：

```text
发现 suspicious dense clusters，其中部分在标签揭示后高度富含 bot。
```

不能写成：

```text
无监督直接识别所有 bot。
```

### 17.2 Recall 较低

Density 方法更保守。Top 1 precision 是 100%，但 recall 只有 18.46%。

这意味着它更适合作为：

```text
高置信 botnet seed discovery
```

而不是完整召回所有 bot 的检测器。

### 17.3 参数敏感

HDBSCAN 对：

```text
min_cluster_size
min_samples
SVD 维度
```

敏感。

Dense component scan 对：

```text
kNN k
cosine threshold
min edge density
max cluster fraction
```

敏感。

后续需要做参数稳定性实验。

### 17.4 当前只用语言

本实验只用账号聚合文本的 word TF-IDF，没有用：

- 时间行为；
- 回复层级；
- 社交互动图；
- OP 回访；
- 主题迁移；
- 发帖/评论比例。

后续可以把这些作为多视角 density discovery。

---

## 18. 下一步建议

### 18.1 做参数稳定性图

建议系统扫描：

```text
component_knn_k = 20, 30, 50, 80
threshold_quantiles = 0.95, 0.975, 0.99, 0.995
min_edge_density = 0.02, 0.025, 0.03, 0.05
```

看 Top 1 / Top 3 / Top 5 的 precision 和 recall 是否稳定。

### 18.2 扩展到 F1-F5

下一步可以把 F1-F5 加入，观察：

- F1-F5 是否形成自己的高密簇；
- 哪些 family 更分散；
- 哪些 family 和 old bot 聚在一起；
- 哪些 family 更接近 human。

这会变成：

```text
无监督 family discovery
```

### 18.3 加入图结构/时间行为 density

可以构造多视角：

```text
word_tfidf density
style_numeric density
bge density
graph_behavior density
temporal_behavior density
```

如果一个账号群在语言、图结构、时间行为上都高密，那证据会更强。

### 18.4 和监督泛化检测结合

密度方法可以作为“未知 family 发现器”：

1. 在未标注数据中发现 suspicious dense clusters；
2. 人工或少量标签揭示；
3. 把高置信簇作为新 family；
4. 测试监督 detector 是否能泛化到这些新 family；
5. 形成“发现新机制 -> 纳入训练 -> 再发现新机制”的闭环。

---

## 19. 复现命令

正式实验：

```bash
cd "[local-path-removed]"

python \
  "unsupervised/code/run_density_account_discovery.py" \
  --output-dir "unsupervised/outputs/density_original_human_vs_oldbot" \
  --hdbscan-min-cluster-sizes 10,20,30,50,80 \
  --hdbscan-min-samples 5 \
  --component-knn-k 50 \
  --component-threshold-quantiles 0.90,0.95,0.975,0.99 \
  --component-min-size 10 \
  --component-min-edge-density 0.025 \
  --max-cluster-fraction 0.25 \
  --null-repeats 300 \
  --top-n-report 30
```

Smoke test：

```bash
cd "[local-path-removed]"

python \
  "unsupervised/code/run_density_account_discovery.py" \
  --output-dir "unsupervised/outputs/density_smoke_original_100_100" \
  --sample-human 100 \
  --sample-bot 100 \
  --hdbscan-min-cluster-sizes 10,20 \
  --hdbscan-min-samples 5 \
  --component-knn-k 20 \
  --component-threshold-quantiles 0.90,0.95,0.98 \
  --component-min-size 5 \
  --component-min-edge-density 0.025 \
  --max-cluster-fraction 0.25 \
  --null-repeats 50 \
  --top-n-report 20
```

---

## 20. 最终判断

这版实验完成了用户提出的核心设想：

> 不再用 KMeans 强行切类，而是在 TF-IDF 词汇使用空间中寻找局部高密点团。

结果证明：

1. 这种方法能找到非常高置信的 bot 机制簇；
2. 最强 dense component 包含 184 个账号，揭示标签后全是 bot；
3. Top 2 dense components 也都是 100% bot；
4. Top 10 clusters 并集仍有 91.34% precision；
5. 但 recall 比 KMeans 低，说明它更适合做高置信 seed discovery，而不是全量检测。

更准确的研究结论是：

> 在 original_human 与 original_bot 混合数据中，基于 word TF-IDF 的密度式无监督方法可以发现一批局部异常集中的账号机制簇；其中排名最靠前的 dense components 在标签揭示后呈现 100% bot purity。这说明机器人生成机制确实能在账号级词汇分布空间中形成可无监督发现的高密度指纹。

