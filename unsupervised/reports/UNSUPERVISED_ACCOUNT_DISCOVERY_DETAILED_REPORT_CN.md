# unsupervised：第一阶段详细实验汇报

生成时间：2026-06-04  
实验目录：`unsupervised`  
第一阶段对象：`original_human` vs `original_bot`  
核心问题：在不使用标签训练分类器的情况下，能否仅从账号文本分布的集中性中发现一批疑似同机制生成的机器人账号？

---

## 1. 研究背景与问题转向

我们前面做了很多监督检测和生成式 bot 对抗实验，已经反复看到一个现象：只要同一批机器人来自同一套生成机制，它们往往会留下某种“机制指纹”。这个指纹不一定是单个词，也不一定是单一 topic，而可能体现在：

- 多个账号的语言分布过于集中；
- 多个账号在词法、句长、标点、抽象程度、句式组合上互相接近；
- 多个账号在语义嵌入空间里形成高密度小团体；
- 多个账号不像真人那样分散，而是围绕某种生成模板或机制聚集。

这启发了一个新的研究方向：我们不再只问“给一个账号，监督模型能不能判定它是 bot”，而是问：

> 如果没有标签，我们能不能发现一组账号明显来自相似机制，从而把它们标记为 suspicious mechanism cluster？

这里的重点不是直接对单个账号下最终判定，而是发现“机制集中簇”。这和传统监督分类不同：

- 监督分类：训练时知道 human/bot 标签，学习一个边界。
- 无监督机制发现：训练/聚类时不知道标签，只寻找异常高密度、异常一致的账号群。
- 标签揭示：只在最后用于评估我们发现的簇是不是确实富含 bot。

因此，本实验的研究口径是：

> 先无标签发现机制簇，再揭示标签评估 cluster purity、bot recall 和 human false positive。

这个口径更接近现实中的未知 botnet 发现：我们不一定知道哪些账号是 bot，但如果一批账号在语言空间里异常相似，就值得作为一个群体机制被审计。

---

## 2. 第一阶段实验目标

第一阶段只做最小可行验证，不引入静态 F1-F5 family 的复杂性。

实验目标：

1. 新建独立工作区，集中保存无监督检测需要的数据、脚本、输出和报告。
2. 搬运原始 BotSim 数据与静态 family 数据，保证后续可扩展。
3. 第一阶段只使用 `original_human` 和 `original_bot`。
4. 把每个账号作为一个样本，账号文本 `doc` 是该账号聚合评论文本。
5. 聚类、相似图、可疑簇评分阶段完全不使用标签。
6. 最后再揭示标签，评估 suspicious clusters 的 bot purity 和 bot recall。
7. 输出可复现脚本、CSV、图表和中文报告。

本阶段不是为了证明“所有机器人都能被无监督识别”，而是验证一个更稳妥的命题：

> 在原始 old bot 与 human 混合时，是否存在一批无需标签也能被发现的高密度机制簇，并且标签揭示后这些簇显著富含 old bot？

---

## 3. 工作区结构

新建目录：

```text
unsupervised/
  README.md
  code/
    run_unsupervised_account_discovery.py
    plot_unsupervised_results.py
  inputs/
    source_manifest.csv
    raw_original/
      Users.csv
      user_post_comment.json
    static_family/
      combined_sample_index.csv
      combined_bge_embeddings.npy
      generated_comments_all.csv
      generated_profiles_all.csv
      families/
        F1_original_architecture/
        F2_tfidf_informed_language/
        F3_human_anchor_local_tfidf/
        F4_peba_trait_prompt/
        F5_gemini_trait_prompt/
  outputs/
    smoke_original_100_100/
    original_human_vs_oldbot/
  reports/
    UNSUPERVISED_ACCOUNT_DISCOVERY_DETAILED_REPORT_CN.md
```

其中：

- `inputs/raw_original/` 保存 BotSim 原始数据，用于溯源。
- `inputs/static_family/` 保存此前静态 family 实验的整理数据和 F1-F5 结果。
- `inputs/source_manifest.csv` 记录每个输入文件来源、用途、是否第一阶段使用、是否 optional。
- `code/run_unsupervised_account_discovery.py` 是主实验脚本。
- `code/plot_unsupervised_results.py` 负责图表和简版中文结果报告。
- `outputs/original_human_vs_oldbot/` 是完整第一阶段输出。
- `reports/` 保存这份详细研究汇报。

---

## 4. 数据来源与使用边界

### 4.1 已搬运数据

`source_manifest.csv` 当前共记录 `52` 个输入文件。

第一阶段实际使用两个输入：

| 文件 | 第一阶段用途 | 是否可选 |
| --- | --- | --- |
| `inputs/static_family/combined_sample_index.csv` | 账号级样本索引，筛选 `original_human` 和 `original_bot` | 否 |
| `inputs/static_family/combined_bge_embeddings.npy` | BGE 向量缓存，用作可选特征视角 | 是 |

同步搬运但第一阶段不直接使用：

| 文件/目录 | 用途 |
| --- | --- |
| `inputs/raw_original/Users.csv` | 原始用户元数据溯源 |
| `inputs/raw_original/user_post_comment.json` | 原始帖子/评论文本溯源 |
| `inputs/static_family/generated_comments_all.csv` | 后续扩展到 F1-F5 时使用 |
| `inputs/static_family/generated_profiles_all.csv` | 后续 family/persona 级分析使用 |
| `inputs/static_family/families/F1-F5/` | 后续静态 family 无监督发现和泛化分析使用 |

### 4.2 `combined_sample_index.csv` 的 family 分布

当前整理好的账号级索引中，各 family 数量如下：

| family | 账号数 |
| --- | ---: |
| `original_human` | 1907 |
| `original_bot` | 1000 |
| `F1_original_architecture` | 50 |
| `F2_tfidf_informed_language` | 50 |
| `F3_human_anchor_local_tfidf` | 50 |
| `F4_peba_trait_prompt` | 50 |
| `F5_gemini_trait_prompt` | 50 |

第一阶段只保留：

```text
family in {original_human, original_bot}
```

再过滤：

```text
doc 非空
word_count >= 10
```

过滤后的完整实验规模：

| 类别 | 账号数 |
| --- | ---: |
| human | 1674 |
| old bot | 997 |
| 合计 | 2671 |

注意：原始索引中有 `1907` 个 human 和 `1000` 个 old bot，但过滤极短/空文本后，进入实验的是 `1674` 个 human 和 `997` 个 bot。

---

## 5. 标签隔离设计

这是本实验最重要的实现边界。

无监督阶段不得使用标签。具体包括：

- TF-IDF 特征构造不使用标签。
- BGE 嵌入读取不使用标签。
- style numeric 特征构造不使用标签。
- kNN 相似图不使用标签。
- KMeans / MiniBatchKMeans 聚类不使用标签。
- suspicious score 计算不使用标签。
- 可疑簇排名不使用标签。

标签只在最后两个文件中出现：

| 输出文件 | 标签用途 |
| --- | --- |
| `label_revealed_account_metadata.csv` | 保存账号级揭示标签，只用于最终评估和可视化着色 |
| `label_revealed_evaluation.csv` | 计算各 suspicious cluster 的 bot purity、precision、recall |

我还专门做了泄漏检查：

```text
account_features.csv shape = (2671, 36)
label_like_cols = []
```

也就是说，最终完整实验的 `account_features.csv` 中没有任何包含以下关键词的列：

```text
label / family / bot / human
```

这保证后续如果有人直接拿 `account_features.csv` 做无监督分析，不会误把标签列当成特征。

---

## 6. 账号级样本构造

本实验单位不是单条评论，而是账号。

原因是：我们的无监督假设本身就是“多个账号的语言分布过于集中”。如果以单条评论为单位，可能会混入太多 topic 或短文本噪声；以账号为单位，可以观察一个账号整体的语言习惯。

主输入字段：

- `account_id`：账号唯一标识。
- `account_id_raw`：原始账号标识。
- `doc`：账号聚合文本。
- `word_count`：账号聚合文本词数。
- `family`：只用于筛选第一阶段数据和最后评估。

脚本中的账号选择逻辑：

```python
mask = (
    work["family"].astype(str).isin(TARGET_FAMILIES)
    & work["doc"].str.strip().ne("")
    & word_count.ge(min_words)
)
```

默认参数：

```text
TARGET_FAMILIES = ("original_human", "original_bot")
min_words = 10
random_state = 42
```

如果同一个 `account_id` 出现重复，脚本会自动加后缀去重，例如：

```text
user_x
user_x__dup2
```

这样可以避免相似图和聚类输出中账号 ID 冲突。

---

## 7. 特征视角设计

本实验没有只依赖单一表示，而是用了四个互补视角：

1. `word_tfidf`
2. `char_wb_tfidf`
3. `style_numeric`
4. `bge`

### 7.1 word TF-IDF

目标：捕捉账号级词项和短语分布。

实现：

```text
analyzer = word
ngram_range = (1, 2)
max_features = 50000
stop_words = english
norm = l2
min_df = 2，失败则降到 1
max_df = 0.95
```

解释：

- unigram 捕捉单词特征。
- bigram 捕捉短语组合。
- stop words 去除非常常见英文虚词。
- L2 normalization 让余弦相似度有意义。

这个视角适合发现词汇层面的机制相似，例如多个 bot 账号重复使用类似政治、冲突、抽象判断词。

### 7.2 char_wb TF-IDF

目标：捕捉字符级风格、词形、缩写、标点邻近模式。

实现：

```text
analyzer = char_wb
ngram_range = (3, 5)
max_features = 80000
norm = l2
min_df = 2，失败则降到 1
max_df = 0.95
```

解释：

- `char_wb` 表示在 word boundary 内做字符 ngram。
- 比 word TF-IDF 更能捕捉风格和拼写习惯。
- 对一些口语缩写、标点、常见词形组合更敏感。

这个视角适合发现文本表层形态上的模板化痕迹。

### 7.3 style_numeric

目标：不看具体词义，只看账号文本的统计风格。

当前实现了 22 个基础风格特征，最后标准化并 L2 normalize。

主要包括：

| 特征类别 | 示例 |
| --- | --- |
| 文本长度 | `style_char_count`, `style_nonspace_count`, `style_token_count` |
| 词汇多样性 | `style_unique_token_ratio` |
| 词长 | `style_mean_token_len`, `style_std_token_len` |
| 句子结构 | `style_sentence_mark_count`, `style_mean_tokens_per_sentence` |
| URL/数字 | `style_url_count`, `style_digit_ratio` |
| 大写 | `style_upper_alpha_ratio` |
| 标点 | `style_punct_ratio`, `style_question_ratio`, `style_exclaim_ratio`, `style_comma_ratio` |
| 符号 | `style_quote_ratio`, `style_parenthesis_ratio` |
| 换行 | `style_newline_count` |
| 输入统计 | `style_input_comment_count`, `style_input_word_count`, `style_filter_word_count` |

这个视角非常关键。它不关心 topic，不直接看“Ukraine / military / politics”这类词，而是看整体表达统计是否集中。

### 7.4 BGE embedding

目标：利用语义嵌入观察账号聚合文本在语义空间中的聚集。

实现：

- 优先读取 `combined_bge_embeddings.npy`。
- 若文件不存在或行数与 `combined_sample_index.csv` 不匹配，则自动跳过。
- 当前完整实验中 BGE 缓存可用，所以启用了 `bge` 视角。
- BGE 向量同样做 L2 normalization。

这个视角适合发现语义层面的集中性，而不是简单词面重复。

---

## 8. 相似图构造

每个特征视角都构建一个 kNN 相似图。

默认参数：

```text
knn-k = 20
metric = cosine
algorithm = brute
```

输出文件：

```text
account_similarity_edges.csv
```

完整实验输出规模：

```text
213680 行 x 7 列
```

计算方式：

1. 对每个账号，在同一视角空间里找最近的 `20` 个账号。
2. 排除自己。
3. 保存：
   - `view`
   - `source_account_id`
   - `target_account_id`
   - `rank`
   - `cosine_similarity`

这个相似图不是最终检测器，而是为了：

- 检查账号之间是否出现异常密集邻接；
- 支持后续多视角一致性分析；
- 后续可以扩展成真正的图社区发现方法。

---

## 9. 聚类方法

本阶段使用 KMeans 系列作为最简单、稳定、可复现的无监督基线。

默认聚类数：

```text
k in {8, 12, 20, 30}
```

每个视角都跑这 4 个 k。

视角和算法：

| 视角 | 算法 |
| --- | --- |
| `word_tfidf` | `MiniBatchKMeans` |
| `char_wb_tfidf` | `MiniBatchKMeans` |
| `style_numeric` | `KMeans` |
| `bge` | `KMeans` |

为什么有的是 MiniBatchKMeans？

脚本中规则是：

```python
use_minibatch = sparse.issparse(view.matrix) or n_samples > 5000 or n_features > 10000
```

word/char TF-IDF 是高维稀疏矩阵，因此使用 MiniBatchKMeans；style 和 BGE 是较低维稠密矩阵，因此使用普通 KMeans。

完整实验一共产生：

```text
4 个 view x 4 个 k = 16 组聚类
每组聚类产生 k 个 cluster
总机制簇数 = 8+12+20+30，乘以 4 个视角 = 280 个 cluster
```

输出文件：

```text
cluster_assignments.csv
```

完整输出规模：

```text
42736 行 x 7 列
```

解释：

```text
2671 个账号 x 16 组聚类 = 42736 行
```

---

## 10. Suspicious mechanism cluster 评分

聚类本身只是把账号分组。我们还需要判断哪些簇“异常集中”。

本实验对每个 cluster 计算：

1. 簇内平均相似度；
2. 簇半径；
3. 与同规模随机账号组的 null baseline 对比；
4. 多视角支持数量；
5. 簇大小校正；
6. 综合 suspicious score。

### 10.1 簇内平均相似度

对于一个 cluster，假设其中账号向量为：

```text
x_1, x_2, ..., x_n
```

向量已经 L2 normalize。

簇内平均相似度近似计算为：

```text
mean_internal_similarity =
( ||sum_i x_i||^2 - sum_i ||x_i||^2 ) / (n * (n - 1))
```

直观解释：

- 如果一个簇里账号语言非常相似，向量方向接近，`||sum_i x_i||` 会很大。
- `mean_internal_similarity` 越高，说明这个簇越集中。

### 10.2 簇半径

脚本定义：

```text
mean_radius = 1 - ||sum_i x_i|| / n
```

直观解释：

- 如果簇内账号高度一致，`||sum_i x_i|| / n` 接近 1，半径就小。
- 如果账号方向很散，半径就大。

因此，可疑簇通常表现为：

```text
mean_internal_similarity 高
mean_radius 低
```

### 10.3 随机 null baseline

仅看一个 cluster 的密度不够，因为小簇天然可能更集中，大簇天然可能更分散。

所以脚本对每个 `(view, cluster_size)` 做随机基线：

```text
随机抽取同样大小的账号组
重复 null_repeats 次
计算随机组的 mean_internal_similarity 和 mean_radius
```

完整实验参数：

```text
null_repeats = 300
```

然后计算：

```text
density_z_vs_random =
(observed_density - null_density_mean) / null_density_std
```

以及：

```text
radius_z_vs_random =
(null_radius_mean - observed_radius) / null_radius_std
```

解释：

- density z 越高，说明这个簇比随机同规模账号组更相似。
- radius z 越高，说明这个簇比随机同规模账号组更紧。

### 10.4 多视角支持

如果一个 cluster 在某个视角中出现，同时其他视角的 cluster 与它有较高成员重合，那么它更值得关注。

实现方式：

```text
同一个 k 下，比较不同 view 的 cluster 成员集合
如果 Jaccard overlap >= 0.5，则认为有一个 view 支持
```

输出字段：

```text
view_support_count
```

### 10.5 综合 suspicious score

最终评分公式：

```text
suspicious_score =
    density_z_vs_random
  + radius_z_vs_random
  + 0.5 * max(0, view_support_count - 1)
  + 0.25 * log1p(cluster_size)
```

这个分数不使用标签。

它表达的是：

- 比随机组更集中；
- 半径更小；
- 被多个视角支持；
- 簇规模不是太小。

输出文件：

```text
suspicious_cluster_scores.csv
```

完整输出规模：

```text
280 行 x 19 列
```

---

## 11. 最后揭示标签的评估方式

无监督 ranking 完成后，才读取 `eval_label`。

标签定义：

```text
original_human -> 0
original_bot -> 1
```

评估文件：

```text
label_revealed_evaluation.csv
```

包含两类行：

### 11.1 单簇评估行

`row_type = cluster`

对每个 suspicious cluster 计算：

| 指标 | 含义 |
| --- | --- |
| `selected_account_count` | 该簇账号数 |
| `human_count` | 揭示后该簇 human 数 |
| `bot_count` | 揭示后该簇 bot 数 |
| `bot_rate` | `bot_count / selected_account_count` |
| `bot_precision` | 与 `bot_rate` 相同，表示如果把该簇全当 bot，精度是多少 |
| `bot_recall` | 该簇覆盖了全部 bot 的比例 |

### 11.2 Top-N 并集评估行

`row_type = top_clusters_union`

对 suspicious ranking 的前 N 个簇取账号并集，计算：

```text
N in {1, 3, 5, 10, 20, 50, 280}
```

这样可以观察：

- 如果只取最可疑的几个簇，precision 是否很高；
- 如果取更多簇，recall 是否上升；
- precision 和 recall 如何权衡。

---

## 12. 输出文件说明

完整实验输出目录：

```text
outputs/original_human_vs_oldbot/
```

主要文件：

| 文件 | 行列规模 | 说明 |
| --- | ---: | --- |
| `account_features.csv` | 2671 x 36 | 无标签账号特征和各视角维度记录 |
| `label_revealed_account_metadata.csv` | 2671 x 9 | 只用于最终揭示评估的账号标签元数据 |
| `account_similarity_edges.csv` | 213680 x 7 | 每个视角的 kNN 相似边 |
| `cluster_assignments.csv` | 42736 x 7 | 每个账号在每个 view/k 下的 cluster assignment |
| `suspicious_cluster_scores.csv` | 280 x 19 | 无标签 suspicious cluster 排名 |
| `label_revealed_evaluation.csv` | 287 x 13 | 最后揭示标签后的 purity/precision/recall |
| `UNSUPERVISED_ACCOUNT_DISCOVERY_CN.md` | - | 自动生成的简版中文结果报告 |

图表目录：

```text
outputs/original_human_vs_oldbot/figures/
```

图表：

| 图 | 说明 |
| --- | --- |
| `cluster_density_null_distribution.png` | 观测簇密度与随机 null baseline 对比 |
| `suspicious_cluster_purity.png` | suspicious clusters 揭示标签后的 bot purity |
| `feature_view_agreement_heatmap.png` | 不同特征视角的相似结构一致性 |
| `account_embedding_umap.png` | 账号数值特征降维图，当前为 PCA fallback |

---

## 13. Smoke test 结果

Smoke test 设置：

```text
sample-human = 100
sample-bot = 100
null-repeats = 50
kmeans-k = 8,12
knn-k = 10
```

输出验证：

```text
account_features.csv shape = (200, 36)
label_like_cols = []
label counts = {human: 100, bot: 100}
```

Top-N 并集评估：

| Top-N clusters | 选中账号数 | bot precision | bot recall |
| ---: | ---: | ---: | ---: |
| 1 | 44 | 0.9773 | 0.4300 |
| 3 | 81 | 0.9259 | 0.7500 |
| 5 | 115 | 0.6609 | 0.7600 |
| 10 | 158 | 0.5570 | 0.8800 |
| 20 | 196 | 0.4949 | 0.9700 |
| 50 | 200 | 0.5000 | 1.0000 |
| 80 | 200 | 0.5000 | 1.0000 |

Smoke test 的意义：

- 脚本能在小样本下完整跑通。
- 无标签特征输出没有泄漏标签列。
- 结果符合预期：最可疑簇 precision 很高，随着纳入簇变多，recall 上升但 precision 下降。

---

## 14. 完整实验核心结果

完整实验设置：

```text
human = 1674
old bot = 997
total accounts = 2671
knn-k = 20
kmeans-k = 8,12,20,30
null-repeats = 300
views = word_tfidf, char_wb_tfidf, style_numeric, bge
```

### 14.1 Top suspicious clusters 单簇结果

前 20 个 suspicious clusters 揭示标签后的结果：

| rank | view | k | cluster_id | size | human | bot | bot_rate | bot_recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | style_numeric | 8 | 1 | 553 | 20 | 533 | 0.9638 | 0.5346 |
| 2 | style_numeric | 12 | 1 | 539 | 14 | 525 | 0.9740 | 0.5266 |
| 3 | style_numeric | 20 | 15 | 262 | 8 | 254 | 0.9695 | 0.2548 |
| 4 | style_numeric | 20 | 2 | 269 | 7 | 262 | 0.9740 | 0.2628 |
| 5 | style_numeric | 30 | 13 | 256 | 7 | 249 | 0.9727 | 0.2497 |
| 6 | style_numeric | 30 | 1 | 264 | 5 | 259 | 0.9811 | 0.2598 |
| 7 | style_numeric | 12 | 11 | 327 | 50 | 277 | 0.8471 | 0.2778 |
| 8 | style_numeric | 8 | 2 | 357 | 100 | 257 | 0.7199 | 0.2578 |
| 9 | style_numeric | 12 | 0 | 239 | 212 | 27 | 0.1130 | 0.0271 |
| 10 | style_numeric | 8 | 6 | 345 | 345 | 0 | 0.0000 | 0.0000 |
| 11 | style_numeric | 12 | 3 | 318 | 313 | 5 | 0.0157 | 0.0050 |
| 12 | style_numeric | 8 | 4 | 409 | 403 | 6 | 0.0147 | 0.0060 |
| 13 | style_numeric | 12 | 10 | 320 | 320 | 0 | 0.0000 | 0.0000 |
| 14 | word_tfidf | 12 | 7 | 286 | 0 | 286 | 1.0000 | 0.2869 |
| 15 | style_numeric | 20 | 13 | 177 | 9 | 168 | 0.9492 | 0.1685 |
| 16 | style_numeric | 8 | 7 | 303 | 288 | 15 | 0.0495 | 0.0150 |
| 17 | word_tfidf | 30 | 21 | 213 | 0 | 213 | 1.0000 | 0.2136 |
| 18 | word_tfidf | 8 | 2 | 330 | 1 | 329 | 0.9970 | 0.3300 |
| 19 | style_numeric | 20 | 0 | 207 | 54 | 153 | 0.7391 | 0.1535 |
| 20 | word_tfidf | 8 | 5 | 324 | 0 | 324 | 1.0000 | 0.3250 |

最重要的结果是 rank 1：

```text
style_numeric:k=8:c=1
cluster size = 553
human = 20
bot = 533
bot_rate / precision = 0.9638
bot_recall = 0.5346
```

解释：

- 这是一个完全由无标签评分排到第一的 cluster。
- 它只用了账号文本统计风格，没有使用标签。
- 标签揭示后，该簇中 96.38% 是 old bot。
- 仅这一个簇就覆盖了 53.46% 的 old bot。

这说明 old bot 的账号级风格分布确实存在强集中机制。

### 14.2 Top-N suspicious clusters 并集结果

| Top-N clusters | 选中账号数 | human | bot | bot precision | bot recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 553 | 20 | 533 | 0.9638 | 0.5346 |
| 3 | 556 | 23 | 533 | 0.9586 | 0.5346 |
| 5 | 558 | 24 | 534 | 0.9570 | 0.5356 |
| 10 | 1515 | 657 | 858 | 0.5663 | 0.8606 |
| 20 | 2138 | 1190 | 948 | 0.4434 | 0.9509 |
| 50 | 2596 | 1604 | 992 | 0.3821 | 0.9950 |
| 280 | 2671 | 1674 | 997 | 0.3733 | 1.0000 |

如何读这张表：

- Top 1 到 Top 5：precision 非常高，约 95.7%-96.4%，但 recall 约 53.5%。
- Top 10：recall 上升到 86.1%，但 precision 降到 56.6%。
- Top 20：recall 达到 95.1%，但 precision 只有 44.3%。
- Top 50 以后几乎覆盖所有 bot，但也覆盖大量 human，precision 接近总体 bot 占比。

这说明 suspicious ranking 的前几个簇非常有效，但继续向后取簇时，会开始纳入“同样高密度但并非 bot 的 human 簇”。

因此，本方法更适合作为：

```text
高置信机制簇发现 / suspicious cluster discovery
```

而不是直接作为：

```text
低阈值全量账号 bot 分类器
```

---

## 15. 不同特征视角的贡献

### 15.1 高纯度 bot cluster 数

定义：

```text
high purity cluster = bot_rate >= 0.95
```

按视角统计：

| view | cluster 总数 | bot_rate >= 0.95 的簇数 | max bot_rate | max bot_recall | 最靠前 rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge | 70 | 19 | 1.0000 | 0.4413 | 29 |
| char_wb_tfidf | 70 | 19 | 1.0000 | 0.3932 | 55 |
| style_numeric | 70 | 7 | 0.9815 | 0.5346 | 1 |
| word_tfidf | 70 | 20 | 1.0000 | 0.4463 | 14 |

解读：

1. `style_numeric` 不是产生最多纯 bot 簇的视角，但它产生了排名最高、覆盖最大的可疑簇。
2. `word_tfidf` 产生了很多纯 bot 簇，例如 rank 14、17、18、20 都几乎是纯 bot。
3. `bge` 也发现了不少高纯度 bot 簇，但它们在 suspicious score 排名中没有 style_numeric 靠前。
4. `char_wb_tfidf` 有高纯度簇，但排名更靠后。

这说明 old bot 的机制指纹至少有两层：

- 风格统计层：多个 bot 的整体文本统计高度集中。
- 词汇/语义层：部分 bot 在词项、字符和嵌入空间里也形成纯 bot 小团体。

### 15.2 suspicious score 视角分布

各视角 suspicious score 统计：

| view | cluster 数 | min | median | mean | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge | 70 | -40.7885 | 21.8809 | 23.0571 | 93.7634 |
| char_wb_tfidf | 70 | -61.8209 | 0.1733 | 7.7568 | 75.1195 |
| style_numeric | 70 | 21.5113 | 79.4544 | 88.9577 | 250.1366 |
| word_tfidf | 70 | -55.1615 | 17.2649 | 24.6050 | 120.1016 |

`style_numeric` 的分数整体显著更高，说明在本实验里：

> old bot 和 human 在账号级风格统计空间中的密度差异，比在词项或语义空间中更强。

这很关键，因为它支持我们提出的“账号级分布集中性”假设，而不是单纯依赖 topic 或具体词汇。

---

## 16. 图表解读

自动报告生成了四张图。

### 16.1 Cluster density / null distribution

路径：

```text
outputs/original_human_vs_oldbot/figures/cluster_density_null_distribution.png
```

含义：

- 横轴是 cluster ranking 或 cluster score。
- 展示观测簇密度与同规模随机账号组 null baseline 的对比。
- 如果某些簇远高于随机基线，说明它们不是普通随机聚集，而是高度集中。

### 16.2 Suspicious cluster purity

路径：

```text
outputs/original_human_vs_oldbot/figures/suspicious_cluster_purity.png
```

含义：

- 这张图在最后揭示标签后绘制。
- 它不参与无监督发现，只用于解释哪些 suspicious cluster 确实富含 bot。
- 可以看到排名靠前的若干簇 bot purity 很高，但也存在排名靠前的人类高密度簇。

这提醒我们：

> 高密度机制簇不等于 bot 簇；它只是“值得审计的机制簇”。标签揭示后，部分高度对应 bot。

### 16.3 Feature-view agreement heatmap

路径：

```text
outputs/original_human_vs_oldbot/figures/feature_view_agreement_heatmap.png
```

含义：

- 衡量不同视角下相似边/聚类结构是否一致。
- 如果两个视角高度一致，说明它们在捕捉类似机制。
- 如果不一致，说明它们发现的是不同类型的集中性，例如风格集中 vs 语义集中。

### 16.4 Account embedding

路径：

```text
outputs/original_human_vs_oldbot/figures/account_embedding_umap.png
```

当前环境没有 UMAP，所以使用 PCA fallback。

含义：

- 使用数值账号特征降维到二维。
- 颜色来自独立标签揭示元数据，只用于最终评估展示。
- 不参与聚类或 suspicious score。

---

## 17. 主要结论

### 17.1 无监督机制簇发现是可行的

完整实验中，无标签 suspicious ranking 的第一名 cluster：

```text
size = 553
bot = 533
human = 20
precision = 96.38%
recall = 53.46%
```

这已经足以说明：

> 在当前 original old bot 数据中，存在一大批账号在语言风格空间中异常集中；即使不使用标签，也能通过密度和随机基线对比把它们排到最前面。

### 17.2 账号级检测比单评论检测更适合无监督机制发现

本实验没有把每条评论当样本，而是把账号聚合文本当样本。结果证明这是合理的。

原因：

- 单条评论太短，偶然性强。
- 账号聚合后能显露稳定语言分布。
- 多个 bot 账号如果来自相同生成机制，它们的账号级分布更容易形成高密度群体。

### 17.3 不是所有高密度簇都是 bot

Top 9、Top 10、Top 11、Top 12、Top 13 等簇揭示后主要是 human。

这说明：

- 人类账号中也会有高密度群体。
- 可能是相似活跃程度、相似文本长度、相似表达习惯或类似数据采样机制导致。
- 因此不能说“高密度 = bot”。

更严谨的说法是：

> 无监督方法发现的是 suspicious mechanism clusters，其中一部分在标签揭示后高度富含 bot。

### 17.4 Top-N 阈值需要谨慎选择

如果目标是高置信发现：

```text
Top 1-5 更合适
precision 约 95.7%-96.4%
recall 约 53.5%
```

如果目标是尽量覆盖 bot：

```text
Top 20 recall 可达 95.1%
但 precision 只有 44.3%
```

这意味着第一阶段方法可以作为：

- botnet seed discovery；
- 高风险机制簇审计；
- 后续人工/半监督标注入口；
- 新 family 发现工具。

但不宜直接作为最终自动封禁式分类器。

### 17.5 当前 old bot 的最强指纹是风格统计集中性

`style_numeric` 在 suspicious score 上明显领先：

```text
style_numeric max score = 250.1366
word_tfidf max score = 120.1016
bge max score = 93.7634
char_wb_tfidf max score = 75.1195
```

这说明 old bot 的主要可疑性不是单纯 topic，而是：

- 文本长度分布；
- 句长；
- 标点使用；
- 词长；
- 评论数/词数；
- 整体风格统计。

这和我们之前监督实验发现的“bot 存在机制指纹”是一致的。

---

## 18. 对研究问题的回应

最初的问题是：

> 如果我们把每个人的语言看成一个分布，那么人的分布更分散，而一批机器人账号可能更集中。能否在无标签情况下看出这是机器人？

第一阶段实验回答是：

> 可以在无标签情况下发现高度集中的账号机制簇；其中排名最高的一些簇在标签揭示后确实高度富含 old bot。

但更严谨地说：

> 当前方法不是直接无监督识别“每个账号是不是 bot”，而是发现“疑似同机制生成账号簇”。这些簇需要后续揭示标签、人工审计或跨 family 验证。

这比直接声称“无监督识别机器人”更稳。

---

## 19. 与监督检测的关系

这个实验不是替代 TF-IDF/BGE/RGCN 等监督检测，而是补充它们。

监督检测回答：

```text
给定标签训练后，模型能否识别 bot？
```

无监督机制发现回答：

```text
没有标签时，是否存在一批账号自己形成异常集中机制簇？
```

二者可以结合：

1. 用无监督方法发现 suspicious clusters。
2. 对高纯度簇做少量人工审计或标签揭示。
3. 把这些簇作为新 family 或新 botnet seed。
4. 再训练监督检测器。
5. 测试检测器对新 family 的泛化能力。

这正好可以接到我们后续的 family 泛化研究。

---

## 20. 当前方法的局限

### 20.1 KMeans 需要预设 k

当前使用：

```text
k = 8, 12, 20, 30
```

这只是基线方案。真实 botnet 或机制簇数量不一定符合这些 k。

后续应加入：

- HDBSCAN；
- Leiden/Louvain 图社区发现；
- spectral clustering；
- density peaks；
- ensemble clustering。

### 20.2 suspicious score 会发现 human 高密度簇

这是方法本身的自然结果。

高密度不等于 bot。某些 human 也可能因为类似行为/长度/表达习惯形成高密度簇。

所以后续应加入：

- 人类高密度簇基线；
- human cluster 类型解释；
- 多视角一致性更强的判定；
- 时间/图结构特征作为额外证据。

### 20.3 当前只用文本账号分布

第一阶段没有使用：

- 发帖/评论时间；
- 回复层级；
- 用户互动图；
- OP 回访；
- 主题迁移；
- 社区流动；
- 账号生命周期。

因此它只能说明“文本机制簇”可发现，还不能说明“完整社交行为机制簇”可发现。

### 20.4 BGE 是缓存视角

当前 BGE 使用已有 `combined_bge_embeddings.npy`。如果后续重建数据或扩展 family，需要保证：

- embedding 与 `combined_sample_index.csv` 行数一致；
- 行顺序一致；
- family 新增后重新生成 embedding。

脚本已经做了行数检查；如果不匹配，会跳过 BGE。

### 20.5 第一阶段只验证 old bot

old bot 的机制指纹很强，因此这个结果不能直接推广到所有新 bot。

后续必须扩展到：

- F1-F5 静态 family；
- 新 persona-social bot；
- 进化式 family；
- 不同模型平台生成的 bot；
- 混合 bot family；
- held-out family 测试。

---

## 21. 后续扩展方案

### 21.1 扩展到 F1-F5 静态 family

下一步可以在同一框架中加入：

```text
original_human
original_bot
F1_original_architecture
F2_tfidf_informed_language
F3_human_anchor_local_tfidf
F4_peba_trait_prompt
F5_gemini_trait_prompt
```

问题变成：

1. 无监督机制簇能否自动分离不同 family？
2. F1-F5 是否分别形成自己的机制簇？
3. 哪些 family 更接近 human 分布？
4. 哪些 family 和 old bot 混在一起？
5. 是否存在跨 family 的共同 LLM 生成指纹？

### 21.2 做 family discovery，而不是只做 bot/human

后续不应只看 bot_rate，而应看：

```text
cluster family composition
```

例如一个 cluster 可能是：

- 90% F1；
- 70% F3 + 20% F4；
- 80% original_bot；
- 95% original_human。

这样可以把无监督检测升级成：

> 无监督生成机制家族发现。

### 21.3 加入图结构无监督特征

当前只有文本。后续可以对每个账号加入：

- 出度；
- 入度；
- 二级回复率；
- OP 回访率；
- 互惠率；
- 被回复率；
- 跨社区互动率；
- 评论深度；
- 与 bot/human 节点的混合邻接，但注意真实无标签情况下不能用 bot/human 标签，只能用无标签图结构。

然后构建：

```text
text view + style view + semantic view + graph view + temporal view
```

### 21.4 和泛化检测协议结合

最终可以形成这样的实验流程：

1. Round 1：无监督发现 old bot suspicious clusters。
2. 揭示少量标签，确认一批高纯度 bot mechanism cluster。
3. 用这些 cluster 训练监督 detector。
4. Round 2：加入 F1-F5，测试 detector 是否能泛化到 unseen family。
5. 用无监督 discovery 找新 family cluster。
6. Round 3：加入新 family，再测试检测恢复曲线。

这和我们之前讨论的进化式 family benchmark 可以结合起来。

---

## 22. 复现命令

本机推荐解释器：

```bash
python
```

完整第一阶段实验：

```bash
cd "[local-path-removed]"

python \
  "unsupervised/code/run_unsupervised_account_discovery.py" \
  --output-dir "unsupervised/outputs/original_human_vs_oldbot" \
  --null-repeats 300 \
  --kmeans-k 8,12,20,30 \
  --knn-k 20

python \
  "unsupervised/code/plot_unsupervised_results.py" \
  --output-dir "unsupervised/outputs/original_human_vs_oldbot"
```

Smoke test：

```bash
cd "[local-path-removed]"

python \
  "unsupervised/code/run_unsupervised_account_discovery.py" \
  --output-dir "unsupervised/outputs/smoke_original_100_100" \
  --sample-human 100 \
  --sample-bot 100 \
  --null-repeats 50 \
  --kmeans-k 8,12 \
  --knn-k 10

python \
  "unsupervised/code/plot_unsupervised_results.py" \
  --output-dir "unsupervised/outputs/smoke_original_100_100"
```

---

## 23. 验收情况

| 验收项 | 状态 | 证据 |
| --- | --- | --- |
| 新建独立目录 | 已完成 | `unsupervised/` |
| 搬运原始数据 | 已完成 | `inputs/raw_original/Users.csv`, `inputs/raw_original/user_post_comment.json` |
| 搬运静态 F1-F5 数据 | 已完成 | `inputs/static_family/families/F1-F5/` |
| source manifest | 已完成 | `inputs/source_manifest.csv` |
| 第一阶段只跑 original human vs old bot | 已完成 | 输出 metadata：human 1674, bot 997 |
| 无标签特征文件 | 已完成 | `account_features.csv` 无 label/family/bot/human 列 |
| 标签只用于最终揭示 | 已完成 | `label_revealed_account_metadata.csv`, `label_revealed_evaluation.csv` |
| smoke test | 已完成 | `outputs/smoke_original_100_100/` |
| full test | 已完成 | `outputs/original_human_vs_oldbot/` |
| 图表 | 已完成 | `outputs/original_human_vs_oldbot/figures/*.png` |
| 中文结果报告 | 已完成 | `outputs/original_human_vs_oldbot/UNSUPERVISED_ACCOUNT_DISCOVERY_CN.md` |
| 详细研究汇报 | 已完成 | 本文件 |

---

## 24. 最终判断

第一阶段实验已经证明：

1. 在账号级文本分布上，old bot 中存在明显的集中机制簇。
2. 无监督方法可以在不使用标签的情况下，把最大的一批 old bot 集中簇排到最前面。
3. 最高置信簇达到 `96.38%` bot precision，并覆盖 `53.46%` old bot。
4. 继续扩大 suspicious cluster 数量可以提高 recall，但会混入大量 human 高密度簇。
5. 因此，当前方法最适合作为“无监督机制簇发现”和“botnet seed discovery”，而不是直接作为单账号最终分类器。

这条路线值得继续。它和我们之前的泛化性研究目标并不冲突，反而可以成为泛化基准的重要补充：当新 family 出现时，我们不只看监督 detector 能不能判别，也可以看无监督方法能不能自动发现一个新的生成机制簇。

