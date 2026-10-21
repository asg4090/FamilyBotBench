# 现代文本模型可行性实验报告

生成时间：2026-05-05  
工作目录：``

## 1. 目标

本实验回答两个问题：

1. 如果项目希望使用更现代、更强的文本分析工具，当前机器上是否可行？
2. 在只用评论文本的同一口径下，现代 embedding 模型是否比之前的 TF-IDF 更好？

## 2. 环境判断

本机环境：

- `torch` 已安装，版本 2.5.1。
- `transformers` 已安装，版本 4.46.3。
- Apple MPS 可用，可以本机跑小型 Transformer。
- 未检测到 `OPENAI_API_KEY`。
- 未安装 `openai` Python 包。
- 磁盘剩余约 9GB，不适合下载特别大的模型。

因此，本次实际执行采用本地部署路线，不调用 API。

## 3. 选择的模型

实际运行模型：

- `BAAI/bge-small-en-v1.5`

选择理由：

- 它是现代文本 embedding 模型，比传统 BiLSTM 更接近当前文本理解范式。
- 英文效果较好，适合 BotSim-24 里的英文 Reddit 评论。
- 模型体积较小，可以在当前机器上下载和运行。
- 输出 embedding 后，可以用 Logistic Regression 做可解释、稳定的二分类。

没有优先跑大模型的原因：

- 本机剩余空间有限。
- 长文本 Transformer 编码已经比 TF-IDF 慢很多。
- 先用小型强 embedding 模型可以快速判断路线是否可行。

## 4. 数据口径

与之前的“仅评论文本”实验一致：

- 只读取 `comment_1/comment_2[*].comment_body`。
- 不使用用户资料、用户名、帖子文本、图结构、评论分数、subreddit、评论数量等特征。
- 使用严格清洗后的评论文本。
- 每个用户最多保留前 600 个词。

模型输入流程：

```text
严格清洗后的用户评论文本
→ 截断到 600 词
→ BGE tokenizer，最大 512 token
→ BGE encoder
→ mean pooling
→ L2 normalize
→ Logistic Regression
```

## 5. 结果对比

5 个随机种子的测试集平均结果：

| method | variant | mean_accuracy | mean_precision | mean_recall | mean_f1 | std_f1 |
|:--|:--|--:|--:|--:|--:|--:|
| TF-IDF + Logistic Regression | strict_clean_comments_truncated_600w | 0.9725 | 0.9979 | 0.9220 | 0.9582 | 0.0172 |
| TF-IDF + Logistic Regression | strict_clean_comments_nonempty_users_truncated_600w | 0.9736 | 0.9978 | 0.9300 | 0.9626 | 0.0161 |
| BGE-small-en-v1.5 embedding + Logistic Regression | all_users_strict_clean_truncated_600w_embedding | 0.9773 | 0.9875 | 0.9460 | 0.9662 | 0.0119 |
| BGE-small-en-v1.5 embedding + Logistic Regression | strict_nonempty_users_strict_clean_truncated_600w_embedding | 0.9722 | 0.9680 | 0.9560 | 0.9618 | 0.0056 |

## 6. 怎么解读

如果看全体用户：

- BGE-small 平均 F1 = 0.9662。
- TF-IDF 平均 F1 = 0.9582。
- BGE-small 略高。

如果剔除严格清洗后无文本用户：

- BGE-small 平均 F1 = 0.9618。
- TF-IDF 平均 F1 = 0.9626。
- 两者几乎持平，TF-IDF 略高一点。

这说明：

1. 现代 embedding 模型在本项目上可行，而且效果很强。
2. 它比 TF-IDF 更倾向于提高 bot recall，即更愿意把 bot 找出来。
3. 但它不是碾压 TF-IDF，因为 BotSim-24 文本中存在很强的词面/语体差异，TF-IDF 本来就非常适合抓这种信号。

seed=0 的 BGE-small 全体用户混淆矩阵：

| true \ pred | bot | human |
|:--|--:|--:|
| bot | 95 | 5 |
| human | 1 | 190 |

对比之前 TF-IDF seed=0：

| true \ pred | bot | human |
|:--|--:|--:|
| bot | 92 | 8 |
| human | 0 | 191 |

也就是说，BGE-small 多找回了 3 个 bot，但多误判了 1 个人类。

## 7. 推荐

如果你希望项目里用一个“现代、强、可执行”的文本分析工具，我建议分三档：

### 最推荐的本地方案

`BAAI/bge-small-en-v1.5` 或同系列更大模型 + Logistic Regression / MLP。

优点：

- 本机可跑。
- 不需要 API key。
- 效果已经达到 F1 约 0.96。
- embedding 可缓存，后续实验很快。

当前已实际跑通。

### 如果有更强机器

可以尝试：

- `BAAI/bge-base-en-v1.5`
- `intfloat/e5-base-v2`
- `Alibaba-NLP/gte-base-en-v1.5`
- `nomic-ai/nomic-embed-text-v1.5`
- ModernBERT / DeBERTa-v3 微调分类

这些模型可能更强，但下载、显存和运行时间更高。

### 如果可以调用 API

可以用强 embedding API，例如 OpenAI text embedding 或其他商业 embedding API。

优点：

- 部署最省事。
- 通常速度和稳定性好。
- 不占本机磁盘和内存。

限制：

- 当前环境没有 `OPENAI_API_KEY`，所以本次没有实际调用。
- 需要注意成本、隐私和可复现性。

## 8. 最终判断

就当前项目、当前机器、当前数据而言：

> 最可行、性价比最高的现代文本工具是本地 embedding 模型，其中 `BAAI/bge-small-en-v1.5 + Logistic Regression` 已经跑通，并且在全体用户上略优于 TF-IDF。

但如果问“绝对最好”，不能只看这次结果。因为更大的 BGE/E5/GTE、ModernBERT 微调、以及 API embedding 可能更强，只是当前机器和 API 条件下没有必要一开始就上最大方案。

更重要的是：无论用 TF-IDF 还是现代 embedding，本数据集的评论文本本身都已经非常可分。因此后续研究重点不应该只追求更高分，而应该做更严格的去伪影实验，例如固定长度、固定评论数、跨主题划分、跨时间划分、删除 LLM 风格模板词等。

## 9. 输出文件

实验脚本：

- `仅评论文本识别/modern_embedding_text_experiment.py`

输出目录：

- `仅评论文本识别/modern_embedding_run/BAAI_bge-small-en-v1.5/`

核心文件：

- `embedding_summary.csv`
- `embedding_metrics.csv`
- `embedding_predictions.csv`
- `strict_clean_truncated_embeddings.npy`
- `figures/seed0_all_users_confusion_matrix.png`
- `figures/seed0_nonempty_users_confusion_matrix.png`

方法对比表：

- `仅评论文本识别/modern_embedding_run/text_method_comparison.csv`
