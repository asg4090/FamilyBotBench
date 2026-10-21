# BGE 改写策略搜索结果报告：已找到有效降低检出的策略

## 1. 结论先行

这轮实验已经找到一种对 BGE 检测器有效的改写方向：

> 把原本长篇、完整、解释式的 bot 评论，强制改写成短、碎、线程回复式的 Reddit 评论。

最有效的固定策略是：

```text
S8_two_sentence_thread_reply
```

它要求模型把原评论压缩为一到两句短回复，只保留主要立场，不再完整展开背景、平衡论证、政策框架和总结。

在完整 `F1-F5` 共 `250` 个 static family bot 上，针对主检测器 `LOFO BGE`：

| 策略 | 平均 bot 概率 | recall | bot->human 翻转率 | 平均长度比例 |
|---|---:|---:|---:|---:|
| 原文 | 0.736 | 0.852 | - | 1.000 |
| 固定 `S8_two_sentence_thread_reply` | 0.663 | 0.800 | 0.088 | 0.290 |
| 固定 `S7_ultra_short_casual_digest` | 0.692 | 0.828 | 0.076 | 0.355 |
| 每条样本在 `S7/S8` 中选 BGE 概率最低候选 | 0.626 | 0.756 | 0.120 | 0.316 |

所以，固定 `S8` 本身已经让 recall 从 `0.852` 降到 `0.800`；如果允许 detector-aware 地在 `S7/S8` 里选更低概率候选，recall 进一步降到 `0.756`。

这说明 BGE 检测器确实对一种结构性特征敏感：**长篇聚合式、完整解释式、抽象分析式的组织方式**。

## 2. 这轮为什么比之前的固定 treatment 成功

上一轮固定 treatment 的核心结果是：  
很多 treatment 看起来合理，但实际没有稳定降低 BGE 检测概率。

例如在 pilot 中：

| treatment | LOFO mean delta probability | 解释 |
|---|---:|---|
| `T3_de_rights_frame` | -0.008 | 有轻微下降，但样本数只有 19，幅度很小 |
| `T7_short_incomplete_reply` | -0.003 | 概率略降，但 recall 没有稳定下降 |
| `T2_de_fact_checking` | +0.045 | 反而更像 bot |

失败原因不是 API 或 BGE 流程，而是策略没有打中主要变量。  
很多 treatment 只是改变局部语义姿态，例如减少 rights frame、fact-checking、policy frame，但 BGE 仍能从整体文本结构里抓到 bot-like embedding 方向。

这轮改动后，我们不再只改局部术语，而是直接改变文本组织方式：

```text
长篇解释 -> 短句回复
完整论证 -> 单点反应
平衡框架 -> 不完整表达
抽象公共议题分析 -> 线程里随手回一句
```

这正好对应了之前 BGE 几何实验里的发现：BGE 不是只抓词，而是抓 embedding 空间里的“语义姿态 + 组织节奏”。

## 3. 全量实验设置

输入数据：

```text
static_language/outputs/full_5family_50bot_3round_gpt54_20260601/
```

主输出目录：

```text
bge_rewrite_strategy_search_strongcompress_full250_s7s8_20260606/
```

实验对象：

```text
F1-F5 static family bot
每个 family 50 个
共 250 个 bot 文本
```

检测器：

```text
LOFO BGE detector
```

也就是每个 family 都用“未见 family”的检测器测试：

```text
测 F1 -> 训练 original + F2/F3/F4/F5
测 F2 -> 训练 original + F1/F3/F4/F5
...
```

这比原始 `D0_original_only` 更适合泛化性解释，因为它测试的是“检测器见过其他 family 后，能不能泛化到当前 family”。

## 4. 固定策略效果

### 4.1 S8：两句式线程回复

`S8_two_sentence_thread_reply` 的 prompt 核心要求是：

```text
Rewrite as one or two short Reddit reply sentences, 25-45 words total.
Preserve the main stance, but do not try to preserve every supporting point.
No formal caveats, no balanced explainer rhythm, no policy framing unless it is absolutely central.
```

全量结果：

| 指标 | 原文 | S8 改写后 |
|---|---:|---:|
| mean BGE bot probability | 0.736 | 0.663 |
| recall | 0.852 | 0.800 |
| flip rate | - | 0.088 |
| mean length ratio | 1.000 | 0.290 |

它是目前最好的单一固定 LLM 改写策略。

### 4.2 S7：40-60 词 casual digest

`S7_ultra_short_casual_digest` 的 prompt 核心要求是：

```text
Rewrite into 40-60 words total.
Keep the central stance and the most important topic cues, but drop most supporting analysis.
Use casual Reddit wording, contractions, and one clear reaction.
```

全量结果：

| 指标 | 原文 | S7 改写后 |
|---|---:|---:|
| mean BGE bot probability | 0.736 | 0.692 |
| recall | 0.852 | 0.828 |
| flip rate | - | 0.076 |
| mean length ratio | 1.000 | 0.355 |

S7 也能降概率，但比 S8 弱。

### 4.3 S7/S8 detector-aware 选择

如果每条样本都生成 `S7` 和 `S8` 两个候选，然后用本地 BGE 检测器选择概率更低的候选：

| 指标 | 原文 | S7/S8 选优后 |
|---|---:|---:|
| mean BGE bot probability | 0.736 | 0.626 |
| recall | 0.852 | 0.756 |
| flip rate | - | 0.120 |
| mean length ratio | 1.000 | 0.316 |

这个结果最强，但要注意：  
它使用了检测器反馈，因此属于 `detector-aware stress test`，不是普通自然生成策略。

## 5. 分 family 结果

自动在 `S7/S8` 里选最优候选后，各 family 结果如下：

| Family | 原平均概率 | 改写后平均概率 | 平均下降 | recall 原文 | recall 改写后 | flip rate | 平均长度比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 0.911 | 0.749 | 0.162 | 1.00 | 0.96 | 0.04 | 0.206 |
| F2 | 0.623 | 0.539 | 0.084 | 0.78 | 0.62 | 0.18 | 0.396 |
| F3 | 0.695 | 0.575 | 0.120 | 0.84 | 0.66 | 0.18 | 0.240 |
| F4 | 0.873 | 0.761 | 0.112 | 1.00 | 0.98 | 0.02 | 0.417 |
| F5 | 0.580 | 0.506 | 0.074 | 0.64 | 0.56 | 0.18 | 0.319 |

最明显的 recall 降低出现在：

```text
F2: 0.78 -> 0.62
F3: 0.84 -> 0.66
F5: 0.64 -> 0.56
```

F1 和 F4 的原始检出率太高，强压缩后概率明显下降，但很多样本仍在 0.5 阈值以上，所以 recall 降幅较小。

## 6. 和机械截断基线的比较

为了确认 BGE 是否真的对“长度/完整解释结构”敏感，我补跑了无 API 的机械基线：

```text
D_first_40_baseline
D_first_80_baseline
```

结果如下：

| 机械基线 | 平均概率下降 | recall 原文 | recall 后 | flip rate | 平均长度比例 |
|---|---:|---:|---:|---:|---:|
| first 40 words | 0.089 | 0.852 | 0.748 | 0.144 | 0.274 |
| first 80 words | 0.028 | 0.852 | 0.808 | 0.092 | 0.542 |
| first40/first80 选优 | 0.114 | 0.852 | 0.696 | 0.180 | 0.360 |

机械截断比 LLM 改写更能降低 recall，但它不是语义保真的改写，因为它会直接丢掉后半部分主题。

它的意义是诊断性的：

> BGE 确实依赖长文本中累积出来的解释结构、语义姿态和组织方式。  
> 当文本被强行压缩后，embedding 会明显离开原来的 bot-like 检测方向。

LLM 的 `S8` 则是更自然、更可解释的版本：不是硬截断，而是把文本改成短线程回复。

## 7. 典型案例

### 案例 1：F4，S7

```text
probability: 0.895 -> 0.301
drop: 0.594
```

原文模式：

```text
Less urgent compared to when, exactly? Russia is still using drones and missiles...
What does this actually add though? If the point is Ukraine needs more air defense now...
...
```

改写后：

```text
Less urgent than when? Russia is still hitting Ukraine with drones and missiles,
so air defense doesn’t sound lower priority unless this is really about limited US stock.
If that’s the point, just say it. Otherwise it’s the same vague Washington review stuff
while people keep getting hit. Patriot, NASAMS, interceptors, what exactly?
```

变化：

```text
多段解释 -> 一段短质疑
政策框架 -> 具体追问
完整分析 -> 线程式吐槽
```

### 案例 2：F1，S8

```text
probability: 0.906 -> 0.378
drop: 0.528
```

改写后：

```text
Saying Israel has a right to defend itself means nothing if allies won't impose real limits in Gaza;
when US and Europe drop their own standards for an ally, their credibility is shot.
Using a family member's addiction as campaign material is just garbage politics.
```

变化：

```text
完整公共议题分析 -> 两句直接判断
道德/制度框架 -> 口语化立场
解释节奏 -> 快速 thread reply
```

## 8. 对 BGE 的解释

这轮实验说明，BGE 检测 F1-F5 不是只靠某几个词，而是很依赖文本整体在 embedding 空间里的组织形态。

最关键的 bot-like 形态包括：

```text
长篇聚合
多主题完整展开
平衡式解释
政策/制度框架化
抽象总结
过于完整的论证闭合
```

当我们把文本改成：

```text
短
碎
不完整
线程回复式
少解释
少总结
```

BGE bot probability 就会稳定下降。

这和 TF-IDF 的结论不同。  
TF-IDF 更容易解释为“哪些词项有贡献”；BGE 的核心是“整体语义姿态和组织方式”。

## 9. 重要限制

这轮实验已经成功降低检出，但必须准确表述限制。

第一，`S8` 是有效的，但它牺牲了信息完整性。  
它保留主要立场，但不保证保留原文所有主题和所有细节。

第二，`S7/S8` 选优是 detector-aware。  
它用了检测器反馈，因此更像对抗压力测试，不应说成普通 bot 自然生成时一定能达到。

第三，这说明了 BGE 的弱点，但也暴露了一个新的检测方向：  
如果检测器加入长度、信息覆盖、评论层级语境一致性，强压缩策略可能会被另一类特征重新抓住。

## 10. 当前可采用的成功策略

如果我们的目标是“找到一种能有效降低 BGE 检出的改写策略”，当前答案是：

```text
S8_two_sentence_thread_reply
```

如果目标是“构造 detector-aware 的压力测试上界”，当前答案是：

```text
每条样本生成 S7 和 S8 两个候选，再选择 BGE 概率最低者。
```

论文里建议这样表述：

> We find that BGE-based detectors are substantially sensitive to long-form explanatory organization. A controlled compression strategy that rewrites bot comments into short thread-style replies reduces LOFO BGE recall from 0.852 to 0.800 under a fixed strategy, and to 0.756 under detector-aware candidate selection. This suggests that BGE captures not only semantic content, but also discourse-level organization and completeness.

## 11. 关键文件

全量 LLM 强压缩实验：

```text
bge_rewrite_strategy_search_strongcompress_full250_s7s8_20260606/
  strategy_effect_summary.csv
  strategy_family_effect_summary.csv
  strategy_selected_best_summary.csv
  strategy_selected_best_rewrites.csv
  strategy_candidate_predictions.csv
  figures/
```

机械诊断基线：

```text
bge_rewrite_strategy_search_deterministic_full250_20260606/
```

新增脚本：

```text
static_language/code/run_bge_rewrite_strategy_search.py
```

