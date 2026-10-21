# TF-IDF 为什么能分出 F1-F5：详细解读版

这份文档不按“报告格式”写，而是按理解顺序解释：图怎么看、检测器到底学到了什么、五个 family 各自留下了什么指纹。

先说最重要的结论：

**当前实验里，旧的 original-only TF-IDF 并不能 zero-shot 检出 F1-F5；真正能检出的是“看过新 family 后重新训练”的 TF-IDF，以及“看过其他 family 后测试留出 family”的 TF-IDF。**

所以问题要拆成两层：

- 旧 TF-IDF 为什么没检出：F1-F5 已经避开了原始 bot 的很多旧指纹。
- 新 TF-IDF 为什么又能检出：F1-F5 作为一批由同一生成系统产生的数据，仍然各自有稳定词项组合和跨 family 共同词项。

## 1. 先看 protocol_recall_heatmap：到底是哪种 TF-IDF 在检出

![protocol_recall_heatmap](protocol_recall_heatmap.png)

这张图最容易误读。

纵轴是训练协议：

- `original_only_zero_shot`：只用原始 human + 原始 bot 训练，然后直接测 F1-F5。
- `self_family`：用原始数据 + 当前 family 训练，再测当前 family。
- `leave_one_family_out`：用原始数据 + 其他四个 family 训练，留出当前 family 测试。
- `incremental_curriculum`：按 F1 到 F5 的顺序逐步加入训练集，测试后续未见 family。

横轴是 F1-F5。

图里的核心现象：

| 协议 | 结果 | 含义 |
|---|---:|---|
| original-only zero-shot | 全部 0.00 | 旧 TF-IDF 对新 family 完全失效 |
| self-family | 0.92-1.00 | 每个 family 内部都有强可学习指纹 |
| leave-one-family-out | F1/F2/F4 很高，F3/F5 中等 | family 之间存在共享生成指纹，但 F3/F5 更特殊 |
| incremental | F4/F5 有一定检出，F1-F3 很低 | 这个图受固定顺序影响，不等同于最终检测能力 |

也就是说：**不是“旧模型一直很强”，而是“只要检测器看到过同类生成机制，它就很快能学到新指纹”。**

这对我们的研究很重要。它说明静态数据集里，即使 bot 语言改得更像人，检测器仍能在同分布或近分布训练后抓住规律。这正是我们之前讨论“固定数据集没有泛化意义”的证据。

## 2. 再看反事实删除图：检测器是不是只靠话题词

![trained_detector_ablation_recall_drop](trained_detector_ablation_recall_drop.png)

这张图是最关键的“因果验证”。

它不是只看哪些词权重大，而是问：

> 如果把这些词删掉，检测器还抓得住吗？

三种颜色代表三种删除方式：

- 蓝色：删除 family 级别 top 词项。
- 橙色：只删除话题实体，比如 Ukraine、Israel、Gaza 之类。
- 绿色：删除每条文本中对 bot 概率贡献最高的 20 个词项。

结果非常清楚：

- 删除每条文本 top 20 贡献词，recall 大幅下降。
- 删除 family top terms，recall 也大幅下降。
- 只删除 topic entities，几乎不下降，甚至 F3 还略微上升。

这说明：

**TF-IDF 不是简单靠“你提到了 Ukraine / Gaza / Trump 所以你是 bot”。**

它靠的是一组更细碎、更稳定的词项组合。例如：

- `just`
- `actual`
- `exactly`
- `addiction`
- `kids`
- `thread`
- `wording`
- `numbers`
- `line`
- `global`
- `urgent`
- `political`

这些词单独看不一定像 AI，但如果某一批 bot 反复以相似频率、相似组合出现，TF-IDF 就能把它们变成可学习特征。

这也是 TF-IDF 的特点：它不理解语义，但它非常擅长抓“重复的词项分布”。

## 3. 再看 family separability：五个 family 真的有差异吗

![family_separability_confusion_matrix](family_separability_confusion_matrix.png)

这张图不是 human/bot 检测，而是只在 F1-F5 之间分类：

> 给一段 bot 文本，只用 TF-IDF 判断它属于哪个 family。

结果是 family separability accuracy = 0.92。

这说明：**F1-F5 的差异不是我们主观想象出来的，TF-IDF 真的能把它们分开。**

具体看：

- F1：15/15 全部分对。
- F5：15/15 全部分对。
- F2：14/15 分对，有 1 个被误判成 F4。
- F3：13/15 分对，有 1 个像 F1，1 个像 F2。
- F4：12/15 分对，有 3 个被误判成 F2。

这告诉我们：

- F1 和 F5 的 family 指纹最明显。
- F2 和 F4 有一部分重叠。
- F3 因为用了 anchor，对其他 family 的边界更模糊一点，但仍然大体可分。

换句话说，我们确实做出了五个不同 family；但这些 family 的“不同”本身也被检测器捕捉到了。

## 4. D0 top terms heatmap：旧检测器眼里的残留 bot 倾向

![family_top_terms_heatmap](family_top_terms_heatmap.png)

注意：这张图展示的是 `D0_original_only`，也就是旧检测器。

旧检测器最终没有把 F1-F5 判成 bot，因为它的总概率都低于 0.5。但它仍然对某些词给了正向推力。

几个明显模式：

- F1：`political`、`public`、`real` 比较突出。
- F2：`global`、`conflict`、`just` 比较突出。
- F3：整体颜色最浅，说明旧模型眼里 F3 最不像旧 bot。
- F4：`ukraine`、`action` 很突出。
- F5：`just` 极其突出。

这张图的含义不是“这些词导致 F1-F5 被 D0 检出”，因为 D0 recall 为 0。

它真正说明的是：

**F1-F5 已经削弱了旧 bot 指纹，但还残留一些旧模型熟悉的 bot-side 词项。**

## 5. taxonomy stackbar：检测器主要抓的是哪类词

![fingerprint_taxonomy_stackbar](fingerprint_taxonomy_stackbar.png)

这张图把 D0 的正向贡献词分成几类：

- `topic_entity`：话题实体。
- `abstract_analysis`：抽象分析词。
- `formal_discourse`：正式论述词。
- `reddit_colloquial`：Reddit 口语词。
- `stance_emotion`：情绪/立场词。
- `other`：没有进入人工词表的其他词。

最显眼的是绿色 `other` 占比很大。

这不是说模型无法解释，而是说明我们原来的人工分类词表太粗。TF-IDF 抓住的很多有效特征不是典型 AI 词，也不是典型话题词，而是一些普通词的异常分布，例如：

- `actual`
- `exactly`
- `line`
- `thread`
- `wording`
- `numbers`
- `comparison`
- `kids`
- `addiction`

这些词在人类文本里当然也会出现，但问题是：

**在某个 family 里，它们出现得太集中、太稳定、和其他词组合得太一致。**

这就是静态生成 family 最容易被 TF-IDF 抓住的地方。

## 6. F1 到底被什么区分出来

F1 是 original architecture family。self-family detector 对 F1 的 top 贡献词是：

```text
governments
real
addiction
public
line
political
access
just
starts
civilian
civilians
politics
exactly
questions
civilian protection
aid access
question
like
```

F1 的主要指纹是：

**公共议题 + 宏观政治判断 + 稳定的解释性语气。**

它很像一个总在把事情上升到公共政治、政府责任、制度线条的人。

为什么容易被分出？

因为 F1 的词项组合很稳定：

- 公共政治词：`governments`、`public`、`political`、`politics`
- 规范判断词：`line`、`question`、`questions`
- 人道/战争相关词：`civilian`、`civilians`、`aid access`
- 常见生成式缓冲词：`real`、`just`、`exactly`

这不是某一个词的问题，而是一个固定评论姿态：总是把具体新闻变成公共政治原则讨论。

所以 F1 被分出的原因是：**架构仍然留下了“宏观评论员”的统一声音。**

## 7. F2 到底被什么区分出来

F2 是 TF-IDF-informed language family。它本来就是为了避开旧 TF-IDF 设计的。

self-family top 词：

```text
comparison
just
2019
weeks
number
headline
wording
count
actual
exactly
global
kids
thread
conflict
stat
ukraine
versus
addiction
```

F2 的指纹不是传统 AI 腔，而是：

**“校正/比较/数字/措辞”的审稿式口吻。**

它经常像在说：

- 这个 headline 有问题。
- 这个 number 要看 comparison。
- 这个 wording 不准确。
- 2019、weeks、count、stat 才是关键。

这让 F2 看起来比 F1 更口语、更 Reddit，但同时形成了新的模式：

```text
comparison + headline + wording + count + actual + exactly
```

这套词在 F2 中太稳定，所以 self-family TF-IDF 很容易学到。

所以 F2 的问题是：**为了避开旧 AI 词，转向了“事实核查/措辞校正”风格，但这个新风格本身又变成了指纹。**

## 8. F3 到底被什么区分出来

F3 是 human-anchor local TF-IDF family，理论上应该最接近人。

self-family top 词：

```text
addiction
dead
yeah
kids
dead kids
like
ugly
hostages
horrific
just
stuff
dragging
getting
line
families
gets
gross
ceasefire
```

F3 的特别之处是，它确实用了更多 human-like 口语词：

- `yeah`
- `like`
- `just`
- `stuff`
- `gets`
- `gross`

但问题是，它的 anchor 对齐并没有让每个 bot 真的像自己的指定 anchor。看 F3 anchor 图：

![f3_anchor_similarity_vs_probability](f3_anchor_similarity_vs_probability.png)

横轴是 assigned-anchor TF-IDF cosine。整体不高，大部分在 0.05-0.10 左右。

这说明：

**F3 不是逐条复刻指定 anchor 的词汇风格，而是整体吸收了一些 human-like 口语和情绪词。**

F3 为什么仍能被 self-family detector 检出？

因为它形成了新的集中词簇：

```text
dead / kids / dead kids / hostages / horrific / ugly / gross
```

也就是说，F3 不是被“AI 正式腔”检出，而是被一个高情绪、人道灾难式的口语词簇检出。

这对我们特别重要：

**靠 anchor 让语言更像人是有效的，但如果 anchor/生成器反复把文本推向同一类情绪表达，TF-IDF 仍然能抓住。**

## 9. F4 到底被什么区分出来

F4 是 PEBA trait prompt family。

self-family top 词：

```text
addiction
thread
flag
flag protests
ukraine
action
ceasefire aid
kids
correction
wording
small correction
protests
urgent
wrong thread
aid access
urgent action
global
scrutiny
```

F4 的明显指纹是：

**trait prompt 让它更像“有行为习惯的人”，但也制造了一组程序化互动词。**

特别明显的是：

- `thread`
- `flag`
- `flag protests`
- `correction`
- `small correction`
- `wrong thread`
- `urgent action`

这些词不像传统 AI 腔，但很像“prompt 里鼓励它纠正、提醒、指出上下文”的副作用。

所以 F4 的问题不是“不像人”，而是：

**它太稳定地扮演了某种会纠错、会提醒、会插入行动判断的用户。**

这解释了为什么 F4 在 family separability 里有 3 个被误判成 F2。F2 也有 `headline`、`wording`、`comparison` 这种校正型词汇，所以 F2/F4 有相似的“纠偏/校正”倾向。

## 10. F5 到底被什么区分出来

F5 是 Gemini trait prompt family。

self-family top 词：

```text
just
addiction
actually
timeline
numbers
actual
overseas
like
global
exactly
feels
political
urgent
ground
isn
deals
weaponizing
fast
```

F5 最突出的是 `just`，贡献非常高。

这很像一种 Gemini 生成出来的“口语化但仍有条理”的模式：

- `just`
- `actually`
- `actual`
- `exactly`
- `timeline`
- `numbers`
- `feels`

它不是特别正式，但有一种很稳定的解释顺序：

1. 先用 `just/actually` 开口。
2. 再说 `timeline/numbers/actual`。
3. 再落到 `global/political/urgent`。

F5 在 family separability 里 15/15 全部分对，说明它的平台/模型风格非常明显。

所以 F5 的指纹是：

**Gemini 式口语连接词 + 数字/时间线/事实校正词 + 稳定解释节奏。**

## 11. 为什么删话题词没用

我们原本可能以为：

> 检测器是不是因为 bot 都在聊 Ukraine、Gaza、politics 才检出？

反事实删除告诉我们不是。

在 self-family 中：

- F1 删除 topic entities：recall 仍是 1.00。
- F2 删除 topic entities：recall 仍是 0.98。
- F3 删除 topic entities：recall 仍是 0.92。
- F4 删除 topic entities：recall 仍是 0.96。
- F5 删除 topic entities：recall 只从 0.96 到 0.92。

所以话题不是主要原因。

真正主要的是：

```text
普通词 + 口语连接词 + 纠错词 + 情绪词 + 数字/时间线词 + 具体 family 的固定组合
```

这也解释了为什么 TF-IDF 很强：它不需要理解你说什么，只要你总用类似词组，它就能学。

## 12. 最核心的机制：TF-IDF 抓的是“分布稳定性”

我们可以把 TF-IDF 检测器理解成一个非常机械的东西：

```text
如果某些词在 bot family 中稳定出现，
且在人类训练样本中没有同样稳定地出现，
这些词就会得到正系数。
```

所以它抓的不是“AI 意图”，也不是“语义深度”，而是：

- 词频是否稳定。
- 词组是否重复。
- 某些普通词是否在 bot 中异常集中。
- 某些 human-side 词是否缺失。
- 同一个 family 是否形成统一说话习惯。

这就是为什么固定数据集下很难彻底骗过 TF-IDF。

只要生成器有稳定规则，TF-IDF 就能把规则投影到词项空间里。

## 13. 对我们后续生成器的启发

这次分析说明，问题不是简单的“AI 腔太重”。

更准确的问题是：

**每个 family 都有稳定生成机制，而稳定机制会产生稳定词项分布。**

因此后续如果继续优化静态文本，不能只做：

- 删除 AI 风格词。
- 增加 Reddit 口语。
- 换模型。
- 换 prompt。

这些都会生成新的指纹。

更有价值的方向是：

1. 让每个 family 内部的语言分布更宽，不要让所有 bot 共享同一套词项。
2. 让 persona 差异影响真实词汇选择，而不是只影响描述性 prompt。
3. 对 anchor 不要只学整体风格，要控制每个 bot 和 assigned anchor 的局部词汇相似度。
4. 不要让所有 bot 都进入“纠错/校正/数字/时间线/公共政治判断”这几种固定姿态。
5. 检测实验上要强调 held-out family，而不是同 family train/test。

最重要的是论文表述：

> 我们不是证明“bot 可以骗过 TF-IDF”，而是证明“静态生成 family 会被 TF-IDF 学到 family-specific 和 cross-family 指纹；因此更有意义的是构建跨 family 泛化 benchmark”。

这句话更接近我们当前结果。

