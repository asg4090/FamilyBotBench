# 生成评论文本长度与空值审计

- 数据目录：`graph_network/outputs/partial_1202actions_stopped_live_20260604`
- 生成评论数：`790`
- 生成帖子数：`258`
- 空评论：`0`
- 空帖子标题：`0`
- 空帖子正文：`0`

## 结论

没有发现大量空文本。问题主要是评论整体偏短：生成评论中位数约 33 words，按 char/4 粗略估算约 48 tokens。所有评论都在约 80-90 tokens 以下，这与当前 prompt 要求 `Write 1-3 sentences` 直接相关。

此外发现 1 条明显拒答/安全话术残留，应在正式分析前过滤或重生成。

## 短文本比例

- <= 15 words: `2` / `790` = `0.25%`
- <= 20 words: `17` / `790` = `2.15%`
- <= 30 words: `280` / `790` = `35.44%`
- <= 50 words: `788` / `790` = `99.75%`
- <= 80 words: `790` / `790` = `100.00%`

## 残留/异常话术

- `refusal_or_policy`: `2`
- `ai_residue`: `0`
- `json_residue`: `0`
- `template_residue`: `0`

## 长度分位数

|      |   word_count |   char_len |   rough_tokens_chars4 |
|-----:|-------------:|-----------:|----------------------:|
| 0    |        13    |      73    |                18.2   |
| 0.01 |        18.89 |     107.89 |                26.978 |
| 0.05 |        22    |     132    |                33     |
| 0.1  |        24.9  |     143    |                35.8   |
| 0.25 |        28    |     166    |                41.5   |
| 0.5  |        33    |     191    |                47.8   |
| 0.75 |        37    |     218    |                54.5   |
| 0.9  |        40.1  |     241    |                60.2   |
| 0.95 |        43    |     253    |                63.2   |
| 0.99 |        47    |     277    |                69.2   |
| 1    |        58    |     353    |                88.2   |

## 按 action_type 的长度

| action_type   |   ('word_count', 'count') |   ('word_count', 'mean') |   ('word_count', 'median') |   ('word_count', 'min') |   ('word_count', 'max') |   ('char_len', 'count') |   ('char_len', 'mean') |   ('char_len', 'median') |   ('char_len', 'min') |   ('char_len', 'max') |   ('rough_tokens_chars4', 'count') |   ('rough_tokens_chars4', 'mean') |   ('rough_tokens_chars4', 'median') |   ('rough_tokens_chars4', 'min') |   ('rough_tokens_chars4', 'max') |
|:--------------|--------------------------:|-------------------------:|---------------------------:|------------------------:|------------------------:|------------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|-----------------------------------:|----------------------------------:|------------------------------------:|---------------------------------:|---------------------------------:|
| comment1      |                       416 |                    32.54 |                         33 |                      17 |                      54 |                     416 |                 190.43 |                      187 |                    96 |                   330 |                                416 |                             47.61 |                                46.8 |                             24   |                             82.5 |
| comment2      |                       185 |                    31.56 |                         32 |                      13 |                      49 |                     185 |                 186.31 |                      186 |                    73 |                   297 |                                185 |                             46.58 |                                46.5 |                             18.2 |                             74.2 |
| revisit       |                       189 |                    33.97 |                         33 |                      18 |                      58 |                     189 |                 200.56 |                      199 |                   115 |                   353 |                                189 |                             50.14 |                                49.8 |                             28.8 |                             88.2 |

## LLM status by provider

| provider   |   fallback |   ok |   ok_after_retry |
|:-----------|-----------:|-----:|-----------------:|
| gemini     |          1 |  496 |               29 |
| openai     |          0 |  516 |                6 |

## LLM status by call_type

| call_type   |   fallback |   ok |   ok_after_retry |
|:------------|-----------:|-----:|-----------------:|
| comment1    |          0 |  401 |               15 |
| comment2    |          0 |  177 |                8 |
| post        |          1 |  252 |                5 |
| revisit     |          0 |  182 |                7 |

## 最短样本

- `c_gen_full_3family_150bot_24window_live_000123` `comment2` `G2_anchor_behavior` words=`13`: I can't help write a targeted political comment aimed at a specific user.
- `c_gen_full_3family_150bot_24window_live_000114` `comment2` `G1_population_calibrated` words=`15`: It really is always projection with these guys. Every single accusation is just a confession.
- `c_gen_full_3family_150bot_24window_live_000646` `comment2` `G3_contextual_interactive` words=`17`: Also good. People involved with Jan. 6 should not be getting rewarded with a seat in Congress.
- `c_gen_full_3family_150bot_24window_live_000016` `comment1` `G3_contextual_interactive` words=`17`: Of course she did. Cannon is doing absolutely everything she can to drag this out past November.
- `c_gen_full_3family_150bot_24window_live_000140` `comment1` `G3_contextual_interactive` words=`17`: New fear unlocked. I already side-eye pork if it’s even slightly undercooked, and this definitely isn’t helping.
- `c_gen_full_3family_150bot_24window_live_000068` `comment1` `G1_population_calibrated` words=`18`: It’s definitely scary. We're basically one direct hit on a coalition destroyer away from a full-blown regional war.
- `c_gen_full_3family_150bot_24window_live_000187` `comment2` `G1_population_calibrated` words=`18`: Honestly, it's about time. Seeing the state seize his tacky gold-plated towers would be the ultimate poetic justice.
- `c_gen_full_3family_150bot_24window_live_000312` `revisit` `G3_contextual_interactive` words=`18`: Absolutely. They don't want solutions, they want content. The fundraising emails practically write themselves during these fake crises.
- `c_gen_full_3family_150bot_24window_live_000502` `comment1` `G3_contextual_interactive` words=`19`: About damn time, but you already know Ken Paxton is going to sue the city the second this passes.
- `c_gen_full_3family_150bot_24window_live_000520` `comment1` `G1_population_calibrated` words=`19`: Uh, most of us? The mental gymnastics these people do to defend him is honestly hilarious at this point.
- `c_gen_full_3family_150bot_24window_live_000499` `comment2` `G2_anchor_behavior` words=`19`: It went straight to his legal fees and personal pockets. Anyone who actually donated to that is a sucker.
- `c_gen_full_3family_150bot_24window_live_000345` `comment2` `G2_anchor_behavior` words=`19`: Wait, did someone actually say that on the floor? This entire speaker vote has been an absolute clown show.

## 输出文件

- 异常/短评论清单：`graph_network/results/partial_1202actions_stopped_live_20260604/comment_text_flagged_short_or_residue.csv`
