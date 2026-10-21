# 连续漂移实验中的机器人漂移维度说明表

这张表用于论文正文或附录，目的是把“机器人连续漂移”讲清楚。这里的漂移不是让机器人直接最小化检测器概率，而是在保持人类行为分布约束的前提下，系统性改变机器人群体的数据生成分布。换句话说，我们关心的是：下一代机器人在哪些可解释维度上和上一代不一样，以及检测器是否能泛化到这些未见过的变化。

## 表 1：连续漂移实验中的八个漂移维度（正式表述版）

| 漂移维度 | 操作化定义 | 生成侧控制变量 | 观测变量与审计统计量 |
|---|---|---|---|
| 语言表达 `language` | 指评论文本在词汇选择、句长结构、语体正式度、语用标记与立场表达方式上的分布变化。该维度用于刻画机器人群体在不必改变互动对象的情况下产生的文本风格漂移。 | 通过 persona 中的 `style`、`stance`、`brevity`、`sentence_shape`、`formality`、`hedging`、`question_rate`、`target_mean_words` 等变量控制评论长度、口语化程度、反问/感叹频率、立场强度和表达粗糙度。 | 观测评论长度分布、词项分布、口语标记比例、抽象词比例和疑问/感叹标点频率。审计指标包括 `length_wasserstein`、`lexical_js`、`style_rate_shift`。 |
| 话题内容 `topic_content` | 指机器人群体在主题、议题实体和社区空间上的注意力分布变化。该维度反映机器人是否从一个内容域迁移到另一个内容域，或在主导话题外引入探索性话题。 | 通过 `interest_weights`、`topic_focus`、`dominant_topic`、`topic_sampling` 和 `subreddits` 控制主导议题、溢出议题、探索比例及社区选择。 | 观测评论主题分布、subreddit 分布和 profile preference 分布。审计指标包括 `comment_topic_js`、`subreddit_js`、`profile_preference_js`。 |
| 人口与身份分布 `persona_population` | 指机器人群体在身份属性、社会背景、兴趣偏好和活跃类型上的组成变化。该维度用于避免所有机器人来自同质化 persona 模板。 | 通过 `age`、`region`、`ideology`、`education`、`interest_weights`、`population_cluster`、`persona_diversity_strength` 等变量控制群体构成和个体差异。 | 观测年龄分布、地区分布、意识形态分布、教育水平分布和兴趣偏好分布。审计指标包括 `age_ks`、`gender_js`、`region_js`、`educationlevel_js`、`ideology_js`、`preference_js`。 |
| 行为动作 `action_behavior` | 指机器人在社交环境中选择行动类型的分布变化，包括发帖、一级评论、二级回复、跳过、探索和争论参与。该维度刻画平台行为层面的漂移。 | 通过 `post_tendency`、`reply_tendency`、`exploration_tendency`、`debate_tendency`、`activity_cluster`、`skip_bias` 等变量控制不同动作的倾向和活跃度差异。 | 观测动作类型比例、每用户评论数、每用户发帖数和活跃不均衡程度。审计指标包括 `action_distribution_js`、`comments_per_user_ks`、`posts_per_user_ks`、`activity_gini_shift`。 |
| 图结构 `graph_structure` | 指机器人与其他用户之间形成的有向互动网络结构变化，包括出入度、目标作者多样性、互惠关系和线程局部互动。该维度用于刻画关系层面的分布漂移。 | 通过 `graph_affinity`、`reply_tendency`、`graph_target`、`target_bot_authored_cap` 等变量控制二级回复、弱关系互动、目标作者选择和 bot-authored 内容暴露上限。 | 观测出边权重、入边权重、方向平衡、互惠比例、目标作者数量和入边作者数量。审计指标包括 `total_out_weight_proxy`、`total_in_weight_proxy`、`direction_balance_proxy`、`reciprocal_share_proxy`、`unique_target_authors`、`unique_inbound_authors` 的 KS 距离。 |
| 时间节律 `temporal_rhythm` | 指机器人行为事件在时间轴上的分布变化，包括响应延迟、事件间隔、日内活跃时间、爆发式活跃和沉默期。该维度用于检测时间过程上的泛化能力。 | 通过 `temporal_profile` 中的 `post_delay_min/max`、`comment_delay_min/max`、`burstiness`、`silence_ratio` 等变量控制发帖延迟、评论延迟、活动密度和沉默窗口。 | 观测事件间隔分布、小时分布和事件间隔标准差。审计指标包括 `interarrival_ks`、`hour_js`、`burstiness_shift`。 |
| 记忆与回访 `memory_revisit` | 指机器人基于自身历史、通知队列和线程记忆进行后续浏览或回复的分布变化。该维度刻画从单次响应到多轮社交参与的行为漂移。 | 通过 `revisit_tendency`、`notification_reactivity`、`thread_memory`、`notification_window` 等变量控制是否回访自己的帖子/评论、是否响应他人回复以及回访窗口长度。 | 观测候选内容来源分布、回访次数、通知创建次数和 own-history 浏览比例。审计指标包括 `candidate_source_js`、`revisit_count_shift`、`notification_count_shift`。 |
| 生成模型混合 `llm_generation` | 指生成端模型来源、模型版本和采样温度的混合分布变化。该维度用于引入生成器层面的异质性，但不向检测器暴露模型标签。 | 通过 `llm_provider`、`llm_model`、`llm_temperature`、`llm_reasoning_effort` 以及 OpenAI-compatible/Gemini-compatible 的模型池配置控制生成模型组合。 | 观测 provider 分布、model 分布和 temperature 分布。审计指标包括 `provider_js`、`model_js`、`temperature_ks`；这些字段仅用于生成审计和分组分析，不作为检测器输入特征。 |

## 表 2：当前连续泛化实验中三个 family 的漂移重点

| Family | 主要目标维度 | 辅助漂移维度 | 尽量冻结或弱化的维度 | 实验含义 |
|---|---|---|---|---|
| `A_language_drift` | `language`、`topic_content` | `persona_population` | `graph_structure`、`temporal_rhythm` | 主要考察文本检测器是否能泛化到新的说话风格和话题偏好。它不是完全不改变其他维度，而是让语言/话题漂移最明显，图和时间尽量保持接近基线。 |
| `C_graph_drift` | `graph_structure`、`memory_revisit` | `action_behavior`、`temporal_rhythm` | `language` | 主要考察图结构检测器是否能泛化到新的互动关系模式。它强调二级回复、弱关系、回访和通知驱动互动，同时尽量不把语言风格作为主要变化来源。 |
| `E_mixed_emergent_drift` | 从 `language`、`action_behavior`、`graph_structure` 中随机选择 2-3 个作为主漂移轴 | `topic_content`、`temporal_rhythm`、`memory_revisit` | 无固定冻结维度 | 用来模拟更接近真实系统演化的混合漂移：文本、行为和图结构会一起变化。这个 family 适合检验检测器面对多因素同时变化时的跨代泛化能力。 |

## 可直接放入论文的方法描述

在连续漂移实验中，我们将机器人生成过程拆分为八个可解释维度：语言表达、话题内容、人口与身份分布、行为动作、图结构、时间节律、记忆回访和生成模型混合。每一代机器人不是通过直接优化检测器输出生成，而是由 mutation plan 指定目标漂移轴、辅助漂移轴和冻结轴，再通过 persona、行为倾向、推荐候选、回访机制和模型采样配置共同改变生成分布。生成完成后，我们比较父代和子代在各维度上的统计距离，只有目标维度发生足够漂移、非目标维度没有严重失控的 family 才进入后续检测器评估。这样设计的目的不是构造一个“不可检测”的机器人，而是构造一组可解释、可复现、可逐代扩展的分布漂移，用来评估不同检测算法的跨 family 和跨 generation 泛化能力。
