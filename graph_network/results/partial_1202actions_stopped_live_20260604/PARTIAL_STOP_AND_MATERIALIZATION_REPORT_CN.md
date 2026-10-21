# graph_network Partial 停机与物化报告

## 结论

本次长跑已安全终止，并已从终止时的 trace 快照物化出一个可分析的完整 partial 数据集。

后续图网络实验建议直接使用：

`graph_network/outputs/partial_1202actions_stopped_live_20260604`

对应验收报告目录：

`graph_network/results/partial_1202actions_stopped_live_20260604`

## 为什么需要物化

原始 full runner 在运行中只实时写入：

- `action_trace.jsonl`
- `events.jsonl`
- `llm_trace.jsonl`
- `memory_trace.jsonl`

而以下标准数据集文件本来要等完整 24 个 time window 全部跑完后才统一写出：

- `users.csv`
- `posts.csv`
- `comments.csv`
- `labels.csv`
- `persona_profiles.csv`
- `graph/edge_table.csv`

因此不能直接强杀后立即分析。现在已经新增并运行 `materialize_partial_from_trace.py`，用 trace 快照重建上述标准入口文件。

## 最终停止快照规模

- action rows: `1202`
- generated bots: `150`
- generated posts: `258`
- generated comments: `790`
- memory rows: `2405`
- events: `2916`
- LLM calls: `1048`

Action 分布：

- `post`: `258`
- `comment1`: `416`
- `comment2`: `185`
- `revisit`: `189`
- `skip`: `154`

LLM 调用分布：

- `openai`: `522`
- `gemini`: `526`
- `ok`: `1012`
- `ok_after_retry`: `35`
- `fallback`: `1`
- fallback rate: `0.0954%`

## 图导出结果

已生成：

- `graph/node_table.csv`
- `graph/edge_table.csv`
- `graph/homogeneous_user_graph.pt`
- `splits/*.json`

图统计：

- users: `3057`
- posts: `34604`
- comments: `48915`
- nodes: `86576`
- edges: `186501`
- homogeneous user edges: `54067`

关键边类型全部存在：

- `user_writes_post`: `34604`
- `user_writes_comment`: `48915`
- `comment_replies_post`: `45127`
- `comment_replies_comment`: `3788`
- `user_comments_on_user_post`: `24310`
- `user_replies_to_user_comment`: `3279`
- `user_receives_reply`: `26186`
- `user_revisits_thread`: `292`

## Split 构造

已生成 `14` 个 split，包括：

- `original_only_zero_shot`
- `lofo_g1`
- `lofo_g2`
- `lofo_g3`
- `label_scarcity_*`
- `time_forward`
- `incremental_curriculum_order1/2/3`

Family 规模：

- `G1_population_calibrated`: `50`
- `G2_anchor_behavior`: `50`
- `G3_contextual_interactive`: `50`

## 完整性检查

`check_dataset_integrity.py` 结果为 `PASS`。

通过项包括：

- persona profile 数量与 generated bot 数量一致。
- `post/comment1/comment2/revisit/skip` 五类 action 均存在。
- `notification_created` 存在。
- `browse_context` 和 `post_action_state` memory 均存在。
- GPT 与 Gemini 均真实调用成功。
- fallback rate 低于 20%。
- 生成文本没有 dry-run 模板残留。
- 评论无空文本。
- 评论无 JSON 包裹残留。
- 图导出所需关键边类型全部存在。

## 注意事项

这是一份 partial 数据集，不是完整 3600 action 的全量数据集。它已经足够做一轮初步图网络泛化分析，但结论应标注为 partial run / stopped snapshot。

原始中断前 trace 仍保留在：

`graph_network/outputs/full_3family_150bot_24window_live`

该目录本身仍是未完成运行目录，不建议直接作为分析入口。分析请使用上面的 `partial_1202actions_stopped_live_20260604`。
