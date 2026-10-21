# graph_network完整性检查报告

- run_dir: `graph_network/outputs/partial_1202actions_stopped_live_20260604`
- generated users: `150`
- profiles: `150`
- generated posts: `258`
- generated comments: `790`
- llm calls: `1048`
- llm fallback rate: `0.10%`

## Action 分布

|          |   0 |
|:---------|----:|
| post     | 258 |
| skip     | 154 |
| comment2 | 185 |
| comment1 | 416 |
| revisit  | 189 |

## LLM Provider 分布

|        |   0 |
|:-------|----:|
| gemini | 526 |
| openai | 522 |

## 图边检查

|                              |     0 |
|:-----------------------------|------:|
| user_writes_post             | 34604 |
| user_writes_comment          | 48915 |
| comment_replies_post         | 45127 |
| user_comments_on_user_post   | 24310 |
| user_receives_reply          | 26186 |
| comment_replies_comment      |  3788 |
| user_replies_to_user_comment |  3279 |
| user_revisits_thread         |   292 |

## 验收项

|                                | 0    |
|:-------------------------------|:-----|
| profiles_match_generated_users | True |
| has_post_action                | True |
| has_comment1_action            | True |
| has_comment2_action            | True |
| has_revisit_action             | True |
| has_skip_action                | True |
| has_notification_created       | True |
| has_memory_browse_context      | True |
| has_memory_post_action_state   | True |
| required_edges_present         | True |
| live_llm_calls_present         | True |
| live_llm_has_openai            | True |
| live_llm_has_gemini            | True |
| fallback_rate_under_20pct      | True |
| no_template_posts              | True |
| no_template_comments           | True |
| no_empty_comments              | True |
| no_json_residue_comments       | True |

总体结论：`PASS`
