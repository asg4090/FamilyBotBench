# graph_network完整性检查报告

- run_dir: `graph_network/outputs/partial_plus_g4_slot_conditioned_20260606`
- generated users: `200`
- profiles: `200`
- generated posts: `339`
- generated comments: `1068`
- llm calls: `1048`
- llm fallback rate: `0.10%`

## Action 分布

|          |   0 |
|:---------|----:|
| post     | 339 |
| skip     | 195 |
| comment2 | 285 |
| comment1 | 524 |
| revisit  | 259 |

## LLM Provider 分布

|        |   0 |
|:-------|----:|
| gemini | 526 |
| openai | 522 |

## 图边检查

|                              |     0 |
|:-----------------------------|------:|
| user_writes_post             | 34685 |
| user_writes_comment          | 49519 |
| comment_replies_post         | 45303 |
| user_comments_on_user_post   | 24763 |
| user_receives_reply          | 27026 |
| comment_replies_comment      |  4216 |
| user_replies_to_user_comment |  3610 |
| user_revisits_thread         |   417 |

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
