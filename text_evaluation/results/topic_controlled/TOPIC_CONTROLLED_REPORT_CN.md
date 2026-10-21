# Semantic-Cluster-Controlled Detection 实验报告

## 研究问题

检测器是不是主要依靠语义簇 shortcut，而不是 bot 语言特征？

## 协议

用 BGE embedding + KMeans 做无监督语义簇，`k=8/12/16`；主解释以 `k=12` 为中心。这个实验不是严格 topic control，因为聚类可能混入话题、风格、人机差异和 family 差异。

## Topic-Matched Detection

```text
 k    model status  matched_total_retained  baseline_global_test_bot_recall  controlled_matched_test_bot_recall  bot_recall_drop_vs_global  bot_recall_drop_vs_uncontrolled_matched_test
 8 tfidf_lr     ok                     494                         0.942857                            0.866667                   0.076190                                     -0.100000
 8   bge_lr     ok                     494                         0.960000                            0.900000                   0.060000                                     -0.033333
12 tfidf_lr     ok                     602                         0.942857                            0.925926                   0.016931                                     -0.074074
12   bge_lr     ok                     602                         0.960000                            0.870370                   0.089630                                      0.018519
16 tfidf_lr     ok                     420                         0.942857                            0.821429                   0.121429                                     -0.107143
16   bge_lr     ok                     420                         0.960000                            0.785714                   0.174286                                      0.000000
```

## Within-Topic 状态

```text
 k    model              status  n
 8   bge_lr                  ok  6
 8   bge_lr skipped_small_topic  2
 8 tfidf_lr                  ok  6
 8 tfidf_lr skipped_small_topic  2
12   bge_lr                  ok  7
12   bge_lr skipped_small_topic  5
12 tfidf_lr                  ok  7
12 tfidf_lr skipped_small_topic  5
16   bge_lr                  ok  8
16   bge_lr skipped_small_topic  8
16 tfidf_lr                  ok  8
16 tfidf_lr skipped_small_topic  8
```

## Leave-One-Topic-Out 状态

```text
 k    model              status  n
 8   bge_lr                  ok  7
 8   bge_lr skipped_small_topic  1
 8 tfidf_lr                  ok  7
 8 tfidf_lr skipped_small_topic  1
12   bge_lr                  ok 11
12   bge_lr skipped_small_topic  1
12 tfidf_lr                  ok 11
12 tfidf_lr skipped_small_topic  1
16   bge_lr                  ok 15
16   bge_lr skipped_small_topic  1
16 tfidf_lr                  ok 15
16 tfidf_lr skipped_small_topic  1
```

## k=12 Topic 关键词预览

```text
 topic_id  topic_total_count  human_count  bot_count                                                                                                                                                 top_tfidf_terms
        0                261          228         33                                                           china; people; said; time; oh; good; year; like; know; new; just; great; state; really; korean; going
        1                325            1        324         political; accountability; implications; legal; just; trump; justice; landscape; integrity; public; discourse; narrative; democracy; like; isn; broader
        2                322          294         28                                       trump; biden; said; president; election; vote; like; people; republican; party; just; republicans; don; court; state; gop
        3                216          197         19                                     israel; hamas; people; israeli; biden; gaza; war; just; genocide; like; don; hostages; palestinian; good; idf; palestinians
        4                177          177          0                                                                                                                                               no_nonempty_terms
        5                166            5        161                                          people; kids; just; dead; exactly; yeah; count; number; line; like; thread; actual; governments; gaza; headline; weeks
        6                181           10        171   political; accountability; just; implications; trump; legal; like; justice; isn; democracy; integrity; landscape; narrative; issues; political landscape; gop
        7                313            2        311 conflict; ukraine; international; implications; military; ongoing; global; peace; situation; just; geopolitical; lives; russia; humanitarian; dynamics; crucial
        8                277          186         91                                         just; people; addiction; like; family; actually; actual; police; right; getting; money; years; going; said; thing; news
        9                208          118         90                            ukraine; russia; russian; putin; conflict; just; war; military; like; ukrainian; ongoing; international; china; people; global; nato
       10                235          215         20                                      israel; hamas; gaza; israeli; palestinian; said; people; idf; iran; war; palestinians; article; don; world; news; hostages
       11                476          474          2                                           trump; biden; people; just; like; don; republicans; vote; going; think; republican; election; good; know; want; right
```

## 解释

- k=12 `tfidf_lr`：baseline bot recall `0.943`，topic-controlled recall `0.926`，相对全局 drop `0.017`。
- k=12 `bge_lr`：baseline bot recall `0.960`，topic-controlled recall `0.870`，相对全局 drop `0.090`。
- 如果控制 topic 后 recall 下降，说明存在话题 shortcut；如果下降不大，说明检测器还抓到 topic 之外的信号。
