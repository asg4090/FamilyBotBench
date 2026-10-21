# LOFO 正式化实验报告

## 研究问题

如果某个 family 完全不进入训练集，检测器能否泛化检出它？

## 协议

每次留出一个 family，训练集为 `original_train + 其余 4 个 family`，测试集包括 held-out family 以及 original_test。

## Held-out Bot Recall

```text
heldout_family_short  BGE+LR  TF-IDF+LR
                  F1    1.00       1.00
                  F2    0.84       1.00
                  F3    0.86       0.64
                  F4    1.00       0.96
                  F5    0.76       0.60
```

## 完整指标

```text
    model              heldout_family heldout_family_short train_families  heldout_bot_recall  heldout_mean_bot_probability  original_test_accuracy  original_test_bot_recall  original_test_human_recall  original_test_macro_f1
TF-IDF+LR    F1_original_architecture                   F1    F2+F3+F4+F5                1.00                      0.728757                0.975945                      0.93                    1.000000                0.972868
   BGE+LR    F1_original_architecture                   F1    F2+F3+F4+F5                1.00                      0.907182                0.975945                      0.96                    0.984293                0.973274
TF-IDF+LR  F2_tfidf_informed_language                   F2    F1+F3+F4+F5                1.00                      0.722660                0.975945                      0.93                    1.000000                0.972868
   BGE+LR  F2_tfidf_informed_language                   F2    F1+F3+F4+F5                0.84                      0.640660                0.975945                      0.96                    0.984293                0.973274
TF-IDF+LR F3_human_anchor_local_tfidf                   F3    F1+F2+F4+F5                0.64                      0.537687                0.975945                      0.93                    1.000000                0.972868
   BGE+LR F3_human_anchor_local_tfidf                   F3    F1+F2+F4+F5                0.86                      0.710650                0.975945                      0.96                    0.984293                0.973274
TF-IDF+LR        F4_peba_trait_prompt                   F4    F1+F2+F3+F5                0.96                      0.750473                0.975945                      0.93                    1.000000                0.972868
   BGE+LR        F4_peba_trait_prompt                   F4    F1+F2+F3+F5                1.00                      0.870787                0.975945                      0.96                    0.984293                0.973274
TF-IDF+LR      F5_gemini_trait_prompt                   F5    F1+F2+F3+F4                0.60                      0.549630                0.975945                      0.93                    1.000000                0.972868
   BGE+LR      F5_gemini_trait_prompt                   F5    F1+F2+F3+F4                0.76                      0.603986                0.979381                      0.96                    0.989529                0.977036
```

## 解释

- TF-IDF 最难泛化到 `F5`，held-out recall = `0.600`。
- BGE 最难泛化到 `F5`，held-out recall = `0.760`。
- 这组结果是比 curriculum 顺序更干净的跨 family 泛化证据。
