# Family Separability 实验报告

## 研究问题

五个 bot family 在文本空间中是否能彼此区分？如果能，说明生成策略留下 family 指纹。

## Test 指标

```text
               model feature     lr_mode  accuracy  macro_f1
tfidf_multinomial_lr   tfidf multinomial      1.00  1.000000
        tfidf_ovr_lr   tfidf         ovr      1.00  1.000000
  bge_multinomial_lr     bge multinomial      0.95  0.949804
          bge_ovr_lr     bge         ovr      0.95  0.949804
```

## TF-IDF Confusion Matrix

```text
true_family  F1  F2  F3  F4  F5
         F1   8   0   0   0   0
         F2   0   8   0   0   0
         F3   0   0   8   0   0
         F4   0   0   0   8   0
         F5   0   0   0   0   8
```

## BGE Confusion Matrix

```text
true_family  F1  F2  F3  F4  F5
         F1   8   0   0   0   0
         F2   1   7   0   0   0
         F3   0   0   8   0   0
         F4   0   0   1   7   0
         F5   0   0   0   0   8
```

## D2 Cross-Round

```text
      protocol model feature lr_mode split                 status family family_short  accuracy  macro_f1  precision  recall  f1  support  train_size  eval_size                                                                                                                                                              note
D2_cross_round   ALL     ALL     ALL  test round_metadata_missing    ALL          ALL       NaN       NaN        NaN     NaN NaN        0           0          0 round_metadata_missing: generated_profiles_all.family_round has only one unique value [0]; generated_profiles_all.family_generation has only one unique value [0]
```

## 解释

- TF-IDF test accuracy/macro-F1 达到 1.0，说明 family 之间存在非常强的表层词汇/短语指纹。
- BGE test accuracy 约 0.95，说明语义空间也能区分 family，但 F2/F1、F4/F3 有少量混淆。
- 这不是简单的好消息：family 多样化同时也制造了可识别的 family fingerprint。
