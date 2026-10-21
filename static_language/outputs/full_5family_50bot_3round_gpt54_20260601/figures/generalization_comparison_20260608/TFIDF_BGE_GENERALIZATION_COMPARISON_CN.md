# TF-IDF vs BGE 泛化性对比图说明

数据来源：`full_5family_50bot_3round_gpt54_20260601`。本轮主实验有 5 个静态语言 family，每个 family 50 个 bot。当前数据已经足够做一轮清晰的算法泛化性对比，但还不够完全消除 curriculum 顺序偏差；如果要写成更强的论文结论，建议后续补全多顺序 curriculum 或全排列平均。

## 核心数字

| 比较项 | TF-IDF recall | BGE recall | 解释 |
|---|---:|---:|---|
| original-only zero-shot | 0.000 | 0.204 | 只用原始 bot 训练，直接测 F1-F5 |
| single-family in-family | 0.976 | 0.880 | 训练某 family，测试同 family |
| single-family cross-family | 0.186 | 0.537 | 训练某 family，测试未见 family |
| LOFO held-out | 0.848 | 0.892 | 训练其他 4 个 family，测试留出的第 5 个 |
| single-family generalization gap | 0.790 | 0.343 | 同族 recall 减跨族 recall，越小越泛化 |

## 图表

1. `01_transfer_heatmaps_tfidf_vs_bge.png`：单 family 训练到其他 family 的迁移热力图。
2. `02_generalization_summary_bars.png`：四种设置下 TF-IDF 和 BGE 的总 recall 对比。
3. `03_lofo_recall_by_family.png`：留一族实验，每个 held-out family 的 recall。
4. `04_incremental_recovery_curve.png`：按 F1 -> F5 顺序逐步加入训练集后，对剩余未见 family 的平均 recall。
5. `05_curriculum_prepost_by_detector.png`：每一轮加入当前 family 前后，当前 family recall 的恢复幅度。
6. `06_generalization_gap_by_train_family.png`：每个单独训练 family 的泛化缺口。

## 初步解释

- 单 family 迁移里，BGE 的跨 family recall 为 0.537，明显高于 TF-IDF 的 0.186。这说明 BGE 在只见过一个 bot family 时更能迁移。
- TF-IDF 的同 family recall 为 0.976，高于 BGE 的 0.880，但跨 family 后掉到 0.186。这正是“表层指纹强、跨族泛化弱”的形状。
- LOFO 中，两者都较高：TF-IDF 0.848，BGE 0.892。这说明一旦训练集覆盖了 4 个 family，两个检测器都能恢复；BGE 均值略高。
- incremental curve 目前只有 F1 -> F5 一个顺序，因此适合展示“恢复过程”，不适合单独作为无偏结论。这个图像只说明在当前顺序下，BGE 早期恢复更快，TF-IDF 后期依靠新增 family 追上很多。
