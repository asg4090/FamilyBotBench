# 有意义顺序实验结果报告

本实验没有枚举 120 个全排列，只按照计划文档中的代表性 curriculum 顺序测试检测器泛化恢复。

## 实验顺序

- `engineering_order`: F1 -> F2 -> F3 -> F4 -> F5
- `hard_to_easy`: F2 -> F3 -> F5 -> F4 -> F1
- `easy_to_hard`: F1 -> F4 -> F5 -> F3 -> F2
- `provider_shift_first`: F5 -> F1 -> F2 -> F3 -> F4
- `anchor_first`: F3 -> F2 -> F5 -> F4 -> F1
- `trait_first`: F4 -> F5 -> F2 -> F3 -> F1

## 输出文件

- `selected_order_prepost_metrics.csv`：每条顺序、每一步、每个模型的加入前/加入后 recall。
- `selected_order_summary.csv`：每条顺序的平均 before/after recall。
- `selected_family_hardness_summary.csv`：family 在不同顺序下的稳定难度。
- `selected_position_effect_summary.csv`：family 放在不同位置时的 before recall。
- `selected_bridge_effects.csv`：某 family 加入训练后，对后续未见 family recall 的逐项影响。
- `selected_bridge_effect_summary.csv`：每个 family 的平均 bridge effect。
- `figures/selected_orders_before_recall_curves.png`：代表性顺序的 before recall 曲线。
- `figures/selected_orders_after_recall_curves.png`：代表性顺序的 after recall 曲线。
- `figures/selected_orders_family_hardness_boxplot.png`：family 顺序敏感性箱线图。
- `figures/selected_orders_position_effect.png`：位置效应图。
- `figures/selected_orders_bridge_effect.png`：family bridge effect 图。

## 各顺序平均结果

### TFIDF

| 顺序 | mean before recall | mean after recall | mean recovery gain | min before recall |
|---|---:|---:|---:|---:|
| engineering_order | 0.336 | 1.000 | 0.664 | 0.000 |
| hard_to_easy | 0.484 | 0.996 | 0.512 | 0.000 |
| easy_to_hard | 0.360 | 1.000 | 0.640 | 0.000 |
| provider_shift_first | 0.336 | 0.992 | 0.656 | 0.000 |
| anchor_first | 0.544 | 0.984 | 0.440 | 0.000 |
| trait_first | 0.420 | 0.992 | 0.572 | 0.000 |

### BGE

| 顺序 | mean before recall | mean after recall | mean recovery gain | min before recall |
|---|---:|---:|---:|---:|
| engineering_order | 0.640 | 0.952 | 0.312 | 0.020 |
| hard_to_easy | 0.664 | 0.948 | 0.284 | 0.000 |
| easy_to_hard | 0.648 | 0.952 | 0.304 | 0.320 |
| provider_shift_first | 0.628 | 0.936 | 0.308 | 0.100 |
| anchor_first | 0.604 | 0.936 | 0.332 | 0.000 |
| trait_first | 0.556 | 0.948 | 0.392 | 0.160 |

## Family 稳定难度

### TFIDF

| family | before mean | before std | min | max | after mean |
|---|---:|---:|---:|---:|---:|
| F1 | 0.510 | 0.537 | 0.000 | 1.000 | 1.000 |
| F2 | 0.367 | 0.385 | 0.000 | 1.000 | 0.997 |
| F3 | 0.220 | 0.209 | 0.000 | 0.440 | 0.987 |
| F4 | 0.653 | 0.454 | 0.000 | 0.960 | 0.993 |
| F5 | 0.317 | 0.242 | 0.000 | 0.600 | 0.993 |

### BGE

| family | before mean | before std | min | max | after mean |
|---|---:|---:|---:|---:|---:|
| F1 | 0.887 | 0.148 | 0.700 | 1.000 | 1.000 |
| F2 | 0.323 | 0.313 | 0.000 | 0.840 | 0.927 |
| F3 | 0.633 | 0.322 | 0.000 | 0.860 | 0.947 |
| F4 | 0.797 | 0.347 | 0.160 | 1.000 | 0.987 |
| F5 | 0.477 | 0.278 | 0.100 | 0.760 | 0.867 |

## Bridge Effect

Bridge effect 衡量的是：把某个 family 加入训练后，对后续仍未见 family 的 recall 平均提升。

### TFIDF

| added family | mean bridge gain | std | min | max | n |
|---|---:|---:|---:|---:|---:|
| F1 | 0.111 | 0.146 | 0.000 | 0.500 | 11 |
| F2 | 0.277 | 0.170 | 0.000 | 0.700 | 14 |
| F3 | 0.333 | 0.244 | 0.020 | 0.680 | 12 |
| F4 | 0.182 | 0.192 | 0.000 | 0.540 | 10 |
| F5 | 0.114 | 0.119 | 0.000 | 0.300 | 13 |

### BGE

| added family | mean bridge gain | std | min | max | n |
|---|---:|---:|---:|---:|---:|
| F1 | 0.175 | 0.159 | 0.020 | 0.460 | 11 |
| F2 | 0.316 | 0.241 | 0.000 | 0.800 | 14 |
| F3 | 0.192 | 0.227 | 0.000 | 0.780 | 12 |
| F4 | 0.190 | 0.154 | 0.000 | 0.500 | 10 |
| F5 | 0.185 | 0.199 | 0.000 | 0.660 | 13 |

## 初步解释

before recall 代表当前 family 尚未加入训练集时的检出能力，因此它测的是跨 family 泛化；after recall 代表把当前 family 加入训练后是否能恢复，因此它测的是模型容量和特征是否足以吸收该 family。

如果一个 family 在多条顺序中 before recall 均值低且方差低，它更像稳定难例；如果它放后面才变高，说明它依赖前置 family 的覆盖，不一定本身容易。
