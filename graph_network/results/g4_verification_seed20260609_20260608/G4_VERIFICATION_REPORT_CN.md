# G4 Verification Report

## Replay Removal

- replay human comments removed: `326`
- slot-conditioned notification events removed: `326`
- rebuilt full user-user edges: `55399`
- noReplay user-user edges: `54284`
- rebuilt full human->G4 edges: `930`
- noReplay human->G4 edges: `0`

## lofo_g4 Condition Metrics

| condition                 | model          |   bot_recall |   bot_mean_probability |   dropped_feature_count |
|:--------------------------|:---------------|-------------:|-----------------------:|------------------------:|
| drop_comment_owner_bundle | GraphStats_EBM |         0.16 |            0.188943    |                       7 |
| drop_comment_owner_bundle | GraphStats_LR  |         0    |            0.00339679  |                       7 |
| drop_comment_owner_bundle | GraphStats_RF  |         0.9  |            0.617358    |                       7 |
| drop_context_bundle       | GraphStats_EBM |         0.26 |            0.312413    |                       7 |
| drop_context_bundle       | GraphStats_LR  |         0    |            0.029794    |                       7 |
| drop_context_bundle       | GraphStats_RF  |         0.92 |            0.643905    |                       7 |
| drop_direction_balance    | GraphStats_EBM |         0.24 |            0.264693    |                       1 |
| drop_direction_balance    | GraphStats_LR  |         0.04 |            0.152111    |                       1 |
| drop_direction_balance    | GraphStats_RF  |         0.88 |            0.597414    |                       1 |
| drop_g4_signature_bundle  | GraphStats_EBM |         0.16 |            0.268495    |                      18 |
| drop_g4_signature_bundle  | GraphStats_LR  |         0.36 |            0.418729    |                      18 |
| drop_g4_signature_bundle  | GraphStats_RF  |         0.18 |            0.292673    |                      18 |
| drop_receive_bundle       | GraphStats_EBM |         0.02 |            0.0148354   |                       9 |
| drop_receive_bundle       | GraphStats_LR  |         0    |            0.0290341   |                       9 |
| drop_receive_bundle       | GraphStats_RF  |         0.02 |            0.259898    |                       9 |
| drop_reciprocity          | GraphStats_EBM |         0.92 |            0.829316    |                       1 |
| drop_reciprocity          | GraphStats_LR  |         0.52 |            0.495173    |                       1 |
| drop_reciprocity          | GraphStats_RF  |         0.94 |            0.714531    |                       1 |
| full_rebuilt              | GraphStats_EBM |         0.2  |            0.215168    |                       0 |
| full_rebuilt              | GraphStats_LR  |         0    |            0.0329942   |                       0 |
| full_rebuilt              | GraphStats_RF  |         0.92 |            0.606601    |                       0 |
| g4_no_replay              | GraphStats_EBM |         0    |            0.000319474 |                       0 |
| g4_no_replay              | GraphStats_LR  |         0    |            0.00480967  |                       0 |
| g4_no_replay              | GraphStats_RF  |         0    |            0.0304803   |                       0 |

## Ablation Test Metrics

| condition                 | model          |   bot_recall |   bot_mean_probability |   dropped_feature_count |   recall_drop_vs_full |
|:--------------------------|:---------------|-------------:|-----------------------:|------------------------:|----------------------:|
| drop_comment_owner_bundle | GraphStats_EBM |         0.16 |            0.188943    |                       7 |                  0.04 |
| drop_comment_owner_bundle | GraphStats_LR  |         0    |            0.00339679  |                       7 |                  0    |
| drop_comment_owner_bundle | GraphStats_RF  |         0.9  |            0.617358    |                       7 |                  0.02 |
| drop_context_bundle       | GraphStats_EBM |         0.26 |            0.312413    |                       7 |                 -0.06 |
| drop_context_bundle       | GraphStats_LR  |         0    |            0.029794    |                       7 |                  0    |
| drop_context_bundle       | GraphStats_RF  |         0.92 |            0.643905    |                       7 |                  0    |
| drop_direction_balance    | GraphStats_EBM |         0.24 |            0.264693    |                       1 |                 -0.04 |
| drop_direction_balance    | GraphStats_LR  |         0.04 |            0.152111    |                       1 |                 -0.04 |
| drop_direction_balance    | GraphStats_RF  |         0.88 |            0.597414    |                       1 |                  0.04 |
| drop_g4_signature_bundle  | GraphStats_EBM |         0.16 |            0.268495    |                      18 |                  0.04 |
| drop_g4_signature_bundle  | GraphStats_LR  |         0.36 |            0.418729    |                      18 |                 -0.36 |
| drop_g4_signature_bundle  | GraphStats_RF  |         0.18 |            0.292673    |                      18 |                  0.74 |
| drop_receive_bundle       | GraphStats_EBM |         0.02 |            0.0148354   |                       9 |                  0.18 |
| drop_receive_bundle       | GraphStats_LR  |         0    |            0.0290341   |                       9 |                  0    |
| drop_receive_bundle       | GraphStats_RF  |         0.02 |            0.259898    |                       9 |                  0.9  |
| drop_reciprocity          | GraphStats_EBM |         0.92 |            0.829316    |                       1 |                 -0.72 |
| drop_reciprocity          | GraphStats_LR  |         0.52 |            0.495173    |                       1 |                 -0.52 |
| drop_reciprocity          | GraphStats_RF  |         0.94 |            0.714531    |                       1 |                 -0.02 |
| full_rebuilt              | GraphStats_EBM |         0.2  |            0.215168    |                       0 |                  0    |
| full_rebuilt              | GraphStats_LR  |         0    |            0.0329942   |                       0 |                  0    |
| full_rebuilt              | GraphStats_RF  |         0.92 |            0.606601    |                       0 |                  0    |
| g4_no_replay              | GraphStats_EBM |         0    |            0.000319474 |                       0 |                  0.2  |
| g4_no_replay              | GraphStats_LR  |         0    |            0.00480967  |                       0 |                  0    |
| g4_no_replay              | GraphStats_RF  |         0    |            0.0304803   |                       0 |                  0.92 |
