# Pretty Generalization Heatmaps

- Transfer figures use a two-level x-axis: top = `train source`, bottom = `target case`.
- Example: top label `C2+E2` above bottom label `A2` means `train on C2 and E2 -> test on A2`.
- Example: top label `all g1-2` above bottom label `A3` means `train on all earlier generations 1-2 from A/C/E -> test on A3`.
- `two_layer_axis_demo.png` is a schematic that shows how one column represents one explicit `source -> target` transfer task.
- Rows: detector algorithms
- Left panel: `incremental_generation`
- Right panel: `heldout_family_same_generation`
- Graph transfer is also exported as two separate figures: `graph_detector_incremental_generation.png` and `graph_detector_heldout_family_same_generation.png`.
- Cell value: target-bot recall
- Gap figure uses `incremental recall - held-out-family recall`; positive values mean the detector benefits more from earlier-generation exposure than from same-generation cross-family transfer.