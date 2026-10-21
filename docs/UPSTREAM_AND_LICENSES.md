# Attribution and license scope

## FamilyBotBench

Copyright (c) 2026 FamilyBotBench contributors. Original persona/family/drift modules, detection and evaluation programs, provider adapters, configuration helpers, tests and documentation use the MIT license in `LICENSE`.

Original modifications to upstream-derived files are covered only to the extent those modifications can be licensed independently. The MIT grant does not relicense upstream code, original or derived datasets, or copied figures/reports. `license_manifest.json` identifies the categories for individual files.

The accompanying `research report.pdf` is the research manuscript by muum. It is separate from the MIT software grant; no additional open-content license is asserted for the manuscript.

## BotSim and BotSim-24

FamilyBotBench builds on the work of Boyu Qiao, Kun Li, Wei Zhou, Shilong Li, Qianqian Lu and Songlin Hu:

- [BotSim repository](https://github.com/QQQQQQBY/BotSim)
- [BotSim-24 dataset](https://github.com/QQQQQQBY/BotSim/tree/main/BotSim-24-Dataset)
- [BotSim paper](https://arxiv.org/abs/2412.13420)

The inherited Reddit environment and utilities are under `RedditBotSim/Action/`, `RedditBotSim/AgentDesicionCenter/`, `RedditBotSim/Environment/`, and `RedditBotSim/Utils/`. Original Reddit-derived data, simulation seed data, graph tensors and derived research datasets retain their upstream provenance.

No explicit license file was present at upstream reference `5c3558f4c8d63605b0400d0beb1f71c6c497e73e`. Upstream code/data redistribution permission therefore remains unspecified. Confirm the relevant permissions before redistributing that material. The root MIT license applies to original FamilyBotBench contributions only.

## Dependencies and model assets

Python dependencies retain their own licenses. Pretrained transformer model weights are not included. The embedding model identifier is `BAAI/bge-small-en-v1.5`; existing derived embedding caches are part of the research data. Provider and source-data conditions also apply when generating or reusing new datasets.
