# Graph inputs

`original_botsim_snapshot/` contains the original BotSim-24 account/content tables, Reddit seed tables, and graph edge tensors used by the graph pipeline. Its `BotSim-24-Dataset/` contains the full account evaluation source; `RedditData/` contains simulator tables; `mini_sample/` contains graph tensors and their account metadata.

The graph YAML defines each input location. Preserve the account-to-node ordering when replacing these inputs. New graph-family interactions are added by the generator and exported into the selected output directory.
