# SelfGPT Training Ledger

| Run | Date | Type | Params | Corpus | Epochs | Time | Init Loss | Final Loss | Val Loss | Notes |
|-----|------|------|--------|--------|--------|------|-----------|------------|----------|-------|
| 1 | 2026-11-03 | pretrain | 5M | 39万字 | 3K | 184s | 9.82 | 0.95 | - | Day 1 |
| 2 | 2026-11-03 | sft | 5M | 12条 | 150 | 5s | 2.35 | 0.61 | - | First SFT |
| 3 | 2026-11-03 | pretrain | 43M | 530万字 | 10K | 2668s | 9.86 | 1.30 | - | Scale up |
| 4 | 2026-11-04 | sft | 43M | 512条 | 1K | 48s | 11.90 | 1.01 | - | Underfit |
| 5 | 2026-11-04 | sft | 43M | 512条 | 1K | 47s | 11.65 | 4.64 | - | Bad data |
| 6 | 2026-11-04 | sft | 43M | 2012条 | 4K | 179s | 10.03 | 0.07 | - | Overfit |
| 7 | 2026-11-04 | sft | 43M | 5012条 | 8K | 359s | 11.17 | 0.46 | - | Best 43M |
| 8 | 2026-11-04 | pretrain | 105M | 530万字 | 10K | 5965s | 9.83 | 6.70 | - | ❌ lr too high |
| 9 | 2026-11-04 | pretrain | 105M | 530万字 | 10K | 6323s | 9.85 | 0.51 | - | lr 3e-4 fix |
| 10 | 2026-11-04 | sft | 105M | 4332条 | 6K | 575s | 10.36 | 0.08 | - | Overfit |
| 11 | 2026-11-05 | pretrain | 105M | 670万token | 10K | 6221s | 9.84 | 2.88 | - | Undertrained |
| 12 | 2026-11-09 | pretrain | 105M | 670万token | 30K | 19192s | 9.85 | 0.81 | - | Best pretrain |
| 13 | 2026-11-09 | sft | 105M | 4332条 | 3K | 290s | 7.32 | 0.19 | - | No val set |
| 14 | 2026-11-11 | sft | 105M | 72K token | 3K | 321s | 3.86 | 0.07 | 2.30 | Stop tag fix, overfit |
| 15 | 2026-11-12 | sft | 105M | 72K token | 1600 | 168s | 3.63 | 0.04 | 1.65 | ✅ Best SFT |