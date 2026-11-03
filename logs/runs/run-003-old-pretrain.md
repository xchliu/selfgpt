## Run 3 — 2026-11-02 15:29

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/zhuge-tan.txt (393,436 chars → 688,796 tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 1000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 13,824,337
- Time: 235.0s
- Initial loss: 11.0327
- Final loss: 2.2150
- Eval loss: 2.2047

### Loss Curve (sampled)
```
step     0: train=11.0327  eval=10.7938
step   100: train=3.6045  eval=3.6279
step   200: train=3.1992  eval=3.1277
step   300: train=2.8613  eval=2.8844
step   400: train=2.8258  eval=2.7420
step   500: train=2.6099  eval=2.5887
step   600: train=2.5230  eval=2.5212
step   700: train=2.3924  eval=2.3924
step   800: train=2.2829  eval=2.3718
step   900: train=2.2375  eval=2.3323
step   999: train=2.2150  eval=2.2047
```

### Sample Outputs
```
Prompt: AI时代
AI时代，你知道他说份一起值得位，不就就越越来的一批自己的，这个视视觉没问。

**�

Prompt: 我们
我们起来话是"他的东西"，但需要的触频"。

**第二个人，AI做了一个人：**你没有一个项目

Prompt: 银行
银行，是知道建议。

**第一次：**

**第一件事：给它让AI帮我只能帮你的那下，我们不是
```

### Analysis
- Train/eval gap: 0.0103 (no overfitting)
- Loss change: 11.03 -> 2.21 (delta=8.82)

---

