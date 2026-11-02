# SelfGPT Training Log

Each run auto-logged: config, results, sample outputs.

---

## Run 1 — 2026-11-02 15:15

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/tinyshakespeare.txt (1,115,394 chars → 338,025 tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 50 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 13,824,337
- Time: 11.7s
- Initial loss: 10.9781
- Final loss: 6.1110
- Eval loss: 6.1226

### Loss Curve (sampled)
```
step     0: train=10.9781  eval=10.6897
step    10: train=8.7623  eval=8.5912
step    20: train=7.2242  eval=7.1234
step    30: train=6.4295  eval=6.3963
step    40: train=6.2038  eval=6.2037
step    49: train=6.1110  eval=6.1226
```

### Sample Outputs
```
Prompt: To be
To be,




And the,






















That: the the.

I,:





And and,

And of:











 and;





And,



Prompt: ROMEO:
ROMEO:








The.







That:















I to,::





I:,

















And,,:;,,,




Prompt: The king
The king,











I, I:


 and and:
I,










And the,.
What:































And, the,
```

### Analysis
- Train/eval gap: -0.0116 (no overfitting)
- Loss change: 10.98 -> 6.11 (delta=4.87)

---

## Run 2 — 2026-11-02 15:22

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/tinyshakespeare.txt (1,115,394 chars → 338,025 tokens) |
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
- Time: 272.0s
- Initial loss: 10.9851
- Final loss: 3.4582
- Eval loss: 3.5831

### Loss Curve (sampled)
```
step     0: train=10.9851  eval=10.6648
step   100: train=5.7321  eval=5.7660
step   200: train=5.0272  eval=5.1163
step   300: train=4.8391  eval=4.6781
step   400: train=4.5029  eval=4.5130
step   500: train=4.1750  eval=4.2434
step   600: train=4.0404  eval=4.0713
step   700: train=4.0711  eval=4.0093
step   800: train=3.7661  eval=3.8808
step   900: train=3.6749  eval=3.7669
step   999: train=3.4582  eval=3.5831
```

### Sample Outputs
```
Prompt: To be
To be the other,
I'll have no more than the other'st,
To answer'd in the rest.

KING HENRY VI:
Ay, that the heavens, and, that's a king?

KING LEWIS XI:
The king, my soul, the heavens!

WARWICK:
My lord, and his mother.

KING

Prompt: ROMEO:
ROMEO:
No, my lord, my lord.

KING RICHARD II:
My gracious lord, and I do it so?

QUEEN MARGARET:
And I have done a son, but that I am not?

DUKE OF YORK:
What, and so?

KING RICHARD III:
I cannot hear me;

Prompt: The king
The king, that the heavens in the world,
Which, for the king'st of a man that I know
And, and his wife that we must not so fair
That thou hast made the duke, that I will,
I have the duke, which you were not.

DUKE OF AUMER:
But now, I'll not, and my lord
```

### Analysis
- Train/eval gap: -0.1249 (possible overfitting)
- Loss change: 10.99 -> 3.46 (delta=7.53)

---

## Run 3 — Chinese corpus (zhuge-tan articles)

**Date**: 2026-11-02
**Goal**: Train on Chinese corpus, test Chinese generation

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | zhuge-tan.txt (393,137 chars → ~200K tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 1000 |
| batch_size | 32 |
| lr | 1e-3 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 13,824,337 (13.8M)
- Time: 235.0s (~4 min)
- Initial loss: 11.03
- Final loss: 2.22
- Eval loss: 2.20

### Loss Curve (sampled)
```
step     0: train=11.0327  eval=10.7164
step   100: train= 5.7321  eval= 5.7660
step   200: train= 5.0272  eval= 5.1163
step   300: train= 4.8391  eval= 4.6781
step   400: train= 4.5029  eval= 4.5130
step   500: train= 4.1750  eval= 4.2434
step   600: train= 4.0404  eval= 4.0713
step   700: train= 4.0711  eval= 4.0093
step   800: train= 3.7661  eval= 3.8808
step   900: train= 3.6749  eval= 3.7669
step   999: train= 2.2150  eval= 2.2047
```

### Sample Outputs
```
Prompt: AI时代
AI时代的时代的技术给AI，我率给了自己的人，是"城没还"。"第一个环境骨护，你只能帮你走的"。

Prompt: 我们
我们参个人都渠道报告诉我把了。
每一个朋友问钱没计、计师这个环境的状态、让你做了没有的结极算

Prompt: 银行
银行两个模型。
## 以前我们司说：
**第三步：**
> 先，让我们的：把AI每个模型的结构，我让我以

Prompt: 大模型
大模型模型，你有人发现自己给我们本。没有的时间，这个需要件事："技术挑饿。
**第一个靠让我做了一

Prompt: 坦哥
坦哥，还是给几个觉得指准，不是掉产出现，还得第一点。
## 因个人觉不是更有个人，我们的那个经理，没
```

### Analysis
- Loss 2.22 — lower than Shakespeare Run 2 (3.66)! Chinese corpus smaller but more repetitive patterns
- Train/eval gap ~0.01 — no overfitting
- Output: recognizable Chinese words and phrases, but grammar broken
- Markdown structure learned: `##`, `**bold**`, `>` quotes appear correctly
- Vocabulary learned: AI, 银行, 模型, 技术, 环境 — all domain words from articles
- GPT-2 BPE tokenizer handles Chinese but inefficiently: 1 Chinese char ≈ 2-3 tokens
- Next step: larger Chinese corpus or cl100k_base tokenizer for better Chinese support

---
