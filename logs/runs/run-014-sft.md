# Run 014 — sft

**Date**: 2026-11-11 11:53
**Goal**: SFT: learn dialogue format
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | sft |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/sft/sft-clean.json |
| Corpus size | 72,218 chars -> 72,218 tokens |
| embed_dim | 768 |
| num_layers | 16 |
| num_heads | 4 |
| head_size | 96 |
| block_size | 128 |
| epochs | 3000 |
| batch_size | 4 |
| lr | 0.0001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 105,018,496
- Time: 321.0s
- Initial loss: 3.8573
- Final loss: 0.0661
- Eval loss: 0.0661
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 3.86 -> 0.07 (delta=3.79)

## Loss Curve

```
step     0: train=3.8573 val=2.9152
step   300: train=0.6761 val=1.4522
step   600: train=0.1435 val=1.4801
step   900: train=0.0789 val=1.5435
step  1200: train=0.0576 val=1.5773
step  1500: train=0.0173 val=1.6391
step  1800: train=0.0129 val=1.6286
step  2100: train=0.0165 val=1.7079
step  2400: train=0.0086 val=1.7975
step  2700: train=0.0412 val=2.0195
step  2999: train=0.0661 val=2.2951
```

## Sample Outputs

```
Q: 什么是大模型？
A: 大模型就是用海量文本训练出来的语言模型。它不是一个能理解你的程序,而是一个概率机器--给定前文,预测下一个词。参数越大、数据越多,预测越准,看起来就像在思考。

Q: 银行怎么落地AI？
A: 银行落地AI不要从大模型开始,从场景开始。先找一个痛点--信贷审批、客服、风控、运营,然后用AI解决它。不要追求技术先进,追求业务效果。最好的落地不是上了什么模型,而是省了多少人、提了多少效。

Q: AI能替代程序员吗？
A: 不能完全替代。AI能省时间,但有时候能省时间。AI能省时间,结果自己花的时间更多了。* 不是AI写得不快,是我对AI输出的容忍度太低。
```
