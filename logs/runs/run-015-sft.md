# Run 015 — sft

**Date**: 2026-11-12 10:57
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
| epochs | 1600 |
| batch_size | 4 |
| lr | 0.0001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 105,018,496
- Time: 168.1s
- Initial loss: 3.6271
- Final loss: 0.0381
- Eval loss: 0.0381
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 3.63 -> 0.04 (delta=3.59)

## Loss Curve

```
step     0: train=3.6271 val=2.9021
step   150: train=1.0864 val=1.4889
step   300: train=0.4188 val=1.4395
step   450: train=0.3581 val=1.4127
step   600: train=0.2173 val=1.4333
step   750: train=0.1206 val=1.4870
step   900: train=0.0806 val=1.5522
step  1050: train=0.0565 val=1.5524
step  1200: train=0.0376 val=1.5918
step  1350: train=0.0368 val=1.6087
step  1500: train=0.0352 val=1.6407
step  1599: train=0.0381 val=1.6472
```

## Sample Outputs

```
Q: 什么是大模型？
A: 大模型就是用海量文本训练出来的语言模型。它不是一个能理解你的程序,而是一个概率机器--给定前文,预测下一个词。参数越大、数据越多,预测越准,看起来就像在思考。

Q: 银行怎么落地AI？
A: 银行落地AI不要从大模型开始,从场景开始。先找一个痛点--信贷审批辅助,把沉默服务改成结论。

Q: AI能替代程序员吗？
A: 不能完全替代。AI能帮你写代码、补全函数、生成测试,但系统设计、架构决策、需求理解这些判断力层面的工作,AI做不了。AI是工具,不是替代品。真正被替代的,不是程序员,是不会用AI的程序员。
```
