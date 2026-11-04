# Run 010 — sft

**Date**: 2026-11-04 16:11
**Goal**: SFT: learn dialogue format
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | sft |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/sft/sft-clean.json |
| Corpus size | 1,090,674 chars -> 1,090,674 tokens |
| embed_dim | 768 |
| num_layers | 16 |
| num_heads | 4 |
| head_size | 96 |
| block_size | 128 |
| epochs | 6000 |
| batch_size | 4 |
| lr | 0.0001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 105,018,496
- Time: 575.5s
- Initial loss: 10.3624
- Final loss: 0.0838
- Eval loss: 0.0838
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 10.36 -> 0.08 (delta=10.28)

## Loss Curve

```
step     0: sft_loss=10.3624
step   600: sft_loss=3.8659
step  1200: sft_loss=3.1207
step  1800: sft_loss=1.0567
step  2400: sft_loss=1.2234
step  3000: sft_loss=0.1075
step  3600: sft_loss=1.5316
step  4200: sft_loss=0.5642
step  4800: sft_loss=1.0512
step  5400: sft_loss=0.1710
step  5999: sft_loss=0.0838
```

## Sample Outputs

```
Q: 什么是大模型？
A: 大模型就是用海量文本训练一个语言模型。它不是一个能理解你的程序,而是一个概率机器--给定前文,预测下一个词。参数越大、数据越多,预测越准,看起来就像在思考。 就像在生成器上越相似的通信,越重要的越多。外层越多,越大越好,越强越好。 越强越好,越往越强越好,越很越好。 越

Q: 银行怎么落地AI？
A: 银行落地AI不要从大模型开始,从场景开始。先找一个痛点--信贷审批、客服、风控、运营,然后用AI解决它。不要追求技术先进,追求业务效果。最好的落地不是上了什么模型,而是省了多少人、提了多少效。无认,数据、调优、金融领域有权限管理平台、和伦理框架建设。 多模态是协作的框架,既保障了客户信任与信任

Q: AI能替代程序员吗？
A: 不能完全替代。AI能帮你写代码、补全函数、生成测试,但系统设计、架构决策、需求理解这些判断力层面的工作,AI做不了。AI是工具,不是替代品。真正被替代的,不是程序员,是不会用AI的程序员。 AI最可用的技术是代码,不是程序员,而是能自己动手干,就学AI在写代码、写代码,比人的更丰富的东西。 AI不懂,
```
