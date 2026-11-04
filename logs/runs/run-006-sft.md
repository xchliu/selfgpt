# Run 006 — sft

**Date**: 2026-11-04 11:52
**Goal**: SFT: learn dialogue format
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | sft |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/sft/sft-auto.json |
| Corpus size | 300,910 chars -> 300,910 tokens |
| embed_dim | 512 |
| num_layers | 12 |
| num_heads | 4 |
| head_size | 64 |
| block_size | 128 |
| epochs | 4000 |
| batch_size | 4 |
| lr | 0.0001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 43,260,544
- Time: 178.5s
- Initial loss: 10.0309
- Final loss: 0.0707
- Eval loss: 0.0707
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 10.03 -> 0.07 (delta=9.96)

## Loss Curve

```
step     0: sft_loss=10.0309
step   400: sft_loss=5.8481
step   800: sft_loss=4.7041
step  1200: sft_loss=4.8827
step  1600: sft_loss=3.3805
step  2000: sft_loss=2.9046
step  2400: sft_loss=1.8181
step  2800: sft_loss=1.4588
step  3200: sft_loss=0.6917
step  3600: sft_loss=0.1426
step  3999: sft_loss=0.0707
```

## Sample Outputs

```
Q: 什么是大模型？
A: 大模型就是用海量文本训练出来的语言模型。它不是一个能理解你的程序,而是一个概率机器——给定前文,预测下一个词。参数越大、数据越多,预测越准,看起来就像在思考。 有的产品经常从“噪声调时”变成“噪声”,而这三个干能叫"、结果那个分三件,可能就永远 ⁇ 编。 更可怕的是,这瞬间根本不会一夜暴,不需要你愿意

Q: 银行怎么落地AI？
A: 银行要是"终财报、查交易的OA,流程减毛鸿板,评论区说"到"QM 信前订阅房。翻译官是奢侈品, ⁇ 出品"。改了,监管就能宣布你脸儿已经没有因为AI生成了。 但效果就常见问题,它已经离不开基础。 我把决策句分得自己智,又让AIRAG防联控起来——它知道哪头。

Q: AI能替代程序员吗？
A: AI可以给你一个功能、一个聊天,让AI直接说清楚。 它能理解用户的问题,能模拟这些判断力、人人有关,每个人都有AI职责——你可以说是你的目标,才有判断。 它能给你一面含义,你就能能掌握出预算的习惯。 长期持有的AI培训师,不断给你一个比AI的工程师,生成全是你躲避。后来没人关心,开源模型都看得住淡。 23 -
```
