# Run 007 — sft

**Date**: 2026-11-04 12:03
**Goal**: SFT: learn dialogue format
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | sft |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/sft/sft-final.json |
| Corpus size | 1,142,915 chars -> 1,142,915 tokens |
| embed_dim | 512 |
| num_layers | 12 |
| num_heads | 4 |
| head_size | 64 |
| block_size | 128 |
| epochs | 8000 |
| batch_size | 4 |
| lr | 0.0001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 43,260,544
- Time: 359.3s
- Initial loss: 11.1730
- Final loss: 0.4592
- Eval loss: 0.4592
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 11.17 -> 0.46 (delta=10.71)

## Loss Curve

```
step     0: sft_loss=11.1730
step   800: sft_loss=3.6302
step  1600: sft_loss=4.0873
step  2400: sft_loss=2.7887
step  3200: sft_loss=3.0074
step  4000: sft_loss=2.6193
step  4800: sft_loss=1.8743
step  5600: sft_loss=0.6067
step  6400: sft_loss=1.1786
step  7200: sft_loss=0.9173
step  7999: sft_loss=0.4592
```

## Sample Outputs

```
Q: 什么是大模型？
A: 大模型就是用海量文本训练出来的语言模型。它不是一个能理解你的程序,而是一个概率机器——给定前文,预测下一个词。参数越大、数据越多,预测越准,看起来就像在思考。我越需要的时候,它就能在短。它还有三个月,这次新模型是:它是设计用于进化的模型剪出来的。 但AI的产物太毒了,它变成同样的,跟这个才知道它怎么知道什么。

Q: 银行怎么落地AI？
A: 银行要是"你换的机会,准确理解"AI查资料*——刚的人跟我说。 说个智能体论、AI能干什么,但这不是别人问的。翻译官是,AI不是数据。 评论区说说你“你确定吗”,作为一个可以告诉你“肯定”或“回答”,而不是提供“肯定”或“回答”。它取决于你的故事和来源。如果你有特定的肯定或否定,我会帮助你识别相关信息。需要补充一些基本

Q: AI能替代程序员吗？
A: 你不需要知道噪声——你想知道"当人启动集",不需要额外安装"说明没有标准答案,不再是系统上搭定,才能在办公室。有一次我将动你的培训不能解决问题,没有人能告诉你"业务方闻。AI能怎样分析疲劳,但大部分人还是隐性的。 --- 下一篇清楚,哪些是你自己的产品信息、下一个阶段? 下一篇给你一篇给你一个口琴演奏家的讲了。结果你的短视频
```
