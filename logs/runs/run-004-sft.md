# Run 004 — sft

**Date**: 2026-11-04 10:19
**Goal**: SFT: learn dialogue format
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | sft |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/sft/sft-auto.json |
| Corpus size | 74,830 chars -> 74,830 tokens |
| embed_dim | 512 |
| num_layers | 12 |
| num_heads | 4 |
| head_size | 64 |
| block_size | 128 |
| epochs | 1000 |
| batch_size | 4 |
| lr | 0.0001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 43,260,544
- Time: 47.6s
- Initial loss: 11.9011
- Final loss: 1.0050
- Eval loss: 1.0050
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 11.90 -> 1.01 (delta=10.90)

## Loss Curve

```
step     0: sft_loss=11.9011
step   100: sft_loss=6.1819
step   200: sft_loss=6.0676
step   300: sft_loss=4.5403
step   400: sft_loss=4.0540
step   500: sft_loss=2.7667
step   600: sft_loss=2.9552
step   700: sft_loss=1.1437
step   800: sft_loss=1.1820
step   900: sft_loss=1.5042
step   999: sft_loss=1.0050
```

## Sample Outputs

```
Q: 什么是大模型？
A: 大模型就是用海量文本训练出来的语言模型。它不是一个能理解你的程序,而是一个概率机器——给定前文,预测下一个词。参数越大、数据越多,预测越准,看起来就像在思考。 目标类型:算法物理学让那个算法越强。它是测量值越大、斯里得算法、GPT能才能看到波动,能互相更新量级、更好运行。 你能讲出一个更有创造力和自然语言理解师自由发挥

Q: 银行怎么落地AI？
A: 笔记里有一段话,AI再着对、内容的觉得很重要。把自己的判断力、不涉及AI原理、比他们强地往里。 比如你说“这件事是出客户留物品、帮我设计,代表我们公司的产品经理并在产品上给你。通过这一方法,我标注“这完全复制粘贴,我连微”这个版本控制时间。我们以该版本,我比客户更便宜了。 现在。有点像那,

Q: AI能替代程序员吗？
A: 你可能需要跟AI做得个做什么焦虑,不妨碍AI替代。 不是试试“chpided”是一半年AIgewary。他不知道怎么打开任务管理架。他听过几道。什么事写的不清效果。 还有哪了好?这听起来像带它? 比如,你还得枯燥,或者你偶尔跑过去。 这当然知道,AI应用不够。但现在我做过时事“AI”
```
