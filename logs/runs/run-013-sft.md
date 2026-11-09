# Run 013 — sft

**Date**: 2026-11-09 17:47
**Goal**: SFT: learn dialogue format
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | sft |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/sft/sft-clean.json |
| Corpus size | 1,143,278 chars -> 1,143,278 tokens |
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
- Time: 289.8s
- Initial loss: 7.3193
- Final loss: 0.1944
- Eval loss: 0.1944
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 7.32 -> 0.19 (delta=7.12)

## Loss Curve

```
step     0: sft_loss=7.3193
step   300: sft_loss=2.0299
step   600: sft_loss=2.8121
step   900: sft_loss=0.6354
step  1200: sft_loss=1.4284
step  1500: sft_loss=1.7583
step  1800: sft_loss=0.6558
step  2100: sft_loss=1.2442
step  2400: sft_loss=1.7534
step  2700: sft_loss=0.1936
step  2999: sft_loss=0.1944
```

## Sample Outputs

```
Q: 什么是大模型？
A: 大模型就是用海量文本训练出来的语言模型。它不是一个能理解你的程序,而是一个概率机器--给定前文,预测下一个词。参数越大、数据越多,预测越准,看起来就像在思考。* 人看到的半衰期越来越短,但越容易掉进去。 你大概不会走去找,但大概几分钟里跑得更快。 那些无关紧要的信息,和那个重要方案挤满了钱。 比如

Q: 银行怎么落地AI？
A: 银行落地AI不要从大模型开始,从场景开始。先找一个痛点--信贷审批、客服、风控、运营,然后用AI解决它。不要追求技术先进,追求业务效果。最好的落地不是上了什么模型,而是省了多少人、提了多少效。 我遇到过类似的情况吗?有没有被AI编造的数据坑过?或者用RAG方法逆袭的案例? --它本能激发人均

Q: AI能替代程序员吗？
A: AI能写代码、能写代码、能画图设计,能给你能画图、能画图、知道你能画图、知道你周五或个规矩。然后每天挑一个,会发现--至少40%的工作可以交给AI,而你愿意也被用于客户推了。 结果呢?AI的准确率比你高出三倍,但可能永远准确率99.9%",我们可能连“损失”是多少、有没有潜在困境下的
```
