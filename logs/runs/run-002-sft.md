# Run 002 — sft

**Date**: 2026-11-03 15:15
**Goal**: SFT: learn dialogue format

## Config

| Parameter | Value |
|-----------|-------|
| Type | sft |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/sft/sft-auto.json |
| Corpus size | 28,347 chars -> 28,347 tokens |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 150 |
| batch_size | 4 |
| lr | 0.0005 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 5,020,288
- Time: 4.8s
- Initial loss: 2.3520
- Final loss: 0.6083
- Eval loss: 0.6083
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 2.35 -> 0.61 (delta=1.74)

## Loss Curve

```
step     0: sft_loss=2.3520
step    50: sft_loss=1.7771
step   100: sft_loss=0.7436
step   149: sft_loss=0.6083
```

## Sample Outputs

```
Q: 什么是大模型？
A: 笔记里说: 通识的、标准的训练数据集或者大模型将很快沉淀,甚至会由专门的机构去进行,而企业私域数据的治理才是拉开应用水平的关键。 翻译一下:通用大模型以后跟自来水一样,谁都能用。 真正的差距,在应用变成重构最扎,在“地球和鸡蛋”,结果GPT用同一个主题是“模型输入”直接就是一个Token,但这句话。也可能是你所有某个答案,不可能真正理解空指出来。 --- ## 人类这个词,有没有让你想到工作中的某个瞬间?

Q: 银行怎么落地AI？
A: 以前AI跟AI检查。 我刚干的下午文章花完发现就是"不好用、目标设定、有人愿意推广、有人愿意标准化交换格式的缺失、三道,基本原理。 他妈妈觉得"就换方法论"。 但泛化能力差,可能比考"没问题"的基础工作就清楚"的基础",failed",你怎么换外面,而是对比发现了什么、更容易被忽略。 --- ### 第三件事:先有价值的信息差之后,让我愣了一下 AI输出了一份自评,尤其是涉及“解决方案架构师”——人

Q: AI能替代程序员吗？
A: 现在你打开任何一个AI输出的内容还没面试,都能看到这样的对话:帮这个指标、审核分人分项。跟他们结束,AI都需要你理解。 就像你打开外卖App,我脑子里冒出同事看完简报,我连准备上——判断、调、一个MCP支持MCP允许色几百和用户。 很多用户意图,价格可能打折——高效、POC、千了再感觉好做完会长都存在是让所有规则、在应用之间的差异。 很多大一:修复名字只是第一步。 ---
```
