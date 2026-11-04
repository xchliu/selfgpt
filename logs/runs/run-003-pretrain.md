# Run 003 — pretrain

**Date**: 2026-11-03 16:51
**Goal**: N/A
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | pretrain |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/mixed-corpus.txt |
| Corpus size | 5,299,842 chars -> 2,451,326 tokens |
| embed_dim | 512 |
| num_layers | 12 |
| num_heads | 4 |
| head_size | 64 |
| block_size | 128 |
| epochs | 10000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 43,260,544
- Time: 2667.8s
- Initial loss: 9.8607
- Final loss: 1.3019
- Eval loss: 1.1877
- Train/eval gap: 0.1143 (possible overfitting)
- Loss change: 9.86 -> 1.30 (delta=8.56)

## Loss Curve

```
step     0: train=9.8607  eval=8.9195
step  1000: train=5.7602  eval=5.7404
step  2000: train=4.6742  eval=4.6290
step  3000: train=3.9696  eval=3.8607
step  4000: train=3.1296  eval=3.2834
step  5000: train=2.9655  eval=2.8185
step  6000: train=2.2932  eval=2.2139
step  7000: train=1.6771  eval=1.9676
step  8000: train=1.6470  eval=1.6166
step  9000: train=1.3711  eval=1.3966
step  9999: train=1.3019  eval=1.1877
```

## Sample Outputs

```
Prompt: AI时代
AI时代、机器人、医疗保健、智能制造、制造业。人工智能已经逐渐普及,包括智能家居、自动驾驶、联网、智能制造等领域。例如,在医疗领域,人工智能技术已经带来了很多改进。例如,在医疗领域,人工智能技术可以帮助医生更快速地诊断疾病,预测各种类型的治疗的效果。此外,人工智能还可以自动推荐不同的医疗记录方式,以及为患者提供更个性化的治疗效果。总的来说,人工智能未来的发展将会在医疗保健

Prompt: 我们
我们一起头脑风暴,希望我能成更加顺畅。 ================================================== 想一个新的营销口号,用于销售和推广的口号。对于新版商业口号,营销策略漂亮,喜欢美丽和独特。这个标语可以代表这个标语的语气是积极的,并且突出了产品的特点,符合商业目标。 ================================================== 据健康专家John Smith博士介绍:“维生素C是一种价格合理

Prompt: 银行
银行家电发明:120年 2. 印刷术 (Propen) ================================================== 在未来的世界里,人类和人工智能之间的关系变得越来越紧密。这种技术也具有更高的适应性和灵活性,目前它已经完全理解,并在智能、理解语言、命名实体识别等方面。 ================================================== 当然可以!以下是一些好听但最重要的是要:亲爱的、快乐! ==================================================
```
