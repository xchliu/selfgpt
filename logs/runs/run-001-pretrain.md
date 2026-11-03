# Run 001 — pretrain

**Date**: 2026-11-03 15:14
**Goal**: N/A

## Config

| Parameter | Value |
|-----------|-------|
| Type | pretrain |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/clean-corpus.txt |
| Corpus size | 394,644 chars -> 186,819 tokens |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 3000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 5,020,288
- Time: 184.2s
- Initial loss: 9.8214
- Final loss: 0.9494
- Eval loss: 0.9137
- Train/eval gap: 0.0357 (no overfitting)
- Loss change: 9.82 -> 0.95 (delta=8.87)

## Loss Curve

```
step     0: train=9.8214  eval=9.6892
step   300: train=6.9003  eval=6.8639
step   600: train=5.4767  eval=5.5350
step   900: train=4.4289  eval=4.3290
step  1200: train=3.6921  eval=3.4652
step  1500: train=2.7783  eval=2.6957
step  1800: train=1.9972  eval=2.1428
step  2100: train=1.6262  eval=1.6909
step  2400: train=1.3486  eval=1.3346
step  2700: train=1.1064  eval=1.0715
step  2999: train=0.9494  eval=0.9137
```

## Sample Outputs

```
Prompt: AI时代
AI时代,而是当成一个**学会服务核心方法。AI,你教得很好,你带着你怎么解决问题,但至少他知道行里除了画画,却有错的地方。** 我原来以为做得好的AI,我把它叫做“边界”。人跟AI审”。 长期主义教会AI做,在你每一次AI的,而你只需要看一眼,而你只需要海量判断。 --- ## 这个认知陷阱,叫“递归信任”字,AI

Prompt: 我们
我们是在AI,而是**面壁的能力**AI的时候,而你只需要表述技术盲。 --- ## 二、AI时代变了 笔记里判断 > **AI工具,认知,而是给孩子在这个暑假。 AI可以24小时不休息,AI的。比如我想说的那个场景,路要不错,用来想出在哪儿?需要什么,那就认真看——是因为真的老同事。只是像流水线,因为AI产品经理。 **

Prompt: 银行
银行AI,至少比很多事情,找到它的分析,哪怕只是一个demo走向规模不小。 ## 三、旧场景,不行 数据容易被忽略。 AI的 说我观察到的矛盾点,管理者总是倾向于信任。 **第二个,是老师也没。** AI那么用它,技术用历史数据Claude能算,就像你不会问家里水龙头接真实数据。。** 花三天,本能就是平时能在用世界模型都是开源模型扎,
```
