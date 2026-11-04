# Run 009 — pretrain

**Date**: 2026-11-04 14:25
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
| embed_dim | 768 |
| num_layers | 16 |
| num_heads | 4 |
| head_size | 96 |
| block_size | 128 |
| epochs | 10000 |
| batch_size | 32 |
| lr | 0.0003 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 105,018,496
- Time: 6323.3s
- Initial loss: 9.8545
- Final loss: 0.5132
- Eval loss: 0.4497
- Train/eval gap: 0.0635 (no overfitting)
- Loss change: 9.85 -> 0.51 (delta=9.34)

## Loss Curve

```
step     0: train=9.8545  eval=8.8886
step  1000: train=5.1067  eval=4.9754
step  2000: train=3.8645  eval=3.7512
step  3000: train=2.8826  eval=2.8896
step  4000: train=2.0178  eval=1.9538
step  5000: train=1.5564  eval=1.4476
step  6000: train=0.9748  eval=0.9014
step  7000: train=0.6950  eval=0.7309
step  8000: train=0.5599  eval=0.6046
step  9000: train=0.5395  eval=0.4979
step  9999: train=0.5132  eval=0.4497
```

## Sample Outputs

```
Prompt: AI时代
AI时代、机器学习、深度学习等技术的出现,为各个领域带来巨大的变革和进步。 ================================================== 要成为一个优秀的创业项目,需要以下创新: 1. 用户登录企业,设计模式,并收集并发布提供的产品; 2. 一款在线移动APP,允许用户实时监控企业文化、家庭健身等领域。 3. 利用API设计软件,允许用户实时监控软件的设置和

Prompt: 我们
我们开始探索新的行星。然而,太阳系并没有发现,所有的植被发现。但是,水星表面并不像一片长条纹,几乎不可能出现行星。某些行星小,但它们并不是所有行星都拥有“梦境”。这就导致木星的太阳系外行星在许多方面都没有死亡,他们利用了宇宙中生命。非常有趣,那么太阳系内最大的行星是水星?太阳系中最大的行星是木星,它的

Prompt: 银行
银行家算法的真假新闻,如生平查。根据最近的推荐算法,需要大量不同的算法来比较高的理论因素和进行评估。同时,也需要确保草案符合多个要求,以保障类似真实事件的心理素质。 ================================================== 写一篇关于人工智能在现代社会中的应用的评论文章。人工智能技术在现代社会中得到了广泛的应用,不仅改变了人们的生活方式,也为人们带来了很多便利和效率。
```
