# Run 011 — pretrain

**Date**: 2026-11-05 15:11
**Goal**: N/A
**Status**: done

## Config

| Parameter | Value |
|-----------|-------|
| Type | pretrain |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/corpus/wiki-mixed.txt |
| Corpus size | 10,425,459 chars -> 6,703,956 tokens |
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
- Time: 6220.7s
- Initial loss: 9.8374
- Final loss: 2.8840
- Eval loss: 2.8284
- Train/eval gap: 0.0557 (no overfitting)
- Loss change: 9.84 -> 2.88 (delta=6.95)

## Loss Curve

```
step     0: train=9.8374  eval=9.5205
step  1000: train=6.2642  eval=6.2959
step  2000: train=5.5092  eval=5.3763
step  3000: train=4.8849  eval=4.9288
step  4000: train=4.6106  eval=4.5249
step  5000: train=3.8895  eval=4.1568
step  6000: train=3.7003  eval=3.8882
step  7000: train=3.4932  eval=3.5079
step  8000: train=3.3514  eval=3.3268
step  9000: train=3.1144  eval=3.0667
step  9999: train=2.8840  eval=2.8284
```

## Sample Outputs

```
Prompt: AI时代
AI时代。 **AI时代的"AI时代的新职位"**。 > 黑暗森林更深"的底线策略和底线是什么?不是答案,是解读。** --- ## 黑暗森林不是科幻,是现实 三体里的黑暗森林法则很简单: **技艺型记忆:半衰期不确定。用久不见,用到新AI。当你在用旧工具“把这三个E

Prompt: 我们
我们怎么做的:**看一眼这个岗位,就够了"这周里那个数据分析了吗?技术跑得比数据快了**。 ### 这是最稀缺的 怕AI替不了什么 怕一:"你怎么办" 回答:AI给的东西怎么做?"——你怎么做的:** 每次聊完,AI给的答案,你得把

Prompt: 银行
银行在"十五"AI真人"里"了,AI出了很多"AI漏斗"。银行AI错了是"智能体力"。 ## 剔除了AI,银行信贷审批的监管 我得说,AI给我一份最新AI项目,要是:把"从"人际交往"缩"需求变更"从银行导数据分析中挑出来:新
```
