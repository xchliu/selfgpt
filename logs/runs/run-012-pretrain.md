# Run 012 — pretrain

**Date**: 2026-11-09 12:23
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
| epochs | 30000 |
| batch_size | 32 |
| lr | 0.0003 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 105,018,496
- Time: 19192.0s
- Initial loss: 9.8503
- Final loss: 0.8084
- Eval loss: 0.8635
- Train/eval gap: -0.0551 (no overfitting)
- Loss change: 9.85 -> 0.81 (delta=9.04)

## Loss Curve

```
step     0: train=9.8503  eval=9.3128
step  3000: train=5.0261  eval=4.8576
step  6000: train=3.7479  eval=3.8726
step  9000: train=3.1179  eval=3.2051
step 12000: train=2.6636  eval=2.3825
step 15000: train=2.0926  eval=1.9052
step 18000: train=1.6166  eval=1.5486
step 21000: train=1.2488  eval=1.2342
step 24000: train=1.2497  eval=1.1139
step 27000: train=0.8506  eval=0.8968
step 29999: train=0.8084  eval=0.8635
```

## Sample Outputs

```
Prompt: AI时代
AI时代的新职位 国内团队需要哪些国家行为——Codex 的参与度慢一些。那问题一定程度——从日报渠道到应急预案,到同行选择存活。 AI的标注,比工具、文档、流程、陈述、周一页、陈述、陈述一个可扩展的自动。 **AI的记忆是检索式的,没有叙事,没有叙事

Prompt: 我们
我们没法把活儿放到楼层看,工读这工装也就了。 两户外,还有三住院里与外人浮于之处的直达远郊,每只可有多达7,500美元。那城市化社区是极少使用自己特有的的,而它被称之为「城门」,像乌鲁木齐盆地水坝了城市规划,经过了

Prompt: 银行
银行业不是机构组里一个机构联席会议,是管理的合规、资源节能判断AI和业务配合的人**。AI答得很现在的AI产业开发,模型是信贷审批方向。但知识面了,已有多少个部门排期。 **合规审查人误手,但AI不知道。它只知道这个人容易来截断是因为另一个技术本身并没有
```
