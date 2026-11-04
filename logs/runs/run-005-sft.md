# Run 005 — sft

**Date**: 2026-11-04 11:51
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
| epochs | 1000 |
| batch_size | 4 |
| lr | 0.0001 |
| optimizer | AdamW |
| device | mps |

## Results

- Params: 43,260,544
- Time: 46.6s
- Initial loss: 11.6549
- Final loss: 4.6361
- Eval loss: 4.6361
- Train/eval gap: 0.0000 (no overfitting)
- Loss change: 11.65 -> 4.64 (delta=7.02)

## Loss Curve

```
step     0: sft_loss=11.6549
step   100: sft_loss=6.6648
step   200: sft_loss=6.1979
step   300: sft_loss=6.0876
step   400: sft_loss=5.2599
step   500: sft_loss=5.2928
step   600: sft_loss=5.2173
step   700: sft_loss=5.0511
step   800: sft_loss=5.5836
step   900: sft_loss=4.3060
step   999: sft_loss=4.6361
```

## Sample Outputs

```
Q: 什么是大模型？
A: 这事不多,不是技术积累的。 比如你不知道哪些数据、工具、模型工程师、模型、写代码。 比如它学不懂模型、写代码、写代码审查、加场景、写代码审查。 AI的AI系统、搭错,自动帮你帮你做一件事:跟你最容易被忽略。 AI没有AI,能给你一些AI工具。 比如你说“我为什么”AI的模式,其实是你愿意把

Q: 银行怎么落地AI？
A: 上周五下午,几点,AI能写几十几千余下的AI助手,但你每天干活。  ⁇   ⁇  说白了,AI  ⁇   ⁇   ⁇  说白了,AI的“AI的“AI的”是“AI”。AI能给你一个真实AI的“AI”的AI,你还在AI做AI时代的人, AI  ⁇  也没法 ⁇ 漏。 AI能干活,AI能答

Q: AI能替代程序员吗？
A: AI的AI项目的判断力。* 工程师的判断:AI的判断和决策行为是:AI的判断力、标准、逻辑都不错,是AI能干活的。AI能帮你帮你完成这个项目,但没人知道什么时候该做。 比如一个AI助手,AI能给你一个AI模型,能给你一个AI助手。你可能有没有“你叫什么?你干的事吗? 你说“你没想”。 你的大脑
```
