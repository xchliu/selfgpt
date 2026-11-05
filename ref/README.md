# 参考资料

## 核心论文
| 论文 | 要点 | 我们用在哪里 |
|------|------|-------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, 2017) | Transformer 原始论文 | Attention/Block 实现 |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford, 2019) | decoder-only + 预训练 | 整体架构 |
| [GELU](https://arxiv.org/abs/1606.08415) (Hendrycks, 2016) | 激活函数 | FeedForward 的激活函数 |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba, 2016) | 层归一化 | 每层 Block + 最终 norm |
| [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (He, 2015) | 残差连接 | Block 中的 x + Sublayer(x) |
| [AdamW](https://arxiv.org/abs/1711.05101) (Loshchilov, 2017) | 优化器 | 训练用的 optimizer |
| [SentencePiece](https://arxiv.org/abs/1808.06226) (Kudo, 2018) | BPE 分词 | 中文分词器 |

## 待读论文（对应 TODO 第 10 项）
| 论文 | 要点 | 我们为什么要读 |
|------|------|---------------|
| [RoPE](https://arxiv.org/abs/2104.09864) | 旋转位置编码 | 支持更长上下文 |
| [Flash Attention](https://arxiv.org/abs/2205.14135) | 注意力加速 | 提速训练 |
| [SwiGLU](https://arxiv.org/abs/2002.05202) | 现代激活函数 | Llama 用的，对比 GELU |
| [GQA](https://arxiv.org/abs/2305.13245) | 分组查询注意力 | Llama 2 用的 |
| [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) | 正则化 | SFT 过拟合问题 |
| [InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155) | RLHF 原始论文 | TODO 第 5 项前置 |

## 代码参考
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的极简 GPT 实现
- [minGPT](https://github.com/karpathy/minGPT) — 更教学化的版本
- [llm.c](https://github.com/karpathy/llm.c) — 纯 C 实现，理解底层

## 视频教程
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy 2小时完整讲解
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — Karpathy 讲 BPE

## 辅助阅读
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar 图解版
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — 哈佛 NLP 逐行注释
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Lilian Weng 综述
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) — 超精简版
- [Position Embedding 综述](https://arxiv.org/abs/2210.07176) — 位置编码全面对比

## 分词器
- [minbpe](https://github.com/karpathy/minbpe) — Karpathy 的 BPE 实现