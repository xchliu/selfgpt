# SelfGPT 模型理论笔记

本文档记录模型中每个组件的设计选择、原理、以及"为什么不选其他方案"。
每一条都关联训练实验中的实际表现。

---

## 1. 损失函数：交叉熵 (Cross-Entropy Loss)

### 代码位置
`train/train.py` — `nn.functional.cross_entropy(logits, targets)`

### 原理
语言模型的本质是：给定前文，预测下一个 token 的概率分布。
交叉熵衡量"模型的概率分布"和"真实分布"之间的差异。

```
Loss = -Σ y_true * log(y_pred)
```

对于 next-token prediction，真实分布是 one-hot（只有一个位置是1，其余是0）：
```
Loss = -log(y_pred[target_token])
```

**直觉理解**：模型给正确 token 分配的概率越高，loss 越低。
- 如果模型给正确 token 100% 概率 → loss = 0
- 如果模型给正确 token 1% 概率 → loss = 4.6
- 随机猜（16000词表）→ loss = ln(16000) ≈ 9.68

### 为什么不用 MSE（均方差）？
MSE 适合连续值回归，分类任务用交叉熵。交叉熵的梯度直接和"错得有多离谱"成正比——
猜得越离谱，梯度越大，学得越快。MSE 在分类任务上梯度太小，学不动。

### 实验关联
- 初始 loss ≈ 9.85 ≈ ln(16000)，说明模型从随机状态开始 ✓
- SFT loss 0.07 = 模型几乎 100% 猜对训练数据 → 过拟合

### 参考资料
- [Cross-Entropy 文章](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-b3b2cf198)

---

## 2. 激活函数：GELU

### 代码位置
`model/block.py` — `nn.GELU()`

### 原理
GELU = Gaussian Error Linear Unit，用高斯分布的累积分布函数做激活：
```
GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
```

**直觉理解**：x 越大，越接近直接通过（x*1）；x 越小，越接近乘0。
比 ReLU 更平滑——ReLU 在 0 处硬切，GELU 是柔和的。

### 为什么不用 ReLU？
| 激活函数 | 公式 | 特点 |
|---------|------|------|
| ReLU | max(0, x) | 简单快，但有"死神经元"问题（负值全部归零） |
| LeakyReLU | max(0.1x, x) | 修了死神经元，但负区斜率固定 |
| GELU | x * Φ(x) | 平滑过渡，负值不全部归零，现代 GPT 标配 |

GPT 系列从 GPT-1 开始就用 GELU。Google 的 BERT 也用 GELU。
现代大模型（GPT-4、Llama）基本都用 GELU 或 SwiGLU（GELU 变体）。

### 实验关联
目前没有对比实验（代码一开始就用了 GELU），是一个待研究点。

### 参考资料
- [GELU 论文](https://arxiv.org/abs/1606.08415)
- [Understanding GELU](https://wandb.ai/ayush-thakur/nerd-studio/reports/What-Is-GELU-Why-Is-It-Used-In-Transformers---Vmlldzo2MTM4NzY0)

---

## 3. 注意力机制：Scaled Dot-Product Attention

### 代码位置
`model/attention.py` — `SingleHeadAttention`

### 原理

核心公式：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**拆解**：
1. Q (Query)：当前 token 在"提问"——"我要找什么样的上下文？"
2. K (Key)：每个 token 在"应答"——"我的内容是什么类型？"
3. QK^T：问题 × 应答 = 关联度矩阵（T×T）
4. /√d_k：缩放，防止点积过大导致 softmax 饱和
5. softmax：关联度 → 概率分布（和为1）
6. ×V：用概率做加权求和，得到融合了上下文的新表示

### 为什么除以 √d_k？
QK^T 的方差随 d_k 增大而增大。如果 d_k=96，点积可能达到几百，
softmax 会变成接近 one-hot（一个1，其余全0），梯度消失。
除以 √96 ≈ 9.8，把数值拉回合理范围。

### Causal Mask 是什么？
```python
mask = torch.tril(torch.ones(T, T))  # 下三角矩阵
scores = scores.masked_fill(mask == 0, float('-inf'))
```
确保每个位置只能看到自己和之前的位置——不能偷看未来。
这是 decoder-only 的核心区别（encoder 没有 mask）。

### 当前限制：只有单头
现在 4 个 head_size=96 的头被当成 1 个头用。
真正的多头注意力应该是：4 个独立的头，每个学不同的注意力模式（语法、语义等）。

### 参考资料
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 4. LayerNorm：层归一化

### 代码位置
`model/block.py` — `nn.LayerNorm(embed_dim)`
`model/gpt.py` — `self.ln_final`

### 原理
对每个 token 的 embedding 维度做归一化：
```
y = (x - mean) / std * γ + β
```
γ 和 β 是可学习参数。

**直觉理解**：让每个 token 的向量"体量"一致，不要有的特别大有的特别小。
没有 LayerNorm，深层网络（16层）的数值会越来越大或越来越小，训练崩掉。

### Pre-norm vs Post-norm
我们用的是 pre-norm（先 norm 再进子层）：
```
x → LayerNorm → Attention → +x   (pre-norm)
```
原始 Transformer 论文用 post-norm：
```
x → Attention → +x → LayerNorm   (post-norm)
```

现代 GPT 都用 pre-norm，因为深层网络 post-norm 训练不稳定。

### 实验关联
Run 8（105M, lr=1e-3）梯度爆炸，可能跟没有 warmup 有关——
pre-norm 比较稳定，但大学习率下初始几步仍可能爆炸。

### 参考资料
- [Layer Normalization 论文](https://arxiv.org/abs/1607.06450)

---

## 5. 残差连接：Residual Connection

### 代码位置
`model/block.py` — `x = x + self.attention(self.ln1(x))`

### 原理
每个子层的输出都加上输入：
```
output = x + Sublayer(x)
```

**直觉理解**：信号可以"跳过"子层直接传递。16层网络，梯度要穿过16层，
残差连接提供了一条"高速公路"，梯度可以绕过子层直接到达前面。

### 为什么重要？
没有残差连接，超过 6-8 层的网络几乎无法训练（梯度消失）。
有了残差连接，可以堆到几十甚至上百层。

### 实验关联
16层网络能训练成功（loss 0.51），残差连接是基础保障。

### 参考资料
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385)

---

## 6. 优化器：AdamW

### 代码位置
`train/train.py` — `torch.optim.AdamW(model.parameters(), lr=lr)`

### 原理
AdamW = Adam + Weight Decay 分离

- **Adam**：自适应学习率，每个参数有自己的学习率（基于历史梯度的移动平均）
- **Weight Decay**：正则化，把参数往0拉，防止参数太大

Adam 把 weight decay 混在梯度里，AdamW 分开处理——效果更好。

### 学习率怎么选？
| 参数量 | 推荐 lr | 原因 |
|--------|---------|------|
| 5M | 1e-3 | 参数少，大步走 |
| 43M | 1e-3 | 还能扛 |
| 105M | 3e-4 | 大了必须降 |
| 200M+ | 1e-4~2e-4 | 更小 |

经验公式：参数翻倍，lr 砍三分之一。

### 为什么没有 warmup？
当前实现没有 warmup（前 N 步从 0 逐渐升到 lr）。
大模型 + 高 lr + 无 warmup = 初始几步梯度爆炸。
105M + lr=1e-3 就是这样炸的。这是个待改进点。

### 参考资料
- [AdamW 论文](https://arxiv.org/abs/1711.05101)
- [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)

---

## 7. Softmax：概率分布

### 代码位置
`model/attention.py` — `torch.softmax(scores, dim=-1)`

### 原理
把任意实数向量变成概率分布（所有元素 > 0，和 = 1）：
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**在注意力中**：关联度分数 → 概率，高关联的 token 权重大。
**在生成中**：logits → 概率分布，采样下一个 token。

### 为什么不用直取 argmax？
argmax 每次取最大值 → 模型每次输出一样 → 死板。
softmax + temperature 可以控制随机性：
- temperature 低：更确定
- temperature 高：更多样

### 实验关联
当 loss 0.07 时，softmax 分布极度集中（几乎 one-hot）→ 模型只会背不会泛化。

---

## 8. FeedForward：两层线性 + 激活

### 代码位置
`model/block.py` — `FeedForward`

### 原理
```
Linear(C, 4C) → GELU → Linear(4C, C)
```

**直觉理解**：注意力层负责"找上下文关联"，FeedForward 层负责"加工信息"。
先扩展到 4 倍维度 → 有更多空间做非线性变换 → 压缩回原维度。

### 为什么是 4 倍扩展？
经验值。原始 Transformer 论文用的 4x，后续基本都沿用。
太小 → 表达力不够；太大 → 参数浪费。

---

## 9. Token + Position Embedding

### 代码位置
`model/gpt.py` — `token_embedding` + `position_embedding`

### 原理
- **Token Embedding**：每个 token id → 一个 embed_dim 维向量。本质是查表。
- **Position Embedding**：每个位置 → 一个 embed_dim 维向量。告诉模型顺序。

两者相加（不是拼接），因为是同一维度的信息融合。

### 为什么 Transformer 需要位置编码？
Attention 机制本身没有顺序概念——打乱输入顺序，attention 输出只是顺序跟着变，内容不变。
位置编码告诉模型"谁在前面谁在后面"。

### 我们用的是可学习位置编码
```python
self.position_embedding = nn.Embedding(block_size, embed_dim)
```
GPT-2 用的就是这个。缺点：block_size 固定，不能超。
RoPE（旋转位置编码）可以支持更长上下文，是待研究点。

### 参考资料
- [Position Embedding 综述](https://arxiv.org/abs/2210.07176)

---

## 10. SFT Loss Mask：只对回答算 loss

### 代码位置
`train/sft.py`

### 原理
预训练：所有 token 都参与 loss 计算（学说话）
SFT：只有 assistant 的 token 参与 loss（学对话）

```
user: 什么是大模型？         ← loss = 0 (不学怎么问)
assistant: 大模型就是...      ← loss = 正常 (学怎么答)
```

### 为什么？
SFT 的目标是教模型"怎么回答"，不是"怎么提问"。
如果 user 的 token 也算 loss，模型会学会模仿用户提问，反而干扰回答能力。

### 实验关联
SFT loss 从 10+ 快速降到 0.08，说明模型在背 assistant 回答。
但这不意味着泛化好——loss 太低反而是过拟合信号。