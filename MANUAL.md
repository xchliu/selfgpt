# SelfGPT — 从零手搓 GPT 操作手册

> 10 天，从零到一个能训练、能生成、能对话的 105M GPT。
> 每一步：**写什么 → 为什么这么写 → 怎么验证**。

---

## 项目背景

SelfGPT 是一个学习型项目——不是做产品，是通过亲手实现每一个组件，真正理解 Transformer 的原理。

**核心理念**：手搓一遍，胜读十遍论文。

**环境要求**：
- Python 3.11+
- PyTorch 2.x（支持 MPS 加速）
- macOS Apple Silicon / Linux / Windows 均可

**快速开始**：
```bash
git clone https://github.com/xchliu/selfgpt.git
cd selfgpt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train/train.py
```

---

## 项目结构

```
selfgpt/
├── MANUAL.md           # 本文件 — 操作手册
├── README.md           # 项目简介（GitHub 用）
├── TODO.md             # 路线图（10方向 50+项）
├── requirements.txt    # 依赖：torch + numpy + tiktoken + sentencepiece
├── .gitignore
│
├── model/              # 模型定义（train 和 serve 共用）
│   ├── tokenizer.py    # 统一分词器（char / tiktoken / sentencepiece 三后端）
│   ├── attention.py    # SingleHeadAttention（QKV + mask + proj）
│   ├── block.py        # TransformerBlock（FF + LayerNorm + 残差）
│   └── gpt.py          # 完整 GPT（Embedding + Position + N×Block + LM Head）
│
├── train/              # 训练
│   ├── train.py        # 预训练循环（105M, lr=3e-4）
│   ├── sft.py          # SFT 微调（loss mask，只对回答算 loss）
│   ├── train_tokenizer.py  # SentencePiece 分词器训练
│   └── log_utils.py    # 三阶段日志（start → progress → finish）
│
├── serve/              # 推理
│   ├── generate.py     # 自回归生成（temperature + top_k）
│   └── chat.py         # 多轮对话
│
├── data_pipeline/     # 数据处理
│   ├── process.py          # 清洗 markdown 文章 → 训练语料
│   ├── extract_wikipedia.py # 从维基百科 JSON 提取纯文本
│   ├── preprocess_pretrain.py # 清洗 JSONL 基础语料
│   ├── gen_sft.py          # 从文章自动生成问答对（13种策略）
│   ├── preprocess_sft.py   # 从外部语料提取对话
│   └── clean_sft.py        # SFT 数据质量清洗
│
├── data/               # 数据（不进 git）
│   ├── raw/               # 原始下载
│   ├── corpus/            # 预训练语料
│   ├── sft/                # SFT 数据
│   ├── tokenizer/         # SP 模型
│   └── archive/           # 历史归档
│
├── docs/              # 文档
│   ├── model-theory.md     # 10 个组件的原理、为什么选它、实验关联
│   ├── data-recipe.md      # 数据配方 + 历史对比
│   ├── data-pipeline.md    # 数据处理流程
│   ├── data-structure.md   # 目录说明
│   └── training-log.md     # 日志机制
│
├── checkpoints/       # 模型权重（不进 git）
│   ├── model.pt            # 预训练模型
│   └── model-sft.pt       # SFT 模型
│
├── logs/              # 训练日志
│   ├── training-log.md     # 总台账（13轮）
│   └── runs/               # 每轮详细
│
├── tests/             # 测试
└── ref/               # 参考资料（论文、教程链接）
```

---

## 阶段总览

| 阶段 | 名称 | 你要写的 | 输出文件 | 核心概念 |
|------|------|----------|----------|----------|
| 1 | Tokenizer | 字符级分词器 | `src/tokenizer.py` | token, vocab, encode/decode |
| 2 | Self-Attention | 单头自注意力 | `src/attention.py` | Q/K/V, causal mask, softmax |
| 3 | Transformer Block | 完整一层 | `src/block.py` | 残差, LayerNorm, FeedForward |
| 4 | GPT 拼装 | 完整模型 | `src/gpt.py` | embedding, position, LM head |
| 5 | 训练 | 训练循环 | `src/train.py` | cross-entropy, batch, gradient |
| 6 | 生成 | 自回归生成 | `src/generate.py` | temperature, top-k, 自回归 |
| + | 对话 | 多轮封装 | `src/chat.py` | 对话格式, 上下文管理 |

---

## 阶段 1：Tokenizer —— 把文字变成数字

### 目标
写一个字符级 tokenizer，让 `hello` 变成 `[3, 2, 4, 4, 5]`，还能变回来。

### 要写的代码
1. `__init__(self, vocab_path)`：从文件加载词表，建 `char_to_id` / `id_to_char` 映射
2. `encode(text) → list[int]`：字符串 → token id 列表
3. `decode(tokens) → str`：token id 列表 → 字符串
4. `save_vocab(path)`：把词表存成文件（持久化）

### 为什么词表要存文件
Tokenizer 在训练和推理时必须用同一份词表。token id 3 在训练时代表 `e`，推理时也得是 `e`。如果每次重建词表，顺序可能变，decode 直接乱码。

### 验证
```python
from tokenizer import Tokenizer
t = Tokenizer(vocab_path='../data/vocab.txt')
text = "hello"
tokens = t.encode(text)
decoded = t.decode(tokens)
assert decoded == text
print(f"encode('{text}') = {tokens}")
print(f"decode({tokens}) = '{decoded}'")
print(f"词表大小: {t.vocab_size}")
```

### 核心概念
- **token**：模型能认识的最小单位。字符级 = 一个字母/标点就是一个 token
- **vocab**：词表，所有可能的 token 的集合
- **为什么从字符级开始**：BPE 分词器原理复杂，先用字符级理解 tokenization 的本质，后面再升级
- **词表一致性**：tokenizer 和模型参数必须配对保存

### 踩坑提醒
- ❌ `__init__` 里用了未传参的变量，碰巧模块级有同名变量才没炸
- ✅ 显式传参，不依赖闭包

---

## 阶段 2：Self-Attention —— 模型怎么「读懂」上下文

### 目标
实现单头 self-attention：让每个 token 能看到前面所有 token，算出它「应该关注谁」。

### 直觉
读一句话「我把苹果放进冰箱然后吃了」——读到「吃了」时，你脑子里自动回看「苹果」。这个「回头看、挑重点」就是 attention。

### 要写的代码
1. `__init__`：三个线性层 `W_q`, `W_k`, `W_v`（embed_dim → head_size）+ 输出投影 `proj`（head_size → embed_dim）
2. `forward(x)`：
   - 算 Q, K, V：`q = self.W_q(x)`, `k = self.W_k(x)`, `v = self.W_v(x)`
   - 算关联度：`scores = q @ k.transpose(-2, -1) / sqrt(head_size)`
   - Causal mask：`torch.tril` 生成下三角，右上角填 `-inf`
   - Softmax：`weights = torch.softmax(scores, dim=-1)`
   - 加权求和：`out = weights @ v`，再过 `proj` 投影回 embed_dim

### 形状追踪
```
输入 x         (B, T, C)       例: (2, 8, 32)
    ↓
Q, K, V       (B, T, H)       例: (2, 8, 16)
    ↓
scores        (B, T, T)       例: (2, 8, 8)  ← T×T 方阵，每格是关联度
    ↓ mask + softmax
weights       (B, T, T)       ← 每行加起来 = 1
    ↓ @ V
out           (B, T, H)       例: (2, 8, 16)
    ↓ proj
输出          (B, T, C)       例: (2, 8, 32)  ← 投影回 embed_dim
```

### 验证
```python
import torch
from attention import SingleHeadAttention

B, T, C = 2, 8, 32
head_size = 16
x = torch.randn(B, T, C)
attn = SingleHeadAttention(C, head_size)
out = attn(x)
assert out.shape == (B, T, C), f"期望 {(B,T,C)}, 实际 {out.shape}"
print(f"✓ 单头 attention 跑通 — {x.shape} → {out.shape}")
```

### 核心概念
- **Q/K/V**：Query 是「我在找什么」，Key 是「我有什么」，Value 是「我实际提供什么信息」
- **为什么拆三套矩阵**：同一个 token 可以从不同角度被查询和表示，表达能力翻倍
- **scaled dot-product**：除以 sqrt(d) 防止点积过大导致 softmax 梯度消失
- **causal mask**：GPT 是自回归模型，生成第 N 个 token 时只能看前 N-1 个
- **softmax 三重作用**：归一化 + 放大差异（赢家通吃）+ 可导（让 attention 能被训练）
- **proj 层**：把 head_size 维映射回 embed_dim，让残差连接 `x + attn(x)` 形状匹配

### 踩坑提醒
- ❌ 类没继承 `nn.Module`，`self.attention(x)` 报 TypeError
- ✅ `class SingleHeadAttention(nn.Module)` + `super().__init__()`
- ❌ 没加 proj 层，残差连接形状不匹配
- ✅ 加 `self.proj = nn.Linear(head_size, embed_dim, bias=False)`

---

## 阶段 3：Transformer Block —— 把零件拼成一个「层」

### 目标
拼一个完整的 Transformer Block：Attention + FeedForward + LayerNorm + Residual。

### 结构
```
输入 x ──→ LayerNorm ──→ Attention ──→ + ──→ LayerNorm ──→ FeedForward ──→ + ──→ 输出
            ↑                              ↑                        ↑              ↑
            └──────────────────────────────┘                        └──────────────┘
                 残差连接                                         残差连接
```

### 要写的代码
1. `FeedForward`：
   - `nn.Linear(embed_dim, 4 * embed_dim)` → `GELU()` → `nn.Linear(4 * embed_dim, embed_dim)`
2. `TransformerBlock`：
   - `x = x + self.attention(self.ln1(x))` — Pre-norm + Attention + 残差
   - `x = x + self.feedforward(self.ln2(x))` — Pre-norm + FeedForward + 残差

### 验证
```python
import torch
from block import TransformerBlock

B, T, C = 2, 8, 32
x = torch.randn(B, T, C)
block = TransformerBlock(embed_dim=32, num_heads=4, head_size=8)
out = block(x)
assert out.shape == x.shape, f"期望 {x.shape}, 实际 {out.shape}"
params = sum(p.numel() for p in block.parameters())
print(f"✓ TransformerBlock 跑通 — 参数量 {params:,}")
```

### 核心概念
- **Residual 连接**：`x = x + f(x)` 让梯度走「高速公路」，解决深层网络训练不动。本质是告诉模型「学不会就别学，把输入原样传过去也行」
- **LayerNorm**：对每个 token 的特征维度做归一化（减均值/除标准差），消除数值漂移
- **FeedForward**：token 的「独处思考时间」——把向量展开 4 倍，做非线性变换再压回来。注意力负责交流，FF 负责消化
- **Pre-norm**：先归一化再过模块（现代 GPT 做法），比 Post-norm 训练更稳定

### 为什么这样设计
整个 Block 就是：**讨论(attention) → 消化(FF) → 讨论 → 消化 → 重复 N 次**
- 每次讨论前先归一化（LayerNorm）——站好起跑线
- 每次讨论/消化后跟原始信息合并（残差）——防止信息丢失

---

## 阶段 4：GPT 拼装 —— 把所有层串起来

### 目标
把 Embedding + Position + N×Block + LM Head 拼成一个完整模型，能前向传播。

### 结构
```
输入: token id 序列 (B, T)
         │
   Token Embedding        ← 每个 id 查表变成向量
         +
   Position Embedding      ← 告诉模型每个词在第几个位置
         │
   TransformerBlock × N   ← 阶段 3 写的
         │
   LayerNorm               ← 最后归一化一次
         │
   LM Head (Linear)        ← 映射回 vocab 大小
         │
   输出: logits (B, T, vocab_size)
```

### 要写的代码
1. `__init__`：
   - `self.token_embedding = nn.Embedding(vocab_size, embed_dim)`
   - `self.position_embedding = nn.Embedding(block_size, embed_dim)`
   - `self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])`
   - `self.ln_final = nn.LayerNorm(embed_dim)`
   - `self.lm_head = nn.Linear(embed_dim, vocab_size)`
2. `forward(idx)`：
   - `tok_emb = self.token_embedding(idx)`
   - `pos_emb = self.position_embedding(torch.arange(T, device=idx.device))`
   - `x = tok_emb + pos_emb`
   - `for block in self.blocks: x = block(x)`
   - `x = self.ln_final(x)`
   - `return self.lm_head(x)`

### 验证
```python
import torch
from gpt import GPT

model = GPT(vocab_size=127, embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=64)
x = torch.randint(0, 127, (2, 32))
logits = model(x)
assert logits.shape == (2, 32, 127), f"期望 (2,32,127), 实际 {logits.shape}"
params = sum(p.numel() for p in model.parameters())
print(f"✓ GPT 前向传播跑通 — 参数量 {params:,}")
```

### 核心概念
- **Token Embedding**：本质是一个大查表，token_id → 向量。每个维度没有名字，是模型自己训出来的
- **Position Embedding**：Transformer 本身不知道顺序，需要显式告诉它「这是第几个位置」
- **为什么两者相加**：模型同时知道「是什么词」+「在哪个位置」
- **LM Head**：最后一层线性变换，把 hidden state 映射回 vocab 大小，每个值代表「下一个 token 是这个的概率有多大」
- **block_size**：上下文窗口大小，决定模型一次最多看多少 token。超过窗口就只看最后 block_size 个

### 踩坑提醒
- ❌ `torch.arange(T)` 默认在 CPU 上，模型在 MPS 上 → 设备不一致报错
- ✅ `torch.arange(T, device=idx.device)`

---

## 阶段 5：训练 —— 让模型学会「接话」

### 目标
写训练循环，用小语料把模型训出能接话的能力。

### 训练循环的本质
```
1. 从语料切一个 batch 的 (输入, 目标) 对
   输入:  [t0, t1, t2, t3]   ← 给模型看的前文
   目标:  [t1, t2, t3, t4]   ← 整体偏移一位，模型应该预测的下一个 token

2. 前向传播 → logits (B, T, vocab_size)
3. 算 loss（交叉熵）：预测的 vs 真实的下一个 token
4. 反向传播 → 梯度
5. optimizer.step() 更新参数
6. 重复
```

### 要写的代码
1. `load_data(data_path, vocab_path)`：读语料 + 词表，返回 token tensor 和 tokenizer
2. `get_batch(data, batch_size, block_size)`：随机切 batch，输入和目标错开一位
3. `train(model, data, epochs, batch_size, block_size, lr)`：
   - AdamW 优化器
   - 每个 epoch：get_batch → forward → cross_entropy → zero_grad → backward → step
   - 定期 eval（`model.eval()` + `torch.no_grad()`）
4. 保存 checkpoint：`torch.save({'model_state': ..., 'vocab': ..., 'config': ...}, path)`

### 验证
```python
from train import load_data, train
from gpt import GPT

data, tokenizer = load_data('../data/train.txt', '../data/vocab.txt')
model = GPT(vocab_size=tokenizer.vocab_size, embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=64)
losses = train(model, data, tokenizer, epochs=200, batch_size=32, block_size=64, lr=1e-3)
print(f"loss: {losses[0][1]:.4f} → {losses[-1][1]:.4f}")
```

预期输出：
```
轮次    0/200 | 训练 loss: 5.0050 | 评估 loss: 4.8388
轮次   10/200 | 训练 loss: 3.7488 | 评估 loss: 3.6825
...
轮次  199/200 | 训练 loss: 1.9410 | 评估 loss: 1.9329
```

### 核心概念
- **Cross-entropy loss**：衡量预测分布和真实分布的距离。初始值 ≈ ln(vocab_size)（瞎猜水平）
- **batch**：一次前向传播处理多条数据，加速训练
- **gradient**：loss 对每个参数的偏导数，告诉你「往哪调能让 loss 变小」
- **AdamW**：优化器，自动调整每个参数的学习率
- **train/eval 模式**：eval 关闭 dropout，no_grad 不算梯度

### 踩坑提醒
- ❌ MPS 设备不一致：`torch.ones(T,T)` 默认 CPU，模型在 MPS
- ✅ `torch.ones(T, T, device=x.device)` — 所有手动创建的 tensor 都要指定 device

---

## 阶段 6：生成 —— 让模型「说话」

### 目标
给模型一个开头，让它自回归地生成后续内容。

### 要写的代码
1. `generate(model, prompt_ids, max_new_tokens, temperature, top_k)`：
   - 把 prompt 转成 tensor，搬到模型所在设备
   - 循环 `max_new_tokens` 次：
     - 如果序列超过 block_size，只取最后 block_size 个
     - 前向传播 → 取最后一个位置的 logits
     - 除以 temperature
     - top_k 过滤（只保留概率最高的 K 个）
     - softmax → 按概率采样（`torch.multinomial`）
     - 拼到序列末尾
   - 返回完整 token id 列表

### 验证
```python
from generate import generate
from gpt import GPT
from tokenizer import Tokenizer

tokenizer = Tokenizer(vocab_path='../data/vocab.txt')
model = GPT(vocab_size=tokenizer.vocab_size, embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=64)
# 加载训练好的模型
import torch
ckpt = torch.load('../data/model.pt', weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

prompt = tokenizer.encode("To be")
result = generate(model, prompt, max_new_tokens=50, temperature=0.7, top_k=10)
print(tokenizer.decode(result))
```

### 核心概念
- **自回归**：生成第 N 个 token 时，把前 N-1 个 token 作为输入。循环生成
- **temperature**：控制随机性。低 = 保守（选高分 token），高 = 放飞（更有创意但可能胡说）
  - `logits / 0.1` → 差异放大 → 更确定
  - `logits / 2.0` → 差异缩小 → 更随机
- **top-k sampling**：只从概率最高的 K 个 token 里采样，避免选中极低概率 token
- **为什么用 `torch.multinomial` 而不是 `argmax`**：采样有随机性，每次生成不同，更有创意

### 生成对比
```
训练前: abcdF9b5j=5n]"oYc#LCte[           ← 纯乱码（均匀分布）
训练后: To be, do ta halep mes;The ind    ← 不再乱码，单词边界基本对
```

---

## 附加：对话模块

### 目标
在 generate 之上封装多轮对话。

### 要写的代码
`ChatSession` 类：
- `chat(user_input)`：存用户消息 → 拼接历史 → tokenize → generate → 截断 stop token → 存助手回复
- `_build_prompt()`：把历史拼成 `<user>msg</user><assistant>msg</assistant><user>...` 格式
- `reset()`：清空历史
- `show_history()`：打印完整对话

### 验证
```bash
cd src
python chat.py
# 输入消息，输入 quit 退出，输入 reset 清空历史
```

### 核心概念
- Chat 不是新东西，是 generate 的多轮上下文封装
- 每轮把历史拼成一段文本，模型看到的不是「对话」，是连续文本
- 超过 block_size 时只保留最近的对话

---

## 完成之后

六个阶段走完，你手搓了一个完整的 GPT。接下来 1→10 的方向：

| 方向 | 做什么 | 预期效果 | 状态 |
|------|--------|---------|------|
| ✅ 维基语料 | 中文维基百科 1000万字符 | 语法通顺有知识感 | 已完成 |
| ✅ SentencePiece | BPE 分词器 16K 词表 | 中文 token 效率提升 | 已完成 |
| ✅ SFT | 4332条问答对微调 | 能对话回答问题 | 已完成 |
| ✅ 105M参数 | embed_dim=768, 16层 | 更强的表达能力 | 已完成 |
| 多头注意力 | 替单头，多角度关注 | 表达能力提升 | TODO |
| RoPE位置编码 | 替学习式位置编码 | 更好的长文本能力 | TODO |
| Warmup+Cosine | 学习率调度 | 训练更稳定 | TODO |
| KV Cache | 推理优化 | 生成加速 | TODO |
| 更大SFT数据 | 用大模型生成高质量问答 | 理解能力提升 | TODO |
| RLHF/DPO | 人类偏好对齐 | 说好话 | TODO |

---

## 参考资料

详见 `ref/README.md`，包括：
- 核心论文（Attention Is All You Need, GPT-2, GELU, LayerNorm, ResNet, AdamW, SentencePiece）
- 待读论文（RoPE, Flash Attention, SwiGLU, GQA, Dropout, InstructGPT）
- 代码参考（nanoGPT, minGPT, llm.c）
- 视频教程（Karpathy 系列）
- 图解文章（Illustrated Transformer 等）
- 理论文档：`docs/model-theory.md`（10个组件的原理、为什么选它、实验关联）

---

## 理论补充：从 RNN 到 Transformer

理解 attention 为什么被发明，有助于理解它为什么这样设计：

```
2014  Seq2Seq（RNN）：能翻译了，但固定向量瓶颈（长句装不下）
        ↓
2015  Seq2Seq + Attention：回头看所有隐藏状态，瓶颈缓解了，但还是串行
        ↓
2017  Transformer：既然回头看这么好用，RNN 可以扔了，全并行
```

| 架构 | 信息流动 | 致命问题 |
|------|---------|---------|
| Seq2Seq (RNN) | 压缩成一个固定向量 | 长句瓶颈 |
| Seq2Seq + Attention | 固定向量 + 回头看所有状态 | 还是串行 |
| Transformer (Self-Attention) | 每个 token 直接看所有 token | 问题解决了 |

**关键转变**：从一维串行计算（RNN）变成二维并行矩阵计算（Transformer）。`q @ k.T` 这一行代码，就是让每个 token 同时跟所有 token 算关联度。

---

## 遇到问题怎么办

本项目在开发过程中记录了每一步的踩坑和修复。常见问题：

| 问题 | 原因 | 解决 |
|------|------|------|
| `NameError: head_size` | 没存到 `self` | `self.head_size = head_size` |
| `TypeError: not callable` | 没继承 `nn.Module` | `class X(nn.Module)` + `super().__init__()` |
| 残差连接形状不匹配 | head_size ≠ embed_dim | 加 proj 投影层 |
| `Passed CPU tensor to MPS op` | tensor 设备不一致 | 所有 `torch.ones/arange` 加 `device=` |
| KeyError 编码失败 | 字符不在词表里 | 补充 vocab.txt 或代码自动补全 |

---

开始吧，从 `model/tokenizer.py` 下手。