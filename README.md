# SelfGPT 🧠

> 从零手搓一个 GPT，4 天，170K 参数，从乱码到能说话。

## 这是什么

SelfGPT 是一个学习型项目——不是做产品，是通过亲手实现每一个组件，真正理解 Transformer 的原理。

**核心理念**：手搓一遍，胜读十遍论文。

## 4 天从 0 到 1

| Day | 做了什么 | 产出 |
|-----|---------|------|
| 1 | 把文字变成数字 | `tokenizer.py` — 字符级分词器 |
| 2 | 让每个词看到其他词 | `attention.py` — 单头自注意力 |
| 3 | 把零件拼成一层 | `block.py` — Transformer Block |
| 4 | 拼成完整模型 + 训练 + 生成 | `gpt.py` + `train.py` + `generate.py` + `chat.py` |

## 训练结果

```
语料：莎士比亚片段（1074 字符）
模型：170K 参数
200 轮，3.3 秒（MPS 加速）
loss: 5.00 → 1.94
```

生成对比：
```
训练前: abcdF9b5j=5n]"oYc#LCte[              ← 纯乱码
训练后: To be, do ta halep mes;The ind       ← 不再乱码，单词边界基本对
```

## 快速开始

```bash
git clone https://github.com/xchliu/selfgpt.git
cd selfgpt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train/train.py
```

训练完会自动生成一段文本，并保存模型到 `data/model.pt`。

## 交互式对话

```bash
python serve/chat.py
```

## 项目结构

```
selfgpt/
├── MANUAL.md           # 完整操作手册
├── README.md           # 项目简介
├── requirements.txt     # 依赖：torch + numpy
├── model/              # 模型定义（训练推理共用）
│   ├── tokenizer.py    # 字符级 tokenizer
│   ├── attention.py    # SingleHeadAttention
│   ├── block.py        # TransformerBlock
│   └── gpt.py          # 完整 GPT
├── train/              # 训练（推理时不需要）
│   └── train.py        # 训练循环
├── serve/              # 推理（训练时不需要）
│   ├── generate.py     # 自回归生成
│   └── chat.py         # 多轮对话
├── tests/              # 测试
├── data/               # 词表 + 语料 + 模型
└── ref/                 # 参考资料
```

## 你能学到什么

- **Tokenizer**：token、vocab、encode/decode 的本质
- **Attention**：Q/K/V 为什么拆三套矩阵、causal mask 为什么必须加、softmax 为什么可导
- **Transformer Block**：残差连接为什么有效、LayerNorm 为什么稳定、FeedForward 为什么放大 4 倍
- **GPT 拼装**：embedding 查表、位置编码、LM Head 预测
- **训练**：cross-entropy loss、batch、梯度、AdamW
- **生成**：自回归、temperature、top-k sampling
- **理论**：从 RNN → Seq2Seq → Attention → Transformer 的完整推演

## 环境

- Python 3.11+
- PyTorch 2.x（支持 MPS 加速）
- macOS Apple Silicon / Linux / Windows

## 下一步

- [ ] BPE 分词器（替字符级）
- [ ] 多头注意力（替单头）
- [ ] 更大模型 + 更大语料
- [ ] RoPE 位置编码
- [ ] 微调个性化模型

## 参考资料

详见 `ref/README.md` — 论文、代码参考、视频教程、图解文章。

## License

MIT

---

> 没有一步是「理解」，每一步都是概率和线性代数。