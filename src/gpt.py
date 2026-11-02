import torch
import torch.nn as nn
from block import TransformerBlock


class GPT(nn.Module):
    """
    完整 GPT 模型 — decoder-only 语言模型

    输入: token id 序列 (B, T)
    输出: logits (B, T, vocab_size) — 每个位置预测下一个 token
    """

    def __init__(self, vocab_size=100, embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size

        # 1. Token Embedding: 每个 token id → 一个 embed_dim 维向量
        #    本质是一个 (vocab_size × embed_dim) 的大查表
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. Position Embedding: 每个位置 → 一个 embed_dim 维向量
        #    Transformer 本身不知道顺序，必须显式告诉它"这是第几个词"
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        # 3. N 个 Transformer Block 堆叠
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, head_size)
            for _ in range(num_layers)
        ])

        # 4. 最后一次 LayerNorm
        self.ln_final = nn.LayerNorm(embed_dim)

        # 5. LM Head: 把 embed_dim 维向量映射回 vocab_size 维
        #    每一维代表"下一个词是这个 token 的得分"
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        """
        idx: (B, T) — token id 序列
        返回: (B, T, vocab_size) — logits
        """
        B, T = idx.shape

        # token 查表 → (B, T, embed_dim)
        tok_emb = self.token_embedding(idx)

        # 位置查表 → (T, embed_dim)，广播成 (B, T, embed_dim)
        pos = torch.arange(T, device=idx.device)           # [0, 1, 2, ..., T-1]
        pos_emb = self.position_embedding(pos)        # (T, embed_dim)

        # 关键：token 向量 + 位置向量，两者相加
        x = tok_emb + pos_emb

        # 过 N 层 Block
        for block in self.blocks:
            x = block(x)

        # 最后归一化 + 映射回 vocab
        x = self.ln_final(x)
        logits = self.lm_head(x)                      # (B, T, vocab_size)

        return logits