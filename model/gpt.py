import torch
import torch.nn as nn
from model.block import TransformerBlock


class GPT(nn.Module):
    """
    Full GPT model - decoder-only language model

    Input: token id sequence (B, T)
    Output: logits (B, T, vocab_size) - next token at each position
    """

    def __init__(self, vocab_size=100, embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size

        # 1. Token Embedding: token id -> embed_dim vector
        #    essentially a (vocab_size x embed_dim) lookup table
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. Position Embedding: position -> embed_dim vector
        #    Transformer has no sense of order, must be told explicitly
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        # 3. N stacked Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, head_size)
            for _ in range(num_layers)
        ])

        # 4. final LayerNorm
        self.ln_final = nn.LayerNorm(embed_dim)

        # 5. LM Head: embed_dim -> vocab_size
        #    each dim = score for that token being next
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        """
        idx: (B, T) - token id sequence
        Returns: (B, T, vocab_size) - logits
        """
        B, T = idx.shape

        # token lookup -> (B, T, embed_dim)
        tok_emb = self.token_embedding(idx)

        # position lookup -> (T, embed_dim), broadcast to (B, T, embed_dim)
        pos = torch.arange(T, device=idx.device)           # [0, 1, 2, ..., T-1]
        pos_emb = self.position_embedding(pos)        # (T, embed_dim)

        # token + position embeddings, element-wise add
        x = tok_emb + pos_emb

        # pass through N blocks
        for block in self.blocks:
            x = block(x)

        # final norm + project to vocab
        x = self.ln_final(x)
        logits = self.lm_head(x)                      # (B, T, vocab_size)

        return logits