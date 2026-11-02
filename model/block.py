import torch
import torch.nn as nn
from model.attention import SingleHeadAttention


class FeedForward(nn.Module):
    """
    Two linear layers with 4x expansion.
    Input(C) -> Linear(C, 4*C) -> GELU -> Linear(4*C, C) -> Output(C)
    
    Why expand then shrink? Gives room for nonlinear transform -
    expand each token vector, transform, compress back.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),   # expand 4x
            nn.GELU(),                              # GELU activation
            nn.Linear(4 * embed_dim, embed_dim),    # compress back
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A complete Transformer block:
    Pre-norm: LayerNorm before sublayer, then residual.
    
    Pipeline:
      x → LayerNorm → Attention → +x  → LayerNorm → FeedForward → +x
    """
    def __init__(self, embed_dim, num_heads, head_size):
        super().__init__()
        # attention (single-head, will upgrade to multi-head)
        self.attention = SingleHeadAttention(embed_dim, head_size)
        # feedforward sublayer
        self.feedforward = FeedForward(embed_dim)
        # two LayerNorms
        self.ln1 = nn.LayerNorm(embed_dim)   # norm before attention
        self.ln2 = nn.LayerNorm(embed_dim)   # norm before feedforward

    def forward(self, x):
        # pre-norm + attention + residual
        x = x + self.attention(self.ln1(x))
        # pre-norm + feedforward + residual
        x = x + self.feedforward(self.ln2(x))
        return x