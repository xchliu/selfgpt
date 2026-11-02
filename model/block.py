import torch
import torch.nn as nn
from model.attention import SingleHeadAttention


class FeedForward(nn.Module):
    """
    两层全连接，中间放大4倍，最后缩回来。
    输入(C) → Linear(C, 4*C) → GELU → Linear(4*C, C) → 输出(C)
    
    为什么放大再缩小？给模型一个"思考空间"——把每个token的向量展开，
    在里面做非线性变换，再压回原来的维度。
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),   # 放大到4倍
            nn.GELU(),                              # GELU激活，比ReLU平滑
            nn.Linear(4 * embed_dim, embed_dim),    # 缩回来
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    一个完整的Transformer层：
    Pre-norm写法——先LayerNorm再进模块，最后残差连接。
    
    流程：
      x → LayerNorm → Attention → +x  → LayerNorm → FeedForward → +x
    """
    def __init__(self, embed_dim, num_heads, head_size):
        super().__init__()
        # 阶段2写的attention（现在是单头，后面会改成多头）
        self.attention = SingleHeadAttention(embed_dim, head_size)
        # 两层FeedForward
        self.feedforward = FeedForward(embed_dim)
        # 两个LayerNorm——attention前后各一个
        self.ln1 = nn.LayerNorm(embed_dim)   # attention之前的norm
        self.ln2 = nn.LayerNorm(embed_dim)   # feedforward之前的norm

    def forward(self, x):
        # 先norm再做attention，然后把原始x加回去（残差连接）
        x = x + self.attention(self.ln1(x))
        # 先norm再过feedforward，再加回去
        x = x + self.feedforward(self.ln2(x))
        return x