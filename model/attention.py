import torch
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    """
    
    """
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.W_q = nn.Linear(embed_dim, head_size, bias=False)
        self.W_k = nn.Linear(embed_dim, head_size, bias=False)
        self.W_v = nn.Linear(embed_dim, head_size, bias=False)
        # 输出投影：把 head_size 维映射回 embed_dim，这样残差连接才能 x + attn(x)
        self.proj = nn.Linear(head_size, embed_dim, bias=False)
        
    def forward(self,x):
        B,T,C = x.shape
        # 2. 算 Q, K, V（每个形状都是 (B, T, head_size)）
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 3. 算关联分数: q @ k转置 / sqrt(head_size)
        #    形状: (B, T, T) — 每个token对所有token的关联度
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)

        # 4. causal mask: 把右上角全部设成 -inf
        #    提示 torch.tril 生成下三角矩阵
        mask = torch.tril(torch.ones(T, T, device=x.device))       # 下三角矩阵
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. softmax（沿最后一维）
        weights = torch.softmax(scores, dim=-1)   # 沿最后一维(T)做softmax，每行变成概率

        # 6. 加权求和: softmax(scores) @ v
        #    形状: (B, T, head_size)
        out = weights @ v   # (B,T,T) @ (B,T,H) → (B,T,H)
        out = self.proj(out)  # (B,T,H) → (B,T,C) 投影回embed_dim
        return out

