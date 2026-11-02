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
        # output projection: head_size -> embed_dim, so residual x + attn(x) works
        self.proj = nn.Linear(head_size, embed_dim, bias=False)
        
    def forward(self,x):
        B,T,C = x.shape
        # 2. compute Q, K, V (shape: B, T, head_size)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 3. attention scores: q @ k^T / sqrt(head_size)
        #    shape: (B, T, T) — token-to-token affinity
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)

        # 4. causal mask: set upper triangle to -inf
        #    use torch.tril for lower triangular matrix
        mask = torch.tril(torch.ones(T, T, device=x.device))       # lower triangular
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. softmax along last dim
        weights = torch.softmax(scores, dim=-1)   # row-wise probability

        # 6. weighted sum: weights @ v
        #    shape: (B, T, head_size)
        out = weights @ v   # (B,T,T) @ (B,T,H) → (B,T,H)
        out = self.proj(out)  # (B,T,H) -> (B,T,C) project back to embed_dim
        return out

