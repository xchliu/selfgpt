"""测试 SingleHeadAttention"""
import torch
import sys
sys.path.insert(0, '../src')
from attention import SingleHeadAttention

def test_single_head():
    B, T, C = 2, 8, 32
    head_size = 16
    x = torch.randn(B, T, C)
    attn = SingleHeadAttention(C, head_size)
    out = attn.forward(x)
    assert out.shape == (B, T, head_size), f'期望 {(B,T,head_size)}, 实际 {out.shape}'
    print(f'✓ SingleHead Attention 跑通 — 输入{x.shape} → 输出{out.shape}')

if __name__ == '__main__':
    test_single_head()