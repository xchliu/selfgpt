"""Test SingleHeadAttention"""
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
    assert out.shape == (B, T, head_size), f'expected {(B,T,head_size)}, got {out.shape}'
    print(f'OK - attention works: {x.shape} -> {out.shape}')

if __name__ == '__main__':
    test_single_head()