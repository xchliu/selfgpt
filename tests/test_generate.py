"""Test generate"""
import torch
import sys
sys.path.insert(0, '../src')
from gpt import GPT
from generate import generate


def vocab_list():
    return list('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:"\'()[]{}@#$%^&*-+=<>~')


def test_generate_basic():
    """Basic generation"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]  # "abcd"
    result = generate(model, prompt, max_new_tokens=20)

    assert len(result) == 24, f'expected 24 (4+20), got {len(result)}'
    assert result[:4] == prompt, 'prompt should be preserved'
    text = ''.join(vocab[i] for i in result)
    print(f'OK - basic generate: {len(result)} tokens')
    print(f'  output: {text}')


def test_temperature():
    """Temperature comparison"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]
    torch.manual_seed(42)

    low = generate(model, prompt, max_new_tokens=30, temperature=0.1)
    high = generate(model, prompt, max_new_tokens=30, temperature=2.0)

    low_text = ''.join(vocab[i] for i in low)
    high_text = ''.join(vocab[i] for i in high)
    print(f'OK - temperature comparison:')
    print(f'  low(0.1):  {low_text}')
    print(f'  high(2.0): {high_text}')


def test_top_k():
    """Top-k sampling"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]
    torch.manual_seed(42)

    k1 = generate(model, prompt, max_new_tokens=30, top_k=1)    # deterministic
    k10 = generate(model, prompt, max_new_tokens=30, top_k=10)  # top 10

    k1_text = ''.join(vocab[i] for i in k1)
    k10_text = ''.join(vocab[i] for i in k10)
    print(f'OK - top_k comparison:')
    print(f'  top_k=1:  {k1_text}')
    print(f'  top_k=10: {k10_text}')


def test_block_size_limit():
    """Block size limit"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=32)

    prompt = list(range(30))  # 30 tokens, within limit
    result = generate(model, prompt, max_new_tokens=10)
    assert len(result) == 40, f'expected 40, got {len(result)}'
    print(f'OK - block_size limit: {len(result)} tokens')


if __name__ == '__main__':
    test_generate_basic()
    print()
    test_temperature()
    print()
    test_top_k()
    print()
    test_block_size_limit()
    print('\nAll tests passed')