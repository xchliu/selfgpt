"""测试生成功能"""
import torch
import sys
sys.path.insert(0, '../src')
from gpt import GPT
from generate import generate


def vocab_list():
    return list('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:"\'()[]{}@#$%^&*-+=<>~')


def test_generate_basic():
    """基本生成：给个开头，生成一段"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]  # "abcd"
    result = generate(model, prompt, max_new_tokens=20)

    assert len(result) == 24, f'期望 24 个 token (4+20), 实际 {len(result)}'
    assert result[:4] == prompt, 'prompt 应该保留在开头'
    text = ''.join(vocab[i] for i in result)
    print(f'✓ 基本生成跑通 — 共 {len(result)} 个 token')
    print(f'  内容: {text}')


def test_temperature():
    """temperature 对比：低=保守，高=放飞"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]
    torch.manual_seed(42)

    low = generate(model, prompt, max_new_tokens=30, temperature=0.1)
    high = generate(model, prompt, max_new_tokens=30, temperature=2.0)

    low_text = ''.join(vocab[i] for i in low)
    high_text = ''.join(vocab[i] for i in high)
    print(f'✓ temperature 对比:')
    print(f'  低温(0.1): {low_text}')
    print(f'  高温(2.0): {high_text}')


def test_top_k():
    """top_k 限制：只从概率最高的K个里选"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]
    torch.manual_seed(42)

    k1 = generate(model, prompt, max_new_tokens=30, top_k=1)    # 只选最高 → 几乎确定性
    k10 = generate(model, prompt, max_new_tokens=30, top_k=10)  # 前10选一

    k1_text = ''.join(vocab[i] for i in k1)
    k10_text = ''.join(vocab[i] for i in k10)
    print(f'✓ top_k 对比:')
    print(f'  top_k=1:  {k1_text}')
    print(f'  top_k=10: {k10_text}')


def test_block_size_limit():
    """超过 block_size 时只看最后 block_size 个"""
    vocab = vocab_list()
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=32)

    prompt = list(range(30))  # 30 个 token，没超
    result = generate(model, prompt, max_new_tokens=10)
    assert len(result) == 40, f'期望 40, 实际 {len(result)}'
    print(f'✓ block_size 限制跑通 — 生成 {len(result)} 个 token')


if __name__ == '__main__':
    test_generate_basic()
    print()
    test_temperature()
    print()
    test_top_k()
    print()
    test_block_size_limit()
    print('\n全部测试通过 ✓')