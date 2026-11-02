"""
阶段 5：训练脚本

核心循环：
  1. 从语料切一个 batch 的 (输入, 目标) 对
  2. 前向传播 → logits
  3. 算 loss（交叉熵：预测的 vs 真实的下一个 token）
  4. 反向传播 → 梯度
  5. optimizer.step() 更新参数
  6. 重复

训练目标：让模型学会「给定前文，预测下一个 token」
"""
import os
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.tokenizer import Tokenizer
from model.gpt import GPT


# ─── 数据准备 ───────────────────────────────────────────────

def load_data(data_path, vocab_path):
    """
    加载训练语料和词表，返回 token id 列表和 tokenizer
    """
    tokenizer = Tokenizer(vocab_path=vocab_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 检查语料中的字符是否都在词表里
    missing = set(text) - set(tokenizer.char_to_id.keys())
    if missing:
        # 把缺失字符补进词表
        print(f"词表缺少 {len(missing)} 个字符，自动补充: {missing}")
        new_chars = sorted(list(missing))
        tokenizer.vocab.extend(new_chars)
        for ch in new_chars:
            tokenizer.char_to_id[ch] = len(tokenizer.char_to_id)
            tokenizer.id_to_char[len(tokenizer.id_to_char)] = ch

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"语料长度: {len(text)} 字符 → {len(data)} tokens")
    print(f"词表大小: {tokenizer.vocab_size}")

    return data, tokenizer


def get_batch(data, batch_size, block_size):
    """
    从语料中随机切一个 batch

    原理：
      语料: [t0, t1, t2, t3, t4, t5, t6, ...]
      输入:  [t0, t1, t2, t3]   ← 给模型看的前文
      目标:  [t1, t2, t3, t4]   ← 模型应该预测的下一个 token（整体偏移一位）

    所以目标和输入是同一段文本，只是错开一位。
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ─── 训练 ───────────────────────────────────────────────────

def train(model, data, tokenizer, epochs=100, batch_size=32, block_size=64, lr=1e-3, eval_interval=10):
    """
    训练循环

    epochs: 训练多少轮
    batch_size: 每次处理几条数据
    block_size: 每条数据多长（上下文窗口）
    lr: 学习率（步长，太大跳过最优解，太小学不动）
    eval_interval: 每多少轮评估一次
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # MPS 加速（Apple Silicon）
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    data = data.to(device)
    print(f"训练设备: {device}")

    losses = []

    for epoch in range(epochs):
        # ── 1. 取一个 batch ──
        x, y = get_batch(data, batch_size, block_size)
        x, y = x.to(device), y.to(device)

        # ── 2. 前向传播 ──
        logits = model(x)  # (B, T, vocab_size)

        # ── 3. 算 loss（交叉熵）──
        # PyTorch 的 cross_entropy 要 (N, C) 格式，所以展平
        B, T, C = logits.shape
        loss = nn.functional.cross_entropy(
            logits.view(B * T, C),  # 展平成 (B*T, vocab_size)
            y.view(B * T)           # 展平成 (B*T,)
        )

        # ── 4. 反向传播 ──
        optimizer.zero_grad()  # 清掉上一步的梯度
        loss.backward()        # 自动算每个参数的梯度

        # ── 5. 更新参数 ──
        optimizer.step()

        # ── 6. 记录 + 定期评估 ──
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            # 在训练集上再跑几批取平均 loss，更稳定
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for _ in range(5):
                    x_eval, y_eval = get_batch(data, batch_size, block_size)
                    x_eval, y_eval = x_eval.to(device), y_eval.to(device)
                    logits_eval = model(x_eval)
                    eval_loss = nn.functional.cross_entropy(
                        logits_eval.view(B * T, C),
                        y_eval.view(B * T)
                    )
                    eval_losses.append(eval_loss.item())
            avg_eval = sum(eval_losses) / len(eval_losses)
            losses.append((epoch, loss.item(), avg_eval))
            model.train()

            print(f"轮次 {epoch:4d}/{epochs} | 训练 loss: {loss.item():.4f} | 评估 loss: {avg_eval:.4f}")

    return losses


# ─── 主入口 ─────────────────────────────────────────────────

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, '..', 'data', 'train.txt')
    vocab_path = os.path.join(base, '..', 'data', 'vocab.txt')

    # 1. 加载数据
    data, tokenizer = load_data(data_path, vocab_path)

    # 2. 建模型（小模型，适合这台 Mac）
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        head_size=16,
        num_layers=4,
        block_size=64
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params:,}")

    # 3. 训练
    print("\n开始训练...\n")
    start = time.time()
    losses = train(model, data, tokenizer, epochs=200, batch_size=32, block_size=64, lr=1e-3)
    elapsed = time.time() - start

    print(f"\n训练完成，耗时 {elapsed:.1f}s")
    print(f"初始 loss: {losses[0][1]:.4f} → 最终 loss: {losses[-1][1]:.4f}")

    # 4. 保存模型
    ckpt_path = os.path.join(base, '..', 'data', 'model.pt')
    torch.save({
        'model_state': model.state_dict(),
        'vocab': tokenizer.vocab,
        'config': {
            'embed_dim': 64,
            'num_heads': 4,
            'head_size': 16,
            'num_layers': 4,
            'block_size': 64,
        }
    }, ckpt_path)
    print(f"模型已保存: {ckpt_path}")

    # 5. 试试生成
    from generate import generate
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    prompt_ids = tokenizer.encode("To be")
    result = generate(model, prompt_ids, max_new_tokens=50, temperature=0.7, top_k=10)
    text = tokenizer.decode(result)
    print(f"\n生成示例:\n{text}")