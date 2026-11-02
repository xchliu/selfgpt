import torch
import torch.nn as nn
from model.gpt import GPT


@torch.no_grad()  # 推理不算梯度，省内存
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    自回归生成

    model: 训练好的（或没训练的）GPT
    prompt_ids: 开头的 token id 列表，如 [0, 1, 2, 3]
    max_new_tokens: 最多生成多少个新 token
    temperature: 控制随机性。<1 更保守，>1 更放飞，=1 原始概率
    top_k: 只从概率最高的 K 个里采样，None = 不限制

    返回: 完整的 token id 列表（prompt + 生成部分）
    """
    idx = torch.tensor([prompt_ids])  # (1, T)
    idx = idx.to(next(model.parameters()).device)  # 搬到模型所在设备

    for _ in range(max_new_tokens):
        # 1. 如果序列超过 block_size，只取最后 block_size 个（模型窗口有限）
        if idx.shape[1] > model.block_size:
            idx_cond = idx[:, -model.block_size:]
        else:
            idx_cond = idx

        # 2. 前向传播 → logits
        logits = model(idx_cond)          # (1, T, vocab_size)

        # 3. 只看最后一个位置的预测
        logits = logits[:, -1, :]         # (1, vocab_size)

        # 4. temperature：拉大/缩小概率差异
        logits = logits / temperature

        # 5. top_k：只保留概率最高的 K 个，其余设成 -inf
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float('-inf')

        # 6. softmax → 概率
        probs = torch.softmax(logits, dim=-1)   # (1, vocab_size)

        # 7. 按概率采样一个 token（不是 argmax，这样有随机性）
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # 8. 拼到序列末尾，下一轮用
        idx = torch.cat([idx, next_id], dim=1)

    return idx[0].tolist()


if __name__ == '__main__':
    # 用没训练的模型瞎生成一段
    vocab = list('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:"\'()[]{}@#$%^&*-+=<>~')
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]  # "abcd"
    result = generate(model, prompt, max_new_tokens=50, temperature=0.8, top_k=10)

    text = ''.join(vocab[i] for i in result)
    print(f'未训练模型生成:\n{text}')
    print(f'\n（纯乱码就对了——还没训练）')