import torch
import torch.nn as nn
from model.gpt import GPT


@torch.no_grad()  # no gradient computation, save memory
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressive generation

    model: trained (or untrained) GPT
    prompt_ids: starting token id list
    max_new_tokens: max new tokens to generate
    temperature: <1 conservative, >1 creative
    top_k: sample from top K, None = no limit

    Returns: full token id list (prompt + generated)
    """
    idx = torch.tensor([prompt_ids])  # (1, T)
    idx = idx.to(next(model.parameters()).device)  # move to model device

    for _ in range(max_new_tokens):
        # 1. crop context if exceeds block_size
        if idx.shape[1] > model.block_size:
            idx_cond = idx[:, -model.block_size:]
        else:
            idx_cond = idx

        # 2. forward pass -> logits
        logits = model(idx_cond)          # (1, T, vocab_size)

        # 3. take logits at last position
        logits = logits[:, -1, :]         # (1, vocab_size)

        # 4. apply temperature scaling
        logits = logits / temperature

        # 5. top_k: keep only top K, mask rest with -inf
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float('-inf')

        # 6. softmax -> probabilities
        probs = torch.softmax(logits, dim=-1)   # (1, vocab_size)

        # 7. sample (multinomial, not argmax, for randomness)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # 8. append to sequence
        idx = torch.cat([idx, next_id], dim=1)

    return idx[0].tolist()


if __name__ == '__main__':
    # generate with untrained model
    vocab = list('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:"\'()[]{}@#$%^&*-+=<>~')
    model = GPT(vocab_size=len(vocab), embed_dim=64, num_heads=4, head_size=16, num_layers=4, block_size=128)

    prompt = [0, 1, 2, 3]  # "abcd"
    result = generate(model, prompt, max_new_tokens=50, temperature=0.8, top_k=10)

    text = ''.join(vocab[i] for i in result)
    print(f'Untrained model output:\n{text}')
    print(f'\n(Gibberish is expected — model is untrained)')