"""
Training script

Core loop:
  1. sample a batch of (input, target) pairs
  2. forward pass -> logits
  3. compute cross-entropy loss
  4. backward pass -> gradients
  5. optimizer.step() update params
  6. repeat

Goal: predict next token given context
"""
import os
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.tokenizer import Tokenizer
from model.bpe_tokenizer import BPETokenizer
from model.gpt import GPT


# --- Data Preparation ---

def load_data(data_path, use_bpe=True):
    """
    Load corpus, return token ids and tokenizer

    use_bpe=True: BPE tokenizer (GPT-2 vocab)
    use_bpe=False: char-level tokenizer
    """
    if use_bpe:
        tokenizer = BPETokenizer()
    else:
        raise NotImplementedError("char-level tokenizer needs vocab_path, use old version")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Corpus: {len(text)} chars -> {len(data)} tokens")
    print(f"Vocab: {tokenizer.vocab_size}")

    return data, tokenizer


def get_batch(data, batch_size, block_size):
    """
    Sample a random batch from corpus.

    Mechanics:
      corpus: [t0, t1, t2, t3, ...]
      input:  [t0, t1, t2, t3]   <- context
      target: [t1, t2, t3, t4]   <- shifted by one

    Target = input shifted by one position.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# --- Training ---

def train(model, data, tokenizer, epochs=100, batch_size=32, block_size=64, lr=1e-3, eval_interval=10):
    """
    Training loop

    epochs: iterations
    batch_size: sequences per batch
    block_size: context length
    lr: learning rate
    eval_interval: eval every N
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # MPS acceleration (Apple Silicon)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    data = data.to(device)
    print(f"Device: {device}")

    losses = []

    for epoch in range(epochs):
        # 1. sample a batch
        x, y = get_batch(data, batch_size, block_size)
        x, y = x.to(device), y.to(device)

        # 2. forward pass
        logits = model(x)  # (B, T, vocab_size)

        # 3. compute cross-entropy loss
        # cross_entropy expects (N, C) format, flatten
        B, T, C = logits.shape
        loss = nn.functional.cross_entropy(
            logits.view(B * T, C),  # flatten to (B*T, vocab_size)
            y.view(B * T)           # flatten to (B*T,)
        )

        # 4. backward pass
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()        # compute gradients

        # 5. update parameters
        optimizer.step()

        # 6. log + periodic eval
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            # average eval loss over multiple batches
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

            print(f"step {epoch:4d}/{epochs} | train loss: {loss.item():.4f} | eval loss: {avg_eval:.4f}")

    return losses


# --- Main Entry ---

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, '..', 'data', 'train.txt')

    # 1. load data (BPE tokenizer)
    data, tokenizer = load_data(data_path)

    # 2. build small model for this Mac
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        head_size=16,
        num_layers=4,
        block_size=64
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    # 3. train
    print("\nTraining...\n")
    start = time.time()
    losses = train(model, data, tokenizer, epochs=200, batch_size=32, block_size=64, lr=1e-3)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"loss: {losses[0][1]:.4f} -> {losses[-1][1]:.4f}")

    # 4. save model
    ckpt_path = os.path.join(base, '..', 'checkpoints', 'model.pt')
    torch.save({
        'model_state': model.state_dict(),
        'encoding': 'gpt2',
        'config': {
            'embed_dim': 64,
            'num_heads': 4,
            'head_size': 16,
            'num_layers': 4,
            'block_size': 64,
        }
    }, ckpt_path)
    print(f"Model saved: {ckpt_path}")

    # 5. generate sample
    from serve.generate import generate
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    prompt_ids = tokenizer.encode("To be")
    result = generate(model, prompt_ids, max_new_tokens=50, temperature=0.7, top_k=10)
    text = tokenizer.decode(result)
    print(f"\nSample output:\n{text}")