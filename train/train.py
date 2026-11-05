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

Auto-logs each run to logs/training-log.md
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.tokenizer import Tokenizer
from model.gpt import GPT


# --- Training Config ---

CONFIG = {
    "type": "pretrain",
    "data_path": "data/corpus/mixed-corpus.txt",
    "embed_dim": 768,
    "num_heads": 4,
    "head_size": 96,
    "num_layers": 16,
    "block_size": 128,
    "epochs": 10000,
    "batch_size": 32,
    "lr": 3e-4,
    "eval_interval": 10,
    "encoding": "sentencepiece",  # "char" | "gpt2" | "cl100k_base" | "o200k_base" | "sentencepiece"
    "vocab_size": 16000,
    "sp_model_path": "data/tokenizer/mixed-sp.model",
    "sample_prompts": ["AI时代", "我们", "银行"],
}


# --- Data Preparation ---

def load_data(data_path, encoding="cl100k_base"):
    """
    Load corpus, return token ids and tokenizer

    encoding: "char" | "gpt2" | "cl100k_base" | "o200k_base" | "sentencepiece"
    """
    base = os.path.dirname(os.path.dirname(__file__))
    if encoding == "char":
        vocab_path = os.path.join(base, 'data', 'vocab.txt')
        tokenizer = Tokenizer("char", vocab_path=vocab_path)
    elif encoding == "sentencepiece":
        model_path = os.path.join(base, CONFIG.get('sp_model_path', 'data/tokenizer/mixed-sp.model'))
        tokenizer = Tokenizer("sentencepiece", model_path=model_path)
    else:
        tokenizer = Tokenizer(encoding)

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

def train(model, data, tokenizer, epochs=100, batch_size=32, block_size=64, lr=1e-3, eval_interval=10, detail_path=None):
    """
    Training loop

    epochs: iterations
    batch_size: sequences per batch
    block_size: context length
    lr: learning rate
    eval_interval: eval every N
    detail_path: path to detail log file for progress updates
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # MPS acceleration (Apple Silicon)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    data = data.to(device)
    print(f"Device: {device}")

    losses = []
    start_time = time.time()

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

            # update progress in detail log
            if detail_path:
                elapsed_now = time.time() - start_time
                update_progress(detail_path, epoch, epochs, loss.item(), avg_eval, elapsed_now)

    return losses


# --- Logging ---

from train.log_utils import get_run_number, start_log, update_progress, finish_log


# --- Main Entry ---

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base, CONFIG['data_path'])
    log_path = os.path.join(base, 'logs', 'training-log.md')
    os.makedirs(os.path.join(base, 'logs'), exist_ok=True)

    # 1. load data
    data, tokenizer = load_data(data_path, encoding=CONFIG.get('encoding', 'gpt2'))
    corpus_chars = len(open(data_path, 'r', encoding='utf-8').read())
    corpus_tokens = len(data)

    # 2. build model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        head_size=CONFIG['head_size'],
        num_layers=CONFIG['num_layers'],
        block_size=CONFIG['block_size']
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    # 3. phase 1: start log
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    run_num = get_run_number(log_path)
    init_results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'device': device,
        'corpus_chars': corpus_chars,
        'corpus_tokens': corpus_tokens,
        'params': params,
        'elapsed': 0,
        'initial_loss': 0,
        'final_loss': 0,
        'final_eval_loss': 0,
    }
    detail_path = start_log(log_path, run_num, CONFIG, init_results)

    # 4. train (phase 2: progress updates inside train())
    print("\nTraining...\n")
    start = time.time()
    losses = train(model, data, tokenizer,
                   epochs=CONFIG['epochs'],
                   batch_size=CONFIG['batch_size'],
                   block_size=CONFIG['block_size'],
                   lr=CONFIG['lr'],
                   eval_interval=CONFIG['eval_interval'],
                   detail_path=detail_path)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"loss: {losses[0][1]:.4f} -> {losses[-1][1]:.4f}")

    # 5. save model
    ckpt_path = os.path.join(base, 'checkpoints', 'model.pt')
    os.makedirs(os.path.join(base, 'checkpoints'), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'encoding': CONFIG.get('encoding', 'gpt2'),
        'config': {
            'embed_dim': CONFIG['embed_dim'],
            'num_heads': CONFIG['num_heads'],
            'head_size': CONFIG['head_size'],
            'num_layers': CONFIG['num_layers'],
            'block_size': CONFIG['block_size'],
        }
    }, ckpt_path)
    print(f"Model saved: {ckpt_path}")

    # 6. generate samples
    from serve.generate import generate
    samples_text = ""
    for prompt in CONFIG['sample_prompts']:
        prompt_ids = tokenizer.encode(prompt)
        result = generate(model, prompt_ids, max_new_tokens=80, temperature=0.7, top_k=10)
        text = tokenizer.decode(result)
        samples_text += f"Prompt: {prompt}\n{text}\n\n"
        print(f"\n--- Prompt: {prompt} ---\n{text}\n")

    # 7. phase 3: finish log
    loss_curve_lines = []
    step = max(1, len(losses) // 10)
    for i in range(0, len(losses), step):
        ep, tl, el = losses[i]
        loss_curve_lines.append(f"step {ep:5d}: train={tl:.4f}  eval={el:.4f}")
    if len(losses) > 0:
        ep, tl, el = losses[-1]
        last_line = f"step {ep:5d}: train={tl:.4f}  eval={el:.4f}"
        if last_line not in loss_curve_lines:
            loss_curve_lines.append(last_line)

    final_results = {
        'date': init_results['date'],
        'device': device,
        'corpus_chars': corpus_chars,
        'corpus_tokens': corpus_tokens,
        'params': params,
        'elapsed': elapsed,
        'initial_loss': losses[0][1],
        'final_loss': losses[-1][1],
        'final_eval_loss': losses[-1][2],
        'loss_curve': '\n'.join(loss_curve_lines),
    }

    finish_log(log_path, run_num, CONFIG, final_results, samples_text.strip(), detail_path)