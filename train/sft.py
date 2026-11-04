"""
SFT (Supervised Fine-Tuning) script

Difference from pre-training:
- Pre-training: learn to continue text (all tokens contribute to loss)
- SFT: learn to respond in dialogue format (only assistant tokens contribute to loss)

Data format (sft-data.json):
[
  {"conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]},
  ...
]

Training format (ChatML-style):
  <user>question</user><assistant>answer</assistant>

Loss mask:
  <user>question</user><assistant>     <- NO loss (prompt)
  answer</assistant>                    <- YES loss (response)
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.tokenizer import Tokenizer
from model.gpt import GPT

# import logging functions
from train.log_utils import get_run_number, start_log, update_progress, finish_log


# --- SFT Config ---

SFT_CONFIG = {
    "type": "sft",
    "sft_data": "data/sft/sft-clean.json",
    "sp_model": "data/mixed-sp.model",
    "embed_dim": 768,
    "num_heads": 4,
    "head_size": 96,
    "num_layers": 16,
    "block_size": 128,
    "epochs": 6000,
    "batch_size": 4,
    "lr": 1e-4,
    "eval_interval": 50,
    "encoding": "sentencepiece",
    "vocab_size": 16000,
}


# --- Data Preparation ---

def load_sft_data(data_path, tokenizer):
    """
    Load SFT data, convert to token ids with loss mask.

    Returns list of (input_ids, loss_mask) pairs.
    loss_mask[i] = 1 if token i should contribute to loss, 0 otherwise.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        convs = item['conversations']
        full_text = ""
        mask_regions = []  # (start, end) ranges where loss should be computed

        for conv in convs:
            role = conv['role']
            content = conv['content']
            tag_open = f"<{role}>"
            tag_close = f"</{role}>"

            if role == "assistant":
                # Mark this region for loss
                start = len(full_text)
                full_text += tag_open + content + tag_close
                end = len(full_text)
                mask_regions.append((start, end))
            else:
                full_text += tag_open + content + tag_close

        # Encode full text
        token_ids = tokenizer.encode(full_text)

        # Build loss mask (char-level -> token-level approximation)
        # Simple approach: re-encode each part separately to find boundaries
        loss_mask = [0] * len(token_ids)

        # Rebuild token boundaries by encoding incrementally
        prefix = ""
        token_offset = 0
        for conv in convs:
            role = conv['role']
            content = conv['content']
            tag_open = f"<{role}>"
            tag_close = f"</{role}>"

            if role == "assistant":
                # Encode prefix (before assistant content)
                prefix_tokens = len(tokenizer.encode(prefix + tag_open))
                content_tokens = len(tokenizer.encode(prefix + tag_open + content))
                for i in range(prefix_tokens, content_tokens):
                    if i < len(loss_mask):
                        loss_mask[i] = 1

            prefix += tag_open + content + tag_close

        samples.append((token_ids, loss_mask))

    print(f"SFT samples: {len(samples)}")
    total_tokens = sum(len(ids) for ids, _ in samples)
    total_masked = sum(sum(m) for _, m in samples)
    print(f"Total tokens: {total_tokens}, masked (loss): {total_masked}")

    return samples


def get_sft_batch(samples, batch_size, block_size, tokenizer):
    """
    Sample a batch from SFT data.

    Returns:
        x: (B, T) input token ids
        y: (B, T) target token ids (shifted by 1)
        mask: (B, T) loss mask (1 = compute loss, 0 = ignore)
    """
    B = batch_size
    T = block_size

    x = torch.zeros(B, T, dtype=torch.long)
    y = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.float)

    for i in range(B):
        ids, loss_mask = samples[torch.randint(len(samples), (1,)).item()]

        # Truncate or pad to block_size
        if len(ids) > T:
            # Random starting point, but keep it reasonable
            start = 0
            ids = ids[start:start + T]
            loss_mask = loss_mask[start:start + T]
        else:
            # Pad with zeros
            pad_len = T - len(ids)
            ids = ids + [0] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        x[i] = torch.tensor(ids[:T], dtype=torch.long)
        mask[i] = torch.tensor(loss_mask[:T], dtype=torch.float)

    return x, mask


# --- SFT Training ---

def sft_train(model, samples, tokenizer, epochs, batch_size, block_size, lr, eval_interval):
    """
    SFT training loop.

    Key difference from pre-training: only compute loss on assistant tokens.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    print(f"Device: {device}")

    losses = []

    for epoch in range(epochs):
        # 1. get batch
        x, mask = get_sft_batch(samples, batch_size, block_size, tokenizer)
        x = x.to(device)
        mask = mask.to(device)

        # 2. forward pass
        logits = model(x)  # (B, T, vocab_size)

        # 3. compute masked loss
        # Target = input shifted by 1
        # Only compute loss where mask = 1
        B, T, C = logits.shape

        # Shift: predict token i+1 from token i
        # loss_mask[i] = 1 means we care about predicting token i+1
        # So we shift logits and mask accordingly
        shift_logits = logits[:, :-1, :]  # (B, T-1, C)
        shift_targets = x[:, 1:]           # (B, T-1)
        shift_mask = mask[:, 1:]           # (B, T-1) - mask for the target token

        # Flatten
        flat_logits = shift_logits.reshape(-1, C)
        flat_targets = shift_targets.reshape(-1)
        flat_mask = shift_mask.reshape(-1)

        # Compute per-token loss
        loss_per_token = nn.functional.cross_entropy(
            flat_logits, flat_targets, reduction='none'
        )

        # Apply mask: only assistant tokens contribute
        if flat_mask.sum() > 0:
            loss = (loss_per_token * flat_mask).sum() / flat_mask.sum()
        else:
            loss = loss_per_token.mean()

        # 4. backward
        optimizer.zero_grad()
        loss.backward()

        # 5. step
        optimizer.step()

        # 6. log
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            losses.append((epoch, loss.item()))
            print(f"step {epoch:4d}/{epochs} | sft loss: {loss.item():.4f}")

    return losses


# --- Main Entry ---

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))

    # 1. load tokenizer
    sp_model_path = os.path.join(base, SFT_CONFIG['sp_model'])
    tokenizer = Tokenizer("sentencepiece", model_path=sp_model_path)
    print(f"Tokenizer: {tokenizer.vocab_size} vocab")

    # 2. load SFT data
    sft_data_path = os.path.join(base, SFT_CONFIG['sft_data'])
    samples = load_sft_data(sft_data_path, tokenizer)

    # 3. load pre-trained model
    ckpt_path = os.path.join(base, 'checkpoints', 'model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        cfg = ckpt['config']
        model = GPT(
            vocab_size=tokenizer.vocab_size,
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            head_size=cfg['head_size'],
            num_layers=cfg['num_layers'],
            block_size=SFT_CONFIG['block_size']
        )
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded pretrained model: {ckpt_path}")
    else:
        print("No pretrained model found, using random init")
        model = GPT(
            vocab_size=tokenizer.vocab_size,
            embed_dim=SFT_CONFIG['embed_dim'],
            num_heads=SFT_CONFIG['num_heads'],
            head_size=SFT_CONFIG['head_size'],
            num_layers=SFT_CONFIG['num_layers'],
            block_size=SFT_CONFIG['block_size']
        )

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    # 3.5 phase 1: start log
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    log_path = os.path.join(base, 'logs', 'training-log.md')
    os.makedirs(os.path.join(base, 'logs'), exist_ok=True)
    run_num = get_run_number(log_path)
    init_results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'device': device,
        'corpus_chars': sum(len(ids) for ids, _ in samples),
        'corpus_tokens': sum(len(ids) for ids, _ in samples),
        'params': params,
        'elapsed': 0,
        'initial_loss': 0,
        'final_loss': 0,
        'final_eval_loss': 0,
    }
    sft_config_for_start = {**SFT_CONFIG, 'data_path': SFT_CONFIG['sft_data'], 'goal': 'SFT: learn dialogue format'}
    detail_path = start_log(log_path, run_num, sft_config_for_start, init_results)

    # 4. SFT train
    print("\nSFT training...\n")
    start = time.time()
    losses = sft_train(model, samples, tokenizer,
                       epochs=SFT_CONFIG['epochs'],
                       batch_size=SFT_CONFIG['batch_size'],
                       block_size=SFT_CONFIG['block_size'],
                       lr=SFT_CONFIG['lr'],
                       eval_interval=SFT_CONFIG['eval_interval'])
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"SFT loss: {losses[0][1]:.4f} -> {losses[-1][1]:.4f}")

    # 5. save SFT model
    sft_ckpt_path = os.path.join(base, 'checkpoints', 'model-sft.pt')
    torch.save({
        'model_state': model.state_dict(),
        'encoding': 'sentencepiece',
        'sp_model': SFT_CONFIG['sp_model'],
        'config': {
            'embed_dim': cfg['embed_dim'],
            'num_heads': cfg['num_heads'],
            'head_size': cfg['head_size'],
            'num_layers': cfg['num_layers'],
            'block_size': SFT_CONFIG['block_size'],
        }
    }, sft_ckpt_path)
    print(f"SFT model saved: {sft_ckpt_path}")

    # 6. test dialogue
    from serve.generate import generate
    model.eval()

    test_prompts = [
        "什么是大模型？",
        "银行怎么落地AI？",
        "AI能替代程序员吗？",
    ]

    print("\n=== SFT Test ===\n")
    samples_text = ""
    for question in test_prompts:
        prompt_text = f"<user>{question}</user><assistant>"
        prompt_ids = tokenizer.encode(prompt_text)
        result = generate(model, prompt_ids, max_new_tokens=100, temperature=0.6, top_k=5)
        text = tokenizer.decode(result)
        # Extract assistant response
        if "<assistant>" in text:
            response = text.split("<assistant>")[-1]
            if "</assistant>" in response:
                response = response.split("</assistant>")[0]
        else:
            response = text

        response = response.strip()
        print(f"Q: {question}")
        print(f"A: {response}\n")
        samples_text += f"Q: {question}\nA: {response}\n\n"

    # 7. auto-write training log
    loss_curve_lines = []
    step = max(1, len(losses) // 10)
    for i in range(0, len(losses), step):
        ep, tl = losses[i]
        loss_curve_lines.append(f"step {ep:5d}: sft_loss={tl:.4f}")
    if losses:
        ep, tl = losses[-1]
        last_line = f"step {ep:5d}: sft_loss={tl:.4f}"
        if last_line not in loss_curve_lines:
            loss_curve_lines.append(last_line)

    log_path = os.path.join(base, 'logs', 'training-log.md')
    sft_config_for_log = {
        **SFT_CONFIG,
        'data_path': SFT_CONFIG['sft_data'],
        'goal': 'SFT: learn dialogue format',
    }
    results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'corpus_chars': sum(len(ids) for ids, _ in samples),
        'corpus_tokens': sum(len(ids) for ids, _ in samples),
        'params': params,
        'elapsed': elapsed,
        'initial_loss': losses[0][1],
        'final_loss': losses[-1][1],
        'final_eval_loss': losses[-1][1],  # SFT doesn't have separate eval
        'loss_curve': '\n'.join(loss_curve_lines),
    }

    sft_config_for_log = {**SFT_CONFIG, 'data_path': SFT_CONFIG['sft_data'], 'goal': 'SFT: learn dialogue format'}
    finish_log(log_path, run_num, sft_config_for_log, results, samples_text.strip(), detail_path)