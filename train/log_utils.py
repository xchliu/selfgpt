"""
Shared training log utilities.

Two-layer logging:
  1. logs/training-log.md    — master ledger (one-line summary per run)
  2. logs/runs/run-NNN-*.md  — detailed log per run

Both pretrain and SFT use these.
"""
import os
from datetime import datetime


def get_run_number(log_path):
    """Read master ledger, return next run number"""
    if not os.path.exists(log_path):
        return 1
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    count = content.count('| Run ')
    return count + 1


def write_ledger_entry(log_path, run_num, config, results):
    """Append one-line summary to master ledger"""
    header = ""
    if run_num == 1:
        header = """# SelfGPT Training Ledger

| Run | Date | Type | Tokenizer | Vocab | Corpus | Params | Epochs | Time | Init Loss | Final Loss | Notes |
|-----|------|------|-----------|-------|--------|--------|--------|------|-----------|------------|-------|
"""

    entry = f"""| {run_num} | {results['date']} | {config.get('type', 'pretrain')} | {config.get('encoding', 'gpt2')} | {config.get('vocab_size', '?')} | {results['corpus_chars']:,} | {results['params']:,} | {config.get('epochs', '?')} | {results['elapsed']:.0f}s | {results['initial_loss']:.2f} | {results['final_loss']:.2f} | {config.get('notes', '')} |
"""

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(header + entry)
    print(f"Ledger updated: {log_path} (Run {run_num})")


def write_detail_log(runs_dir, run_num, config, results, samples):
    """Write detailed log for a single run"""
    os.makedirs(runs_dir, exist_ok=True)

    run_type = config.get('type', 'pretrain')
    filename = f"run-{run_num:03d}-{run_type}.md"
    filepath = os.path.join(runs_dir, filename)

    entry = f"""# Run {run_num:03d} — {run_type}

**Date**: {results['date']}
**Goal**: {config.get('goal', 'N/A')}

## Config

| Parameter | Value |
|-----------|-------|
| Type | {run_type} |
| Tokenizer | {config.get('encoding', 'gpt2')} |
| Vocab size | {config.get('vocab_size', 'N/A')} |
| Corpus | {config.get('data_path', 'N/A')} |
| Corpus size | {results['corpus_chars']:,} chars -> {results['corpus_tokens']:,} tokens |
| embed_dim | {config.get('embed_dim', 'N/A')} |
| num_layers | {config.get('num_layers', 'N/A')} |
| num_heads | {config.get('num_heads', 'N/A')} |
| head_size | {config.get('head_size', 'N/A')} |
| block_size | {config.get('block_size', 'N/A')} |
| epochs | {config.get('epochs', 'N/A')} |
| batch_size | {config.get('batch_size', 'N/A')} |
| lr | {config.get('lr', 'N/A')} |
| optimizer | AdamW |
| device | {results['device']} |

## Results

- Params: {results['params']:,}
- Time: {results['elapsed']:.1f}s
- Initial loss: {results['initial_loss']:.4f}
- Final loss: {results['final_loss']:.4f}
- Eval loss: {results['final_eval_loss']:.4f}
- Train/eval gap: {results['final_loss'] - results['final_eval_loss']:.4f} ({'no overfitting' if abs(results['final_loss'] - results['final_eval_loss']) < 0.1 else 'possible overfitting'})
- Loss change: {results['initial_loss']:.2f} -> {results['final_loss']:.2f} (delta={results['initial_loss'] - results['final_loss']:.2f})

## Loss Curve

```
{results['loss_curve']}
```

## Sample Outputs

```
{samples}
```
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(entry)
    print(f"Detail log: {filepath}")


def write_log(log_path, run_num, config, results, samples):
    """Write both ledger entry and detail log"""
    runs_dir = os.path.join(os.path.dirname(log_path), 'runs')

    # 1. Master ledger (one-line summary)
    write_ledger_entry(log_path, run_num, config, results)

    # 2. Detailed per-run log
    write_detail_log(runs_dir, run_num, config, results, samples)

    print(f"Run {run_num} logged.")