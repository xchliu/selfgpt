"""
Shared training log utilities.

Three-phase logging:
  1. start_log()    — called at training start: ledger row (running) + detail log header
  2. update_progress() — called during training: update detail log with latest step
  3. finish_log()   — called at training end: update ledger (final results) + detail log (analysis + samples)

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
    count = 0
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('| ') and not line.startswith('| Run') and not line.startswith('|---'):
            parts = line.split('|')
            if len(parts) > 1:
                first_cell = parts[1].strip()
                if first_cell.isdigit():
                    count += 1
    return count + 1


def _ledger_header():
    return """# SelfGPT Training Ledger

| Run | Date | Type | Tokenizer | Vocab | Corpus | Params | Epochs | Time | Init Loss | Final Loss | Status |
|-----|------|------|-----------|-------|--------|--------|--------|------|-----------|------------|--------|
"""


def _ledger_row(run_num, config, results, status):
    date = results.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))
    init_loss = results.get('initial_loss', 0)
    final_loss = results.get('final_loss', 0)
    elapsed = results.get('elapsed', 0)
    corpus = results.get('corpus_chars', 0)
    params = results.get('params', 0)
    return f"| {run_num} | {date} | {config.get('type', 'pretrain')} | {config.get('encoding', 'gpt2')} | {config.get('vocab_size', '?')} | {corpus:,} | {params:,} | {config.get('epochs', '?')} | {elapsed:.0f}s | {init_loss:.2f} | {final_loss:.2f} | {status} |\n"


def start_log(log_path, run_num, config, results):
    """Phase 1: Called at training start.
    Writes ledger row with status=running, and detail log header.
    """
    runs_dir = os.path.join(os.path.dirname(log_path), 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    # 1. Ledger: add row with running status
    header = _ledger_header() if run_num == 1 else ""
    row = _ledger_row(run_num, config, results, "running")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(header + row)
    print(f"Ledger: Run {run_num} started")

    # 2. Detail log: header with config
    run_type = config.get('type', 'pretrain')
    detail_path = os.path.join(runs_dir, f"run-{run_num:03d}-{run_type}.md")
    detail = f"""# Run {run_num:03d} — {run_type}

**Date**: {results['date']}
**Goal**: {config.get('goal', 'N/A')}
**Status**: running

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

## Progress

```
training...
```
"""
    with open(detail_path, 'w', encoding='utf-8') as f:
        f.write(detail)
    print(f"Detail log: {detail_path}")

    return detail_path


def update_progress(detail_path, epoch, total_epochs, train_loss, eval_loss, elapsed):
    """Phase 2: Called during training. Updates the Progress section in detail log."""
    if not os.path.exists(detail_path):
        return

    with open(detail_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pct = (epoch + 1) / total_epochs * 100
    progress_line = f"step {epoch+1}/{total_epochs} ({pct:.1f}%) | train: {train_loss:.4f} | eval: {eval_loss:.4f} | {elapsed:.0f}s"

    # Replace the Progress section content
    # Find "## Progress" section and update it
    import re
    pattern = r'(## Progress\n\n```\n)(.*?)(\n```)'
    replacement = rf'\1{progress_line}\n\3'
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if new_content == content:
        # Pattern not found, append
        new_content = content + f"\n{progress_line}\n"

    with open(detail_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


def finish_log(log_path, run_num, config, results, samples, detail_path=None):
    """Phase 3: Called at training end.
    Updates ledger row with final results, and detail log with analysis + samples.
    """
    runs_dir = os.path.join(os.path.dirname(log_path), 'runs')

    # 1. Update ledger: replace the running row with final results
    with open(log_path, 'r', encoding='utf-8') as f:
        ledger = f.read()

    # Find the row for this run_num and replace it
    old_row_pattern = f"| {run_num} |"
    lines = ledger.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith(old_row_pattern) and 'running' in line:
            # Replace with final row
            new_lines.append(_ledger_row(run_num, config, results, "done").strip())
        else:
            new_lines.append(line)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print(f"Ledger: Run {run_num} done (loss {results['final_loss']:.2f})")

    # 2. Update detail log with results, analysis, samples
    if detail_path is None:
        run_type = config.get('type', 'pretrain')
        detail_path = os.path.join(runs_dir, f"run-{run_num:03d}-{run_type}.md")

    with open(detail_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace status
    content = content.replace("**Status**: running", "**Status**: done")

    # Replace progress with final loss curve
    import re
    final_section = f"""## Results

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

    # Replace everything from "## Results" onwards (or from "## Progress" if no Results yet)
    if '## Results' in content:
        content = content[:content.index('## Results')] + final_section
    elif '## Progress' in content:
        content = content[:content.index('## Progress')] + final_section
    else:
        content = content + '\n\n' + final_section

    with open(detail_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Detail log updated: {detail_path}")