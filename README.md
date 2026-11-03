# SelfGPT 🧠

> From scratch to chat: hand-built GPT, from 170K params to 100M, from gibberish to dialogue.

## What is this

SelfGPT is a learning project — not a product. By implementing every component from scratch, you truly understand how Transformer works.

**Core philosophy**: Build it once, beats reading ten papers.

## Evolution

| Stage | Tokenizer | Vocab | Corpus | Params | Loss | Output Quality |
|-------|-----------|-------|--------|--------|------|----------------|
| 1. Char-level | char | 127 | 1K chars | 170K | 1.9 | Word boundaries |
| 2. BPE (tiktoken) | gpt2 | 50K | 1.1M chars | 13.8M | 3.66 | Coherent dialogue |
| 3. Chinese BPE | cl100k | 100K | 390K chars | 26.7M | 3.55 | Chinese phrases |
| 4. SentencePiece | SP (custom) | 16K | 380K chars | 5M | 0.80 | Narrative style |
| 5. Large corpus | SP (custom) | 16K | 5.3M chars | 5M | 3.95 | Chinese grammar |
| 6. SFT | SP (custom) | 16K | 500 Q&A | 5M | 4.96 | Dialogue format |
| 7. Scaling up | SP (custom) | 16K | 5.3M chars | ~100M | TBD | TBD |

## Training Pipeline

```
Raw articles → data_pipeline/process.py → clean-corpus.txt
                   ↓
JSONL corpus  → data_pipeline/preprocess_pretrain.py → pretrain-clean.txt
                   ↓
Mixed corpus  → train/train_tokenizer.py → SP model (16K vocab)
                   ↓
Pretrain      → train/train.py → model.pt
                   ↓
SFT data      → data_pipeline/gen_sft.py → sft-auto.json (500 Q&A pairs)
                   ↓
Fine-tune     → train/sft.py → model-sft.pt
                   ↓
Chat          → serve/chat.py → interactive dialogue
```

## Quick Start

```bash
git clone https://github.com/xchliu/selfgpt.git
cd selfgpt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Clean corpus
python data_pipeline/process.py --input /path/to/articles --output data/clean-corpus.txt

# 2. Train tokenizer
python train/train_tokenizer.py --corpus data/clean-corpus.txt --vocab_size 16000

# 3. Pretrain
python train/train.py

# 4. Generate SFT data
python data_pipeline/gen_sft.py

# 5. SFT
python train/sft.py

# 6. Chat
python serve/chat.py
```

## Project Structure

```
selfgpt/
├── MANUAL.md               # Full operation manual
├── TODO.md                 # Roadmap
├── requirements.txt        # torch + numpy + tiktoken + sentencepiece
│
├── model/                  # Model definition (shared by train & serve)
│   ├── tokenizer.py        # Unified tokenizer: char / tiktoken / sentencepiece
│   ├── attention.py        # SingleHeadAttention (QKV + mask + proj)
│   ├── block.py            # TransformerBlock (FF + LayerNorm + residual)
│   └── gpt.py              # Full GPT (Embedding + Position + N×Block + LM Head)
│
├── train/                  # Training
│   ├── train.py            # Pretraining loop
│   ├── sft.py              # SFT (supervised fine-tuning with loss mask)
│   ├── train_tokenizer.py  # SentencePiece tokenizer trainer
│   └── log_utils.py        # Three-phase training log (start → progress → finish)
│
├── serve/                  # Inference
│   ├── generate.py         # Autoregressive generation (temperature + top_k)
│   └── chat.py             # Multi-turn dialogue
│
├── data_pipeline/          # Data processing
│   ├── process.py          # Clean markdown articles → training corpus
│   ├── preprocess_pretrain.py  # Clean JSONL corpus → training text
│   └── gen_sft.py          # Auto-generate Q&A pairs from articles (6 strategies)
│
├── data/                   # Data + models
│   ├── clean-corpus.txt    # Cleaned articles
│   ├── pretrain-clean.txt  # Cleaned pretrain corpus
│   ├── mixed-sp.model      # SentencePiece tokenizer model
│   └── sft/                # SFT data
│
├── checkpoints/            # Model weights (gitignored)
│   ├── model.pt            # Pretrained model
│   └── model-sft.pt        # SFT model
│
├── logs/                   # Training logs
│   ├── training-log.md     # Master ledger (one line per run)
│   └── runs/               # Detailed log per run
│
├── tests/                  # Tests
└── ref/                    # References (papers, tutorials)
```

## Key Concepts You'll Learn

- **Tokenizer**: char-level → BPE → SentencePiece, why vocab matters
- **Attention**: Q/K/V, causal mask, softmax (normalization + amplification + differentiability)
- **Transformer Block**: residual connection, LayerNorm, FeedForward
- **GPT**: token embedding, position embedding, LM head
- **Training**: cross-entropy loss, batch, gradient, AdamW, overfitting
- **SFT**: loss mask (only assistant tokens), dialogue format, fine-tuning
- **Data pipeline**: cleaning, deduplication, quality filtering, Q&A generation
- **History**: RNN → Seq2Seq → Attention → Transformer → SFT → RLHF

## Training Log

Every training run is auto-logged in two layers:

1. **Master ledger** (`logs/training-log.md`): one-line summary per run
2. **Detail log** (`logs/runs/run-NNN-*.md`): config, loss curve, sample outputs

Three-phase logging:
- `start_log()`: ledger row (running) + detail log header
- `update_progress()`: detail log updated during training
- `finish_log()`: ledger updated (done) + detail log finalized

## Environment

- Python 3.11+
- PyTorch 2.x (MPS acceleration on Apple Silicon)
- macOS Apple Silicon / Linux / Windows
- Tested on M1 Max 32GB

## Roadmap

- [ ] Multi-head attention
- [ ] RoPE positional encoding
- [ ] Larger corpus (Chinese Wikipedia / literature)
- [ ] KV Cache for inference
- [ ] Streaming output
- [ ] RLHF

## References

See `ref/README.md` — papers, code references, video tutorials, illustrated guides.

## License

MIT

---

> No step is "understanding". Every step is probability and linear algebra.