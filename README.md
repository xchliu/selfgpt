# SelfGPT 🧠

> From scratch to chat: hand-built GPT, from 170K params to 105M, from gibberish to dialogue.

## What is this

SelfGPT is a learning project — not a product. By implementing every component from scratch, you truly understand how Transformer works.

**Core philosophy**: Build it once, beats reading ten papers.

## Evolution

| Stage | Tokenizer | Vocab | Corpus | Params | Loss | Output Quality |
|-------|-----------|-------|--------|--------|------|----------------|
| 1. Char-level | char | 127 | 1K chars | 170K | 5.0 | Word boundaries |
| 2. SentencePiece | SP | 16K | 5.3M chars (t2t) | 5M~43M | 1.30 | Coherent sentences |
| 3. Wiki corpus | SP | 16K | 28.9M chars (wiki+gzh) | 105M | 0.81 | Knowledgeable, fluent |
| 4. SFT | SP | 16K | 4332 Q&A pairs | 105M | 0.19 | Dialogue format |
| 5. Chat | SP | 16K | - | 105M | - | Single Q&A works, multi-turn drifts |

## Training Pipeline

```
Wikipedia JSON → data_pipeline/extract_wikipedia.py → wiki-clean.txt
                        ↓
GZH articles → data_pipeline/process.py → clean-corpus.txt
                        ↓
Merged corpus → train/train_tokenizer.py → wiki-sp.model (16K vocab)
                        ↓
Pretrain → train/train.py → model.pt
                        ↓
GZH articles → data_pipeline/gen_sft.py → sft-auto.json (2012 pairs)
External JSONL → data_pipeline/preprocess_sft.py → sft-final.json (10012 pairs)
                        ↓
Clean → data_pipeline/clean_sft.py → sft-clean.json (4332 pairs)
                        ↓
Fine-tune → train/sft.py → model-sft.pt
                        ↓
Chat → serve/chat.py → interactive dialogue
```

## Quick Start

```bash
git clone https://github.com/xchliu/selfgpt.git
cd selfgpt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Extract Wikipedia corpus
python data_pipeline/extract_wikipedia.py

# 2. Merge corpus
cat data/corpus/wiki-clean.txt data/corpus/clean-corpus.txt > data/corpus/wiki-mixed.txt

# 3. Train tokenizer
python train/train_tokenizer.py --corpus data/corpus/wiki-mixed.txt --vocab_size 16000 --prefix data/tokenizer/wiki-sp

# 4. Pretrain (edit CONFIG in train/train.py to set paths)
python train/train.py

# 5. Generate + clean SFT data
python data_pipeline/gen_sft.py
python data_pipeline/preprocess_sft.py
python data_pipeline/clean_sft.py

# 6. SFT
python train/sft.py

# 7. Chat
python serve/chat.py
```

## Project Structure

```
selfgpt/
├── MANUAL.md               # Full operation manual
├── TODO.md                 # Roadmap (10 directions, 50+ items)
│
├── model/                  # Model definition (shared by train & serve)
│   ├── tokenizer.py        # Unified tokenizer: char / tiktoken / sentencepiece
│   ├── attention.py        # SingleHeadAttention (QKV + mask + proj)
│   ├── block.py            # TransformerBlock (FF + LayerNorm + residual)
│   └── gpt.py              # Full GPT (Embedding + Position + N×Block + LM Head)
│
├── train/                  # Training
│   ├── train.py            # Pretraining loop (105M, lr=3e-4)
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
│   ├── extract_wikipedia.py # Extract plain text from Wikipedia JSON
│   ├── preprocess_pretrain.py  # Clean JSONL corpus → training text
│   ├── gen_sft.py          # Auto-generate Q&A pairs (13 strategies)
│   ├── preprocess_sft.py    # Filter + sample external dialogue data
│   └── clean_sft.py        # Quality filter + punctuation normalize + dedup
│
├── data/                   # Data (gitignored)
│   ├── raw/               # Original downloads
│   ├── corpus/            # Pretrain text
│   ├── sft/               # SFT data
│   └── tokenizer/         # SP models
│
├── docs/                   # Documentation
│   ├── model-theory.md     # 10 components: principle, why, experiment link
│   ├── data-recipe.md      # Data formulas + history
│   ├── data-pipeline.md    # Full pipeline with commands
│   ├── data-structure.md   # Directory layout
│   └── training-log.md     # Log mechanism
│
├── checkpoints/            # Model weights (gitignored)
│   ├── model.pt            # Pretrained model
│   └── model-sft.pt        # SFT model
│
├── logs/                   # Training logs
│   ├── training-log.md     # Master ledger (13 runs)
│   └── runs/               # Detail per run
│
├── tests/                  # Tests
└── ref/                    # References (papers, tutorials)
```

## 13 Training Runs

| Run | Type | Params | Corpus | Epochs | Loss | Time | Note |
|-----|------|--------|--------|--------|------|------|------|
| 1 | pretrain | 5M | 39万字 | 3K | 0.95 | 184s | Day 1 |
| 2 | sft | 5M | 12条 | 150 | 0.61 | 5s | First SFT |
| 3 | pretrain | 43M | 530万字 | 10K | 1.30 | 2668s | Scale up |
| 4 | sft | 43M | 512条 | 1K | 1.01 | 48s | Underfit |
| 5 | sft | 43M | 512条 | 1K | 4.64 | 47s | Bad data |
| 6 | sft | 43M | 2012条 | 4K | 0.07 | 179s | Overfit |
| 7 | sft | 43M | 5012条 | 8K | 0.46 | 359s | Best 43M |
| 8 | pretrain | 105M | 530万字 | 10K | 6.70 | 5965s | ❌ Gradient explosion |
| 9 | pretrain | 105M | 530万字 | 39K | 0.51 | 6323s | lr fix 3e-4 |
| 10 | sft | 105M | 4332条 | 6K | 0.08 | 575s | Overfit |
| 11 | pretrain | 105M | 670万token | 10K | 2.88 | 6221s | Undertrained |
| 12 | pretrain | 105M | 670万token | 30K | 0.81 | 19192s | ✅ Best pretrain |
| 13 | sft | 105M | 4332条 | 3K | 0.19 | 290s | ✅ Best SFT |

## Key Concepts

- **Tokenizer**: char → BPE → SentencePiece, why vocab matters
- **Attention**: Q/K/V, scaled dot-product, causal mask, why /√d_k
- **Transformer Block**: residual connection, pre-norm LayerNorm, GELU FeedForward
- **GPT**: token embedding + position embedding → N×Block → LM head
- **Training**: cross-entropy loss, AdamW, lr scaling with model size, train/eval gap
- **SFT**: loss mask (only assistant tokens), dialogue format, overfitting balance
- **Data pipeline**: Wikipedia extraction, 13-strategy Q&A generation, quality cleaning
- **History**: RNN → Seq2Seq → Attention → Transformer → SFT → RLHF

## Lessons Learned

1. **Gradient explosion**: 105M + lr=1e-3 → loss 6.70. Rule: double params, cut lr by 1/3
2. **Training volume**: 670M tokens × 10K epochs → loss 2.88. Each token needs ≥15 views
3. **SFT balance**: loss 0.07 = memorization. loss 0.19 = best generalization
4. **Data quality**: 2500 clean pairs > 5000 noisy pairs
5. **Generation params**: small model needs low temperature (0.6) and low top_k (5)

## Documentation

- [docs/model-theory.md](docs/model-theory.md) — 10 components with principle and experiment links
- [docs/data-recipe.md](docs/data-recipe.md) — Pretrain/SFT/tokenizer formulas + history
- [docs/data-pipeline.md](docs/data-pipeline.md) — Full pipeline with commands
- [ref/README.md](ref/README.md) — Core papers (7) + pending papers (6)
- [TODO.md](TODO.md) — 10 directions, 50+ items including research topics

## Environment

- Python 3.11+
- PyTorch 2.x (MPS acceleration on Apple Silicon)
- macOS Apple Silicon / Linux / Windows
- Tested on M1 Max 32GB

## Roadmap

- [ ] Multi-head attention (current: single-head)
- [ ] RoPE positional encoding
- [ ] Warmup + cosine decay lr schedule
- [ ] KV Cache for inference
- [ ] Larger SFT data (deepseek-generated)
- [ ] RLHF / DPO

## References

See `ref/README.md` — papers, code references, video tutorials, illustrated guides.

## License

MIT

---

> No step is "understanding". Every step is probability and linear algebra.