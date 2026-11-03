# Run 003 — pretrain

**Date**: 2026-11-03 16:51
**Goal**: N/A
**Status**: running

## Config

| Parameter | Value |
|-----------|-------|
| Type | pretrain |
| Tokenizer | sentencepiece |
| Vocab size | 16000 |
| Corpus | data/mixed-corpus.txt |
| Corpus size | 5,299,842 chars -> 2,451,326 tokens |
| embed_dim | 512 |
| num_layers | 12 |
| num_heads | 4 |
| head_size | 64 |
| block_size | 128 |
| epochs | 10000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

## Progress

```
step 1251/10000 (12.5%) | train: 5.2107 | eval: 5.4108 | 345s

```
