"""
Train a SentencePiece BPE tokenizer on custom corpus.

Usage:
    python train_tokenizer.py --corpus data/zhuge-tan.txt --vocab_size 16000

Output:
    data/zhuge-tan-sp.model
    data/zhuge-tan-sp.vocab

Tune vocab_size to balance token efficiency vs model size:
    8000  — small, fast, but limited expressiveness
    16000 — balanced for Chinese articles
    32000 — LLaMA-level, needs more data
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def train_tokenizer(corpus_path, vocab_size=16000, model_prefix=None, character_coverage=0.9995):
    """
    Train SentencePiece BPE model.

    Args:
        corpus_path: path to training text
        vocab_size: target vocabulary size
        model_prefix: output path prefix (without extension)
        character_coverage: fraction of characters covered by the model
    """
    import sentencepiece as spm

    if model_prefix is None:
        model_prefix = corpus_path.replace('.txt', '-sp')

    print(f"Corpus: {corpus_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Output: {model_prefix}.model / {model_prefix}.vocab")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=character_coverage,
        input_sentence_size=10000,
        shuffle_input_sentence=True,
    )

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    print(f"\nDone. Vocab size: {sp.vocab_size()}")

    # Test samples
    tests = ["AI时代", "银行", "大模型", "坦哥说", "人工智能"]
    print("\n=== Test ===")
    for text in tests:
        tokens = sp.encode(text)
        pieces = [sp.id_to_piece(t) for t in tokens]
        decoded = sp.decode(tokens)
        ok = "OK" if decoded == text else "MISMATCH"
        print(f'  "{text}" -> {len(tokens)} tokens {pieces} [{ok}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer')
    parser.add_argument('--corpus', default='data/zhuge-tan.txt', help='corpus path')
    parser.add_argument('--vocab_size', type=int, default=16000, help='target vocab size')
    parser.add_argument('--prefix', default=None, help='output prefix (default: corpus-sp)')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = os.path.join(base, args.corpus) if not os.path.isabs(args.corpus) else args.corpus
    prefix = args.prefix
    if prefix is None:
        prefix = corpus.replace('.txt', '-sp')

    train_tokenizer(corpus, vocab_size=args.vocab_size, model_prefix=prefix)