"""
Unified Tokenizer — single interface, swappable backends.

Backends:
  - "char":       Character-level tokenizer (from vocab.txt)
  - "gpt2":       tiktoken BPE, GPT-2 vocab (50,257)
  - "cl100k_base": tiktoken BPE, GPT-3.5/4 vocab (100,277)
  - "o200k_base":  tiktoken BPE, GPT-4o vocab (200,019)

Usage:
    tk = Tokenizer("cl100k_base")
    ids = tk.encode("hello")
    text = tk.decode(ids)
    size = tk.vocab_size

All backends expose the same interface. Switching tokenizer
does not affect train.py or chat.py.
"""
import os
from abc import ABC, abstractmethod


class TokenizerBase(ABC):
    """Abstract base — all backends implement this."""

    @abstractmethod
    def encode(self, text):
        """string -> list[int]"""
        pass

    @abstractmethod
    def decode(self, tokens):
        """list[int] -> string"""
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """int"""
        pass


class CharTokenizer(TokenizerBase):
    """Character-level tokenizer. One char = one token."""

    def __init__(self, vocab_path=None, text=""):
        if vocab_path:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                chars = [line.rstrip('\n') for line in f]
            if '\n' not in chars:
                chars.append('\n')
            self.vocab = sorted(chars)
        else:
            self.vocab = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.id_to_char[i] for i in tokens])

    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for ch in self.vocab:
                f.write(ch + '\n')

    @property
    def vocab_size(self):
        return len(self.vocab)


class TiktokenTokenizer(TokenizerBase):
    """tiktoken BPE tokenizer. Pre-trained, no training needed."""

    def __init__(self, encoding="gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding)
        self.encoding_name = encoding

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, tokens):
        return self.enc.decode(tokens)

    @property
    def vocab_size(self):
        return self.enc.n_vocab


class SentencePieceTokenizer(TokenizerBase):
    """SentencePiece BPE tokenizer. Trained on custom corpus."""

    def __init__(self, model_path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, tokens):
        return self.sp.decode(tokens)

    @property
    def vocab_size(self):
        return self.sp.vocab_size()


# --- Supported backends ---
_TIKTOKEN_BACKENDS = {"gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"}


class Tokenizer:
    """
    Unified tokenizer entry point.

    Args:
        backend: "char" | "gpt2" | "cl100k_base" | "o200k_base" | "sentencepiece"
        **kwargs: passed to backend (e.g. vocab_path for "char", model_path for "sentencepiece")
    """

    def __init__(self, backend="cl100k_base", **kwargs):
        if backend == "char":
            self._impl = CharTokenizer(**kwargs)
        elif backend in _TIKTOKEN_BACKENDS:
            self._impl = TiktokenTokenizer(encoding=backend)
        elif backend == "sentencepiece":
            self._impl = SentencePieceTokenizer(**kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Supported: char, sentencepiece, {', '.join(_TIKTOKEN_BACKENDS)}"
            )
        self.backend_name = backend

    def encode(self, text):
        return self._impl.encode(text)

    def decode(self, tokens):
        return self._impl.decode(tokens)

    @property
    def vocab_size(self):
        return self._impl.vocab_size

    @property
    def impl(self):
        """Access underlying backend if needed."""
        return self._impl


if __name__ == "__main__":
    # Test all backends
    tests = ["hello world", "你好世界", "To be or not to be"]

    for backend in ["char", "gpt2", "cl100k_base"]:
        if backend == "char":
            vocab_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vocab.txt')
            tk = Tokenizer("char", vocab_path=vocab_path)
        else:
            tk = Tokenizer(backend)

        print(f"\n=== {backend} (vocab={tk.vocab_size}) ===")
        for text in tests:
            try:
                tokens = tk.encode(text)
                decoded = tk.decode(tokens)
                ok = "OK" if decoded == text else "MISMATCH"
                print(f'  "{text}" -> {len(tokens)} tokens [{ok}]')
            except KeyError:
                print(f'  "{text}" -> KeyError (char not in vocab)')
    print("\nAll backends tested.")