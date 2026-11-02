"""
BPE Tokenizer - based on tiktoken (GPT-2)

Difference from char-level tokenizer:
- char-level: 1 char = 1 token, vocab ~127
- BPE: merge frequent pairs, vocab 50257

Benefit: "To be" from 5 tokens to 2, training efficiency doubles.
Downside: external dependency, large vocab.
"""
import tiktoken


class BPETokenizer:
    """
    BPE tokenizer based on tiktoken

    Uses GPT-2 pre-trained vocab (50257 tokens).
    No need to run merge algorithm, OpenAI already trained it.
    """

    def __init__(self, encoding="gpt2"):
        """
        encoding: GPT-2 uses 'gpt2', GPT-3.5/4 uses 'cl100k_base'
        """
        self.enc = tiktoken.get_encoding(encoding)

    def encode(self, text):
        """string -> token id list"""
        return self.enc.encode(text)

    def decode(self, tokens):
        """token id list -> string"""
        return self.enc.decode(tokens)

    @property
    def vocab_size(self):
        return self.enc.n_vocab


if __name__ == "__main__":
    tk = BPETokenizer()

    print(f"Vocab size: {tk.vocab_size}")

    tests = ["hello world", "你好世界", "To be or not to be"]  # keep CJK in test data

    for text in tests:
        tokens = tk.encode(text)
        decoded = tk.decode(tokens)
        print(f'"{text}" → {tokens} ({len(tokens)} tokens)')
        assert decoded == text
    print("OK - BPE Tokenizer works")