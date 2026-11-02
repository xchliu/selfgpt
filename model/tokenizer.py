import os


class Tokenizer:
    """
    Character-level Tokenizer

    Vocab loaded from file, not hardcoded.
    One char per line in vocab.txt.
    """

    def __init__(self, vocab_path=None, text=""):
        """
        Two init modes:
        1. vocab_path: load from file
        2. text: build from text (legacy)
        """
        if vocab_path:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                # one char per line, strip newline
                chars = [line.rstrip('\n') for line in f]
            # newline cannot be stored as a line, add manually
            if '\n' not in chars:
                chars.append('\n')
            self.vocab = sorted(chars)
        else:
            self.vocab = sorted(list(set(text)))

        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        """string -> token id list"""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, tokens):
        """token id list -> string"""
        return ''.join([self.id_to_char[i] for i in tokens])

    def save_vocab(self, path):
        """Save vocab to file, one char per line"""
        with open(path, 'w', encoding='utf-8') as f:
            for ch in self.vocab:
                f.write(ch + '\n')

    @property
    def vocab_size(self):
        return len(self.vocab)


if __name__ == "__main__":
    # load vocab from file
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vocab.txt')
    t = Tokenizer(vocab_path=vocab_path)

print(f"Vocab size: {t.vocab_size}")
print(f"Vocab: {t.vocab[:20]}... ({t.vocab_size} total)")

    # test encode/decode
    text = "hello"
    tokens = t.encode(text)
    decoded = t.decode(tokens)
assert decoded == text, f"mismatch: {decoded} != {text}"
print(f"encode(text) = {tokens}")
print(f'decode({tokens}) = decoded')
print("OK - Tokenizer works")