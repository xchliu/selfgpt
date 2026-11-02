import os


class Tokenizer:
    """
    字符级 Tokenizer

    词表从文件加载，不再硬编码在代码里。
    vocab.txt 每行一个字符，启动时读取建映射表。
    """

    def __init__(self, vocab_path=None, text=""):
        """
        两种初始化方式：
        1. vocab_path：从文件加载词表（推荐）
        2. text：从文本构建词表（兼容旧代码）
        """
        if vocab_path:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                # 每行一个字符，去掉换行符本身
                chars = [line.rstrip('\n') for line in f]
            self.vocab = sorted(chars)
        else:
            self.vocab = sorted(list(set(text)))

        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        """字符串 → token id 列表"""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, tokens):
        """token id 列表 → 字符串"""
        return ''.join([self.id_to_char[i] for i in tokens])

    def save_vocab(self, path):
        """把词表存成文件，每行一个字符"""
        with open(path, 'w', encoding='utf-8') as f:
            for ch in self.vocab:
                f.write(ch + '\n')

    @property
    def vocab_size(self):
        return len(self.vocab)


if __name__ == "__main__":
    # 从文件加载词表
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vocab.txt')
    t = Tokenizer(vocab_path=vocab_path)

    print(f"词表大小: {t.vocab_size}")
    print(f"词表内容: {t.vocab[:20]}... (共 {t.vocab_size} 个)")

    # 测试 encode/decode
    text = "你好"
    tokens = t.encode(text)
    decoded = t.decode(tokens)
    assert decoded == text, f"encode/decode 不一致: {decoded} != {text}"
    print(f"encode('{text}') = {tokens}")
    print(f"decode({tokens}) = '{decoded}'")
    print("✓ Tokenizer 从文件加载词表跑通")