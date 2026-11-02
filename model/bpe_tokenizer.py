"""
BPE 分词器 — 基于 tiktoken (GPT-2)

和字符级 tokenizer 的区别：
- 字符级：一个字符一个 token，词表 127
- BPE：高频字符对合并，词表 50257

好处："To be" 从 5 个 token 变成 2 个，训练效率翻倍。
坏处：依赖外部库，词表大。
"""
import tiktoken


class BPETokenizer:
    """
    基于 tiktoken 的 BPE 分词器

    使用 GPT-2 的预训练词表（50257 tokens）。
    不需要自己跑合并算法，OpenAI 已经训好了。
    """

    def __init__(self, encoding="gpt2"):
        """
        encoding: GPT-2 用 "gpt2"，GPT-3.5/4 用 "cl100k_base"
        """
        self.enc = tiktoken.get_encoding(encoding)

    def encode(self, text):
        """字符串 → token id 列表"""
        return self.enc.encode(text)

    def decode(self, tokens):
        """token id 列表 → 字符串"""
        return self.enc.decode(tokens)

    @property
    def vocab_size(self):
        return self.enc.n_vocab


if __name__ == "__main__":
    tk = BPETokenizer()

    print(f"词表大小: {tk.vocab_size}")

    tests = ["hello world", "你好世界", "To be or not to be"]

    for text in tests:
        tokens = tk.encode(text)
        decoded = tk.decode(tokens)
        print(f'"{text}" → {tokens} ({len(tokens)} tokens)')
        assert decoded == text
    print("✓ BPE Tokenizer 跑通")