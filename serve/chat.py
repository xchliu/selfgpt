"""简易 Chat 模块 — 多轮对话封装在 generate 之上"""
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.gpt import GPT
from serve.generate import generate


class ChatSession:
    """
    多轮对话管理

    本质就是把每一轮的 user/assistant 消息拼成一段文本，
    喂给 generate() 继续生成。模型看到的不是"对话"，
    是一段连续的文本，格式如下：

      <user>你好</user><assistant>你好，有什么可以帮你的？</assistant><user>今天天气怎么样</user><assistant>

    然后让模型接着 <assistant> 后面生成，直到遇到 </assistant> 停下。
    """

    def __init__(self, model, tokenizer, block_size=128, temperature=0.8, top_k=10):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.temperature = temperature
        self.top_k = top_k
        self.history = []  # [("user", "你好"), ("assistant", "你好呀"), ...]

    def _build_prompt(self):
        """把对话历史拼成一段文本，末尾留 <assistant> 等模型续写"""
        text = ""
        for role, msg in self.history:
            text += f"<{role}>{msg}</{role}>"
        text += "<assistant>"
        return text

    def chat(self, user_input):
        """发一句话，拿到模型回复"""
        # 1. 存入用户消息
        self.history.append(("user", user_input))

        # 2. 拼接历史 → 文本 → token ids
        prompt_text = self._build_prompt()
        prompt_ids = self.tokenizer.encode(prompt_text)

        # 3. 超过窗口就截断前面（只留最近的对话）
        if len(prompt_ids) > self.block_size:
            prompt_ids = prompt_ids[-self.block_size:]

        # 4. 生成
        max_new = 100
        result_ids = generate(
            self.model,
            prompt_ids,
            max_new_tokens=max_new,
            temperature=self.temperature,
            top_k=self.top_k
        )

        # 5. 取新生成的部分，decode 成文本
        new_ids = result_ids[len(prompt_ids):]
        response = self.tokenizer.decode(new_ids)

        # 6. 遇到 </assistant> 就截断（停止标记）
        if "</assistant>" in response:
            response = response.split("</assistant>")[0]

        # 7. 存入助手回复
        self.history.append(("assistant", response))
        return response

    def reset(self):
        """清空对话历史"""
        self.history = []

    def show_history(self):
        """打印完整对话历史"""
        for role, msg in self.history:
            label = "🧑 坦哥" if role == "user" else "🤖 GPT"
            print(f"{label}: {msg}")


# ─── BPE Tokenizer ────────────────────────────────────────
from model.bpe_tokenizer import BPETokenizer

class SimpleTokenizer:
    """使用 BPE 分词器"""
    def __init__(self, encoding="gpt2"):
        self._tk = BPETokenizer(encoding=encoding)
        self.vocab = None  # BPE 不需要手动维护词表

    def encode(self, text):
        return self._tk.encode(text)

    def decode(self, ids):
        return self._tk.decode(ids)

    @property
    def vocab_size(self):
        return self._tk.vocab_size


# ─── 交互式聊天入口 ──────────────────────────────────────────
if __name__ == '__main__':
    # 词表从 data/vocab.txt 加载
    tokenizer = SimpleTokenizer()

    # 优先加载训练好的模型，没有就用随机初始化
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        cfg = ckpt['config']
        model = GPT(
            vocab_size=tokenizer.vocab_size,
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            head_size=cfg['head_size'],
            num_layers=cfg['num_layers'],
            block_size=cfg['block_size']
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"已加载训练好的模型: {ckpt_path}")
    else:
        model = GPT(
            vocab_size=tokenizer.vocab_size,
            embed_dim=64,
            num_heads=4,
            head_size=16,
            num_layers=4,
            block_size=128
        )
        print("未找到 model.pt，使用随机模型（输出乱码）")

    session = ChatSession(model, tokenizer, block_size=128, temperature=0.8, top_k=10)

    print("=" * 50)
    print("  SelfGPT 对话")
    print("  输入 quit 退出，输入 reset 清空历史")
    print("=" * 50)
    print()

    while True:
        user_input = input("🧑 你: ").strip()
        if user_input == 'quit':
            print("再见 👋")
            break
        if user_input == 'reset':
            session.reset()
            print("（对话已清空）\n")
            continue
        if not user_input:
            continue

        response = session.chat(user_input)
        print(f"🤖 GPT: {response}\n")