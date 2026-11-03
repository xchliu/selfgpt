"""Chat module — multi-turn dialogue wrapper on top of generate"""
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.gpt import GPT
from serve.generate import generate


class ChatSession:
    """
    Multi-turn dialogue session

    Builds a text prompt from message history, feeds to generate().
    The model sees continuous text, not structured dialogue:

      <user>hi</user><assistant>hello, how can I help?</assistant><user>how is the weather?</user><assistant>

    Model generates after <assistant> until </assistant> is reached.
    """

    def __init__(self, model, tokenizer, block_size=128, temperature=0.8, top_k=10):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.temperature = temperature
        self.top_k = top_k
        self.history = []  # [("user", "hi"), ("assistant", "hello"), ...]

    def _build_prompt(self):
        """Build prompt text from history, trailing <assistant> for completion"""
        text = ""
        for role, msg in self.history:
            text += f"<{role}>{msg}</{role}>"
        text += "<assistant>"
        return text

    def chat(self, user_input):
        """Send a message, get model reply"""
        # 1. store user message
        self.history.append(("user", user_input))

        # 2. build prompt -> encode -> token ids
        prompt_text = self._build_prompt()
        prompt_ids = self.tokenizer.encode(prompt_text)

        # 3. truncate if exceeds block_size
        if len(prompt_ids) > self.block_size:
            prompt_ids = prompt_ids[-self.block_size:]

        # 4. generate
        max_new = 100
        result_ids = generate(
            self.model,
            prompt_ids,
            max_new_tokens=max_new,
            temperature=self.temperature,
            top_k=self.top_k
        )

        # 5. decode new tokens
        new_ids = result_ids[len(prompt_ids):]
        response = self.tokenizer.decode(new_ids)

        # 6. truncate at stop token
        if "</assistant>" in response:
            response = response.split("</assistant>")[0]

        # 7. store assistant reply
        self.history.append(("assistant", response))
        return response

    def reset(self):
        """Clear dialogue history"""
        self.history = []

    def show_history(self):
        """Print full dialogue history"""
        for role, msg in self.history:
            label = "🧑 You" if role == "user" else "🤖 GPT"
            print(f"{label}: {msg}")


# --- Tokenizer ---
from model.tokenizer import Tokenizer as UnifiedTokenizer

class SimpleTokenizer:
    """Unified tokenizer wrapper"""
    def __init__(self, encoding="cl100k_base", model_path=None):
        if encoding == "sentencepiece":
            self._tk = UnifiedTokenizer("sentencepiece", model_path=model_path)
        else:
            self._tk = UnifiedTokenizer(encoding)

    def encode(self, text):
        return self._tk.encode(text)

    def decode(self, ids):
        return self._tk.decode(ids)

    @property
    def vocab_size(self):
        return self._tk.vocab_size


# --- Interactive Chat Entry ---
if __name__ == '__main__':
    # load vocab from data/vocab.txt
    tokenizer = SimpleTokenizer()

    # load trained model if available, else random init
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model-sft.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        cfg = ckpt['config']
        encoding = ckpt.get('encoding', 'gpt2')
        if encoding == "sentencepiece":
            sp_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mixed-sp.model')
            tokenizer = SimpleTokenizer(encoding="sentencepiece", model_path=sp_path)
        else:
            tokenizer = SimpleTokenizer(encoding=encoding)
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
        print(f"Loaded trained model: {ckpt_path}")
    else:
        model = GPT(
            vocab_size=tokenizer.vocab_size,
            embed_dim=64,
            num_heads=4,
            head_size=16,
            num_layers=4,
            block_size=128
        )
        print("model.pt not found, using random model (output will be gibberish)")

    session = ChatSession(model, tokenizer, block_size=128, temperature=0.8, top_k=10)

    print("=" * 50)
    print("  SelfGPT Chat")
    print("  Type quit to exit, reset to clear history")
    print("=" * 50)
    print()

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input == 'quit':
            print("Bye 👋")
            break
        if user_input == 'reset':
            session.reset()
            print("(history cleared)\n")
            continue
        if not user_input:
            continue

        response = session.chat(user_input)
        print(f"🤖 SelfGPT: {response}\n")