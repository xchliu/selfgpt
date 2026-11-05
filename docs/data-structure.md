# data 目录说明

```
data/
├── raw/                    # 原始下载文件（不进 git）
│   ├── wikipedia-zh-cn-20260501.json   (2.2G, 149万篇)
│   ├── pretrain_t2t_mini.jsonl         (1.2G, 127万行)
│   └── sft_t2t_mini.jsonl              (1.6G, 90万条)
├── corpus/                 # 预训练语料
│   ├── wiki-clean.txt      (1000万字符, 维基百科提取)
│   ├── clean-corpus.txt    (97万字符, 公众号文章)
│   ├── mixed-corpus.txt    (530万字符, 旧合并语料)
│   └── pretrain-clean.txt  (500万字符, t2t_mini清洗)
├── sft/                    # SFT 数据
│   ├── sft-clean.json      (4332条, 最终训练用)
│   ├── sft-auto.json       (2012条, 公众号自动抽取)
│   └── sft-final.json      (10012条, 合并中间产物)
├── tokenizer/             # 分词器
│   ├── mixed-sp.model     (SentencePiece 16K, 当前)
│   └── mixed-sp.vocab
└── archive/               # 历史归档
    ├── chinese-corpus.txt
    ├── chinese-corpus-sp.model/vocab
    ├── zhuge-tan.txt
    ├── zhuge-tan-sp.model/vocab
    ├── tinyshakespeare.txt
    ├── train.txt
    ├── vocab.txt
    └── sft-data.json
```

## 路径对应关系

| 旧路径 | 新路径 |
|--------|--------|
| `data/wikipedia-zh-cn-20260501.json` | `data/raw/wikipedia-zh-cn-20260501.json` |
| `data/pretrain_t2t_mini.jsonl` | `data/raw/pretrain_t2t_mini.jsonl` |
| `data/sft_t2t_mini.jsonl` | `data/raw/sft_t2t_mini.jsonl` |
| `data/wiki-clean.txt` | `data/corpus/wiki-clean.txt` |
| `data/clean-corpus.txt` | `data/corpus/clean-corpus.txt` |
| `data/mixed-corpus.txt` | `data/corpus/mixed-corpus.txt` |
| `data/pretrain-clean.txt` | `data/corpus/pretrain-clean.txt` |
| `data/sft/sft-clean.json` | `data/sft/sft-clean.json` (不变) |
| `data/mixed-sp.model` | `data/tokenizer/mixed-sp.model` |
| `data/mixed-sp.vocab` | `data/tokenizer/mixed-sp.vocab` |