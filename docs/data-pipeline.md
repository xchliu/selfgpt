# 数据处理流程

## 总览

```
原始数据
    ├── 公众号文章 (210篇 .md)
    └── t2t_mini 语料 (1.2GB .jsonl)
         │
         ├── 预训练流水线
         │       ├── process.py → clean-corpus.txt
         │       ├── preprocess_pretrain.py → pretrain-clean.txt
         │       ├── 合并 → mixed-corpus.txt
         │       ├── train_tokenizer.py → mixed-sp.model
         │       └── train.py → model.pt
         │
         └── SFT 流水线
                 ├── gen_sft.py → sft-auto.json
                 ├── preprocess_sft.py → sft-final.json
                 ├── clean_sft.py → sft-clean.json
                 └── sft.py → model-sft.pt
```

---

## 预训练流水线

### Step 1: 清洗公众号文章

```bash
python data_pipeline/process.py \
  --input /path/to/gzh/articles \
  --output data/clean-corpus.txt \
  --min_length 200
```

**输入:** 210 篇 markdown 文章
**处理:**
- 去除 YAML frontmatter
- 去除 MEDIA: 图片引用
- 去除 HTML 标签
- 去重（内容哈希）
- 质量过滤（最小长度 200 字符，平均行长 > 15）
**输出:** `data/clean-corpus.txt` (38万字符)

### Step 2: 清洗基础语料

```bash
python data_pipeline/preprocess_pretrain.py \
  --input data/pretrain_t2t_mini.jsonl \
  --output data/pretrain-clean.txt \
  --max_chars 5000000 \
  --min_length 50
```

**输入:** `pretrain_t2t_mini.jsonl` (1.2GB, 127万行)
**处理:**
- 从 JSONL 提取 text 字段
- 去除开头提示词
- 按问句标记拆分多轮问答
- 质量过滤
- 截取 500 万字符
**输出:** `data/pretrain-clean.txt` (500万字符)

### Step 3: 合并语料

```python
with open('data/pretrain-clean.txt') as f: pre = f.read()
with open('data/clean-corpus.txt') as f: gzh = f.read()
mixed = pre + gzh
# 安全截断到 530 万字符
```

**输出:** `data/mixed-corpus.txt` (530万字符)

### Step 4: 训练分词器

```bash
python train/train_tokenizer.py \
  --corpus data/mixed-corpus.txt \
  --vocab_size 16000 \
  --prefix data/mixed-sp
```

**输出:** `data/mixed-sp.model` + `data/mixed-sp.vocab`

### Step 5: 预训练

```bash
python train/train.py
```

**配置:** 在 `train/train.py` 的 CONFIG 字典中设置
**输出:** `checkpoints/model.pt`
**日志:** `logs/training-log.md` (台账) + `logs/runs/run-NNN-pretrain.md` (详细)

---

## SFT 流水线

### Step 1: 从公众号文章生成问答

```bash
python data_pipeline/gen_sft.py
```

**输入:** `data/clean-corpus.txt`
**策略:** 13种（详见数据配方文档）
**输出:** `data/sft/sft-auto.json` (2,012条)

### Step 2: 从外部语料提取对话

```bash
python data_pipeline/preprocess_sft.py
```

**输入:** `data/sft/sft_t2t_mini.jsonl` (90万条)
**处理:**
- 去除 reasoning_content 字段
- 质量过滤（轮数 2-10，内容长度 50-1000）
- 采样 8000 条
- 与 sft-auto.json 合并
**输出:** `data/sft/sft-final.json` (10,012条)

### Step 3: 清洗 SFT 数据

```bash
python data_pipeline/clean_sft.py
```

**输入:** `data/sft/sft-final.json` (10,012条)
**处理:**
- 问题质量过滤（长度、开头词、中文占比）
- 回答质量过滤（长度、中文占比、标点、占位符）
- 标点规范化（全角统一、去重复）
- 去噪（HTML、markdown、控制字符）
- 相关性检查（关键词重叠）
- 按问题去重
**输出:** `data/sft/sft-clean.json` (4,332条)

### Step 4: SFT 训练

```bash
python train/sft.py
```

**配置:** 在 `train/sft.py` 的 SFT_CONFIG 字典中设置
**输入:** `checkpoints/model.pt` (预训练模型) + `data/sft/sft-clean.json`
**输出:** `checkpoints/model-sft.pt`
**日志:** 台账 + `logs/runs/run-NNN-sft.md`

---

## 文件清单

### 数据文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `data/clean-corpus.txt` | 38万字符 | 清洗后的公众号文章 |
| `data/pretrain-clean.txt` | 500万字符 | 清洗后的基础语料 |
| `data/mixed-corpus.txt` | 530万字符 | 预训练合并语料 |
| `data/mixed-sp.model` | ~500KB | SentencePiece 分词器 |
| `data/sft/sft-data.json` | 4.7KB | 手写问答对 (12条) |
| `data/sft/sft-auto.json` | 1.4MB | 自动抽取问答 (2,012条) |
| `data/sft/sft-final.json` | - | 合并问答 (10,012条) |
| `data/sft/sft-clean.json` | - | 清洗后问答 (4,332条) |
| `checkpoints/model.pt` | ~170MB | 预训练模型 |
| `checkpoints/model-sft.pt` | ~170MB | SFT 模型 |

### 脚本文件

| 脚本 | 功能 |
|------|------|
| `data_pipeline/process.py` | 清洗公众号 markdown 文章 |
| `data_pipeline/preprocess_pretrain.py` | 清洗 JSONL 基础语料 |
| `data_pipeline/gen_sft.py` | 从文章生成问答 (13策略) |
| `data_pipeline/preprocess_sft.py` | 从外部语料提取对话 |
| `data_pipeline/clean_sft.py` | SFT 数据质量清洗 |
| `train/train_tokenizer.py` | 训练 SentencePiece 分词器 |
| `train/train.py` | 预训练 |
| `train/sft.py` | SFT 微调 |
| `train/log_utils.py` | 三阶段训练日志 |