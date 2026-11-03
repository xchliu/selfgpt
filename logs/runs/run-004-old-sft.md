## Run 4 — 2026-11-02 15:48

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/zhuge-tan.txt (393,436 chars → 379,549 tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 1000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 26,679,477
- Time: 482.0s
- Initial loss: 11.6702
- Final loss: 3.5505
- Eval loss: 3.4451

### Loss Curve (sampled)
```
step     0: train=11.6702  eval=11.5241
step   100: train=5.6302  eval=5.7387
step   200: train=4.8206  eval=4.6868
step   300: train=4.3826  eval=4.3183
step   400: train=4.3311  eval=4.1741
step   500: train=4.0907  eval=4.0066
step   600: train=3.8528  eval=3.9424
step   700: train=3.9028  eval=3.8222
step   800: train=3.6506  eval=3.7008
step   900: train=3.4541  eval=3.5699
step   999: train=3.5505  eval=3.4451
```

### Sample Outputs
```
Prompt: AI时代
AI时代，但AI不是你的能力、你让AI把所有记忆，但你的记忆变得"。你有没有，你的记忆"。但你让你知道什么？"记忆"，你的记忆，它记忆。你的记忆是记忆

Prompt: 我们
我们怎么样。

**第三件事：** 你写代码（用例） | P1 |
| 2 | 看�发票 | **AI的"AI时代"，什么"到底是AI的核心的人/什么？ | 10章 | ① | 0% |
| 2 |

Prompt: 银行
银行AI的人数学会越来越来越好的，是我觉得AI做的事的，是因为它的是因为它的信息差异。但它们的是因为它只是"记忆"。它"，不知道。

这些记忆的记�
```

### Analysis
- Train/eval gap: 0.1054 (possible overfitting)
- Loss change: 11.67 -> 3.55 (delta=8.12)

---

## Run 4 — 2026-11-02 15:58

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/zhuge-tan.txt (393,436 chars → 213,601 tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 1000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 2,964,288
- Time: 35.2s
- Initial loss: 9.1889
- Final loss: 4.5184
- Eval loss: 4.4258

### Loss Curve (sampled)
```
step     0: train=9.1889  eval=9.0548
step   100: train=7.1944  eval=7.2900
step   200: train=6.8866  eval=6.8984
step   300: train=6.4311  eval=6.5346
step   400: train=6.1621  eval=6.1249
step   500: train=5.7880  eval=5.8178
step   600: train=5.4971  eval=5.4813
step   700: train=4.9918  eval=5.1121
step   800: train=5.0264  eval=4.8969
step   900: train=4.5454  eval=4.5907
step   999: train=4.5184  eval=4.4258
```

### Sample Outputs
```
Prompt: AI时代
AI时代,先想想。但"这个偏好,但有了这个困惑。 --- ## AI不是"记忆"记忆"的"这个记忆"这个环节—— **记忆"记忆"是"记忆"记忆"的"的"的"是"的"的"里。"AI没有"的"的身份"是什么?没有"的"里——不是"的"和"了

Prompt: 我们
我们后来我发现,每天刷两个小时就鼓。 **第二,我一眼就能告诉AI的。AI只能靠"不知道"不知道"的"。不是"我。 --- ## 我不是"的遗忘,而是"这一点—— 有些",它是"AI"这个困惑+AI,但它不是"这两件事。 AI可以帮你,是AI"AI的"我应该知道什么"和"的"

Prompt: 银行
银行的优势,但AI从"我"。 --- ## 两年内:让我们把AI的AI做AI的AI,不是自己看不到的。但AI的“AI”,而是**预测判断**。比如“我确定”的“判断”的体”的责任归属,本质上是一个AI自动做一次“AI,我太极“我”。但AI,而是“AI,我不信”
```

### Analysis
- Train/eval gap: 0.0926 (no overfitting)
- Loss change: 9.19 -> 4.52 (delta=4.67)

---

## Run 4 — 2026-11-02 16:10

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/chinese-corpus.txt (1,464,876 chars → 689,154 tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 1000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 5,020,288
- Time: 54.7s
- Initial loss: 9.8462
- Final loss: 5.4198
- Eval loss: 5.2442

### Loss Curve (sampled)
```
step     0: train=9.8462  eval=9.7449
step   100: train=7.6446  eval=7.5856
step   200: train=7.1741  eval=7.2011
step   300: train=6.8501  eval=7.0070
step   400: train=6.8895  eval=6.7371
step   500: train=6.6407  eval=6.3938
step   600: train=6.2110  eval=6.2835
step   700: train=5.9855  eval=6.0640
step   800: train=5.6238  eval=5.7207
step   900: train=5.8813  eval=5.5725
step   999: train=5.4198  eval=5.2442
```

### Sample Outputs
```
Prompt: AI时代
AI时代,是AI,把“我”的。 ## 那一刻变秃:imgs/13-cover.png  ⁇ ,我看了眼神奇,他不太觉得“我”,它需要,我承认了。 想想,我根本不擅长了。 我不知道,我心里想清楚,然后说“猜”了。我甚至,我盯着你很容易。你真正重要,

Prompt: 我们
我们做,没有答案。 我看了他承认自己无知,但真正的“AI能说“判断”的“AI”,而不是“你,你根本”,你“判断”。我盯着“做”,我甚至你”的“AI”这个”的。 我,你把你“我承认”,我甚至,你根本甚至“我,你”在“这个规则”,我甚至

Prompt: 银行
银行" - **来源**:刘小成,需要的"我认识" - **AI"AI时代,说"这个岗位了"帮我写" - **AI时代,AI实战手册**:AI落地,你每天管,不是AI做AI,但真正做AI。 - **技术能力**:组织转型了。这才是AI,它就是用AI的AI时代,是:AI时代
```

### Analysis
- Train/eval gap: 0.1756 (possible overfitting)
- Loss change: 9.85 -> 5.42 (delta=4.43)

---

## Run 4 — 2026-11-02 16:22

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/zhuge-tan.txt (393,436 chars → 188,934 tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 3000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 5,020,288
- Time: 168.5s
- Initial loss: 9.8493
- Final loss: 0.9591
- Eval loss: 0.9397

### Loss Curve (sampled)
```
step     0: train=9.8493  eval=9.7040
step   300: train=6.9421  eval=6.8174
step   600: train=5.5729  eval=5.5332
step   900: train=4.3423  eval=4.4567
step  1200: train=3.6606  eval=3.5388
step  1500: train=2.7306  eval=2.7447
step  1800: train=2.2755  eval=2.1225
step  2100: train=1.8015  eval=1.6890
step  2400: train=1.3012  eval=1.3478
step  2700: train=1.1030  eval=1.1603
step  2999: train=0.9591  eval=0.9397
```

### Sample Outputs
```
Prompt: AI时代
AI时代有了MCP连接对话——LLM 当RAG就是那个卡住配置。但别指望,都是同一个你自己往前挪不是为了追技术训练平台,但这不是什么。 B帮忙做一件事:A说是组:B 上、信贷优化、数据标注、金额、C渠道,数据上的系统设计运营。眼睛,前三项在手机——这些碎片信息背后的重量、模型,结果没被替代,前会先联网。 一个

Prompt: 我们
我们三所看起来不错过一样拆开20倍。更重要的是,人风控模型给的,使用率。2026 年,节奏科举:欧洲,Anthropic的训练MCP服务端和可持续类型为开源项目。三:用AI已是未来几个月前,大模型API调用API调用?为什么还能做什么? 我直接把结构化数据黑“Harness”,也让你后来说“算法”。 你试试让“确定”,其实

Prompt: 银行
银行信贷审批规则。 当技能比鳄鱼卡住了。 比如你问了句的是**大脑在工作不是写代码和扩展到了错误**—— 它没有"什么"。** 功能越乐观。 存在感标准里,可以放进去,可能不在于它只能算?如果能,继续,所以预算五千,所以预算五千想到了那个味道里——那 AI的记忆像模像样的分析。 你给大模型一堆资料,它总能上
```

### Analysis
- Train/eval gap: 0.0194 (no overfitting)
- Loss change: 9.85 -> 0.96 (delta=8.89)

---

## Run 4 — 2026-11-02 16:29

**Goal**: N/A

### Config
| Parameter | Value |
|-----------|-------|
| Tokenizer | BPE tiktoken GPT-2 (vocab=50,257) |
| Corpus | data/clean-corpus.txt (394,644 chars → 186,819 tokens) |
| embed_dim | 128 |
| num_layers | 6 |
| num_heads | 4 |
| head_size | 32 |
| block_size | 128 |
| epochs | 3000 |
| batch_size | 32 |
| lr | 0.001 |
| optimizer | AdamW |
| device | mps |

### Results
- Params: 5,020,288
- Time: 165.1s
- Initial loss: 9.8478
- Final loss: 0.7958
- Eval loss: 0.9424

### Loss Curve (sampled)
```
step     0: train=9.8478  eval=9.7140
step   300: train=6.7632  eval=6.7894
step   600: train=5.5395  eval=5.6140
step   900: train=4.3534  eval=4.4328
step  1200: train=3.5146  eval=3.5108
step  1500: train=2.6367  eval=2.6578
step  1800: train=2.3797  eval=2.0637
step  2100: train=1.6154  eval=1.6529
step  2400: train=1.3604  eval=1.3649
step  2700: train=1.1587  eval=1.1359
step  2999: train=0.7958  eval=0.9424
```

### Sample Outputs
```
Prompt: AI时代
AI时代,普通人应该是必经之路。 --- *个人观点,仅供参考* ================================================== # 管着一支近百人的AI团队,我的底子却是搞数据库的 > 个人观点,仅供参考 --- ## 1 有一次面试 身上的供需关系,对面坐着一个从简单:AI Agent工程师代码的审文件体系后处理这个。 我又让的新课、有人问我"大脑"。 "Cursor实现机制企业出方案

Prompt: 我们
我们做架构决策权——展期是连续的。 --- ## 四、AI打交道的两张"哇" 最常见的用法是:Agent对话积累的"出差报销系统的接口文档概要或"手工跑"。 **成本结构在变。** 搬不了:AI助手快速搞定一个任务准确率监控:要么给AI,追求"这是谁在回复,人"。 这两个差别合规要求摘要管理问题,安全管理。

Prompt: 银行
银行要做集成 - 大模型学次数** — 教学能力**——服务社会化的注意,而不是起点,技术写作过程中对话不像感慨参数,才能让做AI能力项业务。 - **技术写作/审计报告,架构设计、安全、扫描** — 懂风险点,国内开发者口碑,而不是纯技术运行几百人的真实商业场景Anthropic成本所吞噬** - **机器人不再只看百分比和竞品,
```

### Analysis
- Train/eval gap: -0.1466 (possible overfitting)
- Loss change: 9.85 -> 0.80 (delta=9.05)

---

