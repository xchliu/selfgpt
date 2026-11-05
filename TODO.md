# SelfGPT TODO

## 1. 数据
- [ ] 加大语料量（当前38万字符，目标100万+）
- [ ] 找中文基础语料（维基百科/新闻/文学作品）补充预训练
- [ ] 数据加工增强：按主题分类、质量评分、领域配比
- [ ] 公众号文章 + 外部中文语料的混合配比实验

## 2. 分词器
- [ ] SentencePiece 词表大小实验（16K vs 32K vs 64K）
- [ ] 对比 cl100k_base vs SentencePiece 在相同语料下的表现
- [ ] 评估不同词表大小对中文 token 效率的影响

## 3. 模型架构
- [ ] 实现多头注意力（MultiHeadAttention），替换单头
- [ ] 加 RoPE 位置编码，替学习式位置编码
- [ ] 实验不同 embed_dim（128 vs 256 vs 512）
- [ ] 实验不同 num_layers（6 vs 8 vs 12）
- [ ] 加 dropout 防过拟合

## 4. 训练
- [ ] 控制过拟合（当前 train 0.80 vs eval 0.94，差值 0.14）
- [ ] 学习率调度（cosine decay）
- [ ] 梯度累积（模拟更大 batch_size）
- [ ] 早停机制（eval loss 连续 N 轮不降就停）
- [ ] 训练/验证集分离（当前全部用于训练）

## 5. RLHF
- [ ] 训练奖励模型（Reward Model）：给回答打分
- [ ] 人类偏好数据收集：同一问题多个回答，人工排序
- [ ] PPO 训练：用奖励模型当信号，强化高分回答
- [ ] DPO 作为简化方案（不需要奖励模型，直接偏好对比）
- [ ] 安全对齐：拒绝有害请求、避免幻觉
- [ ] 三个阶段的完整路线：预训练(说通顺) → SFT(说对话) → RLHF(说好话)

## 6. 生成
- [ ] 多种采样策略对比（top-k vs top-p vs temperature）
- [ ] 生成质量评估方法（perplexity / 人工评分）
- [ ] 批量生成 + 人工筛选最佳 checkpoint

## 7. Chat
- [ ] 对话模板设计（ChatML 格式：system/user/assistant）
- [ ] 系统提示词（system prompt 定义模型人设）
- [ ] 停止符可靠性（当前 </assistant> 截断不够健壮）
- [ ] 上下文窗口管理（超 block_size 时智能截断，不丢关键消息）
- [ ] 多轮对话记忆策略（摘要压缩 vs 滑动窗口）
- [ ] 流式输出（逐字打印，不等全部生成完）
- [ ] 对话历史持久化（保存/加载对话记录）
- [ ] KV Cache 推理优化（避免重复计算前文）

## 8. 工程化
- [ ] 训练日志修复（多次并行训练导致日志覆盖问题）
- [ ] config 支持命令行参数覆盖（不改代码调参）
- [ ] 模型版本管理（不同配方存不同 model.pt）
- [ ] README 更新（新架构、新分词器、新流程）

## 10. 待研究理论/技术点

### 注意力
- [ ] 实现真正的多头注意力（当前4个head各96维，但没拆开跑）
- [ ] 对比单头 vs 多头效果（多头应能学到不同模式：语法/语义/共指）
- [ ] Flash Attention：减少 HBM 读写，提速训练
- [ ] KV Cache：推理时不重复计算前文，加速生成

### 位置编码
- [ ] RoPE（旋转位置编码）：支持更长上下文，不限于固定 block_size
- [ ] ALiBi：外推性好，训练时没见过的长度也能处理
- [ ] 对比可学习位置编码 vs RoPE vs ALiBi 的效果

### 训练优化
- [ ] Warmup：前 N 步 lr 从 0 线性升到目标值，防止初始梯度爆炸
- [ ] Cosine decay：训练后期逐渐降低 lr，精细收敛
- [ ] 梯度裁剪（gradient clipping）：防止梯度爆炸的另一种手段
- [ ] 梯度累积：小显存模拟大 batch_size
- [ ] Dropout：在 attention 和 feedforward 中加 dropout 防过拟合
- [ ] 训练/验证集分离（当前全部用于训练）
- [ ] 早停机制（eval loss 连续 N 轮不降就停）

### 损失与评估
- [ ] Perplexity（困惑度）：比 loss 更直观的评估指标
- [ ] 困惑度 = exp(loss)，表示模型对正确 token 的"困惑程度"
- [ ] 训练曲线分析：loss 震荡说明什么？curve 形状如何判断健康度

### 激活函数对比
- [ ] GELU vs ReLU vs SwiGLU 实验对比
- [ ] SwiGLU：Llama 用的，FeedForward 的变体
- [ ] 为什么现代大模型从 GELU 转向 SwiGLU

### 架构演进
- [ ] 当前是 pre-norm decoder-only GPT，对比其他架构
- [ ] Mamba / SSM：替代 attention 的新架构
- [ ] MoE（混合专家）：稀疏激活，大模型用
- [ ] 分组查询注意力（GQA）：Llama 2 用的

### 分词
- [ ] BPE 合并算法的完整实现（当前直接用 SentencePiece）
- [ ] 词表大小对训练的影响（16K vs 32K）
- [ ] 子词分割对中文的影响（单字 vs 词组）
- [ ] 公众号草稿箱：补全标题、删旧草稿、检查排版
- [ ] 系列文章规划（分词器篇、训练篇、数据加工篇）
- [ ] 训练日志分析文章