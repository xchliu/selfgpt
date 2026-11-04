# 训练日志说明

## 结构

```
logs/
├── training-log.md          # 总台账（每次训练一行摘要）
└── runs/
    ├── run-001-pretrain.md  # 每次训练的详细日志
    ├── run-002-sft.md
    └── ...
```

## 三阶段日志机制

每次训练自动经历三个阶段：

### Phase 1: start_log() — 训练启动时

- 台账写入一行，状态 = `running`
- 详细日志写入 Config（所有参数）
- 训练还没开始，loss 显示 0

### Phase 2: update_progress() — 训练过程中

- 每次 eval 时更新详细日志的 Progress 部分
- 包含：当前步数、百分比、train loss、eval loss、已用时间
- 台账不更新（只写一次）

### Phase 3: finish_log() — 训练结束时

- 台账更新：状态 = `done`，填入最终 loss 和时间
- 详细日志更新：写入 Results（参数量、loss 变化、过拟合分析）、Loss Curve（采样曲线）、Sample Outputs（生成样本）

## 台账格式

```markdown
| Run | Date       | Type     | Tokenizer    | Vocab | Corpus   | Params    | Epochs | Time  | Init Loss | Final Loss | Status |
|-----|------------|----------|--------------|-------|----------|-----------|--------|-------|-----------|------------|--------|
| 1   | 2026-11-03 | pretrain | sentencepiece| 16000 | 394,644  | 5,020,288 | 3000   | 184s  | 9.82      | 0.95       | done   |
| 2   | 2026-11-03 | sft      | sentencepiece| 16000 | 28,347   | 5,020,288 | 150    | 5s    | 2.35      | 0.61       | done   |
```

## 详细日志格式

```markdown
# Run 003 — pretrain

**Date**: 2026-11-03 16:51
**Goal**: N/A
**Status**: done

## Config
（所有训练参数表格）

## Results
- Params: 43,260,544
- Time: 2668.0s
- Initial loss: 9.8607
- Final loss: 1.3019
- Eval loss: 1.1877
- Train/eval gap: 0.1142 (possible overfitting)
- Loss change: 9.86 -> 1.30 (delta=8.56)

## Loss Curve
（采样 loss 曲线）

## Sample Outputs
（生成样本）
```

## 查看方法

```bash
# 看总台账
cat logs/training-log.md

# 看某次训练详情
cat logs/runs/run-003-pretrain.md

# 看训练中的实时进度（detail log 的 Progress 部分）
cat logs/runs/run-NNN-*.md
```

## 注意事项

- 不要手动删除日志文件
- 测试新功能时用临时目录，不影响历史记录
- 台账按 Run 编号递增，不会重复
- 详细日志每次训练覆盖对应的 run 文件