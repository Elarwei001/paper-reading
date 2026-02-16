# Memory-Augmented Transformers

## 🎯 为什么关注这个方向？

超长上下文（1M+ tokens）的核心瓶颈是 KV Cache 显存爆炸。
核心思路：**不全存显存，按需检索** —— 类似人眼的中央凹聚焦机制。

## 📚 关键论文 (待读)

### 基础架构
- [ ] **Memorizing Transformer** (Google, 2022)
  - 外部 KV 存储 + 检索
  - https://arxiv.org/abs/2203.08913

- [ ] **Landmark Attention** (2023)
  - 插入路标 token，分层检索
  - https://arxiv.org/abs/2305.16300

- [ ] **LongMem** (2023)
  - 解耦记忆与推理
  - https://arxiv.org/abs/2306.07174

### 压缩方向
- [ ] **H2O: Heavy-Hitter Oracle** (2023)
  - 基于 attention score 的 KV 驱逐
  - https://arxiv.org/abs/2306.14048

- [ ] **StreamingLLM** (MIT, 2023)
  - Attention Sink + 滑动窗口
  - https://arxiv.org/abs/2309.17453

- [ ] **AutoCompressor** (2023)
  - 学习压缩 token
  - https://arxiv.org/abs/2305.14788

### 检索增强
- [ ] **RETRO** (DeepMind, 2022)
  - 检索增强的语言模型
  - https://arxiv.org/abs/2112.04426

- [ ] **Focused Transformer** (2023)
  - Contrastive learning for better retrieval
  - https://arxiv.org/abs/2307.03170

## 🔑 核心概念

```
传统 Attention: O(n²) 全对全
检索增强 Attention: O(n) 存储 + O(k) 检索 + O(k²) 局部 attention
```

## 💡 灵感来源

人眼视觉系统：
- 中央凹 (Fovea)：高分辨率，聚焦当前
- 周边视觉：低分辨率，感知全局
- 眼动 (Saccade)：按需切换焦点

## 📅 Follow 计划

- 关注 arXiv cs.CL 的 long-context 相关论文
- 关注 Google, DeepMind, Meta AI 的 memory/retrieval 工作
- NeurIPS, ICML, ICLR 相关 workshop

---
*Created: 2026-02-09*
*Status: Active tracking*
