# AI 世界模型综述
> 从像素预测到认知建模

---

## 1. 什么是世界模型？

世界模型 (World Model) 是 AI 系统对外部世界的内部表征，使其能够：
- **预测**：给定当前状态和动作，预测未来会发生什么
- **规划**：在心智中模拟不同行动方案的后果
- **推理**：理解因果关系，而非仅仅统计相关性

### 1.1 两种理解

| 维度 | 感知层世界模型 | 认知层世界模型 |
|------|--------------|--------------|
| 预测对象 | 像素、视频帧 | 抽象概念、因果关系 |
| 代表工作 | Sora, Cosmos, Waypoint | JEPA, 因果推理 |
| 评估方式 | 视觉质量、物理准确性 | 推理正确性、泛化能力 |
| 训练数据 | 视频、游戏录像 | 需要更结构化的知识 |

---

## 2. 主要技术路线

### 2.1 生成式世界模型（当前主流）

**核心思想**：预测下一帧像素

**代表作品**：
- **Sora** (OpenAI, 2024) - 视频生成，隐式学习物理
- **Genie** (Google DeepMind, 2024) - 可交互的 2D 世界
- **NVIDIA Cosmos** (2025) - 机器人控制的世界基础模型
- **Waypoint-1** (Overworld, 2026) - 实时可控视频世界

**优点**：
- 数据丰富（视频无处不在）
- 评估直观（看起来真不真）
- 商业路径清晰

**缺点**：
- 在像素空间预测效率低
- 难以泛化到未见场景
- 不理解"为什么"

### 2.2 JEPA：表征空间预测

**提出者**：Yann LeCun (Meta FAIR)

**核心思想**：不预测像素，预测抽象表征

```
传统生成模型：  输入 → 编码 → 解码 → 预测像素
JEPA：         输入 → 编码 → 在表征空间预测 → 不解码
```

**关键论文**：

1. **理论框架**
   - 📄 *A Path Towards Autonomous Machine Intelligence* (2022)
   - Yann LeCun
   - https://openreview.net/forum?id=BZ5a1r-kVsf
   - 提出认知架构蓝图：世界模型 + 内在动机 + 分层规划

2. **I-JEPA (图像)**
   - 📄 *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*
   - Assran et al., ICCV 2023
   - https://arxiv.org/abs/2301.08243
   - 从图像的一部分预测另一部分的表征

3. **V-JEPA (视频)**
   - 📄 *Revisiting Feature Prediction for Learning Visual Representations from Video*
   - Bardes, LeCun et al., 2024
   - https://github.com/facebookresearch/jepa
   - 纯自监督，不用文字、标注、负样本

**JEPA 的优势**：
- 更高效：不浪费计算在预测每个像素
- 更抽象：学到的是语义表征
- 更接近认知：人类也不是在脑中渲染每个像素

### 2.3 因果世界模型

**核心问题**：从"会发生什么"到"为什么发生"

**代表人物**：Judea Pearl (图灵奖得主)

**关键概念**：
- **因果图 (Causal Graph)**：变量之间的因果关系
- **干预 (Intervention)**：如果我做 X，会发生什么？
- **反事实 (Counterfactual)**：如果当时没做 X，会怎样？

**相关工作**：
- 📄 *Causal Reasoning and Large Language Models* - LLM 的因果推理能力评估
- 📄 *CausalVLM* - 视觉语言模型中的因果推理

---

## 3. 关键论文清单

### 3.1 理论基础

| 论文 | 作者 | 年份 | 核心贡献 |
|-----|------|-----|---------|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | LeCun | 2022 | JEPA 架构蓝图 |
| [World Models](https://arxiv.org/abs/1803.10122) | Ha & Schmidhuber | 2018 | 早期世界模型框架 |
| The Book of Why | Pearl | 2018 | 因果推理入门 |

### 3.2 视觉世界模型

| 论文 | 机构 | 年份 | 核心贡献 |
|-----|------|-----|---------|
| I-JEPA | Meta FAIR | 2023 | 图像表征预测 |
| V-JEPA | Meta FAIR | 2024 | 视频表征预测 |
| Genie | DeepMind | 2024 | 可交互 2D 世界生成 |
| Sora | OpenAI | 2024 | 高质量视频生成 |
| Cosmos | NVIDIA | 2025 | 机器人世界模型 |

### 3.3 语言模型中的世界模型

| 论文 | 核心问题 |
|-----|---------|
| Language Models as World Models | LLM 是否隐式学习了世界模型？ |
| Othello-GPT | GPT 在玩 Othello 时学到了棋盘表征 |
| LLMs and Causal Reasoning | LLM 的因果推理能力边界 |

---

## 4. 开放问题

### 4.1 如何评估"理解"？
当前评估主要靠下游任务表现，但这不能证明真正的理解。

### 4.2 感知与认知如何统一？
底层物理直觉和高层抽象推理需要在同一个框架中整合。

### 4.3 世界模型需要具身吗？
LeCun 认为纯语言模型不够，需要从物理世界学习。但 LLM 展现的推理能力挑战了这一观点。

### 4.4 如何学习因果结构？
从观察数据中学习因果关系是根本性的难题。

---

## 5. 我的观点

当前"世界模型"主要集中在**视觉/物理预测**，原因：
1. 容易定义 loss function
2. 数据丰富
3. 商业应用明确

但真正的世界模型应该是：
> **分层的认知结构**：底层是感知预测，上层是因果推理，两者通过抽象表征连接。

JEPA 是朝这个方向的重要一步——它放弃了像素预测，转向表征空间。但从"预测表征"到"理解因果"，还有很长的路要走。

---

## 6. 推荐阅读顺序

**入门**：
1. Ha & Schmidhuber, "World Models" (2018) - 经典入门
2. LeCun, "A Path Towards Autonomous Machine Intelligence" - 宏观视角

**深入**：
3. I-JEPA 论文 - 理解表征预测
4. V-JEPA 代码 - 动手实验

**扩展**：
5. Pearl, "The Book of Why" - 因果推理思维
6. Genie / Sora 技术报告 - 当前工业界水平

---

*最后更新：2026-02-04*
