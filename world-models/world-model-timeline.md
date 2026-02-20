# AI 世界模型学术演进路线
> 地铁阅读版 🚇

---

## 一、什么是"世界模型"？

AI 的世界模型 = 对环境的内部表征，用于：
- 预测：做了 A 会发生什么？
- 规划：想要 B，应该做什么？
- 想象：不用真的尝试，在"脑中"模拟

---

## 二、学术演进时间线

### 🌱 萌芽期 (1980s-2000s)

**1989 - Schmidhuber 的早期工作**
- 提出用神经网络预测环境
- 奠定"学习世界模型"的思想基础

**关键词**：前馈网络、环境预测

---

### 🔬 理论奠基期 (2000s-2015)

**2004 - Friston: 自由能原理**
- 📄 *The free-energy principle: a unified brain theory?*
- 大脑通过最小化"预测误差"来理解世界
- 影响后来的预测编码、主动推理

**2013 - Kingma & Welling: VAE**
- 📄 *Auto-Encoding Variational Bayes*
- 学习数据的隐空间表征
- 世界模型的技术基础之一

**关键词**：概率建模、隐变量、变分推断

---

### 🎮 强化学习中的世界模型 (2015-2018)

**2015 - DeepMind: 想象力网络**
- 📄 *Imagination-Augmented Agents* (I2A)
- Agent 学习环境模型，用"想象"来规划

**2018 - Ha & Schmidhuber: World Models**
- 📄 *World Models* (重要！入门必读)
- 🔗 https://worldmodels.github.io
- 在隐空间中"做梦"训练 agent
- VAE 编码视觉 + RNN 预测动态 + 控制器决策
- **第一次把"世界模型"这个概念清晰呈现**

**2018 - PlaNet (DeepMind)**
- 📄 *Learning Latent Dynamics for Planning*
- 在隐空间做规划，不重建像素

**关键词**：model-based RL、隐空间规划、想象

---

### 🧠 认知启发期 (2019-2022)

**2019 - Dreamer (DeepMind)**
- 📄 *Dream to Control: Learning Behaviors by Latent Imagination*
- 在"梦"中学习策略
- 后续: DreamerV2 (2021), DreamerV3 (2023)

**2020 - MuZero (DeepMind)**
- 📄 *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*
- 不需要游戏规则，学习抽象的状态转移模型
- AlphaGo 的进化版

**2022 - Yann LeCun: 自主智能蓝图**
- 📄 *A Path Towards Autonomous Machine Intelligence*
- 🔗 https://openreview.net/forum?id=BZ5a1r-kVsf
- 提出 JEPA 架构
- 核心观点：
  - 生成像素是浪费
  - 应该在抽象表征空间预测
  - 需要分层的世界模型

**关键词**：隐空间想象、无模型到有模型、认知架构

---

### 🖼️ 视觉世界模型 (2023-2024)

**2023 - I-JEPA (Meta)**
- 📄 *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*
- 🔗 https://arxiv.org/abs/2301.08243
- 图像内预测表征，不生成像素

**2024 - V-JEPA (Meta)**
- 视频版 JEPA
- 🔗 https://github.com/facebookresearch/jepa
- 从视频学习时空表征

**2024 - Sora (OpenAI)**
- 视频生成模型，隐式学习物理
- 展示了大规模视频预训练的潜力
- 但 LeCun 批评：仍在像素空间生成

**2024 - Genie (DeepMind)**
- 📄 *Genie: Generative Interactive Environments*
- 从视频学习可交互的 2D 世界
- 11B 参数，单张图生成可玩环境

**关键词**：自监督、视频预训练、交互式生成

---

### 🤖 具身世界模型 (2024-2026)

**2024 - UniSim (DeepMind)**
- 📄 *Learning Interactive Real-World Simulators*
- 用视频预训练的通用模拟器

**2025 - NVIDIA Cosmos**
- 机器人世界基础模型
- 动作和状态编码为视频帧
- 🔗 https://github.com/nvidia-cosmos

**2026 - Waypoint-1 (Overworld)**
- 实时交互视频世界模型
- 键鼠控制，零延迟
- 🔗 https://huggingface.co/overworld/Waypoint-1-Small

**关键词**：机器人、仿真到真实、交互式

---

## 三、两条技术路线对比

| | 生成式路线 | JEPA 路线 |
|--|----------|----------|
| 代表 | Sora, Dreamer, Genie | I-JEPA, V-JEPA |
| 预测目标 | 像素/视频帧 | 抽象表征 |
| 优点 | 可视化、直观 | 高效、语义 |
| 缺点 | 计算昂贵 | 难以可视化 |
| 评估 | 容易（看视频） | 困难（靠下游任务） |

---

## 四、推荐阅读顺序

**第一周：入门**
1. World Models (2018) - 网页版图文并茂
2. LeCun 的蓝图论文 - 建立宏观认知

**第二周：深入**
3. Dreamer 系列 - 理解隐空间想象
4. I-JEPA / V-JEPA - 理解非生成式路线

**第三周：前沿**
5. Genie / Sora 技术报告
6. Cosmos / Waypoint - 最新应用

**扩展阅读**
- Friston 的主动推理（偏神经科学）
- Pearl 的因果推理（偏哲学/统计）

---

## 五、关键人物

| 人物 | 机构 | 贡献 |
|-----|------|-----|
| Jürgen Schmidhuber | IDSIA | 世界模型先驱 |
| David Ha | Google | World Models 论文 |
| Yann LeCun | Meta | JEPA 架构 |
| Danijar Hafner | DeepMind | Dreamer 系列 |
| Karl Friston | UCL | 自由能原理 |
| Tim Rocktäschel | DeepMind | Genie |

---

## 六、开放问题

1. **生成 vs 表征**：哪条路线通向真正的理解？
2. **评估难题**：如何衡量"世界模型的好坏"？
3. **因果推理**：从相关性到因果性的鸿沟
4. **具身必要性**：必须有身体才能理解物理世界吗？
5. **语言与世界**：LLM 是否有隐式世界模型？

---

## 七、一句话总结

> 世界模型的演进：从"预测像素"走向"理解结构"，从"被动观察"走向"主动交互"。JEPA 指出了一条不同的路——也许理解世界不需要重建世界。

---

*祝地铁阅读愉快！有问题随时问我 🚇*
