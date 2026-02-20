# Relational Inductive Biases, Deep Learning, and Graph Networks

**Paper**: Relational inductive biases, deep learning, and graph networks  
**Authors**: Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, et al. (DeepMind, Google Brain, MIT)  
**Year**: 2018  
**Venue**: arXiv preprint (highly influential)  
**arXiv**: [1806.01261](https://arxiv.org/abs/1806.01261)

---

## 这篇论文尝试解决什么问题？

### 背景：深度学习的组合泛化困境

当前深度学习在**组合泛化 (Combinatorial Generalization)** 上表现不佳：

> 人类可以用有限的元素（如单词）组合出无限的新组合（如新句子），但深度学习难以做到。

**核心问题**：

- 深度学习追求"端到端"学习，尽量减少先验假设
- 但这导致模型难以捕捉世界的**组合结构**
- 泛化能力受限，无法处理新的组合

### 论文定位

这是一篇**综述 + 立场声明 + 统一框架**：
1. 回顾深度学习中的关系归纳偏置
2. 提出 Graph Networks 作为统一框架
3. 倡导将结构化方法与深度学习结合

---

## 解决思路是什么？

### 1. 关系推理的核心要素

| 概念 | 定义 |
|-----|------|
| **实体 (Entity)** | 具有属性的元素（如物理对象） |
| **关系 (Relation)** | 实体间的性质（如"比...重"） |
| **规则 (Rule)** | 将实体和关系映射到新实体/关系的函数 |

### 2. 归纳偏置 (Inductive Bias)

> 归纳偏置让学习算法在多个等价解中优先选择某些解。

深度学习中的关系归纳偏置：

| 架构 | 实体 | 关系 | 规则 | 归纳偏置 |
|-----|-----|-----|-----|---------|
| **全连接层** | 神经元 | 全连接 | 权重 | 弱（无结构假设） |
| **CNN** | 像素/网格 | 局部 | 卷积核 | 局部性 + 平移不变性 |
| **RNN** | 时间步 | 序列 | 递归函数 | 时间不变性 + Markov |
| **Graph Networks** | 节点 | 边 | 消息传递 | **任意关系结构** |

### 3. 图网络 (Graph Networks) 框架

**统一了多种图上的神经网络方法**：
- Message Passing Neural Networks
- Graph Convolutional Networks
- Interaction Networks
- Non-local Neural Networks
- Relation Networks

**核心操作**：

```
1. Edge update:  e'_k = φ^e(e_k, v_rk, v_sk, u)
2. Node update:  v'_i = φ^v(v_i, ρ^{e→v}(E'_i), u)  
3. Global update: u' = φ^u(u, ρ^{e→u}(E'), ρ^{v→u}(V'))
```

其中 ρ 是聚合函数（如求和、均值），φ 是更新函数（如 MLP）。

### 4. 关键设计原则

1. **灵活的表示**：节点、边、全局属性都可学习
2. **可配置的结构**：图结构可以是输入、输出或两者
3. **可组合的模块**：多个 Graph Network 可以堆叠
4. **可端到端训练**：与其他深度学习组件无缝结合

---

## 效果如何？

### 应用领域

| 领域 | 应用 |
|-----|------|
| **物理推理** | 预测物体运动、碰撞 |
| **多智能体** | 建模智能体交互 |
| **关系推理** | 视觉问答、场景理解 |
| **组合优化** | 图着色、旅行商问题 |
| **化学/生物** | 分子性质预测、蛋白质结构 |
| **社交网络** | 节点分类、链接预测 |

### 开源库

DeepMind 发布了 `graph_nets` 库：
- GitHub: [deepmind/graph_nets](https://github.com/deepmind/graph_nets)

---

## 还有哪些待解决的问题？

### 1. 图结构的学习

- 如果图结构未知，如何从数据中学习？
- 如何处理动态变化的图？

### 2. 可扩展性

- 大规模图的计算效率
- 邻居采样和子图方法

### 3. 与符号推理的结合

- 如何将神经网络与逻辑推理结合？
- 可解释性问题

### 4. 因果推理

- 图结构是否对应因果关系？
- 如何从图学习中得到因果结论？

### 5. 从原始观测到图

> 如何从图像/视频等原始输入中自动发现实体和关系？

这正是 **Object-Centric Learning** 要解决的问题。

---

## 关键概念速查

| 概念 | 含义 |
|-----|------|
| **Combinatorial Generalization** | 用已知元素组合出新组合 |
| **Relational Inductive Bias** | 对实体间关系结构的先验假设 |
| **Graph Network** | 统一的图神经网络框架 |
| **Message Passing** | 沿边传递信息、聚合更新节点 |
| **Permutation Invariance** | 对节点/边顺序不敏感 |

---

## 论文结构导读

| 章节 | 内容 |
|-----|------|
| 1 | 动机：组合泛化的重要性 |
| 2 | 关系归纳偏置的定义和分析 |
| 3 | Graph Networks 框架 |
| 4 | 设计原则和组合模式 |
| 5 | 讨论和未来方向 |

---

## 与其他论文的关系

- **启发**: Interaction Networks, Neural Message Passing
- **后续**: Slot Attention, Object-Centric Learning
- **应用**: 物理模拟 (GNS), 多智能体系统
- **思想延续**: LeCun's "A Path Towards Autonomous Machine Intelligence"

---

## 核心观点总结

1. **拒绝错误的二分法**：不是"手工设计" vs "端到端学习"，而是两者结合
2. **结构即归纳偏置**：架构选择本身就是先验假设
3. **图是通用结构**：可以表示任意实体-关系结构
4. **组合泛化是关键**：这是迈向人类级智能的必要条件
