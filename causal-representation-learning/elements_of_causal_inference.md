# Elements of Causal Inference: Foundations and Learning Algorithms

**Book**: Elements of Causal Inference: Foundations and Learning Algorithms  
**Authors**: Jonas Peters, Dominik Janzing, Bernhard Schölkopf  
**Year**: 2017  
**Publisher**: MIT Press (Adaptive Computation and Machine Learning series)  
**Pages**: 288  
**Free PDF**: Available via OAPEN

---

## 这本书尝试解决什么问题？

### 背景：因果推断的数学化

"因果性"长期以来是哲学概念，难以精确定义。直到近几十年，因果推断才被**数学化**，成为可以从数据中学习的对象。

### 核心问题

> 如何从观测数据中推断因果关系，而不仅仅是统计相关性？

### 目标读者

- 具有机器学习或统计学背景的研究者
- 希望理解因果推断基础的 AI 研究者
- 研究生课程教材

---

## 解决思路是什么？

### 1. 因果模型的形式化

**结构因果模型 (SCM)**：

```
Xᵢ := fᵢ(PAᵢ, Uᵢ), i = 1, ..., n
```

- **Xᵢ**: 观测变量
- **PAᵢ**: Xᵢ 的因果父节点
- **Uᵢ**: 独立噪声（外生变量）
- **fᵢ**: 因果机制

### 2. 从双变量开始

**双变量问题**：给定 (X, Y) 的观测，判断 X→Y 还是 Y→X？

这是因果推断的**最难情况**——因为：
- 没有条件独立性可用
- 需要额外假设来区分因果方向

**解决方案**：利用**统计不对称性** (Statistical Asymmetries)

### 3. 关键方法

#### 加性噪声模型 (ANM)
假设：Y = f(X) + N，其中 N ⊥ X

如果因果方向是 X→Y，那么残差应该与 X 独立；反之则不成立。

#### 信息几何方法 (IGCI)
利用分布的复杂性度量来推断因果方向。

#### 基于独立性的推断
P(原因) 与 P(效果|原因) 应该是"独立"的（不相互透露信息）。

### 4. 多变量因果发现

**条件独立性方法**：
- PC 算法
- FCI 算法（允许隐变量）

**基于得分的方法**：
- BIC/MDL 评分
- 贪婪等价搜索

### 5. 干预与 do-演算

**干预** (Intervention)：强制设置变量值，记作 do(X=x)

**do-演算**：Pearl 的三条规则，用于从观测分布推导干预分布

**可识别性**：什么时候可以从观测数据计算因果效应？

### 6. 反事实推理

**反事实** (Counterfactual)："如果当时不同，会怎样？"

需要完整的 SCM（包括噪声分布）才能回答反事实问题。

---

## 效果如何？

### 书的结构

| 部分 | 内容 | 复杂度 |
|-----|------|-------|
| **双变量** | 因果发现的核心困难 | ⭐⭐⭐ |
| **多变量** | 条件独立性与图结构 | ⭐⭐ |
| **干预** | do-演算与可识别性 | ⭐⭐⭐ |
| **反事实** | 完整的因果推理 | ⭐⭐⭐⭐ |

### 本书特色

1. **自包含**：从基础概率论开始
2. **聚焦 ML 视角**：强调可学习性和算法
3. **双变量问题**：作者十年研究的深入总结
4. **代码示例**：可复制粘贴的代码片段
5. **练习题**：每章末尾有练习

---

## 还有哪些待解决的问题？

### 1. 因果表征学习

本书假设因果变量已给定。**最大的开放问题**：

> 如何从原始观测（图像、文本）中发现高层因果变量？

### 2. 非线性可识别性

- 非线性情况下的因果发现更加困难
- 需要更强的假设或额外信息

### 3. 高维因果推断

- 如何扩展到数千变量？
- 计算复杂度问题

### 4. 隐变量问题

- 如何处理未观测的混淆因子？
- FCI 算法的局限性

### 5. 时间序列因果推断

- 动态系统中的因果发现
- Granger 因果性的局限性

---

## 关键概念速查

| 概念 | 含义 |
|-----|------|
| **SCM** | Structural Causal Model，结构因果模型 |
| **DAG** | Directed Acyclic Graph，有向无环图 |
| **d-separation** | 图上的条件独立性判断规则 |
| **Markov condition** | 给定父节点，变量独立于非后代 |
| **Faithfulness** | 所有条件独立性都来自图结构 |
| **ANM** | Additive Noise Model，加性噪声模型 |
| **IGCI** | Information-Geometric Causal Inference |
| **do(·)** | 干预操作符 |
| **Identifiability** | 因果效应能否从观测数据计算 |

---

## 与其他论文的关系

- **本书** → 因果推断的**数学基础**
- **Causality for Machine Learning** (2019) → 连接因果与 ML 的综述
- **Meta-Transfer Objective** (2019) → 利用本书原理的算法
- **Towards Causal Representation Learning** (2021) → 扩展到表征学习

---

## 阅读建议

### 入门路径
1. **Chapter 1-2**: 动机和基础定义
2. **Chapter 3**: 双变量因果发现（核心难点）
3. **Chapter 4**: 多变量和条件独立性

### 进阶路径
4. **Chapter 5-6**: 干预和 do-演算
5. **Chapter 7**: 反事实推理
6. **Appendix**: 数学预备知识

### 配套资源
- [Brady Neal 的因果推断课程](https://www.bradyneal.com/causal-inference-course) 以本书为主要参考
- Yoshua Bengio 的因果表征学习客座讲座可在线观看

---

## 经典引用

> "Causal inference can be seen as part of the toolbox of every scientist, but the mathematics is relatively recent."

> "The bivariate case turns out to be a particularly hard problem for causal learning because there are no conditional independences."

---

## 总结

这本书是因果推断领域的**入门经典**，特别适合：
- 想要系统学习因果推断基础的 ML 研究者
- 准备进入因果表征学习领域的博士生
- 需要参考数学细节的实践者

**核心价值**：将因果推断从哲学概念变成了**可学习的数学对象**。
