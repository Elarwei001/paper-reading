# Beyond AlphaFold: How Boltz is Open-Sourcing the Future of Drug Discovery

**Source:** [Latent Space Podcast](https://www.latent.space/p/boltz)  
**Guests:** Gabriele Corso & Jeremy Wohlwend (Boltz co-founders)  
**Date:** February 2026  
**Type:** Podcast Transcript  

## TL;DR

Boltz 联合创始人讨论了从 AlphaFold 到 Boltz 的演进。核心观点：单链蛋白结构预测已基本"解决"（通过进化信息），下一个前沿是**蛋白质相互作用建模**和**生成式蛋白质设计**。Boltz 通过开源模型 + 商业平台（Boltz Lab）民主化这些工具。

---

## 1. 单链蛋白结构预测：为什么"基本解决"

### 1.1 Co-evolutionary Hints（共进化提示）

关键突破来自解码**进化相关性**：
- 在不同物种中，如果两个氨基酸位置总是同时突变 → 它们在 3D 空间中很可能相邻
- 这些 hints 帮助模型在能量景观中找到正确的"山谷"

```
进化信息（MSA） → 粗粒度距离矩阵 → 精确 3D 结构
```

### 1.2 结构预测 vs 折叠

| 概念 | 定义 | 进展 |
|------|------|------|
| **结构预测** | 直接预测最终答案 | ✅ 很好 |
| **折叠** | 理解从无序到有序的动力学过程 | ❌ 仍然很差 |

蛋白质不是静态的 — 它们会运动、采取不同形状。理解不同构象状态及其概率仍是开放问题。

---

## 2. AlphaFold 2 → AlphaFold 3 的关键变化

### 2.1 从回归到生成式建模

| AlphaFold 2 | AlphaFold 3 / Boltz |
|-------------|---------------------|
| 回归问题：预测单一坐标 | 生成式扩散：从后验分布采样 |
| 不确定时会"平均"答案 | 可以采样多个答案再选最佳 |
| 难以建模多构象 | 可以表示多个构象状态 |

### 2.2 专用架构 vs Bitter Lesson

**有趣的反直觉发现：**
- 这个领域是少数仍然需要**高度专用架构**的 ML 应用
- 简单的 transformer 性能远不如等变架构（equivariant architectures）
- 原因：分子的 3D 几何约束是内在的

### 2.3 Scaling 在这个领域效果不同

> "Scaling hasn't really worked kind of the same in this field."

与 LLM 不同，简单地增大模型规模不一定带来同比例的性能提升。

---

## 3. Boltz 模型套件

### 3.1 Boltz-1：结构预测

- 类似 AlphaFold 3 的开源替代
- 只训练了一次（计算资源极其有限）
- 训练过程中边修 bug 边训练（"surgery in the middle"）

### 3.2 Boltz-2：亲和力预测

**关键创新 — 统一编码（Unified Encoding）：**
- 结构预测和序列预测合并为同一任务
- 因为氨基酸有不同的原子组成，从原子位置可以同时推断出结构和氨基酸身份
- 避免了离散（序列）和连续（结构）监督信号的冲突

### 3.3 BoltzGen：蛋白质设计

**工作方式：**
```
输入: 目标蛋白 + 设计规格（spec）
      ↓
模型: 用 blank tokens 代替序列
      ↓
输出: 同时生成 3D 结构 + 氨基酸序列
```

**评估流程：**
1. **一致性检查**：用 Boltz-2 预测设计出的蛋白结构，对比 BoltzGen 的预测
2. **亲和力预测**：直接预测结合强度（比 confidence score 更可靠）

---

## 4. 实验验证：关键结果

### 4.1 泛化性验证

为证明模型不是"背诵"训练数据：
- 选择 **9 个在 PDB 中没有已知相互作用的靶点**
- 模型从未见过这些蛋白与其他蛋白的结合
- 结果：**三分之二的靶点**成功设计出纳摩尔级结合物

### 4.2 大规模协作验证

- 与 **25+ 学术和工业实验室**合作
- 验证任务包括：肽设计、纳米抗体设计、蛋白-小分子设计
- 覆盖有序蛋白、无序蛋白等多种模态

---

## 5. Boltz Lab：产品化

### 5.1 三个核心组件

| 组件 | 功能 |
|------|------|
| **Agents** | 蛋白设计 agent、小分子设计 agent（复杂 pipeline 的自动化） |
| **Infrastructure** | GPU 集群，支持大规模并行筛选（数万候选分子） |
| **Interface** | API + Web UI，支持协同排序和筛选 |

### 5.2 性能优势

- 小分子筛选 pipeline 比开源版本**快 10 倍**（专有 GPU kernel 优化）
- 规模化使成本远低于自建

### 5.3 开源 + 商业模式

> "Putting a model on GitHub is definitely not enough to get chemists and biologists to use your model."

- 开源模型推动研究社区进步
- 商业产品提供最佳用户体验（类比：我不会自己部署 LLM，直接用 ChatGPT/Claude）

---

## 6. 与怀疑论者打交道

对于持怀疑态度的药物化学家：

> "At the end of the day, you have to show them something they didn't think was possible."

关键策略：
1. 让他们亲自试用
2. 等实验室结果出来
3. 让他们的同事先被说服

**Boltz 团队的药物化学家 Jeffrey：**
- 现在是团队中运行最多计算的人
- 同时运行多个假设的筛选
- 结合模型输出 + 自己的直觉进行选择

---

## 7. 未来方向

### 7.1 Developability（可开发性）

设计出结合物只是第一步，还需要考虑：
- 毒性
- 稳定性
- 可合成性

### 7.2 Virtual Cell（虚拟细胞）

理解蛋白在细胞环境中的行为，而不仅仅是孤立的分子对相互作用。

---

## 8. Key Quotes

> "The folding process is almost instantaneous, which is a strong signal that we might be able to predict it."

> "One of the interesting things about the protein folding problem is that it used to be studied as a classical NP problem."

> "This field is one of the very few fields in applied ML where we still have architectures that are very specialized."

> "When we say we design new proteins, we should be very clear — these are not drugs ready to be put into a human."

---

## 9. Resources

- **Boltz Lab:** https://boltz.bio
- **Open Source:** GitHub (Boltz-1, Boltz-2)
- **Manifesto:** [Science at the Speed of Inference](https://www.a16z.news/p/science-at-the-speed-of-inference)
- **YouTube:** https://www.youtube.com/watch?v=nP0N1kYLegc

---

## 10. 个人思考

1. **Bitter Lesson 的反例**：在蛋白结构预测中，专用等变架构 >> 通用 transformer。这与 NLP/CV 的趋势相反，说明问题的物理约束足够强时，归纳偏置仍然重要。

2. **开源 + 商业的平衡**：Boltz 的模式值得参考 — 开源基础模型推动社区，商业产品提供 10x 更好的体验。类似 Hugging Face 的路径。

3. **验证的重要性**：在设计领域，你不知道答案是什么，所以验证比预测更难。Boltz 选择"9 个 PDB 中无已知相互作用的靶点"是很聪明的 benchmark 设计。

4. **与 AlphaGenome 的联系**：AlphaGenome 预测 DNA/RNA 调控，Boltz 预测蛋白结构和相互作用。两者可以互补 — 从基因组到蛋白组的完整 pipeline。
