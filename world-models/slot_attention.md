# Object-Centric Learning with Slot Attention

**Paper**: Object-Centric Learning with Slot Attention  
**Authors**: Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, Thomas Kipf  
**Year**: 2020  
**Venue**: NeurIPS 2020  
**arXiv**: [2006.15055](https://arxiv.org/abs/2006.15055)  
**Code**: [github.com/google-research/google-research/tree/master/slot_attention](https://github.com/google-research/google-research/tree/master/slot_attention)

---

## 这篇论文尝试解决什么问题？

### 背景：表示学习的局限

深度学习通常学习**分布式表示 (Distributed Representations)**：
- 输入的信息分散在整个表示向量中
- 难以捕捉场景的**组合性质 (Compositional Properties)**

但人类理解世界的方式是**以对象为中心 (Object-Centric)**：
- 场景由独立对象组成
- 对象可以独立操作和推理
- 新场景 = 已知对象的新组合

### 核心问题

> 如何从低层感知特征（如 CNN 输出）中学习对象中心的表示？

**挑战**：
1. 对象数量可变
2. 对象间无固定顺序（集合性质）
3. 需要无监督或弱监督学习

---

## 解决思路是什么？

### 1. Slot Attention 模块

**核心思想**：通过**竞争性注意力**将输入特征路由到一组"槽" (Slots)

```
输入: CNN 特征 (N 个向量)
    ↓
Slot Attention (迭代注意力)
    ↓
输出: K 个 Slot 向量 (每个可绑定一个对象)
```

### 2. 关键设计

#### 随机初始化 Slots
```python
slots ~ N(μ, σ)  # 从可学习的高斯分布采样
```
- 所有 slots 使用**相同的分布**初始化
- 保证 slot 可以绑定到**任意对象**（不会专门化）

#### Slot-归一化注意力

传统注意力（如 Transformer）：**对输入归一化**
```
attn[i,j] = softmax over inputs (j)
```

Slot Attention：**对 slots 归一化**
```
attn[i,j] = softmax over slots (j)  # 关键区别！
```

**效果**：强制 slots 之间**竞争**来解释输入

#### 迭代更新
```python
for t in range(T):  # 通常 T=3
    attn = softmax(k(inputs) @ q(slots).T / sqrt(D), dim='slots')
    updates = weighted_mean(attn, v(inputs))
    slots = GRU(slots, updates) + MLP(slots)
```

### 3. 数学性质

**命题**：Slot Attention 满足：
- **对输入置换不变**：打乱输入顺序，输出不变
- **对 slots 置换等变**：打乱 slot 顺序，输出同样打乱

这保证了：
- 输入是无序集合（适合图像特征）
- 输出是无序集合（适合对象集合）

### 4. 应用场景

#### 无监督对象发现 (Object Discovery)
```
Image → CNN → Slot Attention → Decoder → Reconstruction
                    ↓
              Per-slot masks (分割掩码)
```
- 训练目标：重建图像
- Slot 自动学会分割对象

#### 集合预测 (Set Prediction)
```
Image → CNN → Slot Attention → MLP (per slot) → Properties
                                        ↓
                              Hungarian Matching with GT
```
- 预测每个对象的属性（位置、颜色、形状等）
- 用匈牙利算法匹配预测和标签

---

## 效果如何？

### 实验数据集

| 数据集 | 特点 | Slot 数量 |
|-------|------|----------|
| CLEVR | 简单 3D 几何体 | 7 |
| Multi-dSprites | 2D 精灵 | 6 |
| Tetrominoes | 俄罗斯方块 | 4 |

### 主要结果

1. **对象发现**：匹配或超越 IODINE, MONet
2. **训练效率**：比 IODINE 快 4-5 倍
3. **内存效率**：更少的 GPU 内存
4. **泛化能力**：能泛化到更多对象、更多 slots

### 与基线对比

| 模型 | 方法 | 时间复杂度 |
|-----|------|----------|
| IODINE | 迭代 encode-decode | 高 |
| MONet | 顺序分割 | 中 |
| Slot Attention | 迭代注意力 | **低** |

---

## 还有哪些待解决的问题？

### 1. 扩展到真实数据

- 论文只在**简单合成数据**上验证
- 真实图像的复杂纹理、光照、遮挡
- 需要额外归纳偏置或监督

### 2. 对象定义的模糊性

- 什么是"对象"？（语义上）
- 模型可能过度分割或欠分割
- 需要与任务对齐

### 3. 时间扩展

- 如何处理视频？
- 需要跨帧跟踪同一对象
- → 这是 SAVi 要解决的问题

### 4. 3D 场景理解

- 处理 3D 几何
- 遮挡推理
- 视角变化

### 5. 与其他模块的整合

- 如何与动力学模型结合？
- 如何与语言对接？
- 如何用于规划？

---

## 关键概念速查

| 概念 | 含义 |
|-----|------|
| **Slot** | 可以绑定到任意对象的表示单元 |
| **Object-Centric** | 以对象为中心的表示方式 |
| **Competitive Attention** | Slots 竞争解释输入特征 |
| **Permutation Equivariance** | 对 slot 顺序置换等变 |
| **Spatial Broadcast Decoder** | 将 slot 广播到空间网格再解码 |

---

## 代码要点

```python
# Slot Attention 核心
def slot_attention(inputs, slots, num_iterations=3):
    for _ in range(num_iterations):
        # 注意力（对 slots 归一化）
        attn = softmax(k(inputs) @ q(slots).T / sqrt(D), dim=-1)
        
        # 加权平均
        attn_normalized = attn / attn.sum(dim=0, keepdim=True)
        updates = attn_normalized.T @ v(inputs)
        
        # GRU 更新
        slots = gru(updates, slots)
        slots = slots + mlp(layer_norm(slots))
    
    return slots
```

---

## 与其他论文的关系

- **理论基础**: Relational Inductive Biases (Graph Networks)
- **前身**: IODINE, MONet, AIR
- **后续**: SAVi (视频扩展), STEVE, Dinosaur
- **应用方向**: 物理推理, 视觉问答, 机器人

---

## 核心贡献总结

1. **简单高效的模块**：可插入任何架构
2. **竞争性注意力**：自动实现对象分割
3. **置换对称性**：slots 可交换，支持可变对象数
4. **无需复杂先验**：不需要对象大小等假设
