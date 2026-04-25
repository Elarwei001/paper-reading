# pgvector、HNSW 与近似最近邻搜索：为什么 PostgreSQL 能做向量检索，但它到底在做什么

> 这篇文章专门讨论 AI 时代里 PostgreSQL 最受关注的一块新能力：向量检索。重点不是“怎么安装 pgvector”，而是解释为什么高维检索困难、ANN 在解决什么问题、HNSW 为什么有效，以及 PG 在这里到底扮演什么角色。

---

## 1. 先说核心问题：为什么向量检索不是普通数据库索引问题

传统数据库索引通常处理的是：
- 精确匹配
- 范围查询
- 排序后的查找

这些任务依赖一个很强的前提：

> 键空间存在比较稳定的一维顺序结构。

比如整数、时间戳、字符串前缀，都可以在某种排序上组织。

但 embedding 检索问的问题是：

> 在一个高维向量空间里，谁和当前 query 最接近？

这类问题的难点在于：
- 没有自然总序
- 邻近关系是高维几何关系
- 距离计算成本不低
- 高维里“最近”与“平均”会发生集中现象

所以它不是传统 B-tree 的自然地盘。

---

## 2. 什么是最近邻搜索

给定一个 query vector \(q\)，以及数据库中的很多向量 \(x_1, x_2, ..., x_n\)，我们想找：

\[
\arg\min_i d(q, x_i)
\]

或者取 top-k 最近邻。

常见距离度量包括：
- Euclidean / L2 distance
- cosine distance
- inner product / maximum similarity

不同 embedding model、不同检索任务，会偏好不同度量。

---

## 3. 为什么精确最近邻在大规模下会变慢

最直接的方法是 brute force：
- 对每个向量都算一次距离
- 排序或维护 top-k

复杂度通常近似：
- O(Nd)

其中：
- N 是向量数
- d 是维度

如果 N 很大，这种方法很快就扛不住。

因此真正的问题不是“能不能查”，而是：

> 能不能在少算很多距离的前提下，还找到足够好的候选。

这就是 ANN 的出发点。

---

## 4. ANN：近似最近邻不是作弊，而是工程上的最优折中

ANN = Approximate Nearest Neighbor。

它的目标不是永远找到理论最优最近邻，而是让下面这几个指标达到较好的平衡：
- latency
- recall
- memory
- build cost
- update cost

所以 ANN 的本质是一种系统折中：

**牺牲一点最优性，换取可用的检索速度。**

这和很多 AI 系统本身的现实很一致：
- 不是每次都要绝对最优
- 但必须足够快、足够稳、足够准

---

## 5. HNSW 到底是什么

HNSW = Hierarchical Navigable Small World。

它不是树，也不是哈希，而更像：
- 多层图结构
- 高层稀疏，低层稠密
- 查询时先在高层快速靠近目标区域，再在低层局部展开

你可以把它想成一个分层导航图。

### 直觉
如果在平面上找目标点：
- 粗看地图，先定位大区域
- 再看街道图，靠近具体地点

HNSW 做的就是类似的事，只不过对象变成了高维向量空间里的近邻图。

---

## 6. HNSW 为什么有效

HNSW 强的原因在于它把“高维搜索”转成了“图导航”：

1. 先从高层稀疏图开始
2. 通过 greedy navigation 快速靠近 query 所在区域
3. 再在更低层图中做局部扩展
4. 最终得到高 recall 的候选集合

它的优点通常是：
- 查询很快
- recall 高
- 在大量真实 embedding workload 上效果稳定

代价是：
- 建图成本高
- 内存占用高
- 在线更新复杂度更高

所以 HNSW 并不是“免费午餐”，而是：

**用更复杂的索引构建和内存开销，换取更好的查询性能。**

---

## 7. pgvector 在 PostgreSQL 里到底做了什么

pgvector 给 PostgreSQL 增加了几件关键能力：
- 向量类型
- 距离操作符
- ANN 索引支持

于是数据库就能理解这种查询：

```sql
SELECT *
FROM chunks
ORDER BY embedding <=> $query
LIMIT 10;
```

从语义上看，这已经不是经典 SQL 里最原生的谓词过滤，而是：

**top-k nearest neighbor query embedded into SQL execution.**

这很重要，因为它把 AI 检索从“外部单独系统”拉回到了数据库查询框架里。

---

## 8. PostgreSQL 做向量检索的真正价值不只是 ANN，而是 ANN 能和结构化过滤融合

这点比“能不能存向量”更重要。

真实系统很少只做：
- nearest neighbors over all vectors

而更常做：
- 在满足 tenant / language / permission / source type / freshness 约束下，做 top-k semantic retrieval

例如：

```sql
SELECT id, chunk_text
FROM chunks
WHERE tenant_id = 42
  AND source_type = 'manual'
  AND updated_at > NOW() - INTERVAL '30 days'
ORDER BY embedding <=> $query
LIMIT 10;
```

这说明 PostgreSQL 的价值不是只在“有 HNSW”，而在于：

**HNSW + SQL filter + transaction semantics + metadata constraints**

这正是很多 RAG 系统最需要的能力组合。

---

## 9. 为什么 PostgreSQL 不一定是最大的向量系统赢家，但会是最常见的起点

如果你只比较：
- 十亿级 ANN
- 极端低延迟
- 大规模分片检索
- GPU 检索优化

专用向量数据库通常会更强。

但如果你看的是：
- 现有系统已经用 PG
- 数据规模还没大到极限
- 结构化过滤很多
- 想要统一运维
- 希望检索与权限、审计、trace 在一个系统里

那 PostgreSQL 往往是最佳起点。

所以更准确的说法是：

**pgvector 让 PostgreSQL 从“不能做向量检索”变成“很多真实场景里已经足够好”。**

这就是它在 AI 时代突然重要的原因。

---

## 10. 从算法和系统角度，向量检索值得继续深挖哪些问题

如果继续往论文/算法层深入，最值得展开的是：
- HNSW 的分层图构造与查询路径
- recall-latency-memory 三元权衡
- IVF vs HNSW 的差异
- 高维空间中的距离集中现象
- metadata-aware ANN 的执行策略
- reranking 在整体 retrieval pipeline 中的位置

也就是下一步最自然的文章其实是：

**《RAG 不是向量搜索，RAG 是多阶段检索系统》**
