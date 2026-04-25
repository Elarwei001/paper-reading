# RAG 的数据库视角：为什么它本质上更像 constrained query planning，而不只是“向量搜索”

> 大多数 RAG 教程都会停在 embedding、chunk 和 top-k 上，但这只是最表层。真正进入生产系统后，RAG 更像一个数据库查询规划问题：你不是在裸跑向量最近邻，而是在有权限、租户、时间、版本、来源和成本约束下，做一个受限检索与候选生成流程。

---

## 1. 经典 RAG 教程为什么会误导人

大家最常看到的 RAG 流程是：

1. 文档切 chunk
2. chunk 做 embedding
3. 用户 query 做 embedding
4. 向量相似度 top-k
5. 把 top-k chunk 拼进 prompt
6. 让模型回答

这个流程没错，但它忽略了生产系统最难的部分：

- 权限
- 元数据过滤
- 文档 freshness
- 版本一致性
- provenance
- reranking
- source grounding
- query decomposition
- 多阶段检索

也就是说，真实 RAG 不是一个裸 ANN 问题，而是一个：

**多约束、多阶段、多目标的检索规划问题。**

---

## 2. 用数据库语言重写 RAG

更准确地说，RAG 可以写成：

> 给定一个 query 和一组外部知识源，在权限、租户、时效性、来源可信度、文档类型、成本预算等约束下，生成一组足够好的 supporting context，供生成模型使用。

如果把它翻译成数据库系统问题，本质上是在做：

- candidate generation
- filter pushdown
- top-k ranking
- optional reranking
- provenance preservation
- budget-aware result assembly

这和数据库里的查询规划非常像。

---

## 3. RAG 的第一层不是相似度，而是约束空间

在真实系统里，查询并不是“在全库上找最像的内容”。

它通常先要满足一些结构约束：
- 当前 tenant
- 当前用户权限
- 允许的数据源
- 时间范围
- 文档语言
- 已发布版本
- 某业务域标签

所以第一层问题其实是：

> 检索空间是什么？

如果这个检索空间定义错了，后面的 embedding 再强都没用。

这就是为什么 metadata retrieval 在生产上往往比 ANN 更关键。

---

## 4. 从查询优化器的角度看 RAG

查询优化器的基本任务是：
- 把逻辑目标转成执行计划
- 选择代价更低、效果更好的路径

RAG 其实也一样。

给定一个 query，系统可能要决定：

1. 先 metadata filter 再 ANN，还是先 ANN 再 filter？
2. top-k 取多少进入 reranker？
3. 是否分 query 多路检索？
4. 是否做 lexical + vector hybrid retrieval？
5. 是否走缓存？
6. 是否要引用最近版本优先？
7. prompt budget 只能放多少 chunk？

这些都已经不是“一个相似度排序”能描述的，而是：

**受约束的执行计划选择问题。**

---

## 5. 为什么说 RAG 更像 constrained query planning

因为它至少同时优化三类目标：

### 1. 正确性目标
- 找到真正相关的内容
- 避免权限越界
- 避免过期文档
- 保留来源链路

### 2. 成本目标
- 检索延迟
- rerank 开销
- prompt token 消耗
- 模型调用代价

### 3. 生成目标
- 提供足够上下文支撑回答
- 避免冗余 chunk
- 让模型更容易 grounding

这和数据库优化器处理：
- I/O cost
- CPU cost
- cardinality estimation
- join order

只是领域不同，但结构非常相似。

---

## 6. RAG 的一个更真实的执行模型

真实 RAG 系统更像这样：

1. **normalize query**
2. **resolve tenant / auth scope**
3. **apply metadata filters**
4. **run candidate retrieval**
   - vector
   - lexical
   - hybrid
5. **rerank candidates**
6. **assemble context under token budget**
7. **attach provenance**
8. **generate answer**
9. **log retrieval trace**

这和数据库里的多阶段执行已经非常接近。

---

## 7. provenance 为什么不是附属需求，而是 RAG 的核心语义之一

很多 AI 产品失败，不是因为没检索到内容，而是因为：
- 无法解释内容从哪来
- 没法追查引用链
- 回答不能审计
- 知识更新后不能重现旧结果

所以 provenance 在 RAG 中不是“UI 上加个 citation”这么简单，而是：

**检索系统必须维护上下文来源的结构性可追踪性。**

从数据库角度，这意味着：
- candidate id
- document id
- chunk offset
- version
- source URL
- retrieval timestamp
- filter set
- scoring path

都应该被记录。

这已经明显是数据库/信息检索系统设计问题，而不是模型内部问题。

---

## 8. 为什么 PostgreSQL 特别适合承载 constrained RAG

因为 PostgreSQL 很擅长同时处理：
- 结构化过滤
- 权限与租户边界
- 时间和版本语义
- 检索 trace 记录
- embedding storage
- ANN 扩展

它未必是最大的向量引擎，但它在 constrained RAG 中有个天然优势：

**结构约束就是它的主场。**

而真实 RAG 最难的地方，恰恰不只是向量空间，而是约束空间。

---

## 9. 一个更强的结论：很多所谓“AI 幻觉问题”，其实部分是 retrieval planning 问题

这句话需要谨慎说，但我认为是对的。

很多生成错误，并不完全是模型“胡说”，而是因为：
- 候选集错了
- 过滤条件错了
- 最新文档没被选进来
- reranking 排错了
- 预算只保留了次优 chunk
- provenance 丢了，模型只能瞎补

所以一部分幻觉，本质上是：

**外部记忆检索计划失败。**

这意味着 RAG 质量的很多提升空间，其实在数据库、检索与执行规划层，而不是只在模型层。

---

## 10. 最后的结论

如果把 RAG 理解成“把向量搜索结果塞给 LLM”，那理解太浅了。

更准确的理解应该是：

**RAG 是一个受结构约束、权限约束、预算约束和来源约束的查询规划与上下文组装系统。**

这也是为什么数据库理论、信息检索理论和 AI 系统工程，在这里重新汇合。

如果你想真正提高 RAG 质量，往往不能只盯着 embedding model，而要回到这些更基础的问题：
- 检索空间如何定义
- 约束如何下推
- candidate 如何生成
- provenance 如何保持
- 结果如何在预算内组装

这些问题，本质上都很像数据库系统问题。
