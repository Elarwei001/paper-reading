# PostgreSQL 的原理与内核实现：从关系代数、Heap/WAL/MVCC 到 AI 时代的向量检索、RAG 与元数据检索

> 这篇文章继续往“更硬核”的方向推进。目标不是告诉你怎么在 PostgreSQL 里建一张表，而是解释它作为一个数据库系统，是如何在物理存储、事务语义、索引结构、查询执行与可恢复性上运作的；以及为什么这些传统数据库原理，在 AI 时代反而重新变得重要。

---

## 1. 先立一个总框架：PostgreSQL 解决的不是“存数据”，而是“如何在并发、崩溃和约束下维护可解释状态”

理解 PostgreSQL，最容易误入的一个坑是把它看成：

- 一个 SQL 接口
- 一个表格系统
- 一个后端程序经常连的东西

这些都没错，但都太表面。

如果从数据库系统的角度看，PostgreSQL 本质上在解决的是：

> 在并发事务、部分失败、系统崩溃、索引维护、约束校验和复杂查询同时存在时，如何维护一个**对外看起来一致、对内可恢复、并且可高效访问**的状态机。

所以它的核心不是“存了什么”，而是：

1. **逻辑模型**如何表达数据与约束
2. **物理模型**如何组织页、tuple、索引和日志
3. **并发模型**如何处理同时读写
4. **恢复模型**如何在掉电或崩溃后重建一致状态
5. **查询模型**如何把 SQL 编译成可执行的算法流程

这五件事构成了 PostgreSQL 的真正骨架。

---

## 2. 从 CRUD 到数据库内核：表面动作与内部语义的对应关系

应用开发者说 CRUD：
- Create
- Read
- Update
- Delete

但 PostgreSQL 内核不会把它们理解成“前端按钮动作”。

更准确的对应关系是：

- **Create data** → tuple insertion + index maintenance + WAL logging
- **Read** → planner + executor + MVCC visibility check
- **Update** → old tuple invalidation + new tuple version insertion
- **Delete** → logical invalidation + delayed physical reclamation

也就是说，所谓 CRUD，在 PostgreSQL 内部其实对应的是：

- 记录的**物理布局变化**
- 事务状态的**可见性变化**
- 索引的一致性维护
- WAL 可恢复性的记录

所以这里面最重要的一件事是：

**PostgreSQL 的“更新”并不是覆盖式修改，而是版本化状态转移。**

这件事决定了后面你对 MVCC、VACUUM、HOT update、索引维护，甚至 AI 时代下频繁 embedding 重建这类操作的理解。

---

## 3. PostgreSQL 的物理存储基石：Heap、Page、Tuple 到底是什么

### 3.1 Heap 不是堆排序里的 heap

PostgreSQL 中，普通表的默认存储结构叫 **heap table**。  
这里的 heap 不是二叉堆，也不是优先队列，而更接近：

> 一组没有聚簇顺序承诺的数据页集合。

也就是说：
- 记录被放到页（page）里
- 页被组织成表文件
- 记录的物理顺序不天然等于主键顺序

这和 InnoDB 那种以聚簇索引为主的组织方式在哲学上不一样。

PostgreSQL 的表更像：
- heap 存真实 tuple
- 各种 index 指向 heap tuple 的位置（TID）

### 3.2 Page 是最关键的物理单位

PostgreSQL 的 I/O 和 buffer 管理，核心单位不是“行”，而是 **page**。

通常一个 page 是 8KB。

这意味着：
- 磁盘读写以 page 为单位
- shared buffer cache 以 page 为单位
- 扫描、索引命中、局部性，最终都落到 page 行为上

所以在数据库系统里，性能问题常常不是“SQL 写得漂不漂亮”，而是：

- 访问了多少 page
- page 是否能命中 cache
- page 内部布局是否导致碎片
- 页分裂是否频繁

### 3.3 Tuple 是逻辑记录，但它有很重的事务头

一条 PostgreSQL 记录不是简单的用户数据 blob。  
它前面有 tuple header，里面带事务和可见性相关信息。

PostgreSQL 源码中，`HeapTupleHeaderData` 定义在：
- `src/include/access/htup_details.h`

关键片段像这样：

```c
struct HeapTupleHeaderData
{
    union
    {
        HeapTupleFields t_heap;
        DatumTupleFields t_datum;
    } t_choice;

    ItemPointerData t_ctid;
    uint16 t_infomask2;
    uint16 t_infomask;
    uint8  t_hoff;
    uint8  t_bits[FLEXIBLE_ARRAY_MEMBER];
};
```

这里最值得关注的是：
- `t_ctid`
- `t_infomask`
- `t_infomask2`

这些字段控制：
- 当前 tuple 是否被更新过
- 是否可见
- 是否带 null bitmap
- 是否是 HOT updated tuple
- 是否与其他版本链接

这意味着 PostgreSQL 不是把“事务信息”放在表外，而是直接把它嵌进 tuple 头里。

这也是 MVCC 得以工作的关键之一。

---

## 4. Insert 到底发生了什么：不只是“写进去”，而是一次受 WAL 与可见性约束的状态注入

现在来看你截图里圈出的部分。那几行没错，但确实太浅。下面我们把它展开。

## 4.1 一次 INSERT 的主要步骤

一个 `INSERT` 在 PostgreSQL 里，大致经历：

1. 准备 tuple header
2. 如有必要，处理 TOAST
3. 找到可插入的 page/buffer
4. 把 tuple 放进 page
5. 更新可见性图（如果需要）
6. 标记 buffer dirty
7. 写 WAL 记录
8. 提交时满足 durability 顺序约束

这不是抽象推测，而是能直接在源码里对应到。

### 关键源码入口：`heap_insert`

文件：
- `src/backend/access/heap/heapam.c`

函数：

```c
void
heap_insert(Relation relation, HeapTuple tup, CommandId cid,
            uint32 options, BulkInsertState bistate)
```

### 第一步：准备 tuple header

源码里先调用：

```c
heaptup = heap_prepare_insert(relation, tup, xid, cid, options);
```

而在 `heap_prepare_insert()` 里，会做这些操作：

```c
tup->t_data->t_infomask &= ~(HEAP_XACT_MASK);
tup->t_data->t_infomask2 &= ~(HEAP2_XACT_MASK);
tup->t_data->t_infomask |= HEAP_XMAX_INVALID;
HeapTupleHeaderSetXmin(tup->t_data, xid);
HeapTupleHeaderSetCmin(tup->t_data, cid);
HeapTupleHeaderSetXmax(tup->t_data, 0);
```

这几行非常关键。

### 它们分别在干嘛

#### `HeapTupleHeaderSetXmin(..., xid)`
把当前插入事务的事务号写入 tuple 的 `xmin`。

含义是：
- **谁创建了这个版本**

#### `HeapTupleHeaderSetCmin(..., cid)`
记录当前命令 ID。

它帮助系统在一个事务内部，区分不同语句对可见性的影响。

#### `HeapTupleHeaderSetXmax(..., 0)`
初始化 xmax。  
因为这个 tuple 现在还没有被删除或更新，所以 xmax 为空。

#### `HEAP_XMAX_INVALID`
明确告诉系统：
- 当前 tuple 的 xmax 无效
- 也就是它还没有被“结束”

所以所谓“给 tuple 写入事务可见性信息”，不是一句口号，而是很具体地把：
- 创建事务
- 当前命令
- 删除事务状态
- 可见性标志

写进 tuple header。

---

## 4.2 “找到有空间的页”到底是什么意思

源码中下一步：

```c
buffer = RelationGetBufferForTuple(relation, heaptup->t_len,
                                   InvalidBuffer, options, bistate,
                                   &vmbuffer, NULL,
                                   0);
```

这不是“随便找个位置写入”，而是在做一个页管理问题：

> 对于长度为 `heaptup->t_len` 的新 tuple，在哪个数据页上还有足够 free space，并且这个页的使用是正确的？

这涉及：
- free space map
- buffer manager
- page-level locking
- visibility map

也就是说，数据库在这里已经进入真正的存储引擎逻辑，而不再是 SQL 语义层。

### 为什么 page 选择很重要

因为 page 选择会影响：
- 写放大
- 更新局部性
- HOT update 是否可能发生
- vacuum 行为
- cache locality

所以“插入到哪一页”不是小问题，而是物理设计问题。

---

## 4.3 真正插入 tuple 到页中

接下来源码会调用：

```c
RelationPutHeapTuple(relation, buffer, heaptup,
                     (options & HEAP_INSERT_SPECULATIVE) != 0);
```

这一步才是真正把 tuple 写到 page 内部。

可以把它理解成：
- 在页内分配一个 item slot
- 把 tuple body 复制进去
- 更新行指针（line pointer）
- 给 tuple 一个物理位置 TID

这个 TID 后面就会被索引引用。

所以 PostgreSQL 的索引并不是“直接存行对象”，而是通常存：
- 索引键
- 指向 heap tuple 的 TID

---

## 4.4 WAL 为什么必须在这里出现

源码里的 WAL 部分是 INSERT 逻辑最核心的系统点之一：

```c
if (RelationNeedsWAL(relation))
{
    ...
    XLogBeginInsert();
    ...
    recptr = XLogInsert(RM_HEAP_ID, info);
    PageSetLSN(page, recptr);
}
```

### WAL 的本质
WAL = Write-Ahead Logging。  
它不是“顺手写个日志”，而是可恢复数据库的核心公理之一：

> 在真正依赖数据页持久化之前，必须先把足够恢复该修改的信息写入日志。

换句话说：
- data page 可以晚点落盘
- 但 WAL 必须先保证可恢复性

### 为什么要这样设计
因为系统可能在任何时刻崩溃。  
如果你先改了数据页，但日志没来得及记录，崩溃后系统就无法知道这次修改是不是应该存在。

所以 PostgreSQL 遵守的是：

**log first, data later**

### `XLogInsert` 在干什么
它把这次 heap insert 组织成一条 WAL record，交给 WAL 子系统。

这条记录通常包括：
- 哪个页被改了
- 插入了哪个 offset
- tuple header 信息
- tuple data
- 必要时 full page image

这就是为什么崩溃恢复能重放 INSERT。

---

## 5. Read 为什么不简单：PostgreSQL 读取的其实不是“值”，而是“当前快照下可见的版本”

一个初学者容易误解：
- 读取就是从磁盘拿一行出来

但对 PostgreSQL 来说，真正的问题是：

> 这条 tuple 对当前事务快照可见吗？

这就进入 MVCC 的核心。

## 5.1 MVCC 的真正含义

MVCC = Multi-Version Concurrency Control。  
它不是单纯“有多个版本”这么简单，而是一种并发读写语义：

- 读事务读自己的快照
- 写事务产生新版本
- 不要求所有读写相互阻塞

这意味着数据库中的“记录”其实常常不是单一对象，而是：
- 一个版本链上的某个版本
- 当前事务快照挑选出来的那个逻辑可见版本

## 5.2 xmin / xmax 是怎么参与可见性判断的

可以把 tuple header 里的关键字段理解成：

- `xmin`：谁创建了我
- `xmax`：谁结束了我（delete/update）

于是读一条记录时，系统要判断：
- 创建我的事务是否已提交
- 对当前快照来说它是否太新
- 删除我的事务是否已提交
- 如果我被更新，是否应该追到更“新”的版本

所以 SELECT 背后的本质是：

**读取 tuple + 运行一个可见性判定程序。**

这就是为什么上文说：
- R = planner + executor + visibility check

你如果没有把 visibility check 理解进去，就没有真正理解 PostgreSQL 的 read path。

---

## 6. Update 为什么不是原地覆盖：版本链才是 PostgreSQL 的核心更新语义

很多存储系统里的 update 可以想成：
- 把旧值改成新值

但 PostgreSQL 的 update 逻辑更接近：

1. 旧 tuple 标记失效
2. 新 tuple 作为一个新版本插入
3. 两者通过 `ctid` 等形成链式关系

也就是说：

**UPDATE 本质是 DELETE + INSERT 的版本化组合。**

这会带来几个重要后果：

### 1. 更新是追加式的
不是改写，而是产生新版本。

### 2. 可见性由快照决定
老事务还能看到旧版本，新事务看到新版本。

### 3. 系统会积累 dead tuples
所以 VACUUM 必不可少。

### 4. HOT update 成为关键优化
如果更新不涉及索引键，PostgreSQL 可以尽量避免新索引项写入，只在 heap 页内维护链。

这里已经可以看出数据库内核设计的味道了：
- 不是追求“写一次最省空间”
- 而是追求“在并发和恢复语义下最稳妥”

---

## 7. Delete 为什么不是立刻删除：逻辑失效先于物理回收

DELETE 在 PostgreSQL 里，首先做的是：
- 让 tuple 对未来事务不可见

而不是：
- 立刻从页里把字节抹掉

所以删除其实分成两层：

### 逻辑层
- `xmax` 设置
- 可见性变化

### 物理层
- VACUUM 未来某个时刻回收空间

这就是为什么 PostgreSQL 里的“表大小”和“有效数据量”不是一回事。  
如果你频繁 UPDATE / DELETE，却不理解 VACUUM，就会看到表膨胀。

---

## 8. VACUUM 不是清洁工，而是 MVCC 的必要配套机制

如果 PostgreSQL 采用版本化更新，那它迟早要回答一个问题：

> 旧版本何时回收？

答案就是 VACUUM。

VACUUM 至少做几件事：
- 识别 dead tuples
- 回收可重用空间
- 更新 visibility map
- 维持系统不因旧事务 ID 耗尽而崩坏

所以从理论上讲，VACUUM 不是“性能优化选项”，而是 MVCC 设计的组成部分。

这件事在 AI 场景里尤其重要。为什么？
因为 AI 系统常有大量：
- embedding refresh
- chunk 重建
- 元数据修正
- retrieval cache 更新
- trace record 写入

这些 workload 会让表更容易变成：
- append-heavy
- update-heavy
- vacuum-sensitive

所以如果你想用 PostgreSQL 做 AI 基础设施，不理解 VACUUM，后面很容易踩坑。

---

## 9. PostgreSQL 的查询为什么强：本质是一个代价优化的执行系统

一个成熟数据库的差异，不只是“支持 SQL”，而是：

> 能否把逻辑查询变成一个低代价的执行计划。

PostgreSQL 的查询路径大致是：

1. parser
2. rewrite
3. planner/optimizer
4. executor

### planner 在做什么
planner 本质上在做：
- join order 决策
- scan path 选择
- index use 决策
- sort / hash / aggregate 策略决策
- cost estimation

从算法角度，它是在巨大的计划空间中寻找较优解。

这件事对传统 CRUD 很重要，
对 AI 检索也同样重要，因为 metadata retrieval 本质上也是一个查询规划问题。

---

## 10. 向量搜索为什么不是 PostgreSQL 的“附加功能”，而是它查询模型的一次空间扩展

传统 PostgreSQL 擅长：
- 精确匹配
- 范围查询
- 结构化过滤

但 embedding retrieval 问的是另一类问题：

> 在高维向量空间里，找到最相近的 k 个对象。

这类问题本质上不是关系代数里最原生的那一套，而更接近：
- metric search
- top-k nearest neighbor retrieval
- approximate nearest neighbor search

pgvector 的意义，不只是给 PG 增加一个字段类型，而是把数据库的查询对象从：
- 一维有序键
扩展到：
- 高维相似度空间

这是一次检索空间的扩展。

---

## 11. 为什么 B-tree 不能天然解决向量检索

B-tree 的前提是：
- 数据存在可比较总序
- 邻近关系和顺序有稳定关系

但 embedding 空间中：
- 维度很高
- 没有一个自然全序保留“相似性”
- 高维中距离集中现象明显

所以 B-tree 擅长：
- `id = ?`
- `created_at > ?`

却不擅长：
- “和这个 1536 维 query 最接近的 20 个向量”

因此向量检索需要新的索引结构与近似搜索策略。

---

## 12. pgvector 背后的 ANN 直觉：为什么要用近似最近邻

### 12.1 精确搜索的复杂度问题

如果对每个向量都算一次距离，复杂度通常是：

- O(Nd)

当：
- N 很大
- d 很高
- 查询很多

代价就不现实。

### 12.2 近似最近邻的核心思想

ANN 不追求绝对最优，而追求：
- 足够高的 recall
- 更低延迟
- 更少内存/计算开销

也就是说，它把检索问题转成一个典型工程折中：

- latency
- recall
- build cost
- update cost
- memory footprint

### 12.3 HNSW 的直觉

HNSW 是最值得理解的一个结构。

你可以把它理解成：
- 一个分层近邻图
- 高层是稀疏、快速跳转的导航层
- 低层是更细的局部近邻结构

查询时：
1. 从高层快速逼近目标区域
2. 再在低层做局部扩展
3. 最后得到近似最近邻

它为什么强？
因为它把“高维空间里暴力找近邻”的问题，转成了“图导航问题”。

这是一种非常经典的算法思想转换。

---

## 13. RAG 在数据库视角下，本质上不是“找相似文本”，而是“带结构约束的 top-k 检索”

很多 RAG 教程只讲：
- chunk
- embed
- top-k
- 拼到 prompt

但如果你从数据库系统的角度看，真实问题是：

> 在一堆受权限、租户、时间、类型、版本、来源约束的数据上，做 top-k 语义检索。

也就是：

```text
arg top-k over semantic similarity
subject to relational constraints
```

这个视角很重要，因为它说明：
- RAG 不是单纯向量数据库问题
- 它是**向量检索 + 关系约束 + 查询规划**的复合问题

而 PostgreSQL 的优势正是：
- 相似度检索能做
- 结构化过滤本来就强
- 事务和权限语义成熟

所以对许多 AI 系统来说，PG 不是“顺便做点向量”，而是：

**把 retrieval 放回数据库系统问题本身。**

---

## 14. metadata retrieval 为什么在 AI 时代反而比“纯向量”更关键

语义相似度只是检索的一部分。  
真正上线的 AI 系统几乎都会面对 metadata filtering。

例如：
- 仅限当前 tenant
- 仅限用户有权访问的文档
- 仅限最新版本
- 仅限某产品线
- 仅限某地区政策
- 仅限近 30 天更新

这些约束如果做不好，AI 系统就会：
- 泄露不该看的数据
- 用过期信息回答
- 失去审计能力
- 难以重现实验

所以从生产角度看：

**metadata retrieval 往往比纯向量 ANN 更决定系统可用性。**

这也是 PostgreSQL 在 AI 系统里重新重要的一个根本原因。

---

## 15. 为什么 AI 时代的数据库问题，本质上是“外部记忆系统”问题

如果从 world model / agent 的角度再往上提一层，可以这么看：

- 模型参数提供压缩后的统计记忆
- 数据库提供显式外部记忆
- 检索器决定什么记忆被激活

那么数据库在 AI 时代的角色就不只是存储，而是：

**神经系统外部记忆的可计算载体。**

这会带来新的系统要求：
- freshness
- provenance
- tenant isolation
- retrieval traceability
- deterministic replay
- embedding regeneration consistency

这些要求本质上都更像数据库问题，而不是单纯模型问题。

所以 AI 应用的基础设施竞争，某种意义上正在从：
- 谁的模型更强
转向：
- 谁能更好地组织外部记忆系统

而 PostgreSQL 的旧长处，正好在这里重新变得重要。

---

## 16. 最后的总结：为什么 PostgreSQL 在 AI 时代不是“旧系统复活”，而是“旧原理找到新战场”

如果把这篇文章压成一句话，那就是：

**PostgreSQL 在 AI 时代的重要性，不在于它突然多了一个向量插件，而在于它原本就解决了状态、一致性、约束、查询和恢复这些最难的系统问题，而 AI 恰好把这些问题重新推到了前台。**

所以你现在再看 PostgreSQL，就不该只把它理解成：
- 用户表数据库
- CRUD 后端

而应该把它理解成：

- 一个版本化状态系统
- 一个可恢复的事务引擎
- 一个代价优化的查询执行平台
- 一个能从一维键检索扩展到高维语义检索的数据库系统
- 一个可能承载 AI 外部记忆与 RAG 约束检索的统一底座

这才是它在 AI 时代最值得重新认识的地方。

---

## 17. 如果继续写下去，下一篇最值得展开什么

从这里继续往下，我认为最值得拆成独立文章的是：

1. **PostgreSQL 的 MVCC、VACUUM 与可见性判定源码导读**  
2. **pgvector / HNSW 的算法与工程实现**  
3. **RAG 作为 constrained query planning 的数据库视角**  
4. **Agent memory systems as database systems**  
5. **Provenance、auditability 与 grounded generation 的系统设计**

这些方向都会比“怎么用数据库”更接近你要的：

**算法、数据结构、系统认识，以及论文式的原理讨论。**
