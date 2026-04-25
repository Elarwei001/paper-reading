# PostgreSQL 的 MVCC 与 VACUUM：从可见性判定到版本回收的源码导读

> 这篇文章聚焦 PostgreSQL 最核心、也最容易被误解的一套机制：MVCC 与 VACUUM。目标不是讲概念名词，而是解释它们如何共同构成 PostgreSQL 的并发与存储语义。

---

## 1. 为什么 PostgreSQL 必须用 MVCC

任何数据库都要面对一个根问题：

> 当多个事务同时读写同一份数据时，系统如何既保证正确性，又不让所有操作互相堵死？

最粗暴的办法是：
- 写的时候锁住
- 读的时候等锁

这能保证正确，但吞吐会很差。

PostgreSQL 走的是另一条路：

**让数据保留多个版本，让不同事务按自己的快照看到不同版本。**

这就是 MVCC。

---

## 2. PostgreSQL 的 MVCC 不是抽象理念，而是 tuple header 里的物理事实

MVCC 在 PostgreSQL 里不是“外面有个版本表”，而是每个 tuple 自带事务元信息。

关键结构在：
- `src/include/access/htup_details.h`

源码片段：

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

理解 PG 的 MVCC，最重要的是这几个域：
- `xmin`
- `xmax`
- `ctid`
- `infomask`

其中：
- `xmin` 表示创建这个 tuple 版本的事务
- `xmax` 表示使这个版本失效的事务
- `ctid` 通常指向当前版本或更新后的新版本
- `infomask` 记录提交/失效/锁等状态位

所以 PostgreSQL 的一行记录，不只是业务数据，还是一个带事务历史的对象。

---

## 3. Read 为什么本质上是“可见性判定”

很多人以为 SELECT 就是把行从磁盘拿出来。  
其实 PostgreSQL 需要额外回答：

> 这条记录对当前事务快照可见吗？

判断逻辑依赖：
- `xmin` 对当前快照是否已提交且不太新
- `xmax` 是否为空 / 无效 / 未提交 / 对当前快照还没生效
- 当前事务是不是自己创建或修改了它

这就是所谓 visibility check。

所以数据库里的“读”，其实是：

**取出一个物理版本，然后解释它在当前时间切片下是否存在。**

---

## 4. Update 实际上是版本追加，不是原地覆盖

在 PostgreSQL 中，UPDATE 的逻辑更接近：

1. 旧版本 tuple 写上 `xmax`
2. 生成一个新版本 tuple
3. `ctid` 连接版本链
4. 新事务看到新版本，旧事务还可能看到旧版本

所以 UPDATE 不是“改值”，而是：

**创建一个新版本，并让旧版本在未来某些快照里失效。**

这点是理解 VACUUM 的前提，因为 dead tuples 就是这么来的。

---

## 5. Dead tuple 为什么不可避免

既然 UPDATE / DELETE 都不是立刻物理删除，数据库里就会残留一些：
- 对所有活跃事务都已不可见
- 但物理上仍占空间

的旧版本，这些就是 dead tuples。

它们不是 bug，而是 MVCC 的自然副产物。

只要你选择：
- 并发友好
- 读写分离快照语义

你就必须处理版本残留问题。

---

## 6. VACUUM 的本质不是打扫卫生，而是 MVCC 的闭环补偿机制

VACUUM 做的事情，不是简单“把垃圾删掉”，而是：

1. 判断哪些 dead tuples 已经不可能再被任何快照访问
2. 回收这些 tuple 占用的空间
3. 维护 visibility map
4. 避免事务 ID wraparound
5. 控制表和索引膨胀

所以从系统设计角度，VACUUM 是：

**MVCC 设计的必要后处理机制。**

没有 VACUUM，MVCC 最终会让系统积累越来越多历史残留。

---

## 7. 为什么 VACUUM 要关心 xmin horizon

判断某个 dead tuple 是否可以回收，关键不是“它老不老”，而是：

> 是否还存在某个事务快照可能看见它？

这就需要系统估计一个“最老仍可能需要这些旧版本的事务边界”，也就是常说的 oldest xmin / horizon 一类概念。

如果一个 dead tuple 仍可能被旧快照看到，就不能提前回收。

这说明 VACUUM 不是机械扫描，而是在做：

**基于全局事务视图的安全回收。**

---

## 8. PostgreSQL 为什么害怕 XID wraparound

事务 ID 不是无限整数，而是有限空间循环使用。

如果系统长期运行，又不及时冻结/清理旧事务信息，就会出现一个致命问题：
- 很老的事务 ID 和新的事务 ID 在数值上重新重合
- 系统会失去对可见性的正确解释

所以 VACUUM 还有一个非常底层但极重要的职责：

**防止事务 ID 循环导致可见性语义崩塌。**

这也是为什么 PostgreSQL 不把 VACUUM 当作“有空再跑的优化项”。

---

## 9. HOT update 为什么重要

HOT = Heap-Only Tuple。

当 UPDATE 不涉及索引键时，PostgreSQL 可能只在 heap 内维护版本链，而不必给所有索引都新增一条项。

这很重要，因为它减少了：
- 索引写放大
- 索引膨胀
- 更新成本

也就是说，PostgreSQL 在版本化更新的高代价模型里，提供了一种很关键的局部优化：

**尽量把更新留在 heap 层内部解决。**

这也是物理布局、页空间和更新模式为什么会影响真实性能的原因。

---

## 10. 从 AI 系统角度看，MVCC/VACUUM 为什么反而更重要

在 AI 系统里，常见写操作有：
- embedding refresh
- chunk reindex
- retrieval metadata 修正
- trace / agent memory 持续写入
- workflow state update

这些 workload 通常比传统“纯只读知识库”更动态。

于是会带来：
- 更多 UPDATE
- 更多 dead tuples
- 更频繁 VACUUM 压力
- 更明显表膨胀与索引膨胀

所以如果你想用 PostgreSQL 做 AI 基础设施，理解 MVCC/VACUUM 不是数据库洁癖，而是生产稳定性的前提。

---

## 11. 一个最重要的系统结论

PostgreSQL 之所以能在高并发、强事务语义下运行，不是因为它神奇，而是因为它接受了一个设计现实：

- 更新不是覆盖
- 删除不是立刻消失
- 读取不是直接拿值
- 清理不是顺手附带

而是把这些都变成一套：

**版本化存储 + 快照可见性 + 延迟回收 + 周期性维护**

这就是 MVCC 与 VACUUM 的真正含义。

---

## 12. 下一步最值得继续读什么

如果继续往源码深挖，最值得看的方向是：
- `heapam.c`
- `htup_details.h`
- `vacuumlazy.c`
- `visibilitymap.c`
- `README` 里关于 transaction / snapshot / xlog 的部分

如果继续写文章，最自然的下一篇就是：

**《PostgreSQL 的查询规划器：为什么 SQL 最后会变成一棵执行树》**
