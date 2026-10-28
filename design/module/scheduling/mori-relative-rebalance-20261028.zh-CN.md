# MORI 相对分界置换重构（2026-10-28）

## 1. 本次交付与冻结条件

基于 `db7870d8e3db50d7f14afe200b4a521803fd4ecf`，本次实现迁移触发规则的重构：将真实容量超额修复、
已有余量提升、相对 idleness 置换分开，删除“等待 Reasoning 可以无条件回收任意 Acting”的行为。
这是[状态重构方案](mori-state-refactor-20261028.zh-CN.md)中的调度语义阶段；该文 A/B/C 的完整 native
实例目录、迁移确认和异步规划仍是后续阶段，不能视作本次已交付。

依据用户最终确认：

- 不新增 native 初始排队时间的剔除处理，不改变既有 Reasoning 状态和计时路径。
- 保留默认 100-token decode 预留、块对齐及仅截断推测余量的规则。
- 不恢复客户端 `lineage_id`、escrow、最短驻留、idleness margin 或等待超时提权。
- GPU owner 的请求仍直接准入，不增加第二次容量申请；Reasoning 仍在当前推理完成或确认取消后才降层。

源码：`agentinfer/scheduling/mori/config.py:18-32`、`strategy.py:72-92,141-188,581-645,719-727`。
相对于基线未改函数的 AST 文本校验记录在 `artifacts/mori-relative-rebalance-20261028/validation.json`。
下文 `strategy.py`、`capacity.py` 均位于 `agentinfer/scheduling/mori/`，行号对应本次实现。

## 2. 三种决策路径

[论文 §4.3–4.3.1](https://arxiv.org/html/2606.00866#S4.SS3)分别描述相对分界的 sticky rebalancing、
GPU 超预算降层和有容量时的提升，没有给出多 victim、共享物理块情形的完整交换伪代码。
下面的严格相对交换是本实现对分界描述的具体化，不能称为作者公布的精确算法。

| 前提 | 决策 | 约束 |
| --- | --- | --- |
| 已有 GPU 工作集超预算 | 按 Acting 优先、同状态 idleness 高者优先修复 | Reasoning 只登记 lazy demotion；pending 释放不可用于新准入 |
| 无超额且余量能容纳候选 | 按 CPU 返回、Waiting 返回、新 Program 顺序提升 | 同组 idleness 低者优先；新 Program 先按 context 大小 |
| 无超额、存在候选但余量不足 | 只考虑严格比候选更 idle 的 GPU Acting 驻留者 | 整组边际收益足以容纳候选后才执行；平局保持 placement |
| 所有合格回收对象仍不足 | 跳过此候选，继续检查后续候选 | 不部分回收后白等，不保留独占受益者或未来空槽 |
| 无等待推理且无超额 | 保持 placement | 不因刚进入 Acting 就迁移 |

入口与执行：`strategy.py:206-289`；超额修复：`strategy.py:291-302,581-670`；
顺序：`strategy.py:672-708`；完成边界：`agentinfer/scheduling/runtime.py:501-537`。

这里的比较针对每个被置换的 owner，而非只比较最 idle 的一个。例如 A=0.9、B=0.1、候选 C=0.5，
若仅回收 A 不够，就不能把 B 一起换出。若另一个候选 D 可使用余量或合法交换，则继续放行 D。
回归：`tests/scheduling/test_mori_strategy.py:308-383`。

## 3. 多 owner 共享与回收可行性

对当前集合 S，容量为：唯一已证明 GPU block 的 token 总量，加上各 owner 尚无物理共享证明的私有承诺。
私有承诺包括未知上下文和 decode 余量。相同内容 hash 不能替代“相同实际 block”的证明。
回收 owner 时，仍被其他 owner 需要的块不算可释放收益；候选继续使用的块也必须计入最终集合。
账本：`capacity.py:23-80`；实际 block 证明过滤：`strategy.py:557-566`。

本轮先将 Acting victims 按 idleness 降序排列，再构建 `ReliefProjection`：

1. 保存移除前 0、1、…、V 个 victim 后的容量，并记录每个共享块的最后一个 victim owner 的位置。
2. 有非 victim GPU owner 的块在所有投影边界都保留。
3. 二分找出严格比候选更 idle 的 victim 前缀；先检查整个合格前缀是否足够。
4. 足够时，再二分找最短可行前缀，按原顺序提交实际降层。避免每移除一个 victim 都重扫候选的所有块。
5. 成功准入后，用 Fenwick 树的区间增量更新后续投影；新准入 owner 的共享块成为本轮保留块。

投影只用于计算，不释放任何承诺。提交后仍检查实际 ledger；转换被拒绝就停止本轮后续交换，
保留可使用现有余量的准入路径，下一轮从当前状态重新计算。没有跨轮持有的计划或等待对象。
实现：`capacity.py:83-138`、`strategy.py:242-288`；拒绝回归：`tests/scheduling/test_mori_strategy.py:361-383`。

这依赖当前单 engine writer、同步规划的执行边界。将来移到后台时，必须增加局部版本校验，不能把本轮投影
直接跨线程复用。原有 native refcount、传输 fence 和实际分配校验继续是物理安全边界。

## 4. 开销与验证口径

令 P 为 Program 数、V 为 Acting victim 数、E 为 owner-block 关系数、b 为候选自身 block 数：

- 候选与 victim 各排序一次，约 O(P log P)；账本和回收投影构建约 O(P+E)。
- 一次前缀容量查询 O(log V+b)，无需重新扫描全部 owner 或复制整个账本。
- 对不 fit 的候选，一次全合格前缀检查就跳过；可交换候选用 O(log V) 次查询找最短前缀。
- 每个实际 victim 本轮最多提交一次；每次准入更新约 O((1+b) log V)。

这不是“所有请求路径已 O(1)”或“全轮严格 O(P log P)”的声明。arrival 快照、CPU/GPU repair、diagnostics
仍有全量扫描，实际代价还取决于共享关系数量。增量权威状态与后台规划仍见原方案 A/C。
证据：`agentinfer/scheduling/runtime.py:744-777`、`strategy.py:167-174,291-302,329-345,568-579`。

回归包括：

- 0/0、1/1 平局及候选更 idle 时不置换；不足以完成交换时不部分降层。
- 放不下的 CPU 高优先级候选不阻挡可换入的 Waiting 候选，也不阻挡余量回填。
- 50 个固定随机种子的共享块场景，对照独立全量 union 规划器，验证完整 transition 序列与最终容量。
- 30 次连续准入、每次 21 个 victim 前缀，对照全量 ledger，验证投影与共享折扣变化。
- GPU 增长超额、Reasoning 完成后释放、CPU pending 目标计费、重叠分支、100-token 余量继续回归。
- runtime 在完成瞬间保持平局，工具等待后由下一次周期检查完成交换，无需新请求唤醒。
- 64/128/256/512/1024 个候选与 victims 的 CPU 探针记录实际读取/查询次数，不用模拟数据声明 GPU 收益。

测试源码：`tests/scheduling/test_mori_capacity.py:56-83,143-267`、
`tests/scheduling/test_mori_strategy.py:185-205,308-414`、`tests/scheduling/test_mori_runtime.py:132-154`。
命令、原始输出、源码 hash、未运行项目统一记录于
`artifacts/mori-relative-rebalance-20261028/validation.json`。未运行 GPU 32/16 性能实验。

本次整组 CPU 检查为 **477 passed、26 deselected、14 warnings**，适用 pre-commit hooks 均通过。
不 fit 探针的 factor 读取数为 513/1025/2049/4097/8193，容量查询数为 64/128/256/512/1024；
只描述该场景的操作次数，不是端到端加速比。
原始结果：`artifacts/mori-relative-rebalance-20261028/pytest.txt:1-14`、`probe.json:34-106`。

## 5. 进展边界

删除了“候选仍在就一直等待旧回收计划”的依赖。候选每轮重新评估，不能 fit 就继续后面的请求；GPU tier
请求也不会占着额度却被额外门槛锁住。已授权 Reasoning 的完成通路保持原样。

但严格相对优先级不是无饥饿证明：候选比所有可回收者更 idle 时会等待；所有相关分数恰为 1 的平局也不会
被人为打破。正常持续 Acting 的分数会随工具耗时上升，周期循环会重新比较，但没有固定等待时限。
不能为了声称“绝不死锁”再加入超时抢占、驻留阈值或独占预留。
证据：`strategy.py:249-257`、`tests/scheduling/test_mori_strategy.py:308-321`、
`tests/scheduling/test_mori_runtime.py:132-154`。

native 执行、生命周期、完整驻留覆盖以及后续 A/B/C 中的其他已识别问题，仍以
[论文复核及用户修订](mori-paper-audit-20261028.zh-CN.md)和[状态重构方案](mori-state-refactor-20261028.zh-CN.md)
为后续工作清单；本次没有修改 native offloader 或部署中的 vLLM。
