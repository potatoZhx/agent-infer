# MORI 论文机制收敛与 CPU 写入保护（2026-10-28）

本文补充本日早先的重构文档，以当前工作树为准。早先提交中的设计与实验记录继续保留。
本次基线为 `36e3dab`，另有未提交的失效回收计划修补及回归测试，已保存到
`artifacts/mori-paper-alignment-20261028/before.patch:1-95`。本次保留其回归目的，删除整个持久化受益者机制。

## 调度调整

GPU tier 本身就是当前分支的准入决定。请求直接提交 native，引擎仍按自身 batch 和物理容量调度；
“已准入”不等于已经在 GPU 执行。新 prompt 或 decode 增长唤醒统一容量修复。删除已有 GPU owner
重新申请额度的入口门槛，避免同一 owner 一边被计入 GPU 工作集，一边又被 MORI 留在引擎外。
超额时先回收 Acting；必要时标记 Reasoning，等待当前授权请求完成或 native 确认取消后提交降层。
串行后继不延长这次 drain，独立重叠分支继续独立计费。
证据：`agentinfer/scheduling/mori/strategy.py:141-188,562-626`、
`tests/scheduling/test_mori_lineages.py:86-101,118-170`；论文 [§4.1、§4.3.1](https://arxiv.org/html/2606.00866#S4.SS3.SSS1)。

删除 `relief_candidate`、`relief_for`、最短驻留、额外 idleness margin 和等待超时提权。
配置不再接受 `min_residency_seconds`、`idleness_margin`、`max_wait_seconds`；
保留 diagnostics 中表示实测最长等待时间的同名指标。
证据：`agentinfer/scheduling/mori/config.py:18-32`、`agentinfer/scheduling/mori/factors.py:18-62`、
`agentinfer/agentcache/core/controller.py:82-119`。

等待者按 CPU 返回、Waiting 返回、新请求（小 context 优先）排列，再按 idleness 排序。
每个候选都检查实际可用容量，放不下则继续后面的候选，不保留独占未来回收额度。
推理需求不足以容纳时，可以回收 Acting owner，按 idleness 从高到低选；不会为了等待超时去标记仍在推理的 owner。
同一轮候选和 victims 各排序一次，复用 victim 游标。先构造移除所有可回收 Acting 后的可行性账本，
不能让任何候选立刻进展时不做无效降层。pending Reasoning 的 GPU 承诺仍保留在这个账本中。
证据：`agentinfer/scheduling/mori/strategy.py:205-271`、`tests/scheduling/test_mori_capacity.py:88-133`。

论文没有提供完整调度伪代码。“等待推理需求可以置换 Acting 预留”的具体衔接沿用本地隔离实现，
把 §4.3 的重平衡与 §4.3.1 的状态优先级结合起来；不能宣称是作者源码的逐行复刻。
它不在每次进入 Acting 时自动降层，但有等待推理压力时，不再额外等待一秒或 idleness 差值。
隔离参考：`/data/zx/agent-kvc/research/studies/kvc_theory/mori-real-v031-20261022/joint/mori.py:381-413`。

## 进展与死锁边界

Acting 的 idleness 增长不是超时释放机制。该比例有上界，且表示相对排序；没有等待推理或容量压力时，
缓存可以继续驻留 GPU，这符合 sticky placement。当前测试明确验证时间过去不会自动降层。
证据：`agentinfer/scheduling/mori/strategy.py:731-756`、`tests/scheduling/test_mori_lineages.py:173-190`。

本实现能说明的条件性进展是：若所有 GPU owner 都已结束推理、状态正确、降层转换可提交，且存在一个
不超过 GPU 总预算的等待请求，则它不会被旧回收计划卡住；Acting 承诺可被回收，至少一个候选能够准入。
如果仍有推理运行，不提前花费其容量，等待完成事件再规划。无持久化受益者，也就没有“受益者尚在等待，
但原 victims 已全部结束”的闭环。单个请求不保证无饥饿，native/工具/通知失败也不由这一推论排除。
证据：`agentinfer/scheduling/mori/strategy.py:217-271`、
`tests/scheduling/test_mori_strategy.py:270-325`、`tests/scheduling/test_mori_runtime.py:84-116`。

原现场快照重放：原生请求 0，绑定 17，GPU 承诺 111,024 / 122,192 tokens，原受益者仍差 23,376 tokens。
新排序执行 3 次降层、4 次准入；不再承诺优先放行旧受益者，准入遵循当前队列优先级。
这是状态重放，未执行 GPU 推理。原始输入、重放命令与摘要见
`artifacts/mori-paper-alignment-20261028/validation.json`、`artifacts/mori-paper-alignment-20261028/replay-after.json:1-94`。

## CPU 写入保护

原 SimpleCPU 只有已有缓存的 victim 顺序：INACTIVE、BUSY、IDLE。当 CPU 全部是 IDLE 时，
一个 GPU/BUSY eager 备份仍能选择 IDLE 并替换。ref-count 为零只表示可驱逐，并不表示该块没有缓存。
新增写入准入：

| 写入所属 Program | 可使用的 CPU 目标块 |
| --- | --- |
| GPU / BUSY | 空闲、INACTIVE、BUSY |
| CPU / IDLE | 空闲、INACTIVE、BUSY、IDLE |
| Waiting 或未知 owner / INACTIVE | 空闲、INACTIVE |

共享块按所有 owner 中最强的 CPU 保护分类；同类仍按 native LRU。
允许的块数取自增量 typed index 的分类计数，不在每次 store 时遍历整个池。
证据：`agentinfer/agentcache/mori/simple_cpu_policy.py:82-94`、
`agentinfer/agentcache/mori/typed_lru.py:219-291`、`tests/agentcache/test_mori_simple_cpu_policy.py:85-110`。

普通 HBM allocation 仍可在无效建议时走 native LRU。CPU store 则必须整批验证，拒绝时保持原状态，
不得回退 LRU 绕过保护。覆盖 positional、boundary handoff、finished flush 三条写入路径；拒绝 positional
写入不推进 cursor，也不 pin GPU source 或登记传输。boundary 和已结束请求的跳过备份不保证之后能够重试。
证据：`tests/agentcache/test_mori_native_patch.py:155-294`、
`agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch`、
`agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch`。

论文 §4.3.2 明确 CPU victim 顺序与 IDLE 优先保留目标，没有明确写出 incoming-versus-victim 准入伪代码。
上述过滤是将该目标落实到 vLLM eager-store 的适配规则，不能称为论文逐字规定。
CPU tier 仍不保证完整 KV 已 READY；本次没有新增在降层时强制复制或完成确认的协议，也没有主动 prefetch。
证据：论文 [§4.3.2](https://arxiv.org/html/2606.00866#S4.SS3.SSS2)、
`docs/zh/how-to/use-mori.md:100-128`。

## 部署与验证

SimpleCPU hook 升为 API 2；BlockPool API 1 新增 `store_admission_api_version=1` 能力。
必须从匹配的干净 native 源码应用本次两个 patch 后重建。不能将当前 AgentInfer 和旧 API 1 CPU patch 混用。
本次仅在临时目录应用并运行原生 Python 管理器测试，没有修改已安装 vLLM 或重启实验服务。
证据：`tests/agentcache/test_mori_native_patch.py:101-153,279-294`；
部署说明：`agentinfer/agentcache/patches/README.md`。

测试命令、结果、源码及输入哈希记录于 `artifacts/mori-paper-alignment-20261028/validation.json`。
覆盖调度、lineage、取消、容量共享、CPU 类型与三条 native 写入路径；native 测试使用真实打补丁 BlockPool，
不分配 GPU tensor、不执行 DMA。完整 32/16 GPU 性能实验本次未运行，不能从这些检查推断吞吐改善或全系统无死锁。

## 2026-10-28 后续修订：删除客户端 lineage 协议

`lineage_id` 字段和命名分支后继队列已移除；重叠请求由服务端分配临时独立 owner，完成或确认取消后释放保护。
无需客户端声明分支；后续真实 prompt 决定上下文，服务端不推断重叠结果的采用关系。
保留服务端生成的 `lifecycle_token`，用于拒绝迟到 terminal，不用于分支续接。
证据：`agentinfer/scheduling/request_branches.py:14-58`、`agentinfer/scheduling/runtime.py:178-214,527-560`。

尚需重构的 native 事实目录、增量账本、迁移确认与异步规划见
[状态与执行边界重构方案](mori-state-refactor-20261028.zh-CN.md)；该文明确区分已实现与后续设计。
