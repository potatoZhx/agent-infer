# MORI 状态与执行边界重构方案（2026-10-28）

本文基于 `f96a84fe76c56ddbe8e6e916ba405278af227138`，回答是否需要进一步重构，并记录删除客户端
`lineage_id` 的已实现变更。下文明确标注“已实现”和“建议后续实现”，不将规划中的能力算作现有能力。

## 1. 哪些问题还需要重构

| 问题 | 本轮判断 | 当前依据 |
| --- | --- | --- |
| 客户端 lineage 协议 | 已删除；服务端自行隔离重叠请求 | `identity.py:28-91`；`request_branches.py:14-58` |
| BUSY CPU 备份驱逐 IDLE | 上一提交已修复，保留 API 2 的写入准入及拒绝语义 | `simple_cpu_policy.py:82-94`；native 回归测试 `test_mori_native_patch.py:155-294` |
| 逻辑 tier 与实际驻留不等价 | 需要统一 native 事实、传输执行和覆盖度记录 | `strategy.py:540-560`；`footprints.py:10-53` |
| 相同 hash 对应多个 block | 原生允许的行为；目录必须支持多个实例，不能按 hash 直接去重容量 | native vLLM `block_pool.py:35-97`；`capacity.py:23-83` |
| hash → Program/实例目录 | 建议补齐；radix tree 是可选查询结构，不是权威容量来源 | `footprints.py:10-53`；`runtime_state.py:57-63` |
| 请求关键路径扫描 | 仍需重构增量状态及规划执行边界 | `runtime.py:252-254,744-777`；`strategy.py:167-174,550-560` |
| typed-LRU 重建尾延迟 | 已有增量类型索引，但仍同步重建，需要单独测量与分片 | `typed_lru.py:260-272,317-344` |

表中的简写路径：调度代码在 `agentinfer/scheduling/`（MORI 专用代码在其 `mori/` 下），
缓存代码在 `agentinfer/agentcache/mori/`，native 回归在 `tests/agentcache/`。
本次检查的原生源码位于
`/data/zx-envs/agentinfer-vllm-0.31.0/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:35-97`。

不需要再次推翻论文的排序与容量规则。应该重构的是承载它们的数据及执行边界：
事实更新 → 增量记账 → 选择迁移目标 → native 执行/回报 → 校验并提交状态。
下一阶段不重新引入 escrow、最低驻留、额外 idleness margin 或等待超时提权。
现有规则证据：`agentinfer/scheduling/mori/strategy.py:205-271,562-674`。

## 2. 删除 lineage_id：本轮已实现

删除 `AgentIdentity.lineage_id`、字段解析/序列化、命名分支映射、命名分支的 deferred 后继和激活逻辑。
调度扩展改名为 `uses_request_branches`，只表达“服务端隔离重叠请求”。客户端无需增加分支协议。
旧输入中的 `lineage_id` 与其他未识别字段一样被忽略，不再改变身份或执行顺序。
证据：`agentinfer/scheduling/identity.py:28-91,160-179`、
`agentinfer/scheduling/strategy.py:62-65`、`tests/scheduling/test_mori_branches.py:60-85`。

规则如下：

1. 默认 owner 没有未完成请求时，新请求沿用该 Program；使用真实 prompt 替换 context，可以缩短或切换。
2. 默认 owner 仍有未完成请求时，每个新请求获得独立临时 owner。前一个请求仍在 MORI 等待准入也算重叠。
3. 临时 owner 独立维护 context、执行状态、tier 和预算；完成或确认取消后释放保护，native 可继续缓存。
4. 不猜哪个输出被应用采用，不从完成顺序、请求大小或相似 prompt 推导稳定分支。
5. 不伪造应用层 subagent 或父级阻塞关系；真实 subagent 继续使用已有独立 agent identity。
6. 内部槽位可复用，generation 必须递增；迟到完成/取消不能影响新请求。

依据：`agentinfer/scheduling/request_branches.py:24-58`、`agentinfer/scheduling/runtime.py:178-214,527-560`、
`tests/scheduling/test_mori_branches.py:32-118,275-297`。

`lifecycle_token` 保留，它是 API middleware 生成的服务端响应收据，不是客户端分支协议，也不是分支续接标识。
它只让 terminal 事件指向确切的最新请求，客户端无需实现或设置。
证据：`agentinfer/agentcache/core/api_adapter.py:249-253,488`、`agentinfer/scheduling/runtime.py:539-560`。

代价：没有客户端语义时，临时重叠分支不能承诺跨请求持续保护；但后续请求仍可命中 native 的实际共享块。
后续内容目录用于发现可复用物理缓存，不承担“判断哪个结果是主上下文”的任务。
默认 Program 的外部身份 tombstone 仍随外部 Program 种类增长；本轮槽位复用只约束内部临时 UUID 数量。
证据：`agentinfer/scheduling/request_branches.py:24-32,49-58`、
`tests/scheduling/test_mori_branches.py:275-297`。

## 3. 建议后续实现：一个事实目录，两个容量视图

保留权威边界：native 拥有物理分配、hash、refcount、传输和 READY 事实；scheduler 拥有 Program 身份、
生命周期、目标 tier 和准入状态。统一目录是这些事实的版本化投影，不另起一套互相覆盖的物理状态。
当前 owner/class/footprint 桥接见 `agentinfer/agentcache/mori/runtime_state.py:24-126`。

建议目录结构：

| 实体/关系 | 必须记录 |
| --- | --- |
| 内容节点 | native 完整前缀 hash、模型/缓存 namespace、KV group、块长度 |
| 物理实例 | pool incarnation、tier、block ID、allocation generation、内容版本、READY/在途/失效 |
| owner → 内容需求 | Program generation、prompt version、当前需求；不代表已经驻留 |
| owner → 物理实例 | native 确认的绑定或保护关系、引用状态 |
| 传输 | transfer ID、源/目标实例及版本、目的、字节数、完成/失败 |

同一内容节点可以关联多个 GPU/CPU 实例。同 hash 的两块物理内存仍计两份；只有 native 确认共用同一实例
才能扣减共享容量。若要做最长前缀查询，可以在内容节点上加 radix 结构，先用 hash 映射也能完成一致性任务。
依据：`agentinfer/scheduling/mori/capacity.py:23-83`；上述 native `block_pool.py:35-97`。

逻辑视图计算 owner 的 context、decode 余量和已证实的共享部分；物理视图计算唯一实例、保护者、就绪及在途。
每次请求替换、增长、层级变化、实例绑定/复用、驱逐或复制完成，只更新受影响对象及其反向索引。
保留原容量公式与安全检查，改变维护方式，不额外增加一层申请额度的门槛。
原始每轮账本构造：`agentinfer/scheduling/mori/strategy.py:550-560`；
原始增减计算可复用：`agentinfer/scheduling/mori/capacity.py:44-83`。

native hook 应覆盖分配、命中绑定、hash 注册/失效、驱逐、refcount/可驱逐性、reset 和 transfer 完成/取消。
复用旧 block ID 前必须同步失效旧证明；队列丢事件或版本断档时，该范围暂停共享抵扣并触发重建，不能继续
使用过期减免。目标是单 engine writer 同步更新安全事实，异步部分只读带版本的投影。
当前 hook 边界依据：`agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch:63-136,177-207`、
`agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch:28-147`。

## 4. 建议后续实现：让物理降层可确认、可撤销

上一轮只解决了“可选 BUSY 备份不能挤掉 IDLE”。它没有新增“降层时保证 CPU 副本齐全”的复制协议。
当前逻辑 pause 即改变 tier；native 仍依靠 eager store / on-demand load。
依据：`agentinfer/scheduling/mori/strategy.py:491-507,562-596`、
`agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch:221-240,280-315,395-411`。

建议把 `target_tier`、逻辑保护 tier、实际驻留覆盖分开，native store 明确区分可选备份与已批准的降层复制：

- 可选 GPU 备份沿用 API 2：无法获得不高于自身保护等级的 CPU 空间，就跳过或稍后重试。
- 必要降层在当前 Reasoning 完成或 native 确认取消后，按目录检查源实例仍有效、目标容量是否可分配。
- 无 CPU 配额时直接选择 Waiting；有配额则先预留物理目标块，再复制并跟踪完成。源引用和保护保留到安全边界。
- CPU 实例实际 READY 后提交覆盖信息，才把该次迁移标为完成。不能先从 GPU 账本减掉 pending 的释放量。
- 源已丢失、目标无法获得或复制失败时，撤销相关预留并重新规划/降到 Waiting，明确需要重算。
  不保留没有传输任务、没有执行中推理、也没有可产生进展事件的“等待完成”状态。

这要求 CPU 目标保护与 GPU 源保护可以在传输期间分别表达。不能仅使用当前 Program 的一个 BUSY/IDLE
字段决定所有池的行为，也不能让客户端宣称“必要降层”绕过 CPU 准入。
权限来源应是服务端带 owner generation/prompt version 的已批准迁移，native 再校验实例及目标预留。
当前待补边界：`agentinfer/agentcache/mori/simple_cpu_policy.py:82-94`、
`agentinfer/scheduling/mori/factors.py:35-47`、`agentinfer/scheduling/mori/strategy.py:562-626`。

这是原生迁移执行协议，不是给某个等待候选独占未来容量的新 escrow。实际目标块会在完成/取消/失败时
释放，其他候选仍按当前合法余量继续准入。传输超时仅用于识别任务失败，不作为新的调度优先级规则。

## 5. 建议后续实现：先增量维护，再移出关键路径

当前即使 GPU 快速准入不再重建账本，调用它之前 runtime 仍构造全体快照；新 owner 准入仍扫描账本和等待者，
`before_schedule()` 同步运行全局规划。不能把策略函数的快速返回等同于整个请求路径 O(1)。
依据：`agentinfer/scheduling/runtime.py:252-254,744-777`、
`agentinfer/scheduling/mori/strategy.py:156-174`、`agentinfer/agentcache/core/scheduler.py:239-256`。

建议按以下顺序落地：

1. 持久化增量 ledger、等待集合及计数、owner 索引和 native 目录。请求路径只更新自身及实际变动块，
   GPU owner 仍直接提交；需要选优的事件只唤醒规划。保留全量重算作为测试/审计 oracle，不在每次请求中调用。
2. MORI 使用按对象版本发布的增量规划视图；不改变 Progress-TTL 等策略现有快照接口的行为。
3. 规划器在固定时刻计算 idleness，按论文顺序选候选/victim，产出带相关 owner/prompt/实例版本的计划。
   引擎只校验并提交计划涉及的对象及当前总预算，不重新扫描全体。无关 owner 更新不应导致整批计划失效。
4. native 可分配性在 engine writer 上最终校验。过期建议只失效相关动作，重试不撤销其他已授权请求，
   不建立“必须先放行某个候选”的排他屏障。规划器失败时保留已授权推理的完成通路。
5. 再选择后台执行载体：Python 线程仍会竞争 GIL，不能预先宣称它隐藏了成本。评估辅助进程的事件传输与
   序列化成本，或 native 辅助线程。引擎线程不等待规划 RPC，同步关键路径只做增量事实更新与局部提交。

排序结构依键的性质选择：静态 context、固定优先级和 LRU 可以维护索引/堆；idleness 是随时间变化的函数，
可在规划时刷新排序。只更新旧堆顶不能保证当前全局次序。第一版允许把全体排序放在非请求路径的规划器中，
不为了理论上更小复杂度先引入复杂的时间交叉事件堆。
当前 idleness 计算：`agentinfer/scheduling/mori/strategy.py:731-756`。

typed-LRU 的增量反向索引继续保留。其重建建议分片生成新堆，并记录期间的变更；切换时校验版本、补齐增量，
旧结构在新堆可用之前继续服务。不能简单限量弹出后返回任意同类块并仍声称精确 LRU；也不能用普通 LRU
回退绕过 CPU 写入保护。需要先测 stale-pop 与重建耗时，确定这一项的实际优先级。
当前同步重建证据：`agentinfer/agentcache/mori/typed_lru.py:260-272,317-344`。

## 6. 验收与推进顺序

本轮已实现协议删除，回归覆盖默认串行续接、排队期间重叠、临时分支释放、取消隔离、generation 复用、
迟到响应及 Reasoning 完成后降层。结果见 `artifacts/mori-no-client-lineage-20261028/validation.json`。
当前代码仍是同步规划；第 3–5 节是后续设计，不是已实现能力。

后续建议分为可独立验证的三个阶段：

| 阶段 | 交付 | 必须验证 |
| --- | --- | --- |
| A | native 实例目录和增量账本 | 随机事件序列对照全量重算；hash 重复、ID 复用、reset、取消、共享边际容量 |
| B | 必要降层的原生确认协议 | CPU 全 IDLE、目标 pinned、传输失败/取消、源失效、重复及迟到 ACK，无悬空 pending |
| C | 异步规划与局部提交，按测量处理堆重建 | stale plan 不越权；无全体扫描的到达路径；planner 故障仍允许运行中请求完成 |

性能验收记录 arrival 与 before_schedule 的 CPU 时间及 p50/p95/p99、Program/块读取量、变更块数量、
事件积压、计划失效率、CPU READY 覆盖、D2H/H2D bytes 和重算率；最终用相同 trace 与配置跑 32/16 对照。
只减少源码中的循环数量不足以证明性能改善。本轮未启动 GPU benchmark。

## 7. 重构前论文复核补充（2026-10-28，基于 db7870d）

再次逐项核对后，第 3–5 节仍不足以作为完整实施规格。详见
[论文符合性复核](mori-paper-audit-20261028.zh-CN.md)，其中包含源码索引及五个 CPU-only 复现。
该复核修正“只需重构状态与执行边界”的范围判断，尚未修改运行时代码。

- 增加阶段 0：response stop/end_turn 不应默认代表整个 Program 终止；旧 OffloadingConnector 缺少新的
  写入保护，需收紧 full 能力入口；等待推理置换 Acting、固定 decode buffer 和 max_tokens 上界拒绝必须
  明确其工程性质，不能视为论文已规定；首次 native 执行前的排队时间也不能计入实时 idleness。
  证据见复核 §3.1/3.2/3.5/3.6/3.8。
- 阶段 A 除清空共享证据外，也要撤销旧 prompt 不再需要的 typed owner 保护，保留其他共享者。
  不能仅更换 ledger 数据结构。证据见复核 §3.4。
- 阶段 B 的 READY 目标限定为实际已物化、可缓存的范围；同时覆盖 H2D 和缺失段 prefill 的执行反馈。
  不增加“完整 GPU KV READY 后才提交 native”的外部门槛。证据与执行建议见复核 §3.3。
- 阶段 C 加入 HBM 普通 LRU fallback 的一致性恢复，以及 native 已提交请求均为 BUSY 的不变量验收。
  当前代码没有另加 native batch 类型排序，多副本全局 Waiting/亲和/装箱也不属于本轮范围，见复核 §2/3.7。

保留 generation、native 安全校验、CPU 备份写入保护和无排他受益者的推进路径；不恢复已删除的 lineage
协议、escrow、驻留阈值或等待超时提权。具体执行顺序与新增验收见复核 §5。

## 8. 状态与置换语义再次核对

实施前以[审查报告 §7](mori-paper-audit-20261028.zh-CN.md#7-注释复核状态计时容量估计与跨层置换)为最新解释：

- 请求到达仍进入 Reasoning；只修正所选计时口径的一致性，不把 native 待执行请求改为 Acting。
  此前计时复现使用有历史的返回 Program，不是新 Program 的首个请求；原文未指定 native 事件计时 API。
- 不因论文未给出 estimator 公式而直接删除全部增长预留或单请求可行性检查。固定 100-token 值是工程估计，
  `max_tokens` 上界检查是服务能力约束；应分别说明和验证，保持 GPU owner 不二次申请额度。
- 撤回“等待 Reasoning 无条件置换 Acting”作为默认论文基线的建议。超额修复与有余量提升保持论文规则；
  无余量时的相对分界交换需单独定稿，不能把降层 victim 的状态排序扩展为全局抢占权，也不能虚计候选制造超额。
- 若采用严格相对 idleness 交换，这是 §4.3 的实现解释，必须验证多 victim、共享块、平局和推进条件；
  不恢复额外 idleness margin/驻留阈值/超时优先级，不声称论文已经给出该实现或无死锁证明。

证据：审查报告 §7 所引论文 §4.1–4.3.1；`agentinfer/scheduling/mori/strategy.py:217-271,701-753`。
本节是设计修订，尚未修改上述源码。

## 9. 用户最终确认与本次调度语义落地

本节覆盖 §7/8 中已被用户收回的实施建议：**不处理 native 初始排队计时，暂时保留 100-token 预留**。
不新增执行时间戳协议，也不删除已有计时 hook；现有 `max_tokens` 服务能力检查保持原样。
证据：`agentinfer/scheduling/mori/config.py:18-32`、`strategy.py:72-92,719-727`；
相对基线的未改动校验见 `artifacts/mori-relative-rebalance-20261028/validation.json`。

本次已实现“无条件等待推理置换”的重构：容量超额按原规则修复，有余量按原顺序提升；仅在候选严格比
回收对象更 busy、所有对象均为 Acting、整组边际回收能使候选 fit 时交换。平局保持 placement，无法 fit
就继续后面的候选，不建立跨轮独占计划。共享块通过一次构建的前缀容量投影计算，并在每次准入后更新。
实现：`agentinfer/scheduling/mori/strategy.py:206-289`、`capacity.py:83-138`。

完整算法、复杂度、进展边界和验证索引见
[相对分界置换重构](mori-relative-rebalance-20261028.zh-CN.md)。Reasoning 完成边界、GPU 直通、重叠分支
与 CPU 备份保护继续回归。该交付完成调度语义阶段，不代表 §3–5 的 A/B/C 原生目录、迁移确认和后台规划
已经实现；这些阶段及复核中其他生命周期/兼容路径问题仍保留为后续工作。
