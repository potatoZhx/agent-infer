# MORI 重构前论文符合性复核（2026-10-28）

后续实施状态以文末 §8 为准；§1–7 和归档 probe 是基线审查记录，不能当作新版本的运行结果。

## 1. 审查基线与结论

审查提交：`db7870d8e3db50d7f14afe200b4a521803fd4ecf`。本次只审查并补充设计，不修改运行时代码。
对照 [MORI v1 §3–5](https://arxiv.org/html/2606.00866)、当前源码以及
[状态与执行边界方案](mori-state-refactor-20261028.zh-CN.md)。本地隔离实现是参考实现，不是论文作者代码或规范。

后续解释修订：§7 根据再次核对原文，收窄 §3.8 的论文符合性判断，并替代 §3.5/3.6 中关于无条件
等待压力置换和删除固定预留的建议。此前复现数据与源码事实仍然有效，不能把旧建议当作实施规格。

结论：单 rank 的主要排序规则已实现，但不能称为完整论文复现；现有 A/B/C 重构计划仍有遗漏。
应先明确调度语义和支持范围，再完成物理事实、迁移执行与异步规划。
仅把扫描改成增量索引，不能解决提前结束 Program、兼容路径保护缺失、旧上下文残留保护等问题。
证据：下表、§3 的源码索引以及 `artifacts/mori-paper-audit-20261028/probe.json:2-38`。

下文路径相对于仓库根目录：`S = agentinfer/scheduling/`、`C = agentinfer/agentcache/`；
`plan` 指上述状态与执行边界方案。源码行号冻结在本次审查提交，后续改动可能改变行号。

## 2. 逐项对照

论文机制仅在本表概括；细节以链接原文为准。表中的“已实现”指对应源码路径，不代表 GPU 端到端验证。

| 论文要求 | 当前实现证据 | 计划覆盖及判断 |
| --- | --- | --- |
| Program 跨轮跟踪；subagent 独立，§3.1/4.1 | `S/identity.py:28-91`；`S/runtime.py:180-214` | 基础已有；自动 terminal 破坏连续性的例外未覆盖，见 §3.1 |
| 最近 k 轮空闲比，默认 k=5；排除调度等待，§4.2 | `S/mori/config.py:22-24`；`S/mori/strategy.py:72-92,420-441,475-495,732-753` | 窗口已实现；首次 native 执行前的实时排队计时仍有缺口，见 §3.8 |
| GPU/CPU/Waiting 三层及容量约束，§4.1 | `S/mori/strategy.py:550-560,599-650` | 逻辑账本已有，物理覆盖不保证；plan §3–4 只部分补齐 |
| GPU 内请求直通；其他层等待，§4.1 | `S/mori/strategy.py:156-188`；`C/core/scheduler.py:207-214,252-257` | 已删除第二道 GPU 额度申请；物理执行仍由 native 控制 |
| 超预算降层；Acting 先于 Reasoning，同状态更闲者先，§4.3.1 | `S/mori/strategy.py:599-626,679-690` | 已实现；增长可能暂时超账本，不能描述成每时刻严格不超额 |
| Reasoning 完成后再降层，§4.3.1 | `S/mori/strategy.py:584-595`；`S/runtime.py:501-537` | 当前逻辑降层有完成边界；plan §4 必须保持这一约束 |
| 提升顺序：CPU 返回、Waiting 返回、新请求；相应空闲度/大小排序，§4.3.1 | `S/mori/strategy.py:223-271,654-671` | 已实现；放不下跳过，不保留旧受益者 |
| Sticky placement，§4.3 | `S/mori/strategy.py:229-271,286-294` | 无压力不动；等待推理主动置换 Acting 属具体化规则，见 §3.5 |
| 两池采用相反的类型淘汰顺序，同类 LRU，§4.3.2 | `C/mori/hbm_policy.py:20-30`；`C/mori/simple_cpu_policy.py:24-34` | 正常路径已有；HBM fallback、旧 CPU connector 需要处理 |
| native batch 按类型优先，§4.3.2 | `C/core/scheduler.py:167-214,562-581` | 未单独实现；单 rank 下获准请求均属 GPU/BUSY，可退化等价，必须测试该前提 |
| 异步周期控制，默认 5 秒，§5 | `S/runtime.py:143-154,284-308`；`C/core/scheduler.py:239-257` | 有周期及事件唤醒，但规划同步阻塞 engine；plan §5 已覆盖改造方向 |
| 多副本全局 Waiting、亲和与跨副本装箱，§4.3.1/5 | `S/runtime.py:812-819` | 未实现，A/B/C 计划也未覆盖；本阶段明确限定单 rank，不扩大重构范围 |

本表没有把“Waiting 必须立即物理删除全部副本”或“新增独立预取服务”列为强制实现项。
typed eviction 本身允许低优先级内容在无压力时残留；共享实例也不能因单个 owner 降层而删除。
当前 `CacheClass` 映射及 native 替换依据：`S/mori/strategy.py:763-780`、`C/mori/typed_lru.py:219-291`。
论文 §4.1 的层级抽象与 §4.3.2 的软优先级需要一起解释，不能自行推导出绝对 pin 或独占物理层级。

## 3. 重构前必须补清楚的问题

### 3.1 模型响应结束不等于 Program 结束

当前 OpenAI `stop`、Anthropic `end_turn` 被映射为 TERMINAL，MORI 随后释放空闲分支。
同一 session 后续再来时，generation 更新，原 idleness 历史和缓存保护消失。
证据：`C/core/api_adapter.py:606-637`；`S/runtime.py:550-560,681-688`。
小型复现确认 stop 后 generation 从 0 变为 1、旧 factors 被删除，见 `probe.json:8-14`。

这是对整个 Program 生命周期的额外推断。用于单次任务结束的 trace 时可能恰好成立，但不能推广到
等待用户继续输入或非 tool-call 外部动作的会话。generation 收据只能避免迟到消息误伤，不能证明会话已结束。
原 plan §2 保留 lifecycle 收据，却没有审查信号本身是否足够可靠。

建议：MORI 默认把普通响应完成视为 Acting；只有可靠的 Program 终止事实才释放整个会话。
没有终止事实时，缓存保护仍可经容量决策降级，无需增设 idle TTL。元数据清理与缓存保护回收分别处理；
不为此新增客户端 lineage/生命周期协议。临时重叠 owner 的请求完成释放仍属于服务器兼容语义。

### 3.2 CPU 写入保护没有覆盖所有 full 模式入口

SimpleCPU API 2 已限制 GPU/BUSY 备份的 CPU 替换范围，并明确拒绝回退。
但 `install_vllm_mori()` 仍接受 `OffloadingConnector + MoriCPUCachePolicy`，只检查 `MORI_CPU_POLICY` 标记。
该 policy 的 `evict(n, protected)` 不接收写入方，不限制允许替换的类型。
证据：`C/mori/simple_cpu_policy.py:82-94`、`C/mori/vllm_adapter.py:149-159`、
`C/mori_cpu_policy.py:108-132`。真实 policy API 复现：一个 CPU/IDLE 槽仍能被 GPU/BUSY 写入需要的
驱逐调用取走，见 `probe.json:2-7`。本次未执行完整旧 connector/DMA。

建议第一版 full 仅接受通过统一写入/迁移能力检查的 SimpleCPU；旧路径在补齐同等 contract 前拒绝以 full 启动。
不要把旧路径也描述成“CPU 写入保护已修复”。这项不在原 A/B/C 验收中，应提前加入能力矩阵和启动检查。

### 3.3 物理迁移计划不能只覆盖 GPU→CPU

目前 Acting pause 会立刻改变 class 并减 GPU 承诺，没有确认 D2H。
即便此前 eager 备份被拒绝或仅复制部分缓存，逻辑 CPU tier 仍可成立。
证据：`S/mori/strategy.py:498-507,563-596`、
`C/patches/vllm-0.31-mori-simple-cpu-provider.patch:221-240,280-315,385-395`。
plan §4 已识别此缺口，但它是实现层的迁移确认协议，并非论文指定的 ACK/双保护算法。

需要补充以下执行约束，防止确认机制又变成新的停滞来源：

- 覆盖目标是 native 已物化、可缓存的实际前缀；decode buffer、未计算 token、不可缓存尾部不能成为永远等不到的 ACK。
- D2H 先确认源有效、目标可用；在途源/目标分别保护。失败有完成路径，不能保留无任务的 pending。
- CPU→GPU 的 on-demand load 也需要版本/覆盖反馈。准入允许 native 执行加载或缺失段 prefill；
  READY 检查放在 native 读取 KV/执行计算的边界，不增设“完整 GPU KV 已就绪才允许请求进入 native”的门槛。
- Waiting 新请求原本就需要 prefill，不能套用全量命中条件。保留其他可运行请求的通路。

当前物化边界可见 native patch 中 aligned token 与 hash 检查：
`C/patches/vllm-0.31-mori-simple-cpu-provider.patch:310-341,376-395`。
当前 promotion 只切逻辑 tier：`S/mori/strategy.py:264-271,520-537`。

### 3.4 新目录还必须同步清除旧上下文的保护关系

新 prompt 会替换 context，并清空 footprint，但不清除 typed index 里的旧 owner。
如果 prompt 被裁剪或完全切换，已经不需要的旧块仍可标为 BUSY/IDLE；账本却只计算新 context。
这不意味着 native 绝对不能驱逐旧块，但会挤占高保护类别并误导替换排序。
证据：`S/runtime.py:202-213`；`C/mori/runtime_state.py:65-67`；
`C/mori/hbm_policy.py:38-53`。复现清空 footprint 后 block 7 仍属于原 owner，class 仍为 busy，
见 `probe.json:15-20`。

plan §3 已有 prompt version 和 owner→需求关系，但没有明确旧保护关系的撤销条件。
阶段 A 应同时维护需求、实例使用与缓存保护关系：只移除该 owner 不再需要的保护，不删除其他 owner 的共享实例。
不能只修 ledger 而继续保留过期 typed owner。native 事件的失效与 dirty 唤醒也要一起进入增量事务。
当前 footprint 回调只改索引，不主动唤醒 planner，见 `C/mori/footprints.py:18-44`；周期检查仍是兜底。

### 3.5 等待推理置换 Acting 是策略解释，需要明确保留范围

当前 GPU 总承诺尚未超额，只要排队候选能通过回收 Acting 放入，就可以置换；没有比较候选与 victim 的
相对 idleness。局部复现中 GPU 用量 800/1000、resident idleness=0.01，candidate=0.99，仍执行 pause/resume。
证据：`S/mori/strategy.py:229-271`；`probe.json:21-38`。

这一规则来自本地隔离实现的“等待 Reasoning 优于 Acting”解释：
`/data/zx/agent-kvc/research/studies/kvc_theory/mori-real-v031-20261022/joint/mori.py:384-414`。
论文没有给出这种状态的完整可执行伪代码；不能把隔离实现反向当作论文证明，也不能直接断言它违反原文。
原 plan “保留论文排序”没有明确表达这一政策选择。

建议保留需求驱动的回收入口以保证全 Acting 时仍能推进，同时明确其为论文描述的工程具体化：
只有存在可实际推进的等待推理才回收，候选保持既定层级优先序，victim 同状态按 idleness 排序；
不添加最短驻留、idleness 差值阈值或受益者独占。若选择更严格的相对 idleness 交换条件，应先重新证明进展，
不能简单删除当前入口后仅等待现有 GPU 账本超额。

### 3.6 固定预留与最大输出拒绝仍是额外机制

默认 GPU claim 使用 context+100，再按 block 对齐并截断推测余量；CPU 只按 context 对齐。
此外 bridge 在入队前拒绝 `prompt + max_tokens > HBM quota`，即使真实响应可能提前 EOS。
证据：`S/mori/config.py:22-24`、`S/mori/strategy.py:701-714`、`C/core/scheduler.py:181-194`。
例如配额 1000、prompt 600、max_tokens 500、实际只输出 10，当前仍会提前拒绝。
这属于输出上界检查的静态推论，本次未运行此请求的 GPU 推理。

两者都不是论文公布的具体公式。前者可能降低可同时容纳的 Program 数，后者改变请求接受集合；
将它们作为固定值不能使其自动成为论文规则。原 plan §3 写“保留原容量公式”，因而没有收敛这些差异。

建议论文基线采用实际 context、block 对齐和 native 实际需要的分配粒度；额外 100-token 前瞻不作为固定论文参数。
对单请求可行性，保留真实不支持输入的明确错误，但重新核对 native 的模型长度、最大输出和实际分配约束，
决定是否确有必要保留整个 max_tokens 上界拒绝。不能直接删除全部 native 安全检查，也不能继续称它为论文准入规则。
阶段 A 需要覆盖 decode 连续增长、接近容量和多个已准入请求的 native 抢占；单请求上界可行不证明总并发绝不超额。

### 3.7 HBM fallback 与 native batch 类型前提

HBM provider 建议失效时仍直接走普通 native LRU；这保留可用性，但可能越过类型顺序。
证据：`C/patches/vllm-0.31-mori-hbm-provider.patch:105-129`。
plan §5 只强调 CPU 不得 fallback，尚未规定 HBM 的一致性恢复。
建议优先重建相关有效候选并重试；确需普通 LRU 应作为可观测降级，不能再宣称该次执行符合类型排序。
恢复失败不能阻断已有推理完成。不要为消灭指标上的 fallback 引入永久等待。

当前 native batch 没有另加 BUSY/IDLE key。单 rank 路径里，CPU/Waiting 请求留在 MORI，已提交请求直到完成
都保留 GPU/BUSY，因此该优先级通常退化为同类。建议先测试这个不变量，避免无效增加另一套请求排序器；
将来出现跨类型 native 队列时才补齐。证据：`C/core/scheduler.py:167-214,252-275,574-581`、
`S/mori/strategy.py:156-162,584-595,775-780`。

### 3.8 首次 native 执行前的实时 idleness 仍混入排队时间

admit/resume 会先设置 `reasoning_started_at`，首次 `running=True` 才重置计时。在这两个事件之间，
`_idleness()` 仍用 now 减这个临时起点，将 native 初始排队暂计为 Reasoning。
证据：`S/mori/strategy.py:72-92,520-535,732-753`。
复现中只排队 100 秒、没有执行模型，idleness 从 0.9 降到约 0.0818；首次执行通知又恢复到 0.9。
见 `probe.json:39-44`。这会影响期间的选优，不能以完成时统计已修正为由声称全过程排除了等待。

建议区分已准入、native 待执行、实际执行和抢占等待；首次实际执行前不累积 Reasoning，保留此前完整窗口。
native 执行事件已经存在，修正计时不需要增加客户端信息或新的准入控制。阶段 0/A 增加实时排序回归，
既检查完成后的时间总量，也检查 native 排队期间分数不随排队时长改变。

## 4. 额外实现的处置清单

| 机制 | 性质与建议 | 证据 |
| --- | --- | --- |
| lineage 协议、命名分支后继 | 已删除，保持删除 | `S/request_branches.py:24-32`；`tests/scheduling/test_mori_branches.py:60-85` |
| escrow、驻留阈值、idleness margin、等待超时提权 | 已删除；禁止在异步计划中换名恢复 | `S/mori/config.py:18-32`；`S/mori/factors.py:20-47`；`S/mori/strategy.py:205-271` |
| 临时重叠 owner、可缩短 prompt | 真实流量适配，论文单链模型未定义；保留并明确不保证临时分支跨轮保护 | `S/request_branches.py:24-43`；`S/runtime.py:202-213,527-537` |
| generation、取消 ACK、实例版本、null block 扣除 | 正确性适配，保留；不增加候选排序条件 | `S/runtime.py:516-560`；`C/core/scheduler.py:141-144` |
| 两池共享块最强 owner 分类、按真实实例去重 | 多 owner 适配，保留；不能用相同 hash 推断同一物理实例 | `C/mori/typed_lru.py:313-315`；`S/mori/capacity.py:30-72` |
| CPU 写入方与 victim 等级过滤 | 用户已要求的 eager-store 保护适配，保留并覆盖支持的全部入口 | `C/mori/simple_cpu_policy.py:82-94`；§3.2 |
| eager 备份、native on-demand load | 引擎执行选择；论文未规定具体复制触发 API，不能仅凭 eager 判定违反论文 | `C/patches/vllm-0.31-mori-simple-cpu-provider.patch:155-162`；`docs/zh/how-to/use-mori.md:100-101` |
| 事件唤醒、增量目录、后台规划、版本提交 | 执行优化，保留；改变触发频率但不附加准入或优先级条件 | `S/mori/strategy.py:304-314`；plan §3/5 |
| 永久双份硬保护、全量 KV READY 才进入引擎、强制删除 Waiting 所有副本 | 不应从论文抽象额外推导这些要求；避免在阶段 B 引入 | 本文 §2/3.3；plan §4 |
| `placement_only`、单 rank/single KV group 限制 | 能力边界；不能标为完整多副本论文实现 | `C/mori/vllm_adapter.py:105-109`；`C/core/scheduler.py:141-142`；`S/runtime.py:812-819` |

CPU→Waiting 当前按 idleness 回收逻辑预算，Waiting 可保留低保护残留；无需为了字面层级一致而主动清空缓存。
同样，不根据图示单独加入“Waiting 无推理需求也先提升到 CPU”的投机迁移。
当前实际路径见 `S/mori/strategy.py:223-271,629-650`，论文没有给出该投机动作的具体协议。

## 5. 修订后的落地顺序与验收

1. **阶段 0：先冻结语义。** 修正 response/Program terminal 边界；限定支持的 CPU connector；明确等待压力回收
   的解释；修正首次 native 执行前的计时；重新定稿 context 估算和输入拒绝。不能把所有现有规则都标为论文规定。
2. **阶段 A：统一增量事实。** 沿用原目录/ledger 计划，加上旧 prompt 保护撤销、事件丢失恢复、native 变化唤醒。
3. **阶段 B：迁移执行闭环。** 沿用 D2H 计划，加上实际可缓存范围、H2D/缺失段 prefill 与计算前 fence；
   不增加 MORI 外部 READY 准入门槛，取消/失败都必须能结束事务。
4. **阶段 C：规划移出关键路径。** 不改变上述策略；验证 native 全 BUSY 前提、HBM fallback 恢复、过期局部计划
   与 planner 故障通路。旧设计里的“async”指执行载体，不能只换一个类名。

验收除原 A/B/C 清单外，至少加入：stop 后同 session 继续；CPU 兼容入口不能绕过写入保护；prompt 裁剪不
残留旧保护；部分可缓存尾部不永久 pending；H2D/冷 prefill 不被外部 READY 门槛卡住；增长/传输失败下
仍有可推进事件；回收失败时后续 fit 请求不被阻塞；首次 native 排队不改变实时 idleness。
每项以事件轨迹及账本/native 实例断言验证。

这些条件能防止已知的自锁模式，不能推出任意请求、任意 native 故障下的绝对无死锁或无饥饿保证。
当前仍依赖 native 分配、推理完成、传输结果和调度器持续运行；相关边界见 `S/runtime.py:501-537`、
`C/core/scheduler.py:252-275` 与上述测试场景。

## 6. 验证范围

复现脚本和输出在 `artifacts/mori-paper-audit-20261028/probe.py`、`probe.json`，包含五个 CPU-only 场景。
相关 MORI 调度、分支、容量、旧 CPU policy 与 native patch 回归结果和精确命令见同目录 `validation.json`、
`pytest.txt`。未运行 GPU/DMA、服务重启或性能实验；没有本次吞吐提升结论，也没有修改已安装 native 补丁。
本次文档和脚本尚未提交推送。

## 7. 注释复核：状态、计时、容量估计与跨层置换

### 7.1 请求到达即为 Reasoning；不改变这一状态语义

论文 [§4.1](https://arxiv.org/html/2606.00866#S4.SS1) 明确请求到达进入 Reasoning，响应完成进入 Acting；
[§4.2](https://arxiv.org/html/2606.00866#S4.SS2) 又明确调度等待不计入时间窗口。
因此状态是 Reasoning，不代表状态持续的每一秒都必须进入 idleness 分母。
CPU/Waiting 层也可以有 Reasoning Program，它们有推理需求但尚未准入。
当前到达路径符合状态要求：`S/runtime.py:213-249`；候选条件见 `S/mori/strategy.py:223-226`。

之前 0.9→0.0818 的复现模拟的是**已有历史的返回 Program，在本轮首次 native 执行之前排队**，
不是整个 Program 的第一次请求。脚本显式设置了历史 `(Reasoning=1, Acting=9)`，
见 `artifacts/mori-paper-audit-20261028/probe.py:107-122`。
没有历史的新 Program 在当前实现中 idleness 为 0；即使把排队加进分母，分子仍为 0，
并不会出现该复现的 0.9→0.0818。初始化规则依据：`S/mori/strategy.py:732-753`。

还需收窄原来的论文判定：原文明确排除 MORI 容量门控造成的等待，但没有说明 router 如何获知 native
首次 batch、抢占及 GPU 执行时间戳。不能仅凭原文断言作者实现排除了所有 engine 内部排队。
源码能够确定的问题是当前计时前后不一致：首次执行前临时计入等待，首次执行事件又清除等待；
详见 `S/mori/strategy.py:72-92,520-535,732-753`。
建议保持请求到达即 Reasoning；若采用本实现已有的 native 执行区间口径，应从头到尾一致排除 native 等待。
这是计时口径修正，不是新增请求状态或准入门槛，也不再表述为已证实违反论文指定的 native 计时 API。

### 7.2 估算公式未公开，不能据此认定所有增长预留都应删除

论文 §4.1 使用估计 KV context 并约束聚合工作集，但没有给出 100-token buffer、最大输出预留或精确
下一步分配公式。应区分三件事：

1. 已有 context、块对齐和引擎下一步确实需要的空间：实际资源约束。
2. 固定 100-token buffer：本实现的增长估计，影响 placement 边界；不是另一道 GPU owner 准入锁。
3. `prompt + max_tokens` 上界拒绝：单请求可服务范围检查，与聚合工作集估计不是同一个机制。

源码分别见 `S/mori/strategy.py:156-162,701-714`、`C/core/scheduler.py:181-194`。
论文没给 100 这个值，只能说明参数没有论文复现依据，不能证明估计中保留增长空间违反算法。
因此撤回 §3.6 中仅以论文未规定为由直接删除固定预留的建议：先定义统一 estimator，注明哪些项是 native
实际需求、哪些是预测值，再测量估算误差和抢占。不得借估算器重新给已在 GPU 的 owner 增加二次容量门控。

同样，“可能提前 EOS”只说明上界拒绝比较保守，并不证明服务可以保证完成声明的最大长度。
若实际上下文最终大于单请求物理上限，腾空其他 Program 也解决不了。
应按 native 有效输出上限、模型长度及块粒度核对是否需要拒绝；不依赖未来 EOS，也不把请求上限全部视为
当前实际驻留。该安全检查是否保留不能仅从论文没有该公式作出结论。

### 7.3 撤回把无条件等待压力置换直接作为论文基线的建议

论文 [§4.3.1](https://arxiv.org/html/2606.00866#S4.SS3.SSS1) 的状态优先级限定在**GPU 已超预算时选择
降层对象**。它没有赋予 GPU 层外每个 Reasoning 请求无条件抢占 GPU Acting 的权利。
当前 `_select_pending()` 在没有现存容量债务时，把全部可回收 Acting 当作候选容量，未比较候选与 victim
的 idleness。见 `S/mori/strategy.py:217-271`；数值反例见 `probe.json:21-38`。
把新增候选先虚计进 GPU 再制造超额，也只是换一种方式实现同样的抢占，不能拿来证明原文授权了该动作。

§4.3 的相对分界重平衡描述，比 §4.3.1 的容量触发条款更概括；原文缺少两者衔接的完整伪代码。
因此准确结论是：当前无条件规则缺少论文依据，并可能把更 busy 的驻留者换成更 idle 的候选；不应作为
忠实基线继续默认保留。也不能反过来声称论文明确给出一个精确 `candidate_i < victim_i` 的交换实现。

重构需要区分以下决策：

- GPU 已有工作集超额：按原文状态及 idleness 顺序修复，Reasoning 保持 lazy demotion。
- GPU 有合法余量：按原文 promotion 优先序准入能放入者；不保留排他受益者。
- 无超额、无足够余量：不得只因为候选是 Reasoning 就启动回收。若实现 §4.3 的相位错配重平衡，
  应明确成单独的相对排序规则，而非把所有 Acting 当作低优先级空间。

一种更保守的可执行解释是只回收比候选更 idle 的 Acting，平分时保持 placement，足够容纳候选后才提交
整组交换；不引入额外 margin、最短驻留或超时提权。这是待验证的实现解释，不能标成作者算法。
具体多 victim/共享块及队列优先级的组合必须先在状态机用例中定稿，本轮不据此修改调度代码。

仅删除主动回收并只检查当前 GPU 超额，也不是完整的进展证明。若采用相对分界解释，在窗口的 Reasoning
时间大于 0、等待者 idleness 小于 1、工具未返回且周期检查持续的条件下，Acting 的比例随时间趋近 1，
可能跨过该等待者；何时发生取决于历史，不能保证立即放行或无饥饿。i=1 平局、无计时历史、请求过大、
native 不完成等边界不能由此排除。该条件推导依据现有公式 `S/mori/strategy.py:732-753`，不是论文无死锁定理。

本次修订只更新解释和重构约束，未修改运行时代码、复现脚本或此前测试结果。

## 8. 用户确认后的实施结果

用户已明确不处理 native 内部排队计时，暂时保留 100-token decode 预留。本次不执行 §3.8/§5 的计时
修改建议，也不执行 §3.6 的删除固定预留建议。此前 probe 保留为 `db7870d` 的历史事实，不能据其继续
推进已取消的修改。计时与容量估算函数未改的校验见
`artifacts/mori-relative-rebalance-20261028/validation.json`。

§3.5/§7.3 的无条件置换已在本次重构中删除。现有 GPU 超额修复和有余量提升保持；余量不足的交换要求
每个 Acting victim 严格比候选更 idle，整组边际回收足够后才提交，平局不动。新增前缀投影避免逐候选
扫描全部 victims，并验证共享块和连续准入后的容量。
源码：`S/mori/strategy.py:206-289`、`S/mori/capacity.py:83-138`。

实现规格、CPU 回归、原始证据和边界说明见
[相对分界置换重构](mori-relative-rebalance-20261028.zh-CN.md)。这是调度语义阶段的交付；本报告其他
生命周期、旧 connector、native 目录、迁移确认和异步执行问题没有因此自动解决，仍在后续清单。
本次没有运行 GPU benchmark，也不声称绝对无死锁或无饥饿。
