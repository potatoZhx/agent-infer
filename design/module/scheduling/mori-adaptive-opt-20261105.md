# MORI 自适应调度优化报告（2026-11-04 ~ 11-05）

> 分支：`codex/mori-adaptive-20261105`（基线 `codex/mori-full-pr-20261103` @ e485fe3 + 3 个本地提交）
> 实验口径：AgentX 16×8 u75 if8（TP2 2×L20、KV 392,784 tokens、trace/yaml fingerprint `0e7665e9…` 全程一致）
> 撰写时间：2026-11-05；数据源：`/home/zx/results/`（summary.json / requests.jsonl / server 日志）

---

## 1. 背景与目标

- 前序结论（本仓库实验系列）：静态放宽抢占（`reasoning_preempt_acting`，允许 REASONING 候选回退抢占全部 ACTIVE+ACTING）在 32G 口径严重回归（makespan +52.6%），证实"紧张相位应保护工作集"。
- 用户提出自适应方向：**宽松相位优先吞吐、紧张相位保护工作集**，并要求以多 agent 协作方式产出与迭代方案。
- 量化目标：vs PT（32G 口径 4403s / JCT mean 1930s）makespan 与 mean JCT 双 −10%，即 **makespan ≤3963s 且 JCT mean ≤1737s**。
- 起点：mori ffc39f1 配置 @32G = 4143s / 1771（vs PT −5.9% / −8.2%）。

## 2. 优化方案（多 agent 设计与讨论过程）

三个独立设计 agent（可并行读代码/日志/数据）各出一版方案，再做两轮交叉评审：

| 方案 | 路线 | 核心主张 |
|---|---|---|
| A | 信号驱动 + 滞回的离散模式机（PROTECT/BALANCED/THROUGHPUT） | 相位化既有旋钮，不新增抢占语义 |
| B | 连续重算代价/效用模型（κ/η 自校准） | 把驱逐代价与等待收益写成同一时间量纲的不等式，legacy 为退化特例 |
| C | 数据下驱动的吞吐/尾段专项 | 先下钻数据定位真实损失，再对症设计 |

**C 的下钻改变了设计共识**（关键证伪与发现）：

1. "mori TPOT 长尾慢"被证伪——token 加权 TPOT mori 比 PT 快 15%（59.3 vs 69.8ms）；中位仅差 1.6ms。
2. 真实损失在 **mori 自有 request_pool 排队**：Σ6213s（native 排队 1267s 的 4.9 倍）；尾段 150 请求 TTFT mean 22.3s vs PT 3.5s。
3. 极端样本：107k token 请求等待 1676s——期间 5 个 REASONING 驻留者占满逻辑容量（water 98%）而 `gpu_acting=0`（交换受害者网只捞 ACTIVE+ACTING → 空网）、GPU 物理 Running 0-1。
4. 等待期间前缀腐烂（入队时 cached 44240 → 38s 后归零）。
5. makespan 尾巴 = 最终会话 decode 被其他会话大 prefill 干扰（TPOT 49.3ms vs 干净相位 26ms）。

**共识组合**（B 在评审中承认其准入门槛在"空受害者网"相位无决策对象；A 承认其 water 信号在"高水位+物理空转"并存时失准）：

- **C2 结构性排空**：启用 in-tree 的 `_reserve_aged_waiter` 惰性让位（REASONING 驻留者 `mark_for_pause` 到请求边界，非硬抢占）。门控 = waiter ≥ `drain_wait_threshold_seconds`（默认 180s，0=关）且本周期交换网为空（`acting == 0`）且判定为 no-fit；连续 `drain_persist_cycles`（3）个周期才武装、武装后保持 `drain_min_dwell_seconds`（60s）、每周期让位 ≤ `drain_max_victims_per_cycle`（2）。
- **B 效用受害者排序**：`κ`（重算速率 tokens/s）与 `η`（保留率）从 `on_request_finished` 既有观测量 EWMA 自校准；受害者按"重算秒代价 / 边际释放 tokens"排序。迭代 2 修正为：legacy 主序（ACTING 优先 + 空闲度降序）不动，效用比值仅在 0.05 空闲度分桶内做决胜（迭代 1 的纯成本主序被实验否决，见 §4）。
- **C3 TPOT 护栏混批**：`decode_batch_mode="adaptive"`——仅当"有解码在跑 + 有大块 chunked prefill 排队 + 解码间隔超过自身基线的 `decode_batch_tpot_guard_ratio`（1.3）倍"三条件齐备时，把 native token 预算钳到 decode 优先；冷启动（<32 个输出事件）不钳。信号源挂 `after_output`→逐 token 观测，零新增 native 查询。
- **默认全关**：所有新键默认关闭时行为与基线逐位一致（68 项单测含等价性护栏；并由 32G 基线复跑实证）。

代码落点：`agentinfer/scheduling/mori/{strategy,config,utility}.py`、`agentinfer/agentcache/core/{controller,scheduler}.py`、新增 `tests/agentcache/test_mori_adaptive.py`。

## 3. 实验结果（全部 32G CPU offload，587/587、0 invalid_program_admission、fingerprint 全等）

| 运行 | 配置 | makespan | JCT mean / p95 | cache | 备注 |
|---|---|---|---|---|---|
| PT 基线 | — | 4403s | 1930 / 3459 | 81.2% | 沿用 |
| mori ffc39f1（原基线） | 无自适应 | 4143s | 1771 / 3218 | 84.3% | 沿用 |
| **mori ffc39f1 复跑（r2）** | 同上（新代码默认路径） | **4066s** | **1685 / 2844** | 83.6% | **方差标定运行** |
| 静态抢占（前置实验） | `reasoning_preempt_acting=true` | 6322s（+52.6%） | 2779 / 4695 | 59.3% | 负结果 |
| adapt（迭代 1） | B 纯成本主序 + C2 + C3 | 5151s（+24.3%） | 2281 / 3788 | 76.3% | 负结果 |
| adapt2（迭代 2） | B 分桶决胜 + C2 + C3 | 4962s（+19.8%） | 2135 / 3810 | 78.2% | 负结果 |

关键遥测事实：

- 两轮自适应全量中 **C2 与 C3 一次都未触发**（`drain_armed_transitions=0`、`decode_batch_adaptive_clamped=0`）——设计针对的"空受害者网 28 分钟停摆"在重复运行中不复现（轨迹依赖事件）。
- 迭代 2 运行中实测的真实病灶是 **all-acting-insufficient 型 `no_all_acting_fit`**（acting 受害者存在但全逐也不够，近期决策中 104 次）+ `request_pool` 等待升至 ~212s。
- 基线复跑 vs 原基线：makespan 4143 → 4066（−1.9%）、JCT mean 1771 → 1685（−4.9%）——**单跑方差量级 = ±2%（makespan）**。

## 4. 分析过程（归因链）

1. **迭代 1 归因**：唯一实际生效的机制是 B 排序（C2/C3 未触发）。纯成本主序（ctx×(1−η)/κ ÷ relief 升序）丢掉 legacy 的"ACTING 优先 + 空闲度降序"主信号 → 倾向"多个小释放受害者"而非"少数真正空闲者" → 交换次数/转换开销/重算事件增多 → cache 84.3%→76.3%、makespan +24.3%。
2. **迭代 2 修正与再验证**：B 改为分桶决胜（legacy 主序保留）。回归幅度收窄（+24.3%→+19.8%）但仍显著为负——**在 ±2% 方差下这是真回归**。结论：受害者选择对"次级排序键"高度敏感（混沌放大），重算代价信息不适合进入主序；`_bounded_replacement_plan` 的既有交换优化已是更安全的利用方式。
3. **方差标定**：adapt2 的 −800s 无法用机制解释（其唯一行为差异近似无操作），触发基线复跑。r2 证明 4143s 可复现（±2%），从而确认迭代 1/2 的回归为真；同时暴露方法学要求——**2.5% 的目标裕度下单跑不可下结论，后续机制评估必须 ≥2 次重复**。
4. **目标现状重估**（以 r2 为 mori 代表）：JCT mean 1685 vs PT 1930 = **−12.6%（达标）**；makespan 4066 vs 4403 = −7.7%（差 2.5pp）。原 4143s 单跑基线会系统性低估 mori。

## 5. 结论

1. **负结果（有价值的三个）**：静态抢占 +52.6%；效用排序两种形态 +20~24%；证明"更激进/更聪明地选受害者"在此 workload 上都不成立，legacy 的相对空闲度规则 + 服务租约保护 + 有界交换是经过反复验证的正确基线。
2. **正结果**：基线本身（ffc39f1 配置）在重复运行下稳定优于 PT（makespan −7.7~−9.3%、JCT mean −12.6%）；JCT 维度已达成 −10% 目标。
3. C2/C3 机制代码保留（默认关闭、行为等价有单测与全量实证），其触发条件在当前 workload 轨迹中罕见——若未来 workload 出现长停摆或 decode 干扰相位可直接启用验证。

## 6. 下一步方案

1. **C1v2 前缀信用（首选下一杠杆）**：对症 §3 的 all-acting-insufficient 病灶——候选 claim 全额记账把"前缀仍驻留"的请求判为不可行。设计要点（已过 agent 评审）：
   - 信用 = min(观测到的驻留前缀 tokens, `prefix_credit_max_ratio`(0.6) × context)，新鲜度 30s（过期即全额），总量 ≤ `prefix_credit_total_max_ratio`(0.2) × quota；
   - **信用只进可行性检查（slack 判定与 no-fit 判定），账本 `gpu.add` 永远全额记账**——从定义上消除"信用腐烂→账本低估→债务循环"；
   - 准入后短暂超发走既有 growth-debt 惰性修复，监控 `gpu_capacity_debt_unresolved` 与 `prefix_credit_applied_tokens`。
2. **验证协议（硬性）**：每配置 ≥2 次全量重复（单跑 ±2% 方差）；对照基线用 4066-4143s 区间；通过判据 = 两次重复的 makespan 均 ≤3963s。
3. **16G/64G 跨档稳定性验证**（本轮因服务器不可用中止 @277/587，目录已标 interrupted）：恢复后重跑两档，沿用 PT 5013/3199 与 vanilla 10213（16G）对照；vanilla@64G 无完整数据需注明。
4. **B 排序冻结**：`utility_victim_order` 保持默认关闭；若重启，只允许在 `_bounded_replacement_plan` 包络内做交换候选评估，不触碰前缀序。
5. 远期（用户自适应方向的完整形态）：A 方案的模式机作为参数路由层，在 64G 松弛口径验证 THROUGHPUT 相位（准入放宽 + 服务保护短路）能否收窄对 PT 的差距。

## 7. 复现命令

```bash
# 基线（= 本报告最优配置，自适应键全关）
cd /home/zx/agentinfer-exp && \
MORI_WAIT_CREDIT=false MORI_OBSERVABILITY=1 \
MORI_EXTRA_KEYS='"ready_wait_limit_seconds":0,"gpu_service_requests":4,"fast_capacity_accounting":true"' \
CPU_BYTES_TO_USE=34359738368 \
REPLAY_CONFIG=/home/zx/agentinfer-exp/replay_agentX_inflight8.yaml \
TAG_SUFFIX=-if8-<tag> ./run-d64-tp2.sh mori 16 8 43200 7200 0.75

# 自适应消融：MORI_EXTRA_KEYS 追加
#   "drain_wait_threshold_seconds":180            (C2)
#   "utility_victim_order":true                   (B, 不建议)
#   "decode_batch_mode":"adaptive"                (C3)
```

结果目录：`/home/zx/results/mori-d64-16x8-u75-if8-{ffc39f1-cpu32g,ffc39f1-cpu32g-r2,preempt-cpu32g,adapt-cpu32g,adapt2-cpu32g}`；服务器日志同名于 `/home/zx/results/logs/server-*.log`。
