# 使用 MORI 与 SimpleCPUOffloadConnector

本文说明如何在单个 embedded vLLM rank 上启用 MORI full 模式。当前推荐组合是 vLLM 0.31、
`SimpleCPUOffloadConnector`、CPU backend、eager offload。该路径使用 SimpleCPU 原生 CPU tensor 和异步
D2H/H2D 搬运，不依赖 `OffloadingConnector` 的共享内存分配器。

## 前置条件

- AgentInfer 与待构建的 vLLM 0.31 源码可用；
- 单个 embedded backend rank；
- prefix caching 已启用；
- `kv_offload_backend=cpu`；
- `lazy_offload=false`；
- 每个请求能由服务端 middleware 解析出 Program identity。

full 模式在缺少 HBM hook、SimpleCPU hook 或 DRAM 容量快照时拒绝启动。运行期缺失或格式错误的 identity
返回该请求的终止错误；重叠请求按下述分支规则处理。无效 victim 建议会回退到 native free queue 顺序，并记录告警和
回退次数。需要只验证 placement 时，应显式配置 `"mode":"placement_only"`。

## 给 vLLM 应用两个窄补丁

从 vLLM 0.31 源码根目录依次执行：

```bash
git apply --check \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch
git apply \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch

git apply --check \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch
git apply \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch
```

然后按照该 vLLM 源码树原有方式重新构建或安装。SimpleCPU patch 依赖第一个 patch 提供的实例级 `BlockPool`
provider contract，因此顺序不可交换。未注册 provider 时，native HBM 与 CPU BlockPool 均保持原来的
`popleft_n()` 路径，不维护 MORI typed index。

## 启动服务

仓库提供了可直接使用的示例：

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct \
MORI_CPU_BYTES=17179869184 \
bash examples/serve-mori.sh
```

示例等价于以下关键配置：

```bash
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-mori-lifecycle.sock

AGENTCACHE_CONFIG='{"agentcache":{"strategy":"mori","schedule_interval_seconds":5.0,"mori":{"mode":"full","window_size":5}}}'
KV_TRANSFER_CONFIG='{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184,"kv_offload_backend":"cpu","lazy_offload":false}}'

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --agentinfer \
  --enable-prefix-caching \
  --additional-config "$AGENTCACHE_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
```

`cpu_bytes_to_use` 是 server-wide 容量；SimpleCPU 按 vLLM world size 计算 rank-local 容量。如果部署需要精确指定
每 rank 容量，可使用 vLLM 的 `cpu_bytes_to_use_per_rank`。MORI 从 connector snapshot 读取实际 block 数和
scheduler block size，不根据命令行字节数重复推导容量。

## 发送带 Program identity 的请求

下面的请求由服务端 middleware 从 `X-Claude-Code-Session-Id` 解析 Program identity。后续轮次继续使用同一 header，
无需客户端提供 tier、idleness、未来返回时间或 replay plan。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Claude-Code-Session-Id: mori-demo-session' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Summarize the MORI cache policy."}],
    "max_tokens": 128
  }'
```

subagent 请求应使用独立 agent identity，例如同时携带
`X-Claude-Code-Session-Id: mori-demo-session` 和 `X-Claude-Code-Agent-Id: reviewer-1`。服务端据此生成独立
Program；客户端不应复用 lead Program ID 表示并行 subagent。

## 运行期行为

SimpleCPU adapter 在启动和运行期间完成以下闭环：

1. 从 `SimpleCPUOffloadScheduler` 快照读取 total/resident/pending block 数并换算成 token；
2. 在 CPU BlockPool 注册 typed victim provider，淘汰顺序为 `INACTIVE -> BUSY -> IDLE`；
3. D2H destination 尚在传输时保持 non-evictable，commit 后由 native `free_blocks()` 发布 ready 和 LRU sequence；
4. load hit、already-cached store 和共享 content 会追加服务端解析出的 Program owner；
5. vLLM 在提交前重新验证 victim 数量、唯一性、queue membership、refcount 和 block ID；
6. HBM 继续使用 `INACTIVE -> IDLE -> BUSY`，实际 hash eviction、allocation 和 D2H/H2D 仍由 vLLM 执行。

MORI 当前不主动 prefetch。CPU tier 表示保护优先级；需要再次推理时仍由 SimpleCPU 的 native on-demand load
触发 H2D。

GPU tier 已经是准入决策：其请求直接提交引擎；重叠请求由服务端分配独立分支，分别获取 placement。
上下文增长导致总量超额时，统一循环先降 Acting，再按 idleness 从高到低选择 Reasoning 做 lazy demotion。
Reasoning 完成或 native 确认取消后才释放 GPU 预留。
等待者按 CPU returning、其他 returning、新 Program 的 context/idleness 排序；不能放入的候选不会阻塞
后续候选。

每轮调度重新计算相对 idleness，落实 [MORI 论文 §4.3](https://arxiv.org/html/2606.00866v1#S4.SS3)
描述的层级边界调整。提升先使用已有余量；候选无法放入时，可以降级 idleness 严格更高的 GPU Acting，
即使 GPU 当前尚未超额。执行降级前，先按实际共享块验证整组候选 victim 移除后能否容纳请求；只有已接受
的降级才能释放容量，不能提前兑现 Reasoning 的 pending 释放。真实容量超额仍由独立路径修复，
Reasoning 继续采用 lazy demotion。CPU 不会仅因新候选分数更好而主动置换驻留者，默认 100-token decode
预留继续保留。

另有明确的进展保护：当 GPU 没有 Reasoning owner 时，可放入的等待请求允许回收 Acting 占位，
不再要求其评分严格更优，避免平分或历史分数导致空闲引擎被缓存占位阻塞，包括父 Program 等待子请求
的情况。此规则不增加超时或调参项，首个请求成功准入后即停止生效，也不会抢占 Reasoning；
本轮新准入的请求同样受到保护。

本地实现增加两项优化。首先，CPU 或 WAITING 中暂停的 Reasoning 请求，将本次请求的等待时间临时计入
调度 idleness 分母中的 Reasoning 时间。它仍属于当前周期的原有窗口槽位，准入 GPU 时冻结，Reasoning
结束时清除；完成历史只保存真实 Reasoning 和 Acting 时间。工具执行与 native GPU 抢占不增加此补偿。
层级决策和 diagnostics 使用同一评分函数，原有层级优先顺序与 Typed-LRU 保持不变，不引入历史衰减参数。

其次，Acting owner 已提交到 WAITING 时，当轮退出 MORI 记账并释放 native ownership。后续请求获得新 generation、
空历史及新 Program 的排序优先级。业务 task/parent 关系继续保留；物理共享 KV 与未完成的 native 传输仍由
引擎管理。匹配的 terminal 凭证可以结束旧业务记录，但不能影响后来的 generation。Acting 很久本身不会
触发退出，必须已经离开 GPU 与 CPU 两个逻辑缓存层；等待补偿也不能创造空闲容量或保证等待时间有界。

不再保留候选独占的未来回收额度，也没有额外等待超时、最短驻留或 idleness 差值门槛。

`agentcache.mori` 只接受 `mode` 和 `window_size`。已删除 `min_residency_seconds`、`idleness_margin`、
`max_wait_seconds` 配置；旧 `starvation_timeout_seconds` 仍不支持。diagnostics 的 `max_wait_seconds`
仅表示观测到的最长等待时间，不是配置或延迟保证。

SimpleCPU hook 升级至 API 2，必须同时更新两个 native patch 后重新构建。GPU/BUSY 备份只能使用
空闲、INACTIVE 或 BUSY CPU 块，不得挤掉 IDLE 缓存；写入拒绝不会退回普通 LRU。
依据：`agentinfer/scheduling/mori/strategy.py:209-326`、`agentinfer/scheduling/mori/capacity.py:83-137`、
`agentinfer/agentcache/mori/simple_cpu_policy.py:82-94`。

当前 MORI 要求一个 native KV cache group。有效 HBM 容量扣除永久 null block；prompt 加最大输出超过
有效容量的请求在准入前拒绝。逻辑 GPU 预算是 context 加 100 decode token 后按 block 取整，推测余量
截到总容量；CPU 按 chunk 取整。只有当前 prompt 的有效 native block ID 证据才能用于共享抵扣，未知或
失效证据保守计费。CPU placement 不保证完整 CPU 副本已 READY；native 仍负责 refcount、传输 fence、
实际分配和必要的重算。

### 并发分支与生命周期

串行客户端可沿用原有 Program identity。前一请求未结束时到达的默认重叠请求获得临时独立 owner；该分支
完成即释放保护，native 缓存仍可机会性复用。服务端不推断应用最终采用哪个输出，后续真实 prompt 决定
新的 context，允许裁剪和切换上下文。

不提供 `lineage_id` 协议，也不要求客户端声明分支。旧请求中出现该键时，与其他未识别扩展一样忽略，
不会影响调度；服务端不会再序列化它。同一 Program 的请求只要重叠（包括前一请求仍在等待准入），
就使用独立临时 owner；不存在命名分支的串行后继队列。

临时 owner 不构造应用层的父级阻塞关系，真实 subagent 继续使用独立 agent identity。
API middleware 为每次请求生成内部 `lifecycle_token`，客户端无需设置；迟到的 terminal 信号只能释放
对应最新请求的空闲分支。临时分支完成或取消后释放逻辑保护，物理缓存按 native 规则机会性保留。
证据：`agentinfer/scheduling/request_branches.py:14-58`、`agentinfer/scheduling/runtime.py:527-561`。

协议删除与后续设计边界见
[状态与执行边界重构方案](../../../design/module/scheduling/mori-state-refactor-20261028.zh-CN.md)。

## 启动检查和排障

| 现象 | 原因和处理 |
| --- | --- |
| `requires the MORI BlockPool provider patch` | 未应用第一个 HBM/BlockPool patch，或两个 patch 顺序错误 |
| `requires lazy_offload=false` | 将 SimpleCPU extra config 中的 `lazy_offload` 设为 `false` |
| `requires kv_offload_backend=cpu` | 当前 full 支持不接受 SimpleCPU disk backend |
| `requires server-resolved Program identity` | 请求没有受支持的 body metadata 或 identity header |
| `requires ... register_mori_block_selection_provider` | vLLM HBM hook 未进入正在运行的安装环境 |
| DRAM quota 缺失 | SimpleCPU snapshot hook 不可见，通常是仍在使用未打 patch 的 vLLM |

可以先检查实际导入位置和 hook：

```bash
python - <<'PY'
import inspect
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

print(inspect.getfile(SimpleCPUOffloadScheduler))
print(hasattr(SimpleCPUOffloadScheduler, "register_mori_offload_tier_provider"))
print(hasattr(SimpleCPUOffloadScheduler, "get_mori_offload_tier_snapshot"))
PY
```

源码入口：`examples/serve-mori.sh`、`agentinfer/agentcache/mori/simple_cpu_policy.py`、
`agentinfer/agentcache/mori/vllm_adapter.py`、
`agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch`。架构和不变量见
[MORI 设计](../../../design/module/scheduling/mori-scheduling.md)。
