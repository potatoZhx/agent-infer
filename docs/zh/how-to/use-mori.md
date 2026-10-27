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

full 模式会 fail closed。HBM hook、SimpleCPU hook、DRAM 容量快照或请求 identity 任一缺失时，服务不会静默
降级为 native LRU。需要只验证 placement 时，应显式配置 `"mode":"placement_only"`。

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

SimpleCPU adapter 在启动时完成以下闭环：

1. 从 `SimpleCPUOffloadScheduler` 快照读取 total/resident/pending block 数并换算成 token；
2. 在 CPU BlockPool 注册 typed victim provider，淘汰顺序为 `INACTIVE -> BUSY -> IDLE`；
3. D2H destination 尚在传输时保持 non-evictable，commit 后由 native `free_blocks()` 发布 ready 和 LRU sequence；
4. load hit、already-cached store 和共享 content 会追加服务端解析出的 Program owner；
5. vLLM 在提交前重新验证 victim 数量、唯一性、queue membership、refcount 和 block ID；
6. HBM 继续使用 `INACTIVE -> IDLE -> BUSY`，实际 hash eviction、allocation 和 D2H/H2D 仍由 vLLM 执行。

MORI 当前不主动 prefetch。CPU tier 表示保护优先级；需要再次推理时仍由 SimpleCPU 的 native on-demand load
触发 H2D。

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
