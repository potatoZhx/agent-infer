# MORI `mori-valid-1030` 最优已验证配置

## 版本与适用范围

本分支冻结已通过双 trace 性能晋升门槛的 MORI 算法版本。被验证的算法提交是
`6acbc84c90e8ad2b0771bd306410a39a2265a7eb`（`perf(mori): reuse repair capacity ledgers`），其父提交
`0321565e48a3e31cfee78c2145165864153279e6` 提供 context-aware victim replacement。分支上新增的本文与
入口链接只记录配置，不改变该算法代码。

2026-10-31 的两个独立双 trace 实验 `shared-ledger-v06-02`、`shared-ledger-v06-03` 都满足以下晋升合同：

- AgentX makespan 与 mean task JCT 均低于冻结 PT；
- SWEBench makespan 与 mean task JCT 均低于冻结 PT；
- 两个 workload 的请求、任务、监控与 MORI telemetry 均有效。

后续未通过稳定端到端晋升的 bulk-ledger 构建候选 `f391ed4b` 不属于本分支。

## 共同软件与调度配置

两条 workload 使用同一个模型 `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`、vLLM `0.31.0`、单卡
TP=1、compiled execution（不传 `--enforce-eager`）、async scheduling、16-token blocks、prefix caching、
SHA-256 prefix hash 与 prompt-token details。MORI full 依赖
[`vllm-0.31-mori-hbm-provider.patch`](../../agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch)
和
[`vllm-0.31-mori-simple-cpu-provider.patch`](../../agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch)；
安装与 capability 检查见 [`docs/zh/how-to/use-mori.md`](../../docs/zh/how-to/use-mori.md)。

验证使用的 MORI 配置如下；`enable_wait_credit=false` 是显式实验值，不依赖当时的代码默认值：

```json
{
  "agentcache": {
    "strategy": "mori",
    "schedule_interval_seconds": 5.0,
    "observability": {
      "enabled": true,
      "log_interval_seconds": 5.0
    },
    "mori": {
      "mode": "full",
      "window_size": 5,
      "enable_wait_credit": false
    }
  }
}
```

共同 server 参数：

```text
--tensor-parallel-size 1
--async-scheduling
--block-size 16
--enable-prefix-caching
--prefix-caching-hash-algo sha256
--enable-prompt-tokens-details
--shutdown-timeout 60
--agentinfer
```

## AgentX 16/8 配置

| 项目 | 已验证值 |
| --- | --- |
| Trace | `agentx/data/traces-duration64-ctx120k-gap10min.jsonl.gz` 解压内容 |
| 解压后 trace SHA-256 | `706caa8d19bea70f3ccb54d232e579ce0a5f81386447fd9d1581715354b94e1d` |
| Replay config | `agentx/config/replay_agentX_inflight8.yaml` |
| Replay config SHA-256 | `bbd2632ebf710833e6f1bf782e58ef44601f23f4801d453c7a40f218e46772c2` |
| Trace/session 选择 | `trace_type=agentX`, `sample_seed=0`, `task_num=16` |
| 并发 | `max_concurrency=8`, session 内 `max_inflight_requests=8` |
| 请求覆盖 | 16 tasks、587 requests |
| Endpoint | `/v1/completions` |
| HBM 控制 | `--gpu-memory-utilization 0.445`，目标总 HBM 预算 `63978.095 MiB` |
| 实测 KV 容量 | `31.62 GiB`、345,344 tokens（模型加载约 29.1 GiB） |
| CPU offload | 16 GiB，即 `cpu_bytes_to_use=17179869184` |
| Context/sequence | `--max-model-len 118784 --max-num-seqs 64` |
| 超时 | task 43,200 s；request 7,200 s |

AgentX 的 KV transfer 配置：

```json
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 17179869184,
    "kv_offload_backend": "cpu",
    "lazy_offload": false
  }
}
```

回放命令的关键参数为：

```bash
vllm bench serve --agentinfer replay \
  --config benchmarks/mori-formal/agentx/config/replay_agentX_inflight8.yaml \
  --trace-type agentX \
  --trace-path /path/to/traces-duration64-ctx120k-gap10min.jsonl \
  --base-url http://127.0.0.1:${PORT} \
  --metrics-url http://127.0.0.1:${PORT}/metrics \
  --endpoint /v1/completions \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --interval-mode trace \
  --task-num 16 \
  --max-concurrency 8 \
  --task-timeout-seconds 43200 \
  --request-timeout-seconds 7200 \
  --result-dir /new/result/directory
```

## SWEBench 32/16 配置

| 项目 | 已验证值 |
| --- | --- |
| Trace | `swe-cc-local-trace/traces/cc-2.1.152-ptkvc-64g-terminal-32_16.requests.jsonl` |
| Trace SHA-256 | `d38b0afe97ea0dd0689d32bff6a888831637f3eb963660055d58a8b4d30c7b4a` |
| Replay config | `swe-cc-local-trace/config/replay16-8.yaml`，运行时由 CLI 覆盖为 32/16 |
| 历史运行 config SHA-256 | `57389a1393cd794f7addbfeec2596753e3b787b03c6f11610c2049fe2896b006` |
| Trace/session 选择 | `trace_type=agentinfer`, `sample_seed=0`, `task_num=32` |
| 并发与覆盖 | `max_concurrency=16`、32 tasks、2,435 requests |
| Endpoint | `/v1/messages` |
| HBM 控制 | `--gpu-memory-utilization 0.31` |
| 实测 KV 容量 | `11.19 GiB`、122,208 tokens（模型加载约 29.1 GiB） |
| CPU offload | 32 GiB，即 `cpu_bytes_to_use=34359738368` |
| Context/sequence | `--max-model-len 107632 --max-num-seqs 1024 --max-num-batched-tokens 8192` |
| Tool parser | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| YAML 超时 | task 14,400 s；request 3,600 s；run 18,000 s |

仓库内 SWEBench YAML 与历史运行 YAML 的 replay 语义相同；两者 SHA 不同是因为历史文件还写入了绝对
`result_dir` 和绝对 trace path。正式复现必须继续由 CLI 指定新 `--result-dir`、`--task-num 32` 和
`--max-concurrency 16`，不能复用已有结果目录。

SWEBench 的 KV transfer 配置：

```json
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 34359738368,
    "kv_offload_backend": "cpu",
    "lazy_offload": false
  }
}
```

回放命令的关键参数为：

```bash
vllm bench serve --agentinfer replay \
  --config benchmarks/mori-formal/swe-cc-local-trace/config/replay16-8.yaml \
  --base-url http://127.0.0.1:${PORT} \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --task-num 32 \
  --max-concurrency 16 \
  --result-dir /new/result/directory
```

## 已通过的官方指标

| Workload | 运行 | Makespan / s | Mean task JCT / s | 相对冻结 PT |
| --- | --- | ---: | ---: | --- |
| AgentX | PT | 1872.785859 | 787.513925 | 基线 |
| AgentX | v06-02 | **1791.619339** | **676.190232** | -81.167 / -111.324 s |
| AgentX | v06-03 | 1792.492407 | 686.535558 | -80.293 / -100.978 s |
| SWEBench | PT | 3007.063323 | 1139.397498 | 基线 |
| SWEBench | v06-02 | 2952.098214 | **1100.734778** | -54.965 / -38.663 s |
| SWEBench | v06-03 | **2877.809616** | 1110.726743 | -129.254 / -28.671 s |

AgentX 的最佳 makespan 与 mean JCT 都来自 v06-02。SWEBench 的最佳 makespan 来自 v06-03，最佳 mean JCT
来自 v06-02；因此“最优版本”指同一个算法提交在两次有效复跑中均 4/4 通过，而不是从不同运行拼成一条
虚构结果。

## 验证与证据合同

新结果至少应检查：manifest 为 `completed`、任务/请求覆盖完整、trace/config SHA 匹配、MORI import 来自目标
checkout、四个正式 provider capability 存在、`enable_wait_credit=false`、无 reasoning victim、无 transition
exception/rejection，以及完整的 replacement/repair telemetry。两条 workload 可独立获得 GPU，并行提交；不得
在命令中固定物理 GPU ID。

历史原始证据保存在实验主机的以下追加式目录；它们不是运行时依赖：

```text
/data/zx/agent-kvc/research/studies/kvc_theory/mori-exchange-opt-20261031/
  artifacts/submissions/shared-ledger-v06-02/
  artifacts/submissions/shared-ledger-v06-03/
  artifacts/runs/shared-ledger-v06-agentx-02/
  artifacts/runs/shared-ledger-v06-agentx-03/
  artifacts/runs/shared-ledger-v06-swebench-02/
  artifacts/runs/shared-ledger-v06-swebench-03/
```

其中 `pair-summary.json` 是四门判定，`*-run-contract.json` 是 workload/telemetry 合同，run 目录中的
`fingerprint.json`、`raw/runtime/{server-command.txt,client-command.txt,mori-config.json,kv-transfer-config.json}`
和 `raw/replay/summary.json` 分别固定源码、命令、配置与官方指标。完整长实验没有因本次文档提交再次运行；
本文只归档两个已经完成并通过的独立双 trace 结果。
