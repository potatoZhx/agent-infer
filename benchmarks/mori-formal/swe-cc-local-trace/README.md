# swe-cc-local-trace：MORI 正式实现三后端 Trace Replay

本套件用同一份 AgentBench 本地 trace 比较 Vanilla、Progress-TTL（PT）和 MORI full。默认矩阵复现
2026-10-27/28 正式实现实验的 16 task、8 并行设置；随仓库保存的来源 trace 本身包含 32 个 Session。

套件只比较调度和缓存策略。三臂共用模型、GPU 参数、prefix cache、请求回放配置以及
`SimpleCPUOffloadConnector`；每臂都重启服务，以冷服务开始回放。Runner 不传 `--enforce-eager`，因此不会
强制 vLLM eager execution。`lazy_offload=false` 仅表示 SimpleCPU 使用 eager CPU offload，两者不是同一开关。

## 文件

- `config/replay16-8.yaml`：AgentBench Replay 配置；trace 路径相对 YAML 解析。
- `traces/cc-2.1.152-ptkvc-64g-terminal-32_16.requests.jsonl`：本地 32-Session 请求遥测。
- `traces/SHA256SUMS`：trace 完整性校验。
- `run_matrix.sh`：预检、单臂/三臂运行和官方结果汇总入口。

Trace 只包含 token 数、延迟、缓存计数、身份和时间等请求遥测，不包含原 prompt、messages 或生成文本。
详细来源和结构见 `traces/README.md`。

## 公平性矩阵

| 项目 | Vanilla | PT | MORI full |
| --- | --- | --- | --- |
| vLLM | 原生 0.31.x | 原生 0.31.x | 同版本 0.31.x + 两个正式 MORI provider patch |
| 调度入口 | upstream async scheduler | `--agentinfer` + `progress_ttl` | `--agentinfer` + `mori/full` |
| 请求 identity | Identity middleware | AgentInfer middleware | AgentInfer middleware |
| CPU offloader | SimpleCPU，16 GiB | SimpleCPU，16 GiB | SimpleCPU，16 GiB |
| `lazy_offload` | `false` | `false` | `false` |
| Replay | 同一 trace、seed、16 task、8 并行 | 同左 | 同左 |

共同的关键 server 参数为：TP=1、`gpu-memory-utilization=0.31`、async scheduling、
`max-model-len=107632`、`max-num-seqs=1024`、`max-num-batched-tokens=8192`、block size 16、
prefix caching + SHA-256、prompt token details，以及 Qwen3-Coder tool parser。所有值都由同一组 shell 变量生成，
不会按 arm 分叉。PT 使用当次真实 benchmark 的校准系数；MORI 使用 `mode=full, window_size=5`。

Vanilla/PT 必须导入未打 MORI patch 的原生 vLLM；MORI 必须导入应用了以下两个 patch 的 vLLM：

1. `agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch`
2. `agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch`

`preflight` 会检查 baseline 中四个 MORI hook 全部不存在、MORI 中全部存在，并比较两边的 vLLM 基础版本。
因此可使用两个独立 Python 环境，也可让 `MORI_PYTHONPATH` 指向只包含 patched vLLM 的隔离 overlay；无需改变
Vanilla/PT 环境。正式 patch 的安装说明见 `docs/zh/how-to/use-mori.md`。

## 运行

在仓库根目录准备变量。`MODEL_PATH` 可以是本地模型目录或 vLLM 可解析的模型 ID；`MODEL_NAME` 是三个后端
共同对外提供的名字。

```bash
export MODEL_PATH=/path/to/Qwen3-Coder-30B-A3B-Instruct-FP8
export MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
export BASELINE_PYTHON=/path/to/native-vllm-0.31/bin/python
export MORI_PYTHON=/path/to/patched-vllm-0.31/bin/python
export CUDA_VISIBLE_DEVICES=0

bash benchmarks/mori-formal/swe-cc-local-trace/run_matrix.sh validate
bash benchmarks/mori-formal/swe-cc-local-trace/run_matrix.sh plan
bash benchmarks/mori-formal/swe-cc-local-trace/run_matrix.sh preflight
bash benchmarks/mori-formal/swe-cc-local-trace/run_matrix.sh matrix
```

`validate` 只校验仓库内配置与 trace；`plan` 打印三臂即将执行的完整 server/client 命令；`preflight` 额外导入
两套 vLLM，检查版本与正式 hooks，适合在真正开始长实验前运行。

默认结果目录为 `results/mori-formal/swe-cc-local-trace-<UTC timestamp>`。为便于后续定位，建议显式指定：

```bash
RESULT_ROOT=/data/bench-results/mori-formal-01 \
  bash benchmarks/mori-formal/swe-cc-local-trace/run_matrix.sh matrix
```

脚本拒绝复用已经存在的 `RESULT_ROOT`，避免旧失败产物混入。`matrix` 依次执行 Vanilla、PT、MORI；任一臂失败
即终止并写入 `INVALID`。也可在全新的结果根目录只运行单臂：

```bash
RESULT_ROOT=/data/bench-results/mori-only-01 \
  bash benchmarks/mori-formal/swe-cc-local-trace/run_matrix.sh run mori
```

需要改变压力时可统一覆盖 `TASK_NUM` 和 `MAX_CONCURRENCY`，但这会偏离默认 16×8 正式矩阵。其余常用统一
覆盖项包括 `PORT`、`GPU_MEMORY_UTILIZATION`、`CPU_BYTES`、`TENSOR_PARALLEL_SIZE`、
`MAX_MODEL_LEN`、`SERVER_START_TIMEOUT_SECONDS` 和 `COOLDOWN_SECONDS`。

## 结果与官方指标

每个 arm 的 AgentBench 目录位于 `RESULT_ROOT/runs/{vanilla,pt,mori}`。其中 `manifest.json`、`summary.json`、
`requests.jsonl`、`replay-plan.json`、`replay-execution.json` 和 `evidence/vllm_metrics_*.prom` 是官方产物；
server/client 命令、日志、冷启动 metrics 和 GPU 采样位于 `RESULT_ROOT/runtime/<arm>/`。

三臂成功后，脚本使用官方 CLI 自动生成：

- `comparisons/combined-summary.csv`
- `comparisons/vanilla-vs-pt.json`
- `comparisons/vanilla-vs-mori.json`
- `comparisons/pt-vs-mori.json`

如需对已经完成的结果重新汇总：

```bash
RESULT_ROOT=/data/bench-results/mori-formal-01 \
  bash benchmarks/mori-formal/swe-cc-local-trace/run_matrix.sh compare
```

性能结论应同时检查三臂 manifest 都为 `completed`、请求覆盖一致以及 Replay/metrics 证据完整，不能用失败或
部分运行推导提升。本次套件整理提交只做静态、配置和 trace 完整性验证，不重新消耗 GPU 跑完整矩阵；正式
性能数据应由上述命令产生并与结果目录一并保存。
