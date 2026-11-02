# 复现 MORI history-off-ab09 双 trace 实验

本文记录 `history-off-ab09` 的服务器代码、配置、运行命令和已观察结果。该配置关闭基于工具历史的 gating，使用固定服务预算和快速容量核算。已完成的两组实验均未通过四项指标各改善至少 5% 的验收门槛。

## 代码和外部材料

- 文档和代码分支：`codex/history-off-ab09-repro`。服务器实际运行提交固定为 `ffc39f1bb6ea4562adbebde829264c866d412196`，不是添加本文后的分支 HEAD。
- 客户端固定为 `/data/zx/agent-kvc/.worktrees/mori-exchange-opt`，
  提交 `f391ed4b89471d3c59e24ff52b5ac0bfb95f6fd9`。运行前应保持该 worktree 干净且不改变其提交。
- 实验脚本、trace、历史结果和 Python 环境位于本仓库之外，**不会随本分支推送**。
  下面的命令适用于已经具备这些材料的 `/data/zx/agent-kvc` 环境；
  其他机器需要先按相同路径准备原始材料、模型和环境。不要用不同 trace 或其他模型代替。
- 研究目录：`/data/zx/agent-kvc/research/studies/kvc_theory/joint-rsi-20261101`，
  包含 `run_pair.py`、`run_workload.py`、`assess.py`、`baseline_manifest.json` 和 `resource_reference.json`。
  包装器还使用 `mori-exchange-opt-20261031/harness/run_iteration.sh` 和历史验证器。
- Python：`/data/zx-envs/agentinfer-vllm-0.31.0/bin/python`；服务端 vLLM overlay：`/data/zx-envs/mori-formal-simplecpu-overlay-f51d8f1`；模型：`/data/models/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8/snapshots/dcaee4d4dfc5ee71ad501f01f530e5652438fde0`。
- 需要可用的 `gpu run` 调度器、与历史实验匹配的 NVIDIA L20X GPU、CUDA 环境和研究目录内的校验材料。
  脚本会检查 trace、配置、native vLLM、overlay 的哈希以及端口、服务器提交和 worktree 状态。

在已有工作区获取文档分支，并为实验创建独立的精确提交 worktree：

```bash
git -C /data/zx/agent-kvc/agent-infer fetch origin codex/history-off-ab09-repro
git -C /data/zx/agent-kvc/agent-infer worktree add --detach \
  /data/zx/agent-kvc/.worktrees/history-off-ab09-runtime \
  ffc39f1bb6ea4562adbebde829264c866d412196
git -C /data/zx/agent-kvc/.worktrees/history-off-ab09-runtime rev-parse HEAD
git -C /data/zx/agent-kvc/.worktrees/history-off-ab09-runtime status --short
git -C /data/zx/agent-kvc/.worktrees/mori-exchange-opt rev-parse HEAD
git -C /data/zx/agent-kvc/.worktrees/mori-exchange-opt status --short
```

两条 `rev-parse` 分别应输出 `ffc39f1bb6ea4562adbebde829264c866d412196` 和
`f391ed4b89471d3c59e24ff52b5ac0bfb95f6fd9`；两条 `status` 应为空。
运行器会检查服务器的完整提交号，故不能直接用添加了本文的分支 HEAD 代替。
上述 worktree 路径如果已存在，应先检查并选择未占用的新路径，随后同步更新 `RSI_SERVER_WORKTREE`。

## 固定配置

服务端 MORI 设置为 `mode=full`、`window_size=5`、`enable_wait_credit=false`、
`ready_wait_limit_seconds=0`、`gpu_service_requests=4`、`fast_capacity_accounting=true`；
`history_gated_gpu_service` 保持默认 `false`。调度周期和可观测性日志周期均为 5 秒。
其余设置使用对应提交的默认值。

| 设置 | AgentX | SWE |
| --- | ---: | ---: |
| 任务数 / 并发数 / 请求数 | 16 / 8 / 587 | 32 / 16 / 2435 |
| CPU KV 容量 | 16 GiB | 32 GiB |
| GPU memory utilization | 0.445 | 0.31 |
| GPU KV blocks（block size 16） | 21,584 | 7,638 |
| max model length / max sequences | 118,784 / 64 | 107,632 / 1,024 |
| max batched tokens | 后端默认 | 8,192 |

两个 workload 都使用真实 vLLM 推理、异步调度、SHA-256 prefix caching 和 SimpleCPU offload，
保持原始 replay、依赖释放和工具耗时。`--match-frozen-kv-pool` 依据 `resource_reference.json` 固定块数；
脚本调用 `gpu run --gpus 1` 为每组各申请一张卡，不指定物理 GPU。
AgentX 和 SWE 的 trace/config 路径及 SHA-256 均在 `baseline_manifest.json` 与历史 runner 中。

## 提交两组真实实验

研究目录中的结果和汇总是共享状态；如果其他实验正在写入该目录，先为新运行准备隔离的实验环境。
选取未使用的 label 和两个相邻空闲端口。脚本使用排他创建的 submission 目录，
因此不要重用历史 `history-off-ab09` 的 `01` 或 `02` 结果目录。

```bash
export RSI_SERVER_WORKTREE=/data/zx/agent-kvc/.worktrees/history-off-ab09-runtime
export RSI_STUDY=/data/zx/agent-kvc/research/studies/kvc_theory/joint-rsi-20261101
export RSI_LABEL=history-off-ab09-repro-20261102a

/data/zx-envs/agentinfer-vllm-0.31.0/bin/python "$RSI_STUDY/run_pair.py" \
  --commit ffc39f1bb6ea4562adbebde829264c866d412196 \
  --label "$RSI_LABEL" \
  --attempt 01 \
  --port 19880 \
  --wait-limit 0 \
  --service-requests 4 \
  --fast-capacity-accounting \
  --gpu-memory-threshold 50 \
  --match-frozen-kv-pool
```

这里的 `--port 19880` 分别给 AgentX 和 SWE 使用端口 19880、19881；占用时改为其他相邻空闲端口。
不要加 `--history-gated-gpu-service`。每个 `gpu run` 作业设置 `--timeout 3h`；
总耗时取决于排队和完整推理。`run_pair.py` 等待两组作业并运行 `assess.py`，结果写在：

```text
$RSI_STUDY/artifacts/submissions/$RSI_LABEL-01/
$RSI_STUDY/artifacts/runs/$RSI_LABEL-agentx-01/
$RSI_STUDY/artifacts/runs/$RSI_LABEL-swebench-01/
```

检查 submission 下的两个 `*-submit.log`、`exit-codes.json` 和 `assessment.json`，
以及每组运行目录中的原始 replay summary、服务端日志和运行清单。
完整性和有效性需要检查 `assessment.json` 中的 `valid`、各 workload 的完成状态及错误列表；
成功启动或零退出码本身不是性能验收。自动评估还会更新研究目录的共享候选账本，
必须避免与其他实验并发写入。

原始两次启动参数分别保存在 `launch-ab09-command.json` 和 `launch-confirm09-command.json`。本文使用新 label、端口和独立服务器 worktree；算法设置、工作负载和资源配额与原始配置一致。

## 固定 PT 基线和已有结果

| 运行 | AgentX makespan / mean JCT（秒） | SWE makespan / mean JCT（秒） | 相对 PT 改善：AgentX / SWE（按 makespan、mean JCT） |
| --- | --- | --- | --- |
| 固定 PT 基线 | 1872.785859 / 787.513925 | 3007.063323 / 1139.397498 | — |
| history-off-ab09 01 | 1767.103952 / 680.573469 | 2666.666584 / 1135.669808 | 5.643032% / 13.579500%；11.319906% / 0.327163% |
| history-off-ab09 02 | 1824.423543 / 709.803339 | 2596.716099 / 1105.018093 | 2.582373% / 9.867836%；13.646112% / 3.017332% |

01 的 AgentX makespan 达到 5% 改善，02 没有复现这一结果；两次 SWE mean JCT 均未达到 5%。表格是历史测量，不保证新一次运行得到相同数值。

证据位于研究目录中的 `baseline_manifest.json`、`report-replication-09.md`、`source-audit-09.json`，
以及 `artifacts/submissions/history-off-ab09-01/assessment.json` 和
`artifacts/submissions/history-off-ab09-02/assessment.json`。
评估文件包含原始数据路径、完整性和资源检查结果。
