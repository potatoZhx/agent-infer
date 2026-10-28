# agentx：mori-formal 三后端 AgentX Trace Replay 对比套件

用同一份 AgentX trace（Claude Code 风格 agent 会话遥测）对比 **vanilla / Progress-TTL（PT）/ MORI full**
三个后端。三臂除调度入口外**所有容量与回放配置完全一致**（含统一 SimpleCPU CPU offload），冷启动回放，
适合评估调度与缓存策略本身的差异。

## 数据集（使用哪一份）

**本套件使用 `data/traces-duration64-ctx120k-gap10min.jsonl.gz`**（即
`agentinfer/agentbench/data/` 下同名打包文件的副本，SHA256 见 `data/SHA256SUMS`）。
使用前先解压（保留压缩包）：

```bash
cd benchmarks/mori-formal/agentx
sha256sum -c data/SHA256SUMS
gunzip -k data/traces-duration64-ctx120k-gap10min.jsonl.gz
```

数据集特性（决定 server 参数下限）：

- 64 个 session、2151 个请求、总输入 159.0M tokens；最大单请求上下文 **118,047** tokens
  → `--max-model-len 118784`（不可更小；agentX 回放无 preflight，过小会运行期失败）。
- 会话按 wall 时长升序、`ctx≤120k` 且 `gap≤600s` 过滤后取最短 64 个；可用
  `tools/select_agentx_traces.py --mode duration --count 64 --max-ctx 120000 --max-gap 600` 再生成。
- Replay 以 `sample_seed=0` 的确定性哈希洗牌选取前 N 个 session：`--task-num 16` → 16 sessions
  （587 请求、43.1M tokens）；`--task-num 32` → 32 sessions（1149 请求、85.0M tokens，包含前 16 个）。
  同 seed + 同 task-num 的选择跨运行完全可复现。

## 前提

1. Python 环境：vLLM `0.31.0+cu129`（wheel 安装）+ editable 安装的 agentinfer（本 worktree）。
   脚本用 `PYTHONPATH=$AGENTINFER_ROOT` 让旧 venv 中的 editable agentinfer 解析到本 worktree。
2. vLLM 需先打两个 MORI provider patch（对 vanilla/PT 无行为影响，未注册 provider 时走原生路径）：
   ```bash
   SP=$VENV/lib/python3.12/site-packages
   patch -p1 -b -d $SP < agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch
   patch -p1 -b -d $SP < agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch   # 顺序固定
   ```
3. 硬件：2× GPU（脚本按 TP2 启动，`CUDA_VISIBLE_DEVICES` 默认 `0,1`）；模型
   Qwen3-Coder-30B-A3B-Instruct-FP8（TP2 下 weights ≈14.65 GiB/卡）。
4. 代码基线：`codex/mori-formal` ≥ `d4af0d1`（mori 新配置面：`starvation_timeout_seconds` 已移除，
   新默认 `min_residency_seconds=1.0 / idleness_margin=0.05 / max_wait_seconds=30.0`）。

## 统一容量设置（三臂严格一致）

| 项目 | 三臂共同值 | 说明 |
| --- | --- | --- |
| TP | 2 | 同一组 GPU |
| `--gpu-memory-utilization` | 0.75（默认，可传参覆盖） | HBM KV ≈ 17.9 GiB/卡 ≈ **392k tokens**（三臂 profiling 相同；mori 仅策略记账多预留 1 个 null block） |
| `--kv-transfer-config` | SimpleCPUOffloadConnector，`cpu_bytes_to_use=17179869184`（16 GiB） | SimpleCPU 按 world_size 平分 → 8.00 GB/rank ≈ **175k tokens** CPU 层 |
| 其余 | `--max-model-len 118784 --max-num-seqs 64 --async-scheduling --block-size 16 --enable-prefix-caching --prefix-caching-hash-algo sha256 --enable-prompt-tokens-details --shutdown-timeout 60` | 完全一致 |

臂间唯一差异 = 调度入口：

- vanilla：`python -m vllm.entrypoints.cli.main serve … --scheduler-cls vllm.v1.core.sched.async_scheduler.AsyncScheduler`
- PT：`AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-ttl-$PORT.sock vllm serve … --agentinfer`
- mori：`AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-mori-$PORT.sock vllm serve … --agentinfer --additional-config '{"agentcache":{"strategy":"mori","schedule_interval_seconds":5.0,"mori":{"mode":"full","window_size":5}}}'`

## 运行

所有命令从仓库根目录执行；`run-d64-tp2.sh <arm> <task_num> <concurrency> <task_timeout_s> <request_timeout_s> [util]`，
每臂冷启动 → 健康等待（1800s）→ replay → 优雅停止。

```bash
# 0) 解压数据集（见上）后，先做 smoke（新环境/新代码必做：验证 mori full 启动与 0 拒绝）
bash benchmarks/mori-formal/agentx/run-d64-tp2.sh mori 1 1 43200 7200 0.75

# 1) 组 A 16×8（serial 配置，session 内串行，三臂依次）
for arm in vanilla progress-ttl mori; do
  bash benchmarks/mori-formal/agentx/run-d64-tp2.sh $arm 16 8 43200 7200 0.75 || break
done

# 2) 组 B 32×16（inflight=8 配置；需要 ≥c714745 的 overlapping 修复）
export REPLAY_CONFIG=$PWD/benchmarks/mori-formal/agentx/config/replay_agentX_inflight8.yaml
for arm in vanilla progress-ttl mori; do
  bash benchmarks/mori-formal/agentx/run-d64-tp2.sh $arm 32 16 43200 7200 0.75 || break
done

# 监控（另一个终端）
RESULT_ROOT=/home/zx/results bash benchmarks/mori-formal/agentx/status.sh
```

路径均可通过环境变量覆盖：`AGENTINFER_ROOT / VENV / MODEL_PATH / MODEL_NAME / TRACE_PATH /
REPLAY_CONFIG / RESULT_ROOT / PORT / MORI_MODE`。结果目录默认
`$RESULT_ROOT/<arm>-d64-<N>x<C>`（不可预先存在），日志 `$RESULT_ROOT/logs/{server,replay}-<arm>-d64-<N>x<C>.log`。

### 两种 replay 配置怎么选

- `config/replay_agentX_serial.yaml`（`max_inflight_requests: 1`，默认）：session 内请求严格串行，
  消除同 Program 重叠请求。**旧代码（< c714745）mori 会拒绝重叠请求**（`overlapping_program_requests`，
  现已改名 `invalid_program_admission`），串行配置可在任何版本安全使用；组 A 采用。
- `config/replay_agentX_inflight8.yaml`（`max_inflight_requests: 8`）：保留 trace 原生 session 内并发，
  需要 ≥ c714745 的 overlapping 支持；组 B 采用。

### 超时建议

`--task-timeout-seconds 43200 --request-timeout-seconds 7200`。教训：util 0.75、32×16 下 vanilla 的
超长 session 在 21600s 处仍完不成（曾有 2/32 任务被截断）；43200s 实测余量充足。

## 结果校验与指标

- **一致性门槛（compare 前）**：同组三臂 `replay-plan.json` 的 `workload_fingerprint`、`source_sha256`、
  `tasks[].source_session_id` 必须逐一相等（不同 task_num 的组间指纹不同属预期）：
  ```bash
  python3 - <<'PY'
  import json
  fps={a: json.load(open(f"/home/zx/results/{a}-d64-16x8/replay-plan.json")) for a in ("vanilla","progress-ttl","mori")}
  ok = all(v["workload_fingerprint"]==fps["vanilla"]["workload_fingerprint"] and
          [t["source_session_id"] for t in v["tasks"]]==[t["source_session_id"] for t in fps["vanilla"]["tasks"]]
          for v in fps.values())
  print("workload identical:", ok)
  PY
  ```
- 官方对比：`vllm bench serve --agentinfer compare --baseline <vanilla目录> --candidate <pt或mori目录>`。
- 核心指标：makespan（`run_wall_time_seconds`）、mean JCT、TTFT/latency 分位、TPOT
  （`(latency-ttft)/output_tokens`）、prefix 命中率、`summary.json` 的 vLLM
  `latency_breakdown_seconds`（queue/prefill/decode 分解）；mori 另有
  `sticky_placement_kept / demotion_to_cpu / planning_*` 等策略计数器。

## 已知注意事项

- agentX 回放走 `/v1/completions` 且 `min_tokens=max_tokens`，响应 `finish_reason=length` →
  lifecycle 中间件推断全为 UNKNOWN（三臂一致；仅损失 terminal 提前回收，TTL/空闲机制兜底）。
- lifecycle socket 必设（不设则 `--agentinfer` 的 LifecycleMiddleware 拒绝启动）；脚本已按臂区分路径。
- vanilla 臂通过 `python -m vllm.entrypoints.cli.main` 绕过 agentinfer 的 vllm shim。
- 回放客户端必须使用含 agentX 执行器的代码（main / mori-formal 分支；mori-formal < a77512f 合并点
  的旧提交对 agentX 抛 NotImplementedError）。

## 历史参考结果（本套件的前身运行，HBM-only、util 0.80、serial inflight）

2026-10-27/28 在 2×L20 上的 16×8（vanilla/PT 无 offload、mori full 16 GiB CPU）：
makespan vanilla 8585s / PT 4618s / mori 4760s，mean JCT 3963/2057/2120s，prefix 命中
51.8%/80.1%/80.1%。mori 落后 PT 的 3.1% 全部来自 queue time（+22.6%），decode 反而最优（−4.1%）。
统一 offload + util 0.75 的新矩阵由本套件命令产生。
