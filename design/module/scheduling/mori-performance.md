# MORI typed-LRU 受控性能对比

## 结论

正式 `MoriCPUCachePolicy` 在保持 victim 序列逐项一致的前提下，CPU typed-LRU 路径中位线程 CPU 从隔离实现的
`3.537876127 s` 降到 `0.622228630 s`，即 **5.6858x** speedup、**82.4124%** CPU 时间降低。正式实现内部的
reservation 调优相对原 proposal/restore 路径从 `0.749834859 s` 降到 `0.622228630 s`，再降低
**17.0179%**；5 轮三种路径的 victim digest 都是
`532c985ffe518a66a502df3aff52434d93ad51912730fe6cd51e3a15251b03b7`。

这达到了“正式实现至少接近隔离实现”的受控实现开销门槛，并且不是通过改变 typed eviction 顺序获得的。该结论
只覆盖 CPU typed-LRU 的 owner/class/recency 维护、victim selection、remove 和 reinsert，不把它外推成真实 GPU
端到端 makespan；完整 server run 仍需同时受模型执行、HBM hook、D2H/H2D 和请求交错影响。

## 输入与公平性

输入来自既有 MORI 隔离回放第一轮的真实 `transfers.jsonl`。比较器只读取已提交 D2H `transfer_submit`，用
`max(0, resident + stored - capacity)` 重建 CPU eviction batch。完整 trace 重建出：

| 项目 | 数值 |
| --- | ---: |
| D2H store calls | 16,849 |
| stored chunks | 890,681 |
| CPU capacity | 10,922 chunks |
| eviction calls | 16,795 |
| evicted chunks | 879,759 |
| final resident | 10,922 chunks |

快速实验固定容量不缩小，使用前 500 次 eviction、共 50,193 victims。三种实现接收完全相同的 batch、192 个
Program、每 8 次 eviction 一次 class transition、每次最多 8 个 recency touch，以及每 11 个 key 一个 shared
owner。每轮运行顺序轮换，进程固定到 CPU 40，以 `time.thread_time_ns()` 计费；初始化不计入 timed region。

被比较的三条路径是：

1. `isolated_scan`：直接加载冻结隔离源码中的 `MORICPUTypedPolicy`，它每次扫描 native `_evictable`；
2. `formal_proposal`：正式索引在本次调优前的 `propose_exact -> restore -> remove` 消融；
3. `formal_reservation`：生产路径 `reserve_exact -> validate -> remove`。

任何一轮 victim digest 不同、操作 counter 跨轮变化或 policy 返回不足 victims，脚本都会 fail closed。隔离路径在
每轮检查 5,461,000 keys；正式 reservation 做 3,501 次 owner-driven entry reclassification，不做 eviction-time
全量 key scan。调优同时把 stale repairs 从 25,068 降到 3,561，原因是已提交 victims 不再先回压为 stale heap
records。

## 原始结果

| 路径 | 5 轮线程 CPU / s | 中位数 / s | 相对隔离 |
| --- | --- | ---: | ---: |
| isolated scan | 3.631059563, 3.537876127, 3.546158098, 3.380454819, 3.447668083 | 3.537876127 | 1.0000x |
| formal proposal | 0.837563209, 0.656473763, 0.749834859, 0.640970773, 0.790686765 | 0.749834859 | 4.7181x |
| formal reservation | 0.591023086, 0.736619854, 0.676912101, 0.622228630, 0.589311185 | 0.622228630 | 5.6858x |

完整逐轮 JSON、输入 SHA-256、affinity 和操作 counters 保存在
[`artifacts/mori-typed-lru-benchmark-20261026.json`](artifacts/mori-typed-lru-benchmark-20261026.json)。

## 复现命令

```bash
STUDY=/data/zx/agent-kvc/research/studies/kvc_theory/mori-reproduction-20261022
PYTHONPATH=/data/zx/agent-kvc/.worktrees/mori \
taskset -c 40 \
/data/zx-envs/agentinfer-vllm-0.31.0/bin/python \
tools/benchmark_mori_typed_lru.py \
  --trace "$STUDY/runs/mori-k5-v6-bounded/r1-full-mori/run/transfers.jsonl" \
  --isolated-source "$STUDY/candidate/joint/mori.py" \
  --capacity 10922 \
  --operations 500 \
  --repeats 5 \
  --output design/module/scheduling/artifacts/mori-typed-lru-benchmark-20261026.json
```

脚本不内置宿主路径；`--trace` 与 `--isolated-source` 必须显式传入。原始 artifact 记录两者的绝对路径与 SHA-256，
避免把不同 trace 或不同隔离源码误报为同一实验。

## 没有运行的验证

本轮没有重跑约 8 分钟一轮的 1,290-request 全链路 CPU simulator，也没有据此声称正式实现已经取得新的
makespan/JCT 数字。选择受控 trace-prefix 是为了在提交前快速定位并消除 production policy 的实现开销；正式
server 的端到端性能门槛仍应在应用 vLLM HBM hook 后，用相同请求 trace、容量和 cache report 单独验证。
