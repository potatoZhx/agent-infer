# MORI 慢首 token 诊断

MORI 默认对超过 30 秒仍未输出首 token 的请求记录结构化 WARNING，前缀为
`MORI slow request`，后面是单行 JSON。用于区分 MORI 准入等待、native 首次调度等待，
以及首次调度到首次输出之间的停留；不改变准入、交换或缓存保留策略。
这里的计时从嵌入式 controller 接收请求开始，不包含此前的客户端网络或 API 排队时间。

配置位于 `additional_config.agentcache.mori`，独立于通用 debug observability 开关：

```json
{
  "strategy": "mori",
  "mori": {
    "slow_request_threshold_seconds": 30.0,
    "slow_request_repeat_seconds": 60.0
  }
}
```

`slow_request_threshold_seconds=0` 完全关闭 tracker 和策略诊断上下文。
阈值须有限且非负，重复间隔须有限且大于零。
证据：`agentinfer/scheduling/mori/config.py:19-49`、
`agentinfer/agentcache/core/controller.py:82-128`。

## 一条日志能回答什么

| 字段 | 含义与使用边界 |
| --- | --- |
| `request_id`, `program_id`, `generation`, `external_program_id` | 关联本次请求、内部 owner 及外部业务身份；不记录 prompt 内容 |
| `phase` | `request_pool`、`native_waiting`、`native_prefill`、`native_decode`；来自实际生命周期回调 |
| `request_pool_seconds` | 接收请求到交给 native adapter 的时间 |
| `native_wait_seconds` | 交给 adapter 到首次观察到 native 调度的时间，不含随后 preemption 的等待 |
| `scheduled_to_first_output_seconds` | 首次 native 调度到首次输出的经过时间，包含 preemption，不能当作纯 GPU prefill 耗时 |
| `ttft_seconds`, `native_preemptions` | 首 token 延迟及观察到的抢占次数；缺少首输出回调时 TTFT 为 null |
| `cached_prefix_tokens`, `hbm_cached_prefix_tokens` | 原生回调提供的缓存命中 token 数；未观测到时为 null |
| `context.last_decision` | 最近一次达到阈值后的候选评估，保存当时的 score、排名、容量和置换可行性 |
| `context.last_decision_age_seconds` | 决策证据的年龄；不把过去决策当成当前事实 |
| `context.current_score` | 当前滑动窗口 A/R、保留/丢弃的 cycle 数、partial 时间和等待补偿 |
| `context.latest_snapshot` | 最近已有 snapshot 的时间、年龄、native 使用量、H2D/D2H pending 数量 |
| `context.reconstructed_capacity` | 最近 snapshot 的 views 与当前 factors/footprints 重建的容量，明确区别于当时的精确决策 |

`last_decision.reason=idleness_limited` 表示当前允许释放的 Acting 集合不能容纳候选，
但释放所有可考虑的 Acting 对象后可以容纳；`no_all_acting_fit` 表示后者仍不足。
其他原因包括 `global_capacity_debt`、`claim_exceeds_total_quota`、`exchange_invalid`、
`demotion_rejected`、`promotion_rejected` 和 `promotion_applied`。
容量投影沿用共享块并集，不能用各 owner 的 claim 简单相加代替。
排名字段提供上下文，本身不证明队列优先级是根因。

证据：`agentinfer/scheduling/request_diagnostics.py:187-265`、
`agentinfer/scheduling/mori/diagnostics.py:17-171`、
`agentinfer/scheduling/mori/strategy.py:270-390`。

## 日志量和开销控制

- 正常请求只有常数时间的标量/字典更新，不做 JSON 编码、日志 I/O 或诊断容量扫描。
- 轮询间隔至少 5 秒，每次最多扫描 256 个活动请求、输出 8 条；同一请求至少间隔 60 秒。
  轮转保证后面的请求也能被检查，大量并发时首次告警会晚于阈值。
- 只在慢请求输出告警时重建容量上下文；精确决策证据对每个等待 owner 最多每 5 秒更新一次。
- 首 token 后停止轮询告警。慢 TTFT 的首 token 和结束各保留一条摘要；这些生命周期摘要不共享轮询条数上限。
  首 token 很快、只是 decode 很长的请求不产生日志。
- 每条 JSON 最多 8192 字符，超限丢弃扩展 context 并标记截断；不输出 block 列表或 prompt。
  完成、取消及 owner release 清理活动状态，ProgramRef 的 generation 隔离迟到事件。

证据：`agentinfer/scheduling/request_diagnostics.py:16-185`、
`tests/scheduling/test_request_diagnostics.py:31-165`。

## 验证与复现

CPU 开销脚本对默认开启与显式关闭进行交替配对，计时使用线程 CPU 时钟，
覆盖实际 runtime 生命周期、native/token 回调、正常规划、慢请求规划和日志文件写入。
验收预算事先固定：每请求生命周期增量 10 微秒、每热回调 1 微秒、正常规划 20 微秒；
慢请求诊断每 5 秒规划的增量中位数不超过 5 毫秒。
4096 请求集中释放另行测量，不把生命周期摘要算入轮询的 8 条上限。

```bash
python -m tools.benchmark_mori_diagnostics \
  --output-dir design/module/scheduling/artifacts/mori-diagnostics-20261029
```

命令、原始配对样本、源码 SHA-256、门限判定和抽样日志保存在
`design/module/scheduling/artifacts/mori-diagnostics-20261029/`。
这是 CPU 微基准，不代表完整 GPU workload 的吞吐变化；本次没有重新运行 32/16、16/8 GPU 实验。
证据：`tools/benchmark_mori_diagnostics.py:1-39`；验证结果见同目录的 `overhead.json` 和 `pytest.txt`。

2026-10-29 的 7 轮配对测量全部通过预设预算：生命周期增量中位数 4.502 微秒/请求，
native 热回调增量 108–118 纳秒/次，正常 16 owner 规划增量 0.965 微秒/次；
32/256 owner 的慢请求实际规划及生产 context 增量分别为 1.060/2.677 毫秒/次。
4096 请求集中释放时另写 4096 条摘要，耗时 97.4 毫秒 CPU；这项突发成本不受轮询上限保护。
证据：`design/module/scheduling/artifacts/mori-diagnostics-20261029/overhead-validation.txt:1-12`、
`design/module/scheduling/artifacts/mori-diagnostics-20261029/overhead.json:628-650`。
