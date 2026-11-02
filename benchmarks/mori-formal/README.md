# MORI formal benchmarks

该目录按 workload/trace 家族隔离 MORI 正式实现的可复现实验套件：

- `swe-cc-local-trace/`：SWE/Claude Code AgentInfer 本地请求 trace，比较 Vanilla、Progress-TTL 和 MORI full。
- `agentx/`：AgentX duration64 trace replay 三臂对比套件（统一 SimpleCPU offload、TP2、util 0.75）。

已通过两次 AgentX 16/8 + SWEBench 32/16 双 trace 晋升门槛的冻结版本与真实运行参数见
[`MORI-VALID-1030.zh-CN.md`](MORI-VALID-1030.zh-CN.md)。该记录描述的是提交 `6acbc84c` 的单卡
性能验证矩阵，不应与本目录较早的通用 TP2 示例互换。

后续 AgentX trace 应放入新的同级目录，使用自己的 trace adapter、配置、并行度和校准，不复用
`swe-cc-local-trace` 的合成 prompt 参数。
