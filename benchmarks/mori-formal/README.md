# MORI formal benchmarks

该目录按 workload/trace 家族隔离 MORI 正式实现的可复现实验套件：

- `swe-cc-local-trace/`：SWE/Claude Code AgentInfer 本地请求 trace，比较 Vanilla、Progress-TTL 和 MORI full。

后续 AgentX trace 应放入新的同级目录，使用自己的 trace adapter、配置、并行度和校准，不复用
`swe-cc-local-trace` 的合成 prompt 参数。
