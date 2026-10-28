# Current experiment provenance

本套件由以下本地正式实验整理而来：

```text
/data/zx/agent-kvc/research/studies/kvc_theory/mori-formal-simplecpu-20261027
```

三臂选取的已完成运行是：

| 套件 arm | 当前实验 run | 原 server command SHA-256 |
| --- | --- | --- |
| `vanilla` | `vanilla-redesign-control-01` | `a88674ded13693e5d7bb9613bafb0e2fb8e52fd187132ad9e2c008fb49fae761` |
| `pt` | `pt-refresh-01` | `d7ac7bc64c1fc6cbd79a7444b768f28013253bb0c664abc23c7c5dce8333c3e0` |
| `mori` | `mori-redesign-01` | `89de2c91612335b47dccbd7bd5ca4fabb804769a4218a5149ff065a18693ba81` |

三个 manifest 均为 `completed`，并记录相同的模型、16 task、8 并行和来源 trace。来源 Replay YAML 的
SHA-256 为 `57389a1393cd794f7addbfeec2596753e3b787b03c6f11610c2049fe2896b006`。

2026-10-28 的静态核对结果：

- `run_matrix.sh plan` 生成的三条 server command 与上述三个 runtime command 逐 token 完全相同；
- 三条 AgentBench client command 在只归一化 config/result 绝对路径后完全相同；
- `config/replay16-8.yaml` 与来源 YAML 只在删除旧绝对 `result_dir`、把 trace 改为仓库相对路径两处不同；
- 随套件提交的 trace 与来源 trace 字节相同，SHA-256 为
  `d38b0afe97ea0dd0689d32bff6a888831637f3eb963660055d58a8b4d30c7b4a`；
- Vanilla/MORI 当时使用的 frozen AgentInfer 源码目录与整理时分支的 `agentinfer/` 目录递归比较无差异；
  MORI 仍在 Python path 前置相同的 patched vLLM overlay，Vanilla/PT 仍使用原生 vLLM。

这次整理没有重新运行 GPU 长实验。上述一致性说明验证的是套件忠实复现当前实验设置；新的性能结果仍应由
`run_matrix.sh matrix` 产生，并保留完整的 AgentBench 和 runtime 证据目录。
