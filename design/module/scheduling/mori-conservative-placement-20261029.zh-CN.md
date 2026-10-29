# MORI 保守层级与缓存一致性改动（2026-10-29）

本次实现限定为上下文归属清理、CPU 降级补传和 BUSY 建议后的常规调度唤醒。
实现前已获取并合入并行日志提交 `e169dd5ec8e39fa338841035b4b1fbea3d359d1c`；
上下文替换同时保留日志状态清理和本次缓存归属清理。

## 行为及边界

1. 新 prompt 撤销该 Program 旧上下文的 footprint 和 native ownership，保留其他共享 owner、
   物理 KV、pin、LRU 年龄和 Program 历史。后续实际命中重新挂接所需块。
   每个上下文有独立的不透明凭证；请求清理后仍可用，下一 prompt、release 或 shutdown 后失效。
   证据：`agentinfer/agentcache/mori/runtime_state.py:109-170`、
   `agentinfer/scheduling/mori/strategy.py:76-83`。
2. 保留 eager offload 和现有 admission 比较。已完成请求保存完整、已确认、位置稳定块的内容身份，
   不持有源 pin；真正 BUSY→IDLE 后只登记一次补传意图。
   native 构造 metadata 时再校验当前上下文、CPU 副本、在途复制和 GPU 内容身份。
   目标准入成功后才 pin 源块并进入现有异步传输；失败直接跳过。此顺序避免在 MORI 容量规划中
   分配 CPU 块、改变本轮 footprint。仅传输的 step 也可推进；旧上下文和 reset 可取消未执行意图。
   boundary-state 和 partial-tail 沿用原路径，CPU tier 不等于已有完整 READY 副本。
   证据：`agentinfer/agentcache/mori/simple_cpu_policy.py:41-70`、
   `agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch:288-407,521-577`。
3. 仅实际 HBM 分配的完整 MORI 建议包含 BUSY 时合并一个唤醒标志；下一安全边界提前运行常规调度。
   不等待降级、不强制 victim、不虚构释放容量；重排可以不改变任何层级。
   native 拒绝建议后仍可能出现保守唤醒；仅 fallback 淘汰 BUSY 不触发此信号。
   持续新建议可每个 step 唤醒一次，合并不是限频。
   证据：`agentinfer/agentcache/mori/hbm_policy.py:78-84`、
   `agentinfer/agentcache/core/scheduler.py:242-253`、`agentinfer/scheduling/mori/strategy.py:444-446`。

本次不引入压力水位、主动压力预复制、radix tree、历史衰减或前缀／后缀 LRU 优化。
仍可能保留 GPU/CPU 重复副本；补传是机会性修复，不保证性能提升或缓存严格随层级迁移。
SimpleCPU hook 升至 API 3，部署须更新两个 native patch 并重新构建；旧 API 会在启动时拒绝。
证据：`agentinfer/agentcache/mori/simple_cpu_policy.py:23`、
`agentinfer/agentcache/patches/README.md:47-87`。

## 验证

在合并日志更新后的隔离工作树运行：

```bash
MORI_VLLM_SOURCE=/tmp/vllm-031 python -m pytest -q tests/scheduling tests/agentcache -m cpu_test
pre-commit run --all-files
git diff --cached --name-only -z | xargs -0 pre-commit run --files
git diff --cached --check
```

CPU 回归结果为 **638 passed，26 deselected，16 warnings**，包括上下文替换、共享块、迟到凭证、
BUSY 建议合并及无调整、实际重排、结束时拒绝后降级补传、源块复用、CPU/inflight 去重、
context/reset 取消，以及只传输不推理的推进路径。原始输出见
[pytest.txt](artifacts/mori-conservative-placement-20261029/pytest.txt)。

全量 pre-commit 的 ruff、format、typos 通过；Markdown hook 仅自动添加了既存
`benchmarks/mori-formal/agentx/README.md` 的代码块周围空行，因此返回失败。
该无关改动已撤回，本次修改文件单独运行全部 hooks。全量原始输出见
[pre-commit-all.txt](artifacts/mori-conservative-placement-20261029/pre-commit-all.txt)，
范围内检查输出见 [pre-commit-changed.txt](artifacts/mori-conservative-placement-20261029/pre-commit-changed.txt)。

环境为 Python 3.12.13，依赖导入使用已安装 vLLM 0.32.0。native 测试将两个 patch 应用于
干净的 vLLM 0.31.0 源码（`db9527a46873454610df6dbedf79a36d6bf1a7f6`）副本，再执行 CPU 控制流。
没有修改已安装的 native 包，也未运行 GPU DMA、端到端服务或性能 benchmark；不据此声称吞吐收益。
证据：`tests/agentcache/test_mori_native_patch.py:21-41,131-153`。
