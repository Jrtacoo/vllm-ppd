# Figure 2：扩大 decode batch 的补充实验

用户于 2026-11-12 授权扩大 decode batch。本实验保留 [首轮 batch 1–16](TWO_NODE_PHASE8.md) 的全部数据和图片，新证据位于 `artifacts/two-node-phase8-figure2-large-batch-evidence/`，新图位于 `results/figures/phase8/large-batch/`。不预设 batch 增大必然扩大 full/append 的差距。

## 容量与两组实验

仍使用单 dcn10 RTX 4090、Qwen2.5-7B-Instruct BF16、FA2、eager、prefix cache，GPU utilization 0.85、context 4096、block 16、max batched tokens 4096、seed 42。唯一服务容量变更为 max sequences 256，以容纳 250 个 decode 和 1 个干扰请求。使用既有镜像和独立容器 `ppd-figure2-large-dcn10`，不改原 Proxy/P/D，不覆盖 `/opt/ppd/vllm-source`。

服务报告 KV 容量 **80,224 token / 4.28 GiB**，小于原 max sequences 33 时的 96,480 token。250 条独立 2048-token 历史，仅历史 KV 就需约 27.34 GiB（本模型每 token 57,344 bytes），加上权重后无法装入 24GB 4090。

因此分两组，不能将两者混成同一条扩展曲线：

| 组别 | batch | 背景历史 | 每点重复 |
|---|---|---|---:|
| 独立历史 | 32 | 与首轮相同，各请求的 2048-token 历史内容和 cache_salt 独立 | 5 |
| 共享历史 | 1、8、16、32、64、100、150、200、250 | 同一 trial 的背景请求共享相同 2048-token 历史与 cache_salt；各请求的 4-token 新输入不同，输出独立生成 | 5 |

两组均为 2048 cached + 4 new → 128 output 的背景请求。Full 干扰仍为 0 cached + 1024 new → 1 output，append 为独立的 2048 cached + 1024 new → 1 output。Full/append 的新增 1024 token 相同。同一组同 batch/重复，三场景背景输入相同；不同组不是严格相同 prompt 的配对实验。

共享前缀是为在本卡上保持逻辑上下文长度并达到大 batch 的适配。它会改变 KV 物理布局和读取复用，因此不能称为 250 路独立 KV，也不能据此推断真实多会话容量。其横轴仍需由调度 trace 验证确实有 B 条序列参与 decode；共享 KV 不会把 B 个生成请求合并为一个。

## 与仓库已有微基准的关系

核对 `scripts/tests/interference_benchmark.py` 可见，它也让背景请求使用共享 `context_prefix`，因此本次共享历史方式有仓库实现依据。但该脚本默认背景和干扰均输出 64 token，并一起启动；本次为延续首轮控制，仍采用背景输出 128、干扰输出 1、等每个背景已生成至少 8 token 再注入。其 append 还复用背景的同一段历史，本次 append 历史单独预热。已有脚本不能证明论文原图所有实际运行条件，不能声称此次逐参数一致。具体核对记录为证据的 `existing-benchmark-review.json`。

## 容量试探与验收

第一次独立历史先导随机先执行 batch 40 的 append 场景。82 条请求全部成功返回，但背景/干扰缓存命中与执行重叠不满足预定条件，流程停止。40 条历史已经超过 80,224-token cache，出现缓存挤出，因此不作为有效 batch=40 性能点。完整请求、SSE、trace 和日志保留在 `capacity-independent40/` 等文件中。

随后先导验证独立 batch 32，以及共享 batch 64、128、250。只有通过缓存、无抢占、实际混合 B 个 decode、恰好 1024 新 prefill token 等核验的负载才进入正式采集。

客户端连接池上限由固定 100 改为 `max(100, max_batch+16)`，防止 HTTP 连接池将 batch 250 限制为 100。所有请求仍使用新连接，不重试失败。共享组每 trial 只预热两个 prefix（背景与干扰各一个），独立组为 B+1；每组内三场景使用相同的准备过程。各 trial 的 cache_salt 隔离，不跨 trial 复用。

主图沿用首轮的整段 token-ID SSE TPOT 口径，均值与样本标准差按 trial 汇总，退化先与同 batch/重复的纯 decode 配对。新图横轴使用线性刻度，显示到 250。单次 prefill 对 128-token 平均 TPOT 的影响仍会被稀释，另保留混合步骤的模型墙钟时间。观测器只向内存追加，所有请求结束后才导出 trace。

## 结果

最终共享历史 135 个 trial、12675 条请求，独立历史 15 个 trial、985 条请求，全部成功并通过 token、缓存、实际 batch、混合执行和无抢占核验。正式混合步骤共 100 次（共享 90、独立 10），最大实际 decode batch 为 250。

共享历史组：

| Batch | Decode TPOT (ms) | Full TPOT (ms) | Append TPOT (ms) | Full 变化 | Append 变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.040 ± 0.039 | 17.726 ± 0.025 | 17.784 ± 0.056 | +4.03% ± 0.14 pp | +4.37% ± 0.31 pp |
| 8 | 18.260 ± 0.017 | 18.962 ± 0.052 | 19.012 ± 0.032 | +3.84% ± 0.27 pp | +4.12% ± 0.17 pp |
| 16 | 18.949 ± 0.033 | 19.663 ± 0.030 | 19.721 ± 0.023 | +3.76% ± 0.19 pp | +4.07% ± 0.19 pp |
| 32 | 21.414 ± 0.028 | 22.199 ± 0.071 | 22.224 ± 0.027 | +3.66% ± 0.36 pp | +3.78% ± 0.20 pp |
| 64 | 23.017 ± 0.596 | 23.993 ± 0.683 | 23.703 ± 0.038 | +4.31% ± 4.38 pp | +3.04% ± 2.60 pp |
| 100 | 25.311 ± 1.073 | 25.506 ± 0.034 | 25.831 ± 0.619 | +0.91% ± 4.22 pp | +2.23% ± 5.73 pp |
| 150 | 28.020 ± 0.065 | 29.196 ± 0.090 | 29.612 ± 0.845 | +4.20% ± 0.25 pp | +5.69% ± 3.21 pp |
| 200 | 33.700 ± 0.487 | 35.100 ± 0.750 | 35.276 ± 0.822 | +4.18% ± 3.19 pp | +4.69% ± 2.75 pp |
| 250 | 35.985 ± 0.168 | 37.930 ± 0.844 | 37.950 ± 0.662 | +5.41% ± 2.58 pp | +5.46% ± 1.80 pp |

独立历史组（当前配置的有效扩展点）：

| Batch | Decode TPOT (ms) | Full TPOT (ms) | Append TPOT (ms) | Full 变化 | Append 变化 |
|---:|---:|---:|---:|---:|---:|
| 32 | 23.655 ± 0.016 | 24.365 ± 0.023 | 24.408 ± 0.030 | +3.00% ± 0.12 pp | +3.18% ± 0.14 pp |

表中均为五次 trial 均值 ± 样本标准差，pp 为百分点；相对变化先按同 batch/重复与 baseline 配对。batch 250：decode=35.985 ms/token，full=37.930 ms/token，append=37.950 ms/token；full/append 相对变化分别为 +5.41% / +5.46%。

**本轮扩大到 batch 250 后，仍未观察到论文中 full/append 干扰相差一个数量级的趋势，不能证实“首轮差距小仅由 batch 小造成”。** 大 batch 的部分点标准差较大，不对两条接近曲线作显著性判断。

作为单次干扰的补充观察，以下为五个 trial 的步骤墙钟均值，非 CUDA kernel 时间，也不是整段 TPOT：

| Batch | 纯 decode 步骤墙钟 (ms) | Full 混合步骤墙钟 (ms) | Append 混合步骤墙钟 (ms) |
|---:|---:|---:|---:|
| 1 | 16.93 | 102.57 | 107.66 |
| 8 | 18.05 | 104.74 | 111.21 |
| 16 | 18.65 | 108.35 | 114.73 |
| 32 | 20.94 | 114.91 | 121.57 |
| 64 | 22.20 | 129.40 | 135.97 |
| 100 | 24.07 | 150.00 | 156.65 |
| 150 | 26.26 | 173.82 | 180.48 |
| 200 | 31.04 | 203.37 | 210.05 |
| 250 | 33.08 | 225.83 | 235.85 |

这表明一次 prefill 会明显拉长混合步骤，但 full 与 append 仍处于相近量级。主图的 128-token 整段平均稀释了这一次扰动。

正式共享组 observer 开销 P99 为 0.200 ms、最大 229.298 ms，17552 个对应步骤中有 20 个超过 1 ms。大停顿均保留，不能把几个百分点的差异解释为精确的纯 GPU kernel 差异。最终共享组客户端末 token 相对引擎结束的 P99 延迟为 56.77 ms、最大 66.59 ms，未再出现超过 1 秒的尾部延迟；该诊断没有用于事后扣除或修正 TPOT。

两组不能合并为一条“独立 KV 从 1 到 250”的曲线。共享组与原论文的硬件、模型、采样窗口和物理布局仍有差别；本次整段 TPOT 不能直接与论文 batch 200 的 48%/2% 数字对齐，不能把差异仅归因于 batch。完整调度步骤统计在各 run 的 verification.json 中。

最终客户端尾 token 的到达延迟（相对于同一时钟的引擎结束时间）见 client-tail-verification.json，不用于事后修正图中 TPOT。观测开销的 P50/P95/P99/最大值及超过 1 ms 的次数也全部保存，不能将 observer 声称为零开销。

图和可独立重画快照见 [large-batch 图目录](../results/figures/phase8/large-batch/README.md)。原始证据见 `artifacts/two-node-phase8-figure2-large-batch-evidence/`，优先读 stage-verification.json、formal/summary.csv、formal/verification.json。

独立实验容器已清理，GPU 无计算进程；原 Proxy 仍运行、原 P/D 仍退出，原容器 ID/Image/完整 State 未改变。PDF 和首轮图片保持原哈希，没有创建镜像或启动下一阶段，未提交或推送 Git。


## 客户端写盘干扰修正

第一轮大 batch 正式采集功能检查全部通过，但发现客户端的同步证据落盘会延迟仍在接收中的 SSE。一个 batch 16 baseline 的引擎 decode 全程约 2409.59 ms，单步最高 22.22 ms，observer 最高 23.47 μs；客户端第一个与最后一个请求的末 token 记录却相差 1168.69 ms，后续请求 TPOT 被抬高。诊断、全部原始数据与当时图片保留在 `attempt-sync-io/`，不作为最终性能结果。

修正后的 `--defer-artifacts` 将 request/SSE/events/response 以及请求 CSV/JSONL 在活跃推理期间缓存在内存，同一批所有在途请求都结束后才序列化、写盘。预热和正式 decode 分别刷新，失败结果也保留。完整重跑相同先导与两组正式计划，使用新的请求命名空间，不选择性重跑偏高样本。引擎观测器和模型服务未变。本轮主图只使用修正后的完整结果。
