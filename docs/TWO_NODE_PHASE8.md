# 阶段八：Figure 2 单卡干扰实验的 RTX 4090 适配

本阶段在 dcn10 单张 RTX 4090 上比较纯 decode、加入一次 cold full-prefill、加入一次 cached append-prefill。它是论文 Figure 2 方法的硬件/模型适配，不使用双机 KV connector，不涉及 RoCE 性能或动态 PPD 决策。

## 实验方法

原论文为单 H100 / Llama-3.1-8B，batch 可到约 250；本实验为单 RTX 4090 24GB / Qwen2.5-7B-Instruct，先测 batch 1、2、4、8、16。每种场景、每档 batch 做 5 次重复，共 75 个 trial；每个重复随机排列 batch 和场景顺序。

| 请求类型 | 已缓存历史 | 本次新增输入 | 输出 |
|---|---:|---:|---:|
| 背景 decode | 2048 | 4 | 128 |
| full-prefill 干扰 | 0 | 1024 | 1 |
| append-prefill 干扰 | 2048 | 1024 | 1 |

以上均为精确 token 数。请求直接使用 tokenizer 词表中的普通文本 token ID，避免文本重新分词的长度漂移。同一 batch/重复在三种场景下使用相同的背景 prompt；full/append 使用相同的 1024 个新输入 token。干扰请求只输出一个 token，从而测量一次 prefill 的影响。

每次 trial 开始前都预热 batch+1 个独立的 2048-token prefix；纯 decode 和 full 场景也预热相同数量的 prefix，保持准备过程一致。各 trial 和请求角色使用独立 `cache_salt`，隔离历史缓存。API usage 与实际调度记录同时验证缓存命中。

只有每个背景请求都已输出至少 8 个 token 后，客户端才发出干扰请求。通过引擎每步调度记录核对：干扰请求确实与指定数量的 decode 共用一个调度步骤，且本次实际计算恰为 1024 个 prefill token。不能将客户端并发数直接视为实际 batch；完整 batch 的出现次数、启动和收尾时的小 batch 均保留在验收文件中。

## 运行设置与隔离

复用 `ppd-qwen25:source-cu129`，镜像 ID 为 `sha256:ae90422b157e0517caafcb7a4a4756bd2eb845eec7963d7b2f01b166cafce0ad`。单实例容器名 `ppd-figure2-dcn10`，HTTP `192.168.1.11:8300`。

BF16、FA2、eager、prefix caching、block 16、context 4096、max sequences 33、max batched tokens 4096、GPU utilization 0.85、seed 42。服务日志报告约 5.15 GiB KV cache / 96,480 tokens。两种干扰路径的 1024 个新 token 都在一个混合步骤内完成，未被拆成多个 prefill chunk。

额外观测器通过 vLLM 的 `--scheduler-cls figure2_scheduler.TraceScheduler` 加载，继承默认调度器，只记录调度输入/输出，不修改其决策。观测脚本挂载到独立的 `/opt/figure2-tools`，没有覆盖镜像的 `/opt/ppd/vllm-source` 或原生扩展。

开始时原 P/D 已停止，Proxy 运行；本阶段不启动或重建它们。只使用单独的实验容器；不创建镜像、不修改全局网络、驱动、固件或其他项目环境。结束时删除实验容器，并核对原有容器状态。

## 重新采集入口

客户端代码为 `scripts/benchmark/figure2_interference.py`（依赖 aiohttp），不管理服务。启动独立服务时，使用上面的参数，并通过独立只读挂载设置 `PYTHONPATH`、`--scheduler-cls figure2_scheduler.TraceScheduler`、`FIGURE2_TRACE=<可写 evidence 目录>/scheduler-trace.jsonl`。本次完整 Docker argv 保存在证据的 `container-command.json`。不能用宿主机源码覆盖镜像的编译目录。

每次重新采集须使用新输出目录和独立请求命名空间；示例仅展示正式负载入口：

```bash
python scripts/benchmark/figure2_interference.py \
  --server http://192.168.1.11:8300 \
  --output artifacts/figure2-new-run/formal --trial-prefix formal-new-run \
  --batches 1 2 4 8 16 --repeats 5
```

实际执行前先按相同方法运行 batch 1、16、repeats=1 的先导。每个 run 所有请求结束后，在 trace 所在目录写入 `export-request`（内容为新的唯一标识），等待 `scheduler-trace.exported` 内容与其相同，再运行 `figure2_analyze.py --evidence <evidence目录> --run formal`。测量期间不能触发导出。`generate.py --evidence-dir <evidence目录>` 只接受通过验收的正式数据；客户端遇失败保留证据并停止，不重试。

## 指标

主图使用 `token_tpot_ms = (最后 token-ID 事件时间 − 首个 token-ID 事件时间) / (输出 token 数 − 1)`。端点覆盖完整 decode 流，仍不能把每个 SSE 事件都当作单独一个 token 的 ITL；每个事件的 token ID 和时间均保留。

每个 trial 先对其背景 decode 请求求 TPOT 均值，再对相同场景/batch 的 5 个 trial 求均值与样本标准差。相对退化先按同一 batch/重复的 baseline 计算 `(场景 TPOT / baseline TPOT − 1) × 100%`，再汇总 5 次重复。误差棒为标准差，不是置信区间。

逐请求另保留 Phase 7 兼容的 `tpot_ms`（到 SSE EOF）及 `content_tpot_ms`、客户端 TTFT/E2E。只输出一个 token 的预热/干扰请求没有 TPOT，字段为空属于预期。

调度 trace 的 `model_wall_ms` 是调度返回至模型输出交回调度器的墙钟耗时，包含引擎与主机开销，不是 CUDA kernel 时间。它用于定位单次混合步骤的影响，不能直接等同整段流的平均 TPOT。

## 观测器修正与原始尝试

第一次尝试已完成相同的 pilot 和正式负载，但同步 JSON 写盘产生偶发最高约 226 ms 的观测开销，可能污染几个百分点的性能差异。该次完整数据、代码版本和图片保存在证据目录 `attempt1/`，不进入最终主图。

最终观测器只在内存中保存调度记录。pilot / 正式请求全部结束后，外部流程才写入 `export-request`；观测器后台线程随后导出 trace，并通过 `scheduler-trace.exported` 确认完成。测量期间不执行 trace JSON 序列化或写盘。使用同一镜像重新运行相同负载；所有尝试保留，不删除或选择性丢弃异常请求。

第二次尝试完成了 60/75 个正式 trial，随后第 61 个 trial 的一个预热请求报 `ServerDisconnectedError`，其余四个预热成功，流程按失败策略终止，没有重试。模型服务仍在运行，无 OOM，服务日志没有 ERROR/Traceback。怀疑是空闲 HTTP keep-alive 连接与服务端关闭连接竞争，但日志不足以确证根因。完整数据保存在 `attempt2/`，不进入最终主图。

最终客户端对所有角色统一设置 `aiohttp.TCPConnector(force_close=True)`，每请求新建连接，完整重跑 pilot 和 75 个 trial。仍使用相同引擎、负载、seed 和随机顺序，独立的 `pilot-final` / `formal-final` 请求 ID 与 cache_salt 隔离旧缓存；未重试失败请求，未混合三次尝试的性能数据。最终 trace 也含该引擎较早的步骤，验收按最终请求 ID 精确匹配，观测开销只统计对应 run 的步骤。

## 结果与证据

最终正式采集 75/75 个 trial，1055/1055 条请求成功（540 预热、465 背景 decode、50 干扰）；先导 6 个 trial、112 条请求全部成功。正式 50 次干扰都在一个步骤内与指定 B 个 decode 混合执行，新增 prefill 恰为 1024 token；full cached=0，append cached=2048，无抢占。所有原始文件、usage、token-ID 事件计时重算及相同 prompt 核验通过。

| Decode batch | 纯 decode TPOT (ms) | Full TPOT (ms) | Append TPOT (ms) | Full 相对变化 | Append 相对变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.129 ± 0.088 | 17.777 ± 0.082 | 17.825 ± 0.069 | +3.78% ± 0.39 pp | +4.06% ± 0.34 pp |
| 2 | 17.827 ± 0.088 | 18.533 ± 0.110 | 18.520 ± 0.096 | +3.96% ± 0.14 pp | +3.89% ± 0.37 pp |
| 4 | 18.333 ± 0.166 | 19.022 ± 0.115 | 19.108 ± 0.153 | +3.76% ± 0.32 pp | +4.23% ± 0.17 pp |
| 8 | 18.870 ± 0.152 | 19.565 ± 0.138 | 19.814 ± 0.385 | +3.69% ± 0.35 pp | +5.01% ± 2.27 pp |
| 16 | 20.330 ± 0.147 | 20.933 ± 0.064 | 21.051 ± 0.120 | +2.97% ± 0.77 pp | +3.55% ± 0.61 pp |

表中为均值 ± 样本标准差；`pp` 为百分点。相对变化先按同 batch/重复的 baseline 求比值，再做汇总，不直接用两组总体均值相除。batch 8 的 append 波动较大（TPOT 样本标准差 0.385 ms，相对变化标准差 2.27 pp），五次样本全部保留，没有删除偏高样本。

本次 full 的平均 TPOT 退化约 2.97%–3.96%，append 约 3.55%–5.01%，两者处于相近量级；**在当前配置、batch 1–16 与本次计时口径下，没有复现原论文“append 干扰显著小于 full”的趋势**。不能据此推广到 H100/batch 200，也不能用于否定其他负载下的差异。硬件、模型、batch 范围、缓存历史及整段流计时均与原图存在适配差异，论文正文未完整定义其微基准的全部参数。

正式 trace 中 full 混合步骤的平均墙钟耗时约 101.93–108.51 ms，append 约 107.83–115.07 ms；同一 trial 的普通 decode 步骤约 16.99–19.70 ms。可见单次 prefill 的即时扰动明显，整段 128-token 平均 TPOT 稀释了这一扰动。append 同样新增计算 1024 token，还要注意 2048-token 历史；本次没有观测到其计算量显著更小的依据，不能预设其必然更快。

最终观测器在正式 run 对应的 9935 个步骤上，采集和内存保存开销均值 9.586 μs、最大 74.043 μs；这些是主机侧观测开销，没有宣称零扰动。先导对应步骤均值 11.704 μs、最大 38.593 μs。服务日志未发现 ERROR/Traceback/CUDA OOM。

结果图、SVG/PDF、GitHub 可独立重画的数据快照见 [Phase 8 图目录](../results/figures/phase8/README.md)。完整证据为 `artifacts/two-node-phase8-figure2-evidence/`；先读 `stage-verification.json`，再读 `formal/verification.json` 和 `formal/summary.csv`。两次早期尝试单独归档，不进入主图。

实验容器已停止删除，GPU 无计算进程；原 Proxy 保持运行，原 P/D 保持退出。两端原容器的 ID、镜像、完整 State 字段与实验前相同，未创建镜像。PDF SHA256 仍为 `3b53ebc8158aae53e2aedb37855b80fe40a0474bc7be37d3797f8226b6ad8c32`。本阶段到此暂停，没有启动离线校准、Figure 6 或新一轮实验。

