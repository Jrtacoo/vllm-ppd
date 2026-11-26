# Phase11：单节点 GPU 资源时间线

状态：90 个正式 trial 已完成并通过验收；图片已生成。

本阶段延续 Phase10 的独立历史请求，目标是观察不同背景 batch 下模型前向期间的 SM 活跃度和显存读写带宽，固定一次干扰，避免混入干扰次数变化的影响。节点为 dcn9 的单张 RTX 4090；模型 Qwen2.5-7B-Instruct，BF16，vLLM 0.13.0+ppd.local，复用 ppd-qwen25:source-cu129 镜像。模型权重已驻留显存；时间线包含推理期间从显存读取权重的 GPU 工作，不包含从磁盘装载模型。

## 实验配置

| 项目 | 设置 |
|---|---|
| 背景 batch | 10、50、100、150、200、250 |
| 场景 | Decode only、Append prefill、Full prefill |
| 重复 | 每个条件 5 次，共 90 个正式 trial |
| 背景请求 | 每个请求独立 256-token 历史 + 32-token 新输入，输出 128 tokens |
| 干扰请求 | 2048-token 历史 + 128-token 新输入，输出 1 token |
| 干扰次数与触发 | 每 trial 一次；所有背景客户端均收到至少第 32 个输出 token 后发送 |
| Append | 同历史缓存标识，命中 2048 tokens，计算新增 128 tokens |
| Full | 相同完整 2176-token 输入，冷缓存标识，重算全部 2176 tokens |
| 调度 | FCFS、continuous batching、chunked prefill、eager |
| 容量限制 | max_num_batched_tokens=2048，max_num_seqs=251，max_model_len=18432 |
| 显存配置 | gpu_memory_utilization=0.94，block_size=16 |
| 隔离 | 每个 trial 清空 prefix cache；三种场景预热同量、同长度的独立历史 |

同 batch、同 repeat 下三种场景使用同一批逻辑输入；随机种子包含请求编号。请求使用精确 token ID，以避免文本分词后的长度漂移；每个请求、角色、trial 的 cache_salt 隔离缓存，Append 只与自己的 warmup 共享标识。Full 输入包含历史与新增的全部 token，并非只发送新增部分。三场景执行顺序按种子打乱。先导不计入正式 90 组。

## 测量口径

主工具为 Nsight Systems 2025.1.3，ad10x 指标集，10 kHz（100 μs）GPU 计数器采样，另采 CUDA/NVTX 时间线。关闭 CPU sampling、context-switch sampling 与 CUDA Event completion tracing。硬件计数器权限只授予本阶段独立容器，未修改驱动或全局 profiling 配置。

真实硬件指标名称：

- `SMs Active [Throughput %]`：SM 活跃度；不能解释成 occupancy 或 FLOPS 利用率。
- `DRAM Read Bandwidth [Throughput %]`：显存读取带宽占 Nsight 定义峰值的百分比。
- `DRAM Write Bandwidth [Throughput %]`：显存写入带宽占 Nsight 定义峰值的百分比。

后两者是 GPU GDDR6X 读写指标，不是显存容量占比，也不是 RoCE 网络带宽；保留工具原始百分比，不臆测换算 GB/s。

独立 Worker 子类标记模型主体 `_model_forward` 和外层 `execute_model`，记录逐步调度 token 映射。NVTX 的 CPU range 通过 CUDA launch API 的 correlationId 关联实际 GPU kernel，取首个 kernel 开始至最后一个 kernel 完成作为 GPU 边界；不能直接使用 CPU range 长度。没有在各 forward 内插入 `cudaSynchronize()`。同步只位于 trial 采样开始前和所有请求完成后的结束阶段；结束时的 trace flush 不计入背景 token 间隔。

完整时间线保存各次重复，不将时序错位的峰值直接平均；事件分析以实际首个干扰 prefill forward 对齐。普通 decode 的对照按同 repeat、同背景输出位置匹配。原始 100 μs 计数器保留，展示图如进行时间分箱必须标明分辨率。

每个请求保留 request JSON、原始 SSE、token 事件与 response JSON；保留失败、禁止隐藏重试。每个捕获保留 `.nsys-rep`、只读分析的 SQLite、Worker JSON 与调度器 trace；跨节点复制核对 SHA256。

## 文件位置

- 原始证据：`artifacts/phase11-resource-timeline-evidence/`。
- 运行脚本：`scripts/benchmark/phase11_*.py`。
- 发布图片：`results/figures/phase11/`（完成后生成）。

仅创建本阶段实验容器；不挂载覆盖镜像内的 `/opt/ppd/vllm-source`，不新增镜像，不修改原有 Proxy。

## 先导验收

先运行两段 BF16 矩阵乘验证重复捕获，再运行 B=10、250 的三种场景，各做一遍采样关闭与开启的配对先导，共 12 个 trial。6 个采样捕获均取得 SM、DRAM read/write，采样间隔中位数约 99,994 ns，未发现超过 150 μs 的间隔或非法百分比；模型主体和外层 GPU 边界均完整。

配对 TPOT 差异为 -7.67%～+2.49%，其中 5 项增加 1.24%～2.49%，1 项减少 7.67%。每条件仅一次，不能从这一结果扣除固定开销或宣称负开销；正式时延属于开启采样后的观测值。capture stop 的 trace flush 发生在全部请求完成后，明显增加控制器耗时，但不加入 127 个背景 token gap 的统计。

Nsight 诊断保留了驱动 CUDA API 13.0 与工具库 12.9 的兼容提示，以及通用 NVTX 捕获提示。主机驱动不作变更；验收直接检查本实验每个 NVTX 模型范围的实际关联 kernel 与硬件采样完整性。不要将这些提示删除后宣称工具无任何 warning。

服务本次实测模型权重占约 14.25 GiB，KV 池约 6.44 GiB，即 120,624 tokens。按每背景最终 416 tokens、5 条预热干扰历史合计 10,240 tokens、Full 冷请求 2176 tokens 粗算，B=250 的保守 KV 需求约 116,416 tokens（不含分配器细节）；最终以请求缓存命中、无抢占和运行检查为准。

先导运行中修正了两项控制器问题：健康检查需要明确禁用 HTTP 代理；`docker top` 的输出格式需要含 PID 列。修正发生在正式实验之前；原始失败控制器日志保留，未隐藏或重试推理请求。另有两次离线提取失败（Python 环境缺少 aiohttp、末尾空清理调用计数）也保留日志；修正提取逻辑后对原始数据重新分析，没有重跑或替换采集结果。

`execute_model` 包含模型主体和 logits 等外层工作，`_model_forward` 仅标记模型主体；图中会区分两种边界。DRAM 计数器观察 GPU 全局显存流量，无法单凭该曲线将权重读取与 KV 读取精确拆开。Full 的两块都包含同批背景 decode；它们不是独立执行的一条纯 prefill 请求。

输入构造沿用 Phase10 的独立历史生成函数：候选干扰历史数为 `max(1, ceil(B/50))`，三个场景都预热全部候选；本阶段始终只发送第 1 条候选干扰请求。其余候选只占用预热缓存，不参与测量期间的 attention。这样保留了 Phase10 的输入构造和预热占用；不能将候选历史数量解释成实际干扰次数。B=250 的容量估计已包含全部 5 条候选历史。

## 正式结果与验收

共 90 个正式 trial、23,100 个请求、3,980,414 个采样时刻、11,758 次实际 GPU 前向。最大采样间隔 100000 ns，超过 150 μs 的间隔为 0，抢占为 0；请求、缓存、Full/Append 计算量、事件重叠与 GPU 边界检查均通过。

| Batch | Decode TPOT ms | Append TPOT ms | Full TPOT ms | Append 混合步 ms | Full 两混合步之和 ms |
|---|---|---|---|---|---|
| 10 | 18.24 | 19.16 | 19.75 | 31.84 | 215.56 |
| 50 | 21.68 | 21.78 | 23.87 | 28.92 | 230.65 |
| 100 | 25.31 | 25.27 | 26.78 | 39.24 | 251.14 |
| 150 | 30.01 | 30.51 | 31.78 | 50.48 | 272.08 |
| 200 | 37.79 | 37.93 | 39.41 | 58.00 | 296.08 |
| 250 | 42.05 | 42.15 | 44.07 | 64.67 | 318.65 |

表内均为 5 次重复的均值。GPU 混合步耗时取首个关联 kernel 至末个关联 kernel，包含内部空隙；Full 两步之和不包含两块之间的调度间隔。


B=10 的 Append 第 1 次重复包含约 35.16 ms 的 GPU kernel 间隙：该步首尾跨度 60.43 ms，kernel 执行区间并集约 23.40 ms；第 3 次重复相应为 24.78 / 23.86 ms。具体间隙来源未隔离。该点保留在全部 5 次的均值和标准差中，因此小 batch 的大误差条不能直接归因于 Append 计算本身。


Full 干扰的 GPU 混合步总耗时随 batch 从约 216 ms 增至 319 ms；整段 TPOT 增量则被 127 个 token 间隔稀释。Full 期间的 SM 活跃度约 96%～97%，显存读带宽占比约 18%～21%；它们与普通 decode 的资源形态不同，但不能仅凭 SM Active 宣称 FLOPS 或 occupancy 饱和。


发布了 13 张图（PNG/SVG/PDF）：6 张整段时间线、6 张前向放大图及 1 张资源汇总。全部样本保留于 artifacts，发布目录含固定第 3 次重复的 18 条原始采样快照和全部 90 组标量统计。实验容器已删除，GPU 无计算任务，原有 Proxy 保留。


继承的注入诊断字段 `target_decode_work` / `actual_min_decode_work` 仍以旧参考点 8 计算，仅为辅助记录，不参与本阶段触发或决定次数；实际触发始终由 `trigger_tokens=32` 与逐请求 `progress` 验证。
