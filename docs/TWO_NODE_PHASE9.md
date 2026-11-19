# 双机阶段九：增大 PD KV 传输量，对比 Socket/TCP 与 RoCE

2026-11-19 UTC。目标是在相同双机模型服务和串行 PD 请求下，将输入从 256 增大到 8,192 tokens，观察 KV 传输和端到端延迟随负载的变化。本阶段不包含静态 PPD、动态策略或并发对照。

## 负载与配对

模型为两端 Qwen2.5-7B-Instruct，BF16，各一张 RTX 4090。dcn9 运行 Proxy 与 P，dcn10 运行 D；所有请求走 1P_1D，动态路由关闭。

| 输入 token | 256 | 512 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|---:|---:|---:|
| KV payload (MiB) | 14 | 28 | 56 | 112 | 224 | 448 |
| blocks (16 token/block) | 16 | 32 | 64 | 128 | 256 | 512 |

KV 每 token 为 28 层 × 4 KV heads × 128 head dimensions × K/V 两份 × BF16 两字节 = 57,344 字节。最终实验逐请求从真实发送/接收 tensor 的 numel 和 element_size 统计 payload，使用完整请求 ID 和 API 响应匹配；不把 external_cache_hit_tokens 当作字节测量。

每个长度五对重复，Socket 与 RoCE 使用完全相同 payload，共 60 个正式请求。每条固定生成 32 tokens、temperature=0、seed=42、ignore_eos=true。串行闭环，前一请求 SSE EOF 后再发下一条。

输入不是自然对话：Session 后是按长度/重复编号构造的 SHA256 nonce，主体重复空格加 data，结尾要求总结。客户端经真实 Decode tokenizer 校准到准确长度；随机 nonce 位于第一个 block 内，使不同请求不共享完整前缀 block。不同长度和重复的输入均独立；相同长度/重复的两种传输才复用同一 prompt。

五轮协议顺序依次为 Socket→RoCE、RoCE→Socket、Socket→RoCE、RoCE→Socket、Socket→RoCE。相邻同协议轮次共用模型进程，所以共六次部署；每次部署单独预热六种长度，并在预热后调用 P/D 的 reset_prefix_cache。该 API 需要临时模型容器开启 VLLM_SERVER_DEV_MODE=1，最终日志记录成功调用。每个跨度均使用所在节点的本地 monotonic clock，不跨主机相减时间戳。

同一部署内部不在每条之间重启服务；正式输入各自独立，缓存可以共存。pilot/此前部署的进程缓存随重建释放。8K pilot 与正式使用相同原始输入文本，但运行于不同进程实例，不复用其 KV；预热使用独立 nonce，与正式 prompt 不共享完整缓存前缀。

## 部署与计时

沿用镜像 ppd-qwen25:source-cu129，未创建镜像。两端 max_model_len=9,216、max_num_batched_tokens=16,384、max_num_seqs=4、block_size=16、GPU memory utilization=0.75、eager=true、prefix caching=true、KV buffer=1 GiB、host pool=2 GiB。参数对两种传输相同，保存在临时节点 JSON，原 configs/two_node JSON 不变。

数据网卡 ens7f0np0，100Gbps。Socket 端显式 transport=socket，对应 NCCL_NET=Socket；RoCE 显式 transport=rdma，对应 NCCL_NET=IB、HCA=rocep202s0f0:1、IPv4 GID3。RoCE 日志确认 NET/IB 与 GPU Direct RDMA Disabled，仍经主机内存中转。HTTP、ZMQ 与 NCCL bootstrap 使用 TCP。

关键口径：

- TTFT：客户端 POST 开始至首段非空正文，包含 Proxy、prefill、KV 传输、D 等待和启动。
- E2E：客户端 POST 至 SSE EOF，包含固定 32-token 输出。
- rx_nccl_sync_sum_ms：接收端 28 次 ncclRecv 和原有 recv_stream.synchronize 的耗时之和。会包含等待发送端，不是纯线速传输时间。
- rx_receive_span_ms：第一层 PUT 元数据到最后一层同步接收完成的跨度。含分配、控制消息、逐层生产与传输间隙；不含第一条 PUT 之前的 NCCL communicator 初始化。
- tx_nccl_sync_sum_ms、tx_send_span_ms：发送端对应累计同步调用和时间跨度，留在逐请求数据中。
- connector_load_ms：原 connector 在加载 KV 时的 recv_time。在 PUT_ASYNC 下网络接收线程可能已完成大部分传输，该值不能作为全程 KV 传输耗时。

为了记录实际同步接收和字节，最终实验在临时停止的 P/D 容器中 docker cp 一个仅增加计时代码的 Python 模块，再启动服务。不修改仓库 vLLM 源码、不修改镜像、不新增 CUDA 同步、不覆盖 /opt/ppd/vllm-source 目录；编译产物保留。副本、原始和修改后 SHA256、全部遥测定义在 evidence/telemetry/provenance.json。

Socket 记录接口 stats64；RoCE 记录 HCA port_*_data，其单位为四字节。前后差值按每轮六请求累计，包含协议开销及少量其他流量，仅作真实通信的辅助证据，不能拆成每请求的实测线速字节数。

## 过程与证据

1. 只读确认旧 P/D 已停、GPU 空闲，旧 Proxy 孤立且 HTTP 状态不响应；重建为 PD。
2. 先执行 Socket/RoCE 各五条 8K 容量 pilot，全部成功，28 层、512 blocks。
3. 完成初版 60 请求。审计发现首次 NCCL 建连混入 256-token 测量，原 connector 计时也不覆盖后台传输，因此保留初版但不用作最终图。
4. 加入同步接收遥测和独立预热。首次尝试清缓存收到 404，尚未发送最终正式请求；保留六条预热，临时容器开放开发模式 API 后重新部署。
5. 最终使用独立预热的 60 请求矩阵；不以期望网络收益筛选或删除数据。

启动过程中还修正了 evidence 目录 root 归属导致的宿主写入失败（运行挂载改用 /tmp）、health 成功空 body 的 JSON 解析问题，以及旧监听 socket 释放前的 Proxy 端口竞态（加入端口可绑定轮询）。这些均发生在对应推理批次开始之前。默认配置、全局网络、驱动、固件和 JITServe 未改动。

证据目录为 artifacts/two-node-phase9-kv-transport-evidence/，最终结果位于 calibrated/。原请求、完整 SSE、事件时间戳、失败、服务日志、metrics、计数器和部署命令均保留。详见该目录 README。

结果图位于 [results/figures/phase9](../results/figures/phase9/README.md)，包含可独立重画的完整 plot-data.json 和 source SHA256。图使用中位数、全部散点与 min/max，不擅自去除长尾。

## 最终结果与验收

最终 60/60 正式请求成功，30 对 payload 完全相同；均为目标输入长度和 32-token 输出，全部匹配到发送/接收遥测。P/D 的本地 prefix cache hits 增量在全部十个轮次中为零。每条恰为 28 层，接收与发送字节一致，16-token block 数与输入长度匹配；正式轮次日志无 NCCL WARN、CUDA OOM、KV missing/mismatch 或运行时 ERROR。

以下为每个条件 5 次测量的中位数，单位 ms。没有删除生成阶段长尾。

| 输入 tokens | 协议 | NCCL发送累计 | NCCL接收累计 | 接收时间跨度 | TTFT | E2E |
|---:|---|---:|---:|---:|---:|---:|
| 256 | Socket | 9.60 | 24.64 | 33.18 | 104.41 | 644.97 |
| 256 | RoCE | 7.35 | 12.04 | 31.29 | 102.85 | 642.81 |
| 512 | Socket | 14.35 | 34.45 | 53.11 | 124.18 | 663.94 |
| 512 | RoCE | 10.52 | 17.13 | 50.88 | 121.16 | 660.70 |
| 1,024 | Socket | 41.58 | 71.50 | 95.94 | 175.95 | 718.56 |
| 1,024 | RoCE | 40.58 | 46.97 | 95.40 | 174.45 | 714.51 |
| 2,048 | Socket | 94.21 | 133.93 | 179.48 | 268.90 | 815.85 |
| 2,048 | RoCE | 91.76 | 99.14 | 177.82 | 266.79 | 810.81 |
| 4,096 | Socket | 186.57 | 264.98 | 361.85 | 470.21 | 1,023.90 |
| 4,096 | RoCE | 182.99 | 191.61 | 359.22 | 469.73 | 1,021.85 |
| 8,192 | Socket | 208.61 | 255.19 | 761.43 | 920.47 | 1,487.06 |
| 8,192 | RoCE | 59.64 | 70.47 | 745.56 | 904.68 | 1,465.30 |

正式 payload 每轮 882 MiB、每协议五轮合计 4,410 MiB。dcn10 计数器接收增量合计：Socket stats64 4,614.23 MiB、RoCE HCA 数据计数 4,662.59 MiB。两类计数器统计层不同，不能把这两个数的差异直接解释成协议开销之差。

8K 请求实测发送/接收 tensor payload 均为 448 MiB。RoCE 的同步发送累计中位数约降低 71.4%，同步接收累计约降低 72.4%；TTFT 中位数降低约 15.8 ms（1.7%），E2E 中位数约降低 21.8 ms（1.5%）。这些比例是两个条件中位数的比较，并非逐对比例的中位数。

从 4K 到 8K，RoCE 的累计同步调用耗时出现下降，但接收跨度与 TTFT 上升。这不表示 8K 传输的数据更少，也不意味着全程传输只需 70 ms：前后 tensor 字节计数已确认翻倍；累计调用只计 NCCL 调用内部的等待与搬运，排除逐层间隙，受 CPU 调度、计算/通信重叠、传输协议选择与生产节奏影响。当前只验证了实际调用时间和完整接收跨度，尚未做 kernel/系统 profiler 来定位该非单调变化的具体成因，不应强行拟合为纯带宽曲线。

整体接收跨度主要随逐层 prefill 的生产节奏增长，两协议接近，说明这次串行 PD 下通信可与模型计算重叠。实测能够说明大 payload 下 RoCE 的 NCCL 调用阻塞时间明显少于 Socket；不能声称模型请求获得等比例加速，也不能把该结果推广为高并发或所有负载下的 RoCE 收益。每条件仅五对合成样本，图中的 min/max 是描述范围，不是置信区间。

阶段结束停止并删除本轮临时 P/D 模型容器，保留 PD Proxy。最终节点状态、镜像 ID、模块 SHA256、GPU 状态及 Proxy status 保存在 calibrated/final-*。不提交或推送 Git，不自动启动下一阶段。

## 24K 补充实验

新增精确24,576-token输入的五对Socket/RoCE串行PD测试，临时配置扩大上下文、KV容量与缓冲区，独立记录。详见 [24K补充报告](TWO_NODE_PHASE9_24K.md)。
