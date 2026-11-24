# Phase 9：PD 的 Socket/TCP 与 RoCE 对照

本阶段仅使用双机 PD，没有 PPD 对照或并发。输入长度为 256、512、1,024、2,048、4,096、8,192 tokens；输出固定 32 tokens。每个长度有 5 对相同输入，共 60 个正式请求。

使用既有 Qwen2.5-7B-Instruct BF16、2×RTX 4090、100Gbps 网络。RoCE 经主机内存中转，GPUDirect RDMA 关闭。每次部署后单独预热六种长度，再清空 P/D prefix cache；所有正式请求有独立 nonce，配对 Socket/RoCE payload 完全相同。五轮交替协议先后顺序，连续同协议轮次共用部署。

## 结果图

- [kv-transport.png](kv-transport.png)：A 为从发送/接收 tensor 的形状与 dtype 统计的真实 KV payload，14 至 448 MiB/请求；B 为 28 层同步 NCCL 接收调用耗时之和；C 为同步 NCCL 发送调用耗时之和；D 为第一层 PUT 元数据到最后一层同步接收完成的时间跨度。
- [request-latency.png](request-latency.png)：客户端 TTFT 和 E2E，TTFT 从 POST 到首段非空正文，E2E 到 SSE EOF。

数据线为每种条件 5 次测量的中位数，误差线为最小/最大值，淡色点为全部原始样本。没有删除长尾或失败请求。PNG 方便 GitHub 预览，SVG/PDF 适合缩放与导出。

NCCL 接收调用会等待发送端，所以 B 不能解释成纯链路延迟；C 额外包含分配、控制消息、层间生产间隔等，也不能用来计算纯网络带宽。原有 connector 的 recv_time 在 PUT_ASYNC 下可能只记录取用已到达 KV 的耗时，未用于主传输图。接口计数器包含协议开销，且每轮六请求合计采样，不能拆成每请求实测线速字节数。

## 数据与重画

最终原始证据：artifacts/two-node-phase9-kv-transport-evidence/calibrated/。初版 60 请求和 10 条容量 pilot 另留在上一级，不参与这两张图。

此目录保留可独立上传 GitHub 的 plot-data.json、generate.py 和依赖文件。JSON 包含全部 60 请求、完整匹配 ID、传输字节、10 个轮次的网卡计数差值、校验结果和证据文件 SHA256。

    /workspace/ppd-paper-tools/bin/python results/figures/phase9/generate.py
    # 仅使用仓库中的快照，不需要原始 artifacts 或模型服务
    /workspace/ppd-paper-tools/bin/python results/figures/phase9/generate.py --from-snapshot

绘图使用既有 /workspace/ppd-paper-tools，不改模型环境。实验方法、最终数值和限制详见 [阶段报告](../../../docs/TWO_NODE_PHASE9.md)。

## 主要观测

60/60 正式请求成功，30 对输入完全匹配，全部收到 28 层 KV，本地 prefix cache hit 增量为零。

8K/448 MiB 时，Socket/RoCE 的同步发送累计中位数为 208.61/59.64 ms，同步接收累计为 255.19/70.47 ms；TTFT 为 920.47/904.68 ms，E2E 为 1487.06/1465.30 ms。传输调用耗时改善明显，整条请求收益约 1%–2%。

RoCE 在 8K 的累计调用耗时低于 4K，但接收时间跨度和 TTFT 继续增长。累计计时不含层间间隔，不能解读为网络已在 70 ms 内传完全部请求；其非单调成因尚未用 profiler 分解。保留该实测趋势，不能把它拟合为纯网络带宽曲线。

## 24K 补充实验

[24K结果与逐请求数据](extension-24k/README.md)：精确24,576-token输入，每条实测1,344MiB KV。配置扩大后单独报告，原六长度图保持原数据。

## 32K热缓存补充实验

[32K热P/冷D结果](extension-hot32k/README.md)：固定1token输出、每次真实传输1.75GiB KV，Socket/RoCE各10次。P仅重算末尾16tokens，D不复用本地前缀。端到端中位数877.37/472.36ms，接收跨度581.17/176.44ms；分别降低46.2%/69.6%。该时延仍含提取、装入、少量计算和软件开销。
