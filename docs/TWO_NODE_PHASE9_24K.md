# Phase 9 扩展：24K 输入的 PD KV 传输

2026-11-19 UTC。本轮将输入扩大为精确的 **24,576 tokens**，仅测试双机 1P_1D、串行请求，Socket/TCP 与 NCCL IB/RoCE 各五次。10/10 正式请求成功，五对输入 payload 完全相同，固定生成 32 tokens。另保留每次部署的一条独立预热，共六条，不计入正式结果。

## 输入与配置

沿用 Phase 9 的 Session SHA256 nonce + 重复 ` data` + 总结指令，由真实 `/tokenize` 校准完整输入长度。nonce 根据本轮独立 namespace、长度和重复编号产生，位于第一个缓存 block 内；不同重复的 prompt 独立，同一重复在 Socket/RoCE 下复用完全相同的 payload。

模型仍为 Qwen2.5-7B-Instruct、BF16，两台各一张 RTX 4090、100Gbps RoCE，P/D 均加载完整模型。使用原镜像 `ppd-qwen25:source-cu129`，没有构建新镜像或用宿主源码覆盖镜像的编译目录。

| 参数 | 原六长度实验 | 本轮 24K |
|---|---:|---:|
| max_model_len | 9,216 | 25,600 |
| max_num_batched_tokens | 16,384 | 24,576 |
| GPU memory utilization | 0.75 | 0.85 |
| KV transfer buffer | 1 GiB | 2 GiB |
| Host memory pool | 2 GiB | 2 GiB |
| max_num_seqs | 4 | 4 |
| 实际客户端并发 | 1 | 1 |
| 输出 tokens | 32 | 32 |

新输入大小超过原上下文、KV 容量和传输缓冲区，因此使用独立临时节点 JSON。默认 `configs/two_node/*.json` 保持原值。本轮启动日志报告每端 GPU KV cache 为 45,648 tokens，足够本轮单条请求。调度保留 vLLM V1 默认 chunked prefill 能力；token budget 等于完整 24K 输入，串行无缓存命中的 prompt 可在一次 prefill 中调度完成。

协议顺序沿用交替设计：S1、R1、R2、S2、S3、R3、R4、S4、S5、R5，相邻同协议共用部署，共六次部署。每次部署先发送独立 nonce 的 24K 预热请求，再清空 P/D prefix cache，随后发正式请求。正式轮次 P、D 的本地 prefix cache hits 增量全部为零。

## 实测字节与计时

每条请求发送、接收均为 **1,409,286,144 bytes = 1,344 MiB = 1.3125 GiB**，全部 28 层、1,536 个 16-token blocks。字节数来自真实 TX/RX tensor 的 `numel × element_size`，与 24,576 × 57,344 bytes/token 一致；这是 KV payload，未包含协议开销。

以下为每协议五次测量的中位数，方括号为完整最小值/最大值，不是置信区间，未删除长尾：

| 指标（ms） | Socket / TCP | RoCE（主机内存中转） |
|---|---:|---:|
| TX 同步调用累计 | 497.57 [460.69, 547.77] | 246.43 [222.71, 248.57] |
| RX 同步调用累计 | 574.21 [526.28, 727.39] | 258.69 [235.59, 260.90] |
| KV 接收跨度 | 2820.75 [2813.40, 2838.14] | 2806.12 [2782.68, 2828.83] |
| KV 发送跨度 | 2818.92 [2812.11, 2830.73] | 2806.77 [2783.23, 2829.21] |
| TTFT | 3180.56 [3165.67, 3199.78] | 3171.26 [3130.26, 3187.35] |
| E2E | 3775.36 [3761.34, 4031.71] | 3766.23 [3725.25, 3782.77] |


两种条件中位数相比，RoCE 的 TX/RX 同步调用累计分别降低 50.5% / 54.9%，接收跨度降低 0.5%，TTFT 降低 0.3%。逐请求数值见 CSV。

沿用原 Phase 9 单模块计时副本，部署到新建、尚未启动的模型容器；没有修改仓库 vLLM 原始源码或镜像，也没有增加 CUDA synchronize。完整 API 响应 ID 与 engine ID 的规范 `-0` 后缀匹配。

- **TX/RX 同步调用累计**：28 层 NCCL send/recv 及其原有 stream.synchronize 的时间之和。RX 会等待对端，不能当作纯链路传输耗时或据此计算纯带宽。
- **KV 接收跨度**：从第一层 PUT 元数据到第28层同步接收完成，包含 tensor 分配、控制消息和层间产出等待；不包含首个 PUT 前的 communicator 初始化，也不包含此前的模型计算。
- **KV 发送跨度**：从首层 send_sync 入口到最后一层同步发送完成，可能与后续层 prefill 计算重叠。
- **TTFT/E2E**：客户端 POST 到首段非空正文 / SSE EOF，包含入口、计算、通信和固定32-token输出对应部分。

RoCE 仍经主机内存中转，GPUDirect RDMA 关闭。通信与逐层模型计算重叠，因此累计调用时间改善与 TTFT 改善不应直接等同。由于本轮调整了 token budget、显存比例及缓冲区，本轮独立报告，不直接接入原六长度曲线。

## 证据与验收

原始证据位于 `artifacts/two-node-phase9-kv-transport-evidence/extension-24k/`：

- `formal-r*/raw/`：原始请求、完整 SSE、事件时间戳和解析响应，保留失败记录的能力与原阶段一致。
- `audited-requests.csv`、`measurements.json`、`verification.json`：10 条请求的 TX/RX 字节和计时、五对输入匹配、路由、缓存和网络后端审计。
- 每轮 `before/after`：P/D 日志、metrics、Proxy 状态、两端网络计数器。计数器仅作通信佐证，不解释为逐请求的精确 wire bytes。
- `plan.json`、`commands/`、`deploy_phase9_24k.py`、`run_phase9_24k.py`、`operator.log`：实验前源文件 SHA256、配置、启动命令和执行记录。
- `audit_24k.py`：只读审计脚本，使用既有 `/workspace/ppd-paper-tools/bin/python` 执行。
- `final-*`、`sync-*`：结束状态、模型容器清理与同步校验。

本轮验收：10/10 成功、五对 payload 一致，每条 TX/RX 字节一致且均28层，所有正式轮次本地 prefix hit 为零、路由仅 PD，日志无运行期 ERROR / NCCL WARN / OOM / KV missing 或 mismatch。P/D 模型容器已停止并删除，Proxy 保留，默认配置与仓库 vLLM 源码哈希保持不变。本轮结束后暂停。

适合上传 GitHub 的精简数据和说明见 [24K 结果目录](../results/figures/phase9/extension-24k/README.md)。

结果已绘为 [六面板对照图](../results/figures/phase9/extension-24k/kv-transport-24k.png)，同时提供 SVG/PDF 和独立重画脚本。图中保留全部五对样本、中位数及最小/最大值，纵轴从零开始。
