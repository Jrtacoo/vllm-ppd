# 双机阶段七：RoCE 模型 KV 推理与逐请求性能验证

2026-11-11 UTC，本阶段完成并暂停。真实 vLLM P2pNcclConnector 已切到 NCCL IB/RoCE v2，通过 PD 和静态 PPD 两轮验证，并完成 192 条正式性能请求，全部成功。每条请求的 TTFT、TPOT、usage、原始请求和 SSE 已保存。`stage-verification.json` 汇总验收通过。

## 部署和实际通信路径

- dcn9 运行 Proxy/P，dcn10 运行 D/pD；最终 Proxy 为静态 `1P_1pD`，动态路由关闭，三个服务保持运行。
- 沿用同一 `ppd-qwen25:source-cu129` 镜像、Qwen2.5-7B-Instruct 权重、BF16、eager、上下文 2048、block 16、max seqs 4、GPU utilization 0.75、512 MiB KV buffer 和 1 GiB host pool。没有生成新镜像或修改模型推理源码。
- 重建 P/D 容器，加入 `/dev/infiniband` 映射、`--transport rdma`、`NCCL_IB_GID_INDEX=3`、`NCCL_IB_ADDR_FAMILY=AF_INET`、`NCCL_IB_ROCE_VERSION_NUM=2`。既有启动器设置 `NCCL_NET=IB`、`NCCL_IB_DISABLE=0`、HCA `=rocep202s0f0:1`、接口 `=ens7f0np0`、`NCCL_NET_GDR_LEVEL=LOC`。
- 两端日志均确认 `Using network IB`、`via NET/IB/`、所选 HCA/RoCE/GID 和 `GPU Direct RDMA Disabled`，没有 Socket 数据 channel；bootstrap、HTTP 和 ZMQ 仍使用 TCP。实际路径为 GPU → 主机内存 → RoCE → 主机内存 → GPU。
- 网卡 IP、MTU、固件、驱动和内核模块保持原样，未加载 `nvidia_peermem`。选定 RDMA 错误计数器前后增量均为零。
- `configs/two_node/*.json` 仍保留 Socket 默认值；本阶段的实际容器命令显式覆盖为 RDMA。`docker restart` 会保留当前 RDMA 命令，重新创建时应使用本阶段 `*-command.json` 或等效参数。宿主机源码没有覆盖镜像中的已编译 vLLM。

## 功能验收

先分别运行一个两轮对话和独立两轮预热，smoke 请求与正式性能请求分开保存，全部请求使用流式客户端保存 TTFT/TPOT。
PD 的两轮均经过 P，D 每次收到全部 28 层 KV。静态 PPD 的第一轮经过 P/跨机 KV，第二轮走 D-direct，P 请求数不增加；D 本地 prefix cache 命中增加。后端计数、路由 ID、输出长度和 usage 均与客户端记录一致。

正式 PD 日志含 104 次 KV 接收（96 正式 + 8 预热）；静态 PPD 含 52 次 KV 接收和 52 次 direct（各 48 正式 + 4 预热）。全部 KV 接收均为 28 层，没有运行时 ERROR、Traceback、CUDA OOM、NCCL WARN 或 KV missing/mismatch。smoke 两种模式分别另有 4/2 次 KV 接收，静态 PPD smoke 另有 2 次 direct。

## 正式性能结果

沿用阶段五原样 `workload.json` 和同一流式客户端：short 第一轮 256 输入 token/每轮输出 64，long 第一轮 768 输入 token/每轮输出 128；并发对话 1/4，每单元 12 组两轮对话。每种模式开始前重建 P/D 清除进程内缓存，每单元另做一组两轮预热。正式请求 192，smoke 请求 4，预热请求 20，共 216 次模型请求。

下表为本次 RoCE 运行的中位数，单位 ms，每个负载/并发/轮次有 12 个正式样本。

| 负载 | 并发 | 模式 | T1 TTFT | T2 TTFT | T1 TPOT | T2 TPOT |
|---|---:|---|---:|---:|---:|---:|
| short | 1 | 1P_1D | 107.26 | 111.03 | 17.41 | 17.41 |
| short | 1 | 1P_1pD | 96.28 | 44.24 | 17.57 | 17.60 |
| short | 4 | 1P_1D | 116.71 | 106.06 | 18.20 | 18.21 |
| short | 4 | 1P_1pD | 118.98 | 42.52 | 18.21 | 18.22 |
| long | 1 | 1P_1D | 158.40 | 139.19 | 17.64 | 17.80 |
| long | 1 | 1P_1pD | 156.27 | 69.37 | 17.54 | 17.59 |
| long | 4 | 1P_1D | 164.16 | 143.51 | 18.42 | 18.52 |
| long | 4 | 1P_1pD | 192.86 | 70.43 | 18.52 | 18.78 |

## 与阶段五 Socket 的比较

跨阶段共有 192 对请求，其中 183 对 payload 完全相同；96 对第一轮输入全部相同。第二轮包括各自实际生成的第一轮回答，因此不默认全部第二轮输入相同。完整全量与相同输入子集见 `socket-rdma-comparison.csv/json`。

下表只比较第一轮，全部输入相同，单位 ms；Socket 来自阶段五历史单次运行，RoCE 来自本阶段。

| 负载 | 并发 | 模式 | Socket TTFT | RoCE TTFT | Socket TPOT | RoCE TPOT |
|---|---:|---|---:|---:|---:|---:|
| short | 1 | 1P_1D | 108.14 | 107.26 | 17.40 | 17.41 |
| short | 4 | 1P_1D | 112.88 | 116.71 | 18.13 | 18.20 |
| long | 1 | 1P_1D | 157.17 | 158.40 | 17.38 | 17.64 |
| long | 4 | 1P_1D | 162.25 | 164.16 | 18.21 | 18.42 |
| short | 1 | 1P_1pD | 102.19 | 96.28 | 17.39 | 17.57 |
| short | 4 | 1P_1pD | 121.07 | 118.98 | 18.06 | 18.21 |
| long | 1 | 1P_1pD | 154.00 | 156.27 | 17.33 | 17.54 |
| long | 4 | 1P_1pD | 195.24 | 192.86 | 18.47 | 18.52 |

本阶段内 PD 与静态 PPD 的第二轮相同输入子集如下；其余请求保留在全量结果中。

| 负载 | 并发 | 相同输入对数 | PD TTFT | PPD TTFT | PD TPOT | PPD TPOT |
|---|---:|---:|---:|---:|---:|---:|
| short | 1 | 12 | 111.03 | 44.24 | 17.41 | 17.60 |
| short | 4 | 11 | 101.62 | 40.98 | 18.21 | 18.23 |
| long | 1 | 12 | 139.19 | 69.37 | 17.80 | 17.59 |
| long | 4 | 7 | 144.27 | 69.34 | 18.52 | 18.78 |

本次结果没有显示 RoCE 对 Socket 的一致性请求延迟改善：第一轮 TTFT 有升有降，TPOT 大体接近。静态 PPD 的第二轮 TTFT 在相同输入子集中仍低于 PD。以上为描述性观测，不能将不同时间的单次差异直接归因于传输协议。

TTFT 从客户端 POST 开始到首段非空正文，含 Proxy/P/KV/D/网络；TPOT=(SSE EOF−首段正文)/(completion_tokens−1)。同时保存 `last_content_ms`、`done_ms`、`content_tpot_ms`。它们与阶段六独立 NCCL 探针的往返耗时口径不同。

这是先 PD 后 PPD 的单次小样本合成负载，Socket 与 RoCE 也在不同时间运行，没有随机交错、长时间重复或统计显著性检验。不要将网络微测试带宽直接换算成模型 TTFT 加速，也不要据此宣称所有工作负载的 RDMA 收益或论文性能已复现。动态 PPD、自然对话质量、缓存驱逐、高 QPS 和持续负载稳定性仍未验证。

## 证据与过程

两端统一位置：`/home/dell/xwd/vllm-ppd/artifacts/two-node-phase7-evidence/`。

- `all-requests.csv/jsonl`：192 条正式请求；`1P_1D/`、`1P_1pD/` 含各自汇总和原始 SSE。
- `smoke-*/`：单独的功能验证及其逐请求 TTFT/TPOT。
- `stage-verification.json`、`verification.json`、`runtime-verification.json`、`host-verification.json`：总验收、客户端/后端、真实模型网络路径、主机错误计数器。
- `socket-rdma-comparison.csv/json`、`transport-paired-requests.json`：历史 Socket 配对比较；`matched-second-turn-comparison.json`：本阶段 PD/PPD 相同第二轮输入子集。
- 实际 Docker argv、启动日志、服务日志、容器状态、测试前后 inventory、客户端命令和实际运维/汇总程序均保留。

原始 SSE、模型输出、API usage、服务日志和 metrics 来自真实运行；CSV、配对表及验收 JSON 是本次脚本计算生成。

初次创建 dcn10 RDMA 容器前，结果子目录创建因 CLI root 的目录归属失败。修正 evidence 目录归属后，按同一已保存 argv 启动 D；没有重试推理请求。首次日志核查还因接收端 GID 日志格式不同而误判，检查器现接受其明确的 `NCCL_IB_GID_INDEX set by environment to 3.`，原始日志未修改。详见 `operations-notes.md`。

期间 CLI 流程发生中断。恢复时两端宿主机 uptime 均超过 7 天，未发现宿主机重启；PD 的 96 条请求及验收已经完成，PPD 容器已经创建但尚未发出正式请求。保留原 PD 证据，等待现有 PPD 模型加载后采集其 96 条请求，没有覆盖已完成测试或重试推理。恢复脚本及输出为 `resume-stage.py`、`resume-operator.log`。

本阶段完成后暂停，未启动下一阶段。
