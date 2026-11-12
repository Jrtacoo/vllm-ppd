# 重跑双机两轮静态基线

本页可单独使用。前提：两端已运行 Proxy/P/D，数据网可访问，模型为 Qwen2.5-7B-Instruct，服务上下文至少2048。不会自动管理服务。

本页调用原有 `scripts/benchmark/two_node_latency.py`，用于阶段5/7式的 PD 与静态PPD对照。它会检查 Proxy 路由和统计，**不能用于独立单卡服务或动态模式**。

全部客户端命令在 **dcn10 Docker宿主机**执行。客户端容器复用已有镜像，不需要GPU。目录与原项目脚本分别挂载，不覆盖镜像的编译目录。

## 1. 准备共享输入清单

确保 D 的 `/tokenize` 可用。选择新目录；多个模式共用同一份 `workload.json`：

```bash
BASE=/home/dell/xwd/vllm-ppd/artifacts/manual-baseline-001
mkdir -p "$BASE"
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$BASE,target=/out" \
  ppd-qwen25:source-cu129 /tools/two_node_latency.py \
  --prepare --manifest /out/workload.json --conversations 12
```

第一轮短/长输入精确为256/768 token，输出分别64/128 token；每种长度测试并发1、4，每组12个两轮会话，因此每种模式96个正式请求。每组另有一段两轮预热，不计入正式 CSV，但保留其 raw。

## 2. 先运行静态 PD

先依照本目录 `02_TWO_NODE.md`，将服务配置为目标传输模式和 `1P_1D`，并确认 P/D 健康、Proxy 注册正确。

以下以 **RDMA** 为例命名结果目录；文件夹名称不会自动切换服务的通信方式。

```bash
BASE=/home/dell/xwd/vllm-ppd/artifacts/manual-baseline-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$BASE,target=/out" \
  ppd-qwen25:source-cu129 /tools/two_node_latency.py \
  --manifest /out/workload.json --routing 1P_1D --output /out/rdma-1P_1D
```

## 3. 再运行静态 PPD

先停止客户端；按 `02_TWO_NODE.md` 保存服务日志并重新创建三个服务，设置相同传输模式、`ROUTING=1P_1pD`，使模型缓存和 Proxy 状态重新开始。等待全部健康和注册后执行：

```bash
BASE=/home/dell/xwd/vllm-ppd/artifacts/manual-baseline-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$BASE,target=/out" \
  ppd-qwen25:source-cu129 /tools/two_node_latency.py \
  --manifest /out/workload.json --routing 1P_1pD --output /out/rdma-1P_1pD
```

`--routing` 只声明并核对预期模式，不改变 Proxy。如果服务模式不符或动态开关打开，脚本会失败。不要修改参数绕过检查。

## 4. 比较 Socket 和 RoCE

按相同方法分别以 `TRANSPORT=socket` 和 `TRANSPORT=rdma` 创建两端服务；重复使用同一 `workload.json`，输出分别用 `socket-1P_1D`、`socket-1P_1pD`、`rdma-1P_1D`、`rdma-1P_1pD` 等新子目录。

每次保留两端 `docker inspect`、`docker logs`。RDMA须在真实请求后看到正确的 NCCL NET/IB/HCA/GID 和实际 KV 收发，不能只根据结果目录的名字判断。

## 5. 读取结果

每个模式目录内：

- `requests.csv/jsonl`：96条正式请求的逐请求计时；预热不在其中。
- `summary.json`：短/长输入、并发1/4、第一/第二轮分别汇总。
- `short-c1-checks.json` 等：实际路由计数、期望值和失败记录，所有 checks 应为 true。
- `*-before/after-status.json`、`*-before/after-*-metrics.txt`：Proxy状态和P/D指标快照。
- `raw/`：完整 payload、SSE、带时间事件、正文及 usage，包括独立命名的预热请求。

默认 API 地址：Proxy=`192.168.1.9:10001`、P=`192.168.1.9:8100`、D=`192.168.1.11:8200`。如需更改，向客户端显式传 `--proxy`、`--prefill`、`--decode`。

相同 manifest 不保证第二轮 payload 一致，因为第二轮拼接第一轮实际生成内容。严格比较须按 `payload_sha256` 匹配，并报告匹配数量。PD和PPD都能命中D本地缓存；不能把对照解释为“无缓存 vs 有缓存”。外部缓存命中 token 也不等于网络上真实传输的 token/字节数。

该实验为闭环并发，不是论文的开放式 QPS 曲线，也没有动态决策。每种模式一次96请求只适合作为基线；正式性能结论还需多轮重复及对应负载设计。
