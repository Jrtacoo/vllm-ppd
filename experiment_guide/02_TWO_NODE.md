# 双机静态 PD / PPD：Socket 与 RoCE

本页可单独使用。所有命令在指定服务器的 **Docker 宿主机**执行；项目目录为 `/home/dell/xwd/vllm-ppd`，本目录为其 `experiment_guide/`。

## 拓扑和前提

| 服务器 | 容器 | HTTP | 其他端口 |
|---|---|---|---|
| dcn9：192.168.1.9 | ppd-dcn9-proxy | 10001 | ZMQ 30001：注册与控制 |
| dcn9：192.168.1.9 | ppd-dcn9-prefill | 8100 | KV 控制 14579 |
| dcn10：192.168.1.11 | ppd-dcn10-decode | 8200 | KV 控制 14580 |

两端已有镜像 `ppd-qwen25:source-cu129`，本地完整权重位于 `/home/dell/xwd/models/Qwen2.5-7B-Instruct`。P/D 都加载完整模型，TP=PP=1；这不是跨机模型权重切分。

两个维度独立选择：

- `ROUTING=1P_1D`：静态 PD，两轮都经 P→D。
- `ROUTING=1P_1pD`：静态 PPD，首轮 P→D，后续轮次直接到同一 pD。
- `TRANSPORT=socket`：NCCL Socket 传输 KV。
- `TRANSPORT=rdma`：NCCL IB 后端通过 Ethernet/RoCE 传输 KV，当前环境使用主机内存中转，未启用 GPUDirect RDMA。

这里没有开启动态决策或校准决策表。D/pD 是相同 consumer 程序，由 Proxy 决定路径。

## 1. 检查，避免 GPU 冲突

两端分别执行：

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}'
docker image inspect ppd-qwen25:source-cu129 --format '{{.Id}}'
nvidia-smi
ip -br addr show ens7f0np0
```

如果 dcn10 有运行中的 `ppd-manual-single`，先 `docker stop ppd-manual-single`。先记录旧服务状态。若只是恢复原有配置，可 `docker start` 现有容器；只有改变配置时才需要下面的重新创建流程。

## 2. 切换配置时，先保存旧日志并移除旧容器

先停止所有客户端请求。下列命令会删除指定的旧服务容器；保留镜像、模型及挂载结果。旧容器不存在的命令会报错，可跳过相应行。

**dcn10：**

```bash
mkdir -p /home/dell/xwd/vllm-ppd/artifacts/manual-old-dcn10
docker inspect ppd-dcn10-decode > /home/dell/xwd/vllm-ppd/artifacts/manual-old-dcn10/container.json
docker stop ppd-dcn10-decode
docker logs ppd-dcn10-decode > /home/dell/xwd/vllm-ppd/artifacts/manual-old-dcn10/decode.log 2>&1
docker rm ppd-dcn10-decode
```

**dcn9：**

```bash
mkdir -p /home/dell/xwd/vllm-ppd/artifacts/manual-old-dcn9
docker inspect ppd-dcn9-proxy ppd-dcn9-prefill > /home/dell/xwd/vllm-ppd/artifacts/manual-old-dcn9/containers.json
docker stop ppd-dcn9-prefill ppd-dcn9-proxy
docker logs ppd-dcn9-prefill > /home/dell/xwd/vllm-ppd/artifacts/manual-old-dcn9/prefill.log 2>&1
docker logs ppd-dcn9-proxy > /home/dell/xwd/vllm-ppd/artifacts/manual-old-dcn9/proxy.log 2>&1
docker rm ppd-dcn9-prefill ppd-dcn9-proxy
```

每次实验换一个日志目录，不覆盖先前记录。原配置若要恢复，应保留本次保存的 `docker inspect`，特别注意其传输模式。

## 3A. 启动 Socket + 静态 PD

**dcn9：**

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
TRANSPORT=socket ROUTING=1P_1D bash start_proxy.sh --run
TRANSPORT=socket ROUTING=1P_1D bash start_prefill.sh --run
```

**dcn10：**

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
TRANSPORT=socket bash start_decode.sh --run
```

去掉 `--run` 可仅预览完整 Docker 命令。

## 3B. 或者启动 RoCE + 静态 PPD

这是 3A 的替代方案；已有同名容器时先按步骤2保存并移除，不要两套同时启动。

两端先检查 `rdma link show`、`ls /dev/infiniband`。已验证的网络为 `ens7f0np0`、HCA `rocep202s0f0:1`、IPv4 RoCEv2 GID3。

**dcn9：**

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
TRANSPORT=rdma ROUTING=1P_1pD bash start_proxy.sh --run
TRANSPORT=rdma ROUTING=1P_1pD bash start_prefill.sh --run
```

**dcn10：**

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
TRANSPORT=rdma bash start_decode.sh --run
```

脚本为 P/D 映射 `/dev/infiniband` 并设置 GID3；镜像内启动器设置 NCCL 的网卡、HCA、传输后端与 GDR 禁用。端口主要用于 HTTP、ZMQ 和连接协调；实际 RoCE 数据通道不能简单等同于 KV 控制端口。

要做 RoCE+PD，将上述 dcn9 两行的 `ROUTING` 改为 `1P_1D`。要做 Socket+PPD，将两端 `TRANSPORT` 改为 `socket`。每次切换都明确指定，避免依赖默认值。

## 4. 等待健康并验证注册

从任一能访问数据网的终端执行：

```bash
curl --noproxy '*' -f http://192.168.1.9:8100/health
curl --noproxy '*' -f http://192.168.1.11:8200/health
curl --noproxy '*' -f http://192.168.1.9:10001/status
```

状态中应包含 P=`192.168.1.9:8100`，D 或 pD=`192.168.1.11:8200`，`config` 为所选路由，`ppd_mode_enabled=false`。注册成功还不证明 KV 实际传输成功；必须发送推理请求。

```bash
curl --noproxy '*' -N --fail-with-body http://192.168.1.9:10001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-7B-Instruct","prompt":"Explain prefix caching.","max_tokens":64,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

## 5. 看日志、停止和再次启动

```bash
# dcn9
docker logs -f ppd-dcn9-proxy
docker logs -f ppd-dcn9-prefill
# dcn10
docker logs -f ppd-dcn10-decode
```

在完成真实请求后查看 P/D 的 NCCL `NET/IB`、HCA、GID 和收发日志，验证 RoCE；仅 `/health` 200 不够。Socket 模式查看 `NET/Socket`。HTTP/ZMQ 继续走 TCP 是正常的。

停止时先停止客户端，再在 dcn10 执行 `docker stop ppd-dcn10-decode`，在 dcn9 执行 `docker stop ppd-dcn9-prefill ppd-dcn9-proxy`。

保持同样参数再次运行：先在 dcn9 `docker start ppd-dcn9-proxy ppd-dcn9-prefill`，再在 dcn10 `docker start ppd-dcn10-decode`。`docker start` 保留创建时参数，不读取新的宿主机 shell 环境；模型配置修改也建议重新创建两端容器。
