# dcn10 本地 D/pD 部署

此文记录 dcn10 consumer 的配置和启动入口。dcn9 的 Proxy/P 与 benchmark 适配现已接续实现，当前环境核验和联调命令见 [dcn9 部署说明](DCN9_DEPLOYMENT.md)。两端真实推理、KV/cache 与 RDMA 验证仍待完成；本文原有 dcn10 环境检查结果需在启动时重新核验。

## 已知拓扑

| 项目 | dcn9 | dcn10 |
|---|---|---|
| 角色 | Proxy + P，下一阶段配置 | D/pD，本阶段提供启动入口 |
| 数据 IP | 192.168.1.9 | 192.168.1.11 |
| 接口 | ens7f0np0 | ens7f0np0 |
| RDMA 端口 | rocep202s0f0/1 | rocep202s0f0/1 |
| GPU | RTX 4090 × 1 | RTX 4090 × 1 |
| 主机内存 | 用户提供为 32 GB | 用户提供为 32 GB |

用户提供的输出确认两端接口均为 100000 Mb/s 全双工，指定接口的双向 ping 无丢包。尚未测量 TCP/RDMA 吞吐、NCCL 传输及推理性能。第二个端口不在初始方案中。

## 本地配置

配置文件：`configs/two_node/dcn10.json`。关键参数：

- 本地 HTTP：`192.168.1.11:8200`。
- 本地 KV ZMQ 控制端口：`192.168.1.11:14580`。
- 向 dcn9 的 `192.168.1.9:30001` 注册。
- 模型路径：`/workspace/models/Qwen2.5-7B-Instruct`，四个 safetensors 分片已在当前挂载目录中看到。
- API 模型名称：`Qwen2.5-7B-Instruct`，客户端请求中的 `model` 应使用该名称。
- BF16、TP=1、PP=1、block size=16、seed=42、prefix caching 开启。
- 最大上下文 2048、最大并发序列 4、每批最多 2048 tokens；历史与输出共同受上下文长度限制。
- vLLM 显存利用率 0.75，KV 接收缓存阈值 512 MiB。
- pinned 主机内存池 1 GiB，覆盖原 connector 的 32 GiB 默认值。
- 初始开启 `--enforce-eager` 降低 CUDA Graph 启动开销；性能实验中两端需采用一致设置，调整后重新采集基线。

这些是启动验证参数，不是经过实测优化的性能配置。原 connector 先分配接收 GPU tensor，再决定是否放入主机池；512 MiB 是缓存阈值，不是显存硬上限。需要保留传输临时显存和进程/模型加载所需主机内存。

## 使用方式

从项目根目录执行。优先激活实际装有依赖的 Python 环境；也可以显式指定 `PYTHON_BIN`。当前 CLI 中可用的 Python 是 `/workspace/conda-env/bin/python3.11`，但尚无 PyTorch/vLLM。

```bash
# 只预览参数，不导入 PyTorch，不分配 GPU，不创建服务或日志。
PYTHON_BIN=/workspace/conda-env/bin/python3.11 \
  bash scripts/server/start_dcn10.sh --dry-run

# 检查当前环境。此检查会导入 PyTorch/CUDA（可能短暂建立 CUDA context），
# 但不会加载模型、启动服务器或连接 dcn9。
PYTHON_BIN=/path/to/runtime/bin/python \
  bash scripts/server/start_dcn10.sh --check

# 依赖和 dcn9 侧准备好后，在前台启动 consumer。
# 使用 Ctrl-C 停止当前前台服务；脚本没有全局 pkill 或远程清理。
PYTHON_BIN=/path/to/runtime/bin/python \
  bash scripts/server/start_dcn10.sh
```

本地模型在不同位置时使用 `--model-path /actual/model/path`，或者修改 JSON。修改仅影响此部署入口，不改变论文原有的单机脚本。

启动器要求使用此工程 `vllm-source/` 的 editable 安装，防止误用不带 D-direct 补丁的其他 vLLM。它会检查本地接口/IP 可绑定、HTTP/KV 端口空闲、模型配置和分片文件存在，以及 PyTorch CUDA 和 connector 来源。检查不校验模型文件内容，也不等价于跨节点测试。

## 通信和 HTTP 代理

默认 Socket 模式显式设置：

```text
VLLM_HOST_IP=192.168.1.11
CUDA_VISIBLE_DEVICES=0
NCCL_SOCKET_IFNAME==ens7f0np0
NCCL_SOCKET_FAMILY=AF_INET
NCCL_NET=Socket
NCCL_IB_DISABLE=1
NCCL_NET_GDR_LEVEL=LOC
SIMULATED_KV_BW_GBps=0
```

上面 `NCCL_SOCKET_IFNAME==ens7f0np0` 表示变量的值是 `=ens7f0np0`，NCCL 使用接口名精确匹配。

启动器保留已有 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY`，合并已有 `no_proxy/NO_PROXY` 并加入两台数据 IP、管理 IP、节点名和回环地址；只影响启动的进程，不修改 shell 配置。预览不会输出代理 URL 或 API key。

RDMA 需要后续两端一起验证：

```bash
PYTHON_BIN=/path/to/runtime/bin/python \
  bash scripts/server/start_dcn10.sh --dry-run --transport rdma
```

RDMA 模式使用 `NCCL_NET=IB`、`NCCL_IB_DISABLE=0` 和精确的 `NCCL_IB_HCA`。仍设 `NCCL_NET_GDR_LEVEL=LOC`，先验证 host-staged RDMA，不假设 4090 支持 GPUDirect RDMA。该模式必须能看到 RDMA 设备、`/dev/infiniband`，且两端的 verbs/NCCL 网络库及网络配置兼容；失败时不会自动将 RDMA 标记为成功或隐式切换到 Socket。

容器应能直接绑定 `192.168.1.11`，通常使用 Linux host 网络；后续测试 RDMA 时还需映射对应设备、配置 pinned 内存和 shared memory 的运行条件。仅固定 ZMQ 端口不代表 NCCL 网络端口也固定。

## 验证和打包

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/conda-env/bin/python3.11 -B \
  -m unittest discover -s scripts/tests -p test_dcn10_deployment.py -v

bash scripts/package_source.sh
```

打包默认输出到工程内部的 `artifacts/`，包含完整源码和 vendored vLLM、此配置及交接文档。排除整个 `artifacts/` 目录、git 历史、模型权重、虚拟环境、日志、数据集、benchmark 结果和 `.env`，避免历史压缩包被重复打包。包不含可直接运行的已安装依赖环境。
