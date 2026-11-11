# dcn9 Proxy/P 与双节点联调

本阶段增加 dcn9 本地启动器、`1P_1D` / `1P_1pD` 路由与 benchmark 客户端适配。dcn10 继续使用 [consumer 启动器](DCN10_DEPLOYMENT.md)。当前尚未完成真实推理、KV 传输、prefix cache 命中或 RDMA 验证。

## 2026-11-10 实际环境核验

以下结果来自当前 dcn9 CLI 容器，不代表宿主机其他容器或 Python 环境：

| 核验项 | 结果 |
|---|---|
| hostname / 系统 | dcn9；Debian 12 容器，共享宿主机 Ubuntu 5.15 内核 |
| GPU | RTX 4090，24564 MiB，驱动 580.95.05；核验时占用 15 MiB |
| 主机 RAM | 30 GiB 总量，核验时约 22 GiB 可用 |
| 数据 / 管理接口地址 | ens7f0np0 = 192.168.1.9；eno8303 = 10.10.60.25 |
| 网口 / RDMA sysfs | 100000 Mb/s，link up；rocep202s0f0/1 ACTIVE；不是吞吐测试 |
| 本地监听端口 | 8100、14579、10001、30001 均可绑定 |
| dcn10 TCP | 192.168.1.11:22 可连接；8200 拒绝连接；未做 SSH 认证或远端进程管理 |
| 模型 | 当前挂载仅有 `/workspace/models/Qwen3-8B`；缺少目标 Qwen2.5-7B-Instruct |
| 推理运行环境 | 初始没有 Python；安装验证工具后仍无 torch/vLLM/CUDA 编译工具链 |
| 容器资源 | `/dev/shm` 64 MiB，memlock 64 KiB；没有 `/dev/infiniband` 和 Docker socket |

为代码验证，在当前 CLI 容器安装了 Python 3.11、venv、ripgrep，以及发行版的 aiohttp、Quart、NumPy、ZMQ 和 msgpack。没有安装推理依赖、下载模型、修改宿主机配置或启动推理服务。通过 pip 下载隔离验证环境依赖的尝试未完成，已终止；`/workspace/dcn9-tools` 是未完成的验证 venv，不能用于推理。APT/PIP 的诊断日志在 `/tmp/dcn9-*.log`，没有变更全局代理设置。

两端实际推理环境仍需确定。必须使用同一源码和兼容的 CUDA/NCCL 环境；预检要求 torch **2.9.1** 和本工程 `vllm-source/` editable 安装，检查 D-direct 补丁。不要安装 PyPI vLLM 来代替该源码。32 GB RAM 主机全量编译需另行评估；可将 `MAX_JOBS=1`、`NVCC_THREADS=1` 用于限制并发，但这不保证能够编译。需要从宿主机准备合适的运行容器时，先选择镜像、模型挂载、host 网络、共享内存和 memlock 配置；RDMA 再加入对应设备映射及 verbs/NCCL 网络支持。

## 本地配置和启动

`configs/two_node/dcn9.json` 与 dcn10 使用相同的 Qwen2.5-7B-Instruct、BF16、seed 42、TP=PP=1、block size 16、prefix caching、eager、2048 上下文/批 tokens、4 seqs、0.75 显存利用率。P 的 pinned 内存池为 1 GiB，KV 缓冲阈值 512 MiB。配置中的模型路径可以不同，但两端权重/tokenizer 必须一致，API 名称固定为 `Qwen2.5-7B-Instruct`。

所有命令从项目根目录执行；`PYTHON_BIN` 应指向实际安装本地 vLLM 源码的环境。

```bash
# 仅预览；不需要 torch，不创建服务。
PYTHON_BIN=/usr/bin/python3 bash scripts/server/start_dcn9.sh --role proxy --dry-run
PYTHON_BIN=/usr/bin/python3 bash scripts/server/start_dcn9.sh --role prefill --dry-run

# 当前 CLI 的 Proxy 检查通过；P 检查会因模型和 torch 缺失失败。
PYTHON_BIN=/usr/bin/python3 bash scripts/server/start_dcn9.sh --role proxy --check
PYTHON_BIN=/usr/bin/python3 bash scripts/server/start_dcn9.sh --role prefill --check

# 实际运行环境和模型就绪后，先分别 --check，再在两个终端前台启动。
PYTHON_BIN=/path/to/runtime/bin/python bash scripts/server/start_dcn9.sh \
  --role proxy --routing 1P_1D
PYTHON_BIN=/path/to/runtime/bin/python bash scripts/server/start_dcn9.sh \
  --role prefill --model-path /actual/Qwen2.5-7B-Instruct --transport socket
```

Proxy 绑定 `192.168.1.9:10001`（HTTP）与 `:30001`（ZMQ）。P 绑定 `:8100`（HTTP）与 `:14579`（KV 控制）。两个入口都是前台单进程，Ctrl-C 停止对应进程；没有全局清理、`pkill`、自动后台化或 SSH。

默认强制 Socket、`NCCL_SOCKET_IFNAME='=ens7f0np0'`、IPv4、GDR 禁用、网络延迟模拟关闭。保留已有 HTTP 代理并合并内部地址 bypass。两端都可以显式 `--transport rdma`，但当前容器 P 的 RDMA 预检失败，不能据此报告 RDMA 已可用。

## 与 dcn10 协调启动顺序

1. 用户先确定两端 Python/容器环境，并补充 dcn9 的 Qwen2.5 模型路径。两端核对相同模型/tokenizer；文件存在检查不等于内容校验，可对配置、tokenizer、分片索引及权重逐一比较 SHA-256。
2. 传递本阶段更新后的完整源码，使两端使用相同版本。在两端分别执行本地 `--check`。此时不要将预检通过视为推理通过。
3. dcn9 启动 Proxy `--routing 1P_1D` 和 P；再由用户在 dcn10 前台运行：

   ```bash
   PYTHON_BIN=/path/to/runtime/bin/python bash scripts/server/start_dcn10.sh \
     --model-path /actual/Qwen2.5-7B-Instruct --transport socket
   ```

4. 查看 `http://192.168.1.9:10001/status`，应为 `prefill=[192.168.1.9:8100]`、`decode=[192.168.1.11:8200]`。两端注册不等于完成了 NCCL 初始化。
5. 预热后发送两轮完整历史。下面脚本会先预热，再检查响应和路由计数，输出保留原始响应与前后状态；有错误时退出非零。执行时不要同时运行其他客户端。

   ```bash
   /path/to/runtime/bin/python scripts/tests/two_node_smoke.py --config 1P_1D \
     --output results/smoke/socket-pd.json
   ```

6. 用 Ctrl-C 停止并重启 **Proxy**，改为 `--routing 1P_1pD`；同一 consumer 会重新注册为 pD，无需换启动器。确认 `ppd_decode=[192.168.1.11:8200]`，再运行 smoke `--config 1P_1pD`。PD 两轮都经 P；静态 PPD 第一轮经 P，第二轮到同一个 pD。客户端保留完整历史。
7. 路由计数只能证明 Proxy 选择。进一步核对 P 实际请求数、connector 发送/接收日志或字节计数、consumer `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` 等指标，证明 KV 传输及真实 prefix 命中。无命中时排查缓存驱逐与重新 prefill。smoke 报告始终将未核验的 KV/cache/RDMA 项保留为 unverified。
8. Socket 正确性通过后才采集性能基线；RDMA 单独做 verbs、TCP/RDMA 有效 payload 吞吐及 NCCL `NET/IB` 日志验证，不假定 4090 支持 GDR，不按双端口 200G 汇报。

## Benchmark 适配

comprehensive 和 ShareGPT 都支持 `--proxy-url`、`--api-model-name`（别名 `--model-name`）、`--tokenizer-path`、`--max-context-tokens`。API 名与权重/tokenizer 路径分别配置。

两节点配置必须显式选择一个 `--config` 并使用 `--skip-startup`。该模式在预热失败、测试点失败和工作负载之间的健康检查失败时退出，所有启动/停止/清理/重启入口也禁止进程管理。它不会调用四卡脚本。ShareGPT 的 `--enable-ppd` 只是客户端实验参数，不能改变已经启动的外部 Proxy。

```bash
/path/to/runtime/bin/python scripts/benchmark/comprehensive_benchmark.py \
  --config 1P_1D --skip-startup --proxy-url http://192.168.1.9:10001 \
  --api-model-name Qwen2.5-7B-Instruct \
  --tokenizer-path /actual/Qwen2.5-7B-Instruct --max-context-tokens 2048 \
  --workload small_tiny --qps 0.5 --output-dir results/two_node/socket-pd

/path/to/runtime/bin/python scripts/benchmark/sharegpt_benchmark.py \
  --config 1P_1pD --skip-startup --proxy-url http://192.168.1.9:10001 \
  --api-model-name Qwen2.5-7B-Instruct \
  --tokenizer-path /actual/Qwen2.5-7B-Instruct --max-context-tokens 2048 \
  --sharegpt-path /actual/sharegpt.json --num-conversations 4 --qps 0.5 \
  --output-dir results/two_node/socket-static-ppd-sharegpt
```

先手动将 Proxy 切换到对应配置。客户端检查 `/status` 的 config、两端实际注册地址和路由角色；静态双卡基线要求动态 PPD 关闭。注册会过期，死实例不会永久留在状态或路由选择中。HTTP 200 的错误体、没有生成 token/usage 的流都不会算成功。

两节点上下文默认限制为 2048，要求本地 tokenizer 可读取，对发送的完整 prompt 加输出预算逐次核验。ShareGPT 的数据筛选也计入输出与格式预算；实际累计历史仍以请求前的精确核验为准。超限请求记为失败，不静默截断历史。synthetic benchmark 的输入 token 标签仍是近似目标，不保证精确等长；结果不能据此当作严格 token 配对实验。使用新输出目录区分模型、transport 与本次 prompt 格式变更，避免旧 checkpoint 混用。失败/未运行的综合测试点不记为成功完成。

默认 `all` 仍维持原单机配置集合，双节点配置可用 `--list-configs` 查看后显式选择。未增加 2R：跨节点 replica 地址和独立启动入口仍需另行实现。

动态决策表仍缺少本硬件与网络的 `1P_1D` / `1P_1pD` 成对基线及 config pair 适配，当前不宣称动态自适应已经完成或验证。512-token bypass 阈值也没有经过实测。

## 已执行的代码验证

```bash
python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v
bash -n scripts/server/start_dcn9.sh scripts/server/start_dcn10.sh scripts/package_source.sh
```

当前 20 项测试全部通过；Python 语法、shell 语法及启动参数预览通过。新增测试使用模拟响应/本地 HTTP 假服务检查外部服务模式、两轮路由和 smoke 报告；没有加载 GPU 模型，不是推理或 RDMA 测试。
