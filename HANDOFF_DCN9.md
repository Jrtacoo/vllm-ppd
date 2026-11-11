# dcn9 Codex CLI 接续说明

最新状态见 [双机阶段五](docs/TWO_NODE_PHASE5.md)：Socket PD/静态 PPD 小样本基线完成，192 条正式请求全部成功，每请求 TTFT/TPOT、token 数和原始 SSE 保存在 `artifacts/two-node-phase5-evidence/`，入口为 `all-requests.csv`。39 对第二轮输入完全一致的子集另有比较；9 对第二轮历史不同，已明确标记。28 项本地测试通过，两端镜像一致，最终服务健康。三个服务保持运行，Proxy 当前为静态 `1P_1pD`，本阶段结束后暂停。此前缓存验证见 [阶段四](docs/TWO_NODE_PHASE4.md)，跨机 KV 验证见 [阶段三](docs/TWO_NODE_PHASE3.md)，运行环境见 [阶段二](docs/TWO_NODE_PHASE2.md)，准备记录见 [阶段一](docs/TWO_NODE_PHASE1.md)。RDMA、动态路由和持续负载性能仍未验证。以下保留前期交接快照，不代表当前状态。

> 2026-11-10 dcn9 接续：本机 Proxy/P、双卡静态路由与 benchmark 外部服务适配已实现。当前环境和启动协调详见 [docs/DCN9_DEPLOYMENT.md](docs/DCN9_DEPLOYMENT.md)。下文保留原始交接快照；其中 dcn9 待办部分已由该文更新。dcn9 当前缺少目标模型与推理运行环境，未进行真实推理、KV/cache 或 RDMA 验证。

## 用户目标与当前授权范围

将论文 PPD 的单机四卡部署迁移到两台服务器，各 1 张 RTX 4090，使用 Qwen2.5-7B-Instruct。用户选择先只在 dcn10 修改本地配置/源码并打包，手动传到 dcn9，再由 dcn9 上的 Codex CLI 接续。不要把此包当成已经完成双机部署或已经通过端到端测试。

上游基线 commit：`28aaa63c6`。原始 PPD connector 已具备跨节点 ZMQ + NCCL 路径；本阶段未更换 connector，未改变论文路由算法。

## 硬件与地址

| 节点 | 数据接口/IP | 管理 IP | 角色 |
|---|---|---|---|
| dcn9 | ens7f0np0 / 192.168.1.9 | 10.10.60.25 | 待实现 Proxy + P |
| dcn10 | ens7f0np0 / 192.168.1.11 | 10.10.60.26 | 本包提供 D/pD 启动入口 |

两台 GPU 均为 4090（各 1 张），用户提供主机内存均为 32 GB。ConnectX-5 双端口；本阶段只选端口 0。两端 RDMA 设备/端口均为 `rocep202s0f0/1`，netdev `ens7f0np0`；ethtool 显示 100000Mb/s、Full、Link yes，双向指定接口 ping 均无丢包。dcn9 第二端口 DOWN；dcn10 另一接口为 192.168.1.12/24，本次不用，避免随意改动该地址。

SSH 入口由用户提供为 `ssh dell@172.28.9.43 -p 10109`（dcn9）、`-p 10110`（dcn10）。dcn10 CLI 的 TCP 测试能接通 dcn9 的 `192.168.1.9:22` 和 `10.10.60.25:22` 并收到 OpenSSH banner；入口转发端口曾超时。未做 SSH 认证，未操作 dcn9。

用户两机已有代理：`http_proxy=https_proxy=http://127.0.0.1:7897`，大写变量同值，原始 `no_proxy` 仅有 localhost、127.0.0.1、::1。dcn10 启动器保留这些代理，补充内部地址到大小写 bypass 变量。

## 本包已经完成的部分

- `configs/two_node/dcn10.json`：单 GPU consumer，HTTP 8200、KV ZMQ 14580、Proxy 注册目标 192.168.1.9:30001。
- `ppd/dcn10.py`：仅本地预览/检查/前台启动；默认 Socket，可显式切到 RDMA；不执行 SSH、不启停 dcn9、不全局清理进程。
- `scripts/server/start_dcn10.sh`：启动入口，支持 `PYTHON_BIN`、`--dry-run`、`--check`、`--model-path`、`--transport`。
- `scripts/tests/test_dcn10_deployment.py`：无需 GPU/第三方 Python 包的部署契约测试。
- `scripts/package_source.sh`：完整源码包和 SHA-256 校验文件。
- `docs/DCN10_DEPLOYMENT.md`：配置、使用、内存限制及通信模式说明。

本地模型在 `/workspace/models/Qwen2.5-7B-Instruct`，已检查配置、tokenizer 和四个 safetensors 分片文件存在，未进行文件内容完整性校验。API 模型名称固定为 `Qwen2.5-7B-Instruct`；模型权重不在源码包内。

当前 CLI 只找到 `/workspace/conda-env/bin/python3.11`，没有 PyTorch/vLLM，因此未启动推理。不能据此断言宿主机其他环境也缺少依赖。未安装任何依赖、未修改全局代理、网卡地址、路由或宿主机服务。

## dcn9 侧待办（本阶段刻意留给接续 CLI）

1. 阅读本文件和 dcn10 部署说明，再检查 dcn9 实际 Python/容器/驱动环境及可用 RAM/显存。为两端选择兼容的同一运行环境。
2. 安装同一份 `vllm-source/`（含 D-direct 补丁）。`setup.sh` 自带 editable 安装，要求预先准备 build 依赖；vendored 项目的 torch pin 为 2.9.1。避免直接 `pip install vllm` 替代本地源码。32 GB 主机应限制构建/模型加载并发；本包没有承诺可在此机器直接完成全量编译。
3. 新增 dcn9 的 P/Proxy 本地启动入口。Proxy HTTP 10001、ZMQ 30001；P 建议 HTTP 8100、KV ZMQ 14579，`VLLM_HOST_IP=192.168.1.9`、`kv_role=kv_producer`、`proxy_ip=192.168.1.9`、`send_type=PUT_ASYNC`，所有 JSON 端口字符串按 connector 预期传递。P 也必须把 `mem_pool_size_gb` 改为 1，不能沿用默认 32 GiB。
4. 两端保持相同模型权重/tokenizer/BF16、block size=16、TP=PP=1、seed=42、prefix caching、上下文/批处理参数。dcn10 初值：max model len=2048、max seqs=4、max batched tokens=2048、GPU util=0.75、KV buffer threshold=512 MiB、enforce eager=true。P 可单独预算传输缓存，但不要让两端 KV 布局不同。
5. 修改 `ppd/comprehensive_proxy.py` 的配置表：添加 `1P_1D: {D:[8200],pD:[]}` 和 `1P_1pD: {D:[],pD:[8200]}`。D/pD 运行同一个 consumer，角色差异由 Proxy 路由决定。该文件在此包中仍为上游版本。
6. 为 benchmark 增加双卡配置、Proxy URL 参数和 API model name。现有 `MODEL_PATH` 默认是 Llama，不能直接用于 Qwen 请求；两台本地权重目录不必相同，但 API 的 `model` 必须匹配 `Qwen2.5-7B-Instruct`。
7. 修复 comprehensive benchmark 的外部服务管理模式：现有 `--skip-startup` 未完全禁止失败路径中的本机清理/重启。不要让双机实验调用四卡单机脚本或全局 pkill。ShareGPT 的同类流程也应检查。
8. 如果要做双 Replica 基线，另加 `2R` 和跨节点的显式 replica 地址。当前 replica 注册假定都在 Proxy 本机，不适合直接复用。

## 联调顺序和验收

先在两端检查本地依赖、模型和地址；部署 Proxy/P/D 后确认 `/status` 同时注册两端地址。两端先显式 Socket，固定 `NCCL_SOCKET_IFNAME='=ens7f0np0'`，关闭网络延迟模拟。预热以建立 NCCL 连接，然后测试两轮完整历史请求。

- PD：两轮均经 P，正确传输 KV 并完成 decode。
- 静态 PPD：第一轮传 KV，第二轮直接到同一个 pD，验证实际 prefix cache 命中和无 P 请求。完整历史仍需要随请求发送，不能只发新增一句话。
- 缓存被驱逐时仍可能重新 prefill；检查命中与传输字节/请求计数，不只看路由名字。
- RDMA：验证实际 TCP/RDMA 吞吐和 NCCL NET/IB 日志；4090 不假定支持 GPUDirect RDMA。dcn10 `--transport rdma` 强制 IB transport、关闭 GDR，需在两端安装 verbs/NCCL 所需网络支持。失败不算通过。
- 吞吐测试要区分网口 100G 协商速率与有效 payload 吞吐，不能按双端口 200G 汇报。
- 最大上下文 2048 仅适合初始正确性测试；原 benchmark 的 large 等工作负载可能累计超限，需先选 small_tiny 或合理扩大上下文/缩小输出再测。

原 `results/comprehensive` 不在仓库中；决策引擎缺数据时默认所有后续轮使用 PPD。应先分别采集 `1P_1D` / `1P_1pD` 的本机型/网络基线，再添加对应 config pair、构建决策表并在 `1P_1D` 上开启动态模式。512-token bypass 阈值也需实测，不能把打开开关当作自适应已验证。

## 已执行验证与未验证项

- 在当前 Python 环境运行新增 stdlib 单元测试，6 项全部通过；两份新增 shell 脚本语法检查通过。
- 执行 Socket/RDMA 参数预览，检查注册地址、端口类型、内存池、代理保留/绕过和 GDR 设置。
- 执行本地 `--check`：接口/IP 可绑定、HTTP/KV 端口空闲、模型文件存在均通过；runtime 检查因当前 Python 缺少 `torch` 返回非零，未启动服务。
- 打包后核对源码文件清单，解压到临时目录再运行测试和参数预览。
- 未执行 vLLM 推理、跨机 NCCL/RDMA 或吞吐/延迟 benchmark；尚无性能结论。

传输源码包及其 `.sha256` 文件。在 dcn9 校验后解压：

```bash
sha256sum -c vllm-ppd-dcn10-<timestamp>.tar.gz.sha256
tar -xzf vllm-ppd-dcn10-<timestamp>.tar.gz
cd vllm-ppd
```
