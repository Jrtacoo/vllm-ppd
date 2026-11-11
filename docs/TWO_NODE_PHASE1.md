# 双机阶段一：源码、模型与部署准备（2026-11-10）

本阶段已完成源码合并、模型内容核对、部署命令生成及代码测试。按用户要求在阶段末暂停；没有构建推理镜像、安装 GPU 推理依赖或启动 Proxy/P/D。

## 实际核验

- dcn10 源码目录：CLI 中 `/workspace/vllm-ppd`，对应宿主机 `/home/dell/xwd/vllm-ppd`。本地文件直接读写。
- dcn9 源码目录：`/home/dell/xwd/vllm-ppd`。已将该目录的 Proxy/P、双节点路由和 benchmark 更新合并到 dcn10，再同步本阶段文件回 dcn9。
- 两端模型宿主机路径均为 `/home/dell/xwd/models/Qwen2.5-7B-Instruct`。dcn10 CLI 映射为 `/workspace/models/Qwen2.5-7B-Instruct`。
- 模型目录顶层 11 个文件（包含全部 4 个 safetensors 分片、tokenizer 和配置）逐文件 SHA-256 完全一致。未把下载工具的 `.cache` 目录计入模型内容。
- 两台均为 RTX 4090 24 GB、驱动 580.95.05。此前端口、网卡及双向连接检查结果见两份节点部署文档。
- 两台宿主机有 Docker，dcn10 CLI 容器没有 Docker CLI/socket；dcn9 可直接 SSH 登录。此前为检查 dcn10 宿主机 Docker，使用过 dcn9 上已有集群密钥绕回 dcn10，该路径不用于源码读写。
- 两端已有的 `vllm/vllm-openai:v0.28.0-cu129` 镜像不适合作为本项目的直接运行环境：在 dcn9 的临时无网络检查容器中读到 Python 3.12.3、torch 2.13.0+cu129、vLLM 0.28.0+cu129。检查容器已经退出并自动删除，没有加载模型或 GPU。

## 依赖兼容性结论

仓库 CUDA/build requirements 明确要求 torch 2.9.1。通过 PyPI release 元数据核对：官方 vLLM 0.13.0 wheel 要求 torch 2.9.0，0.14.0 wheel 要求 torch 2.9.1。

进一步对比官方 v0.13.0、v0.14.0 对应源码：仓库的 `_custom_ops.py`、`CMakeLists.txt` 和 scheduler 均与两个 tag 不完全相同；P2P engine 与两个 tag 相同；CUDA requirements 与 v0.14.0 相同。因此，不能仅靠版本号或 torch pin 断言某个官方 release wheel 的原生算子适配此源码。

本阶段提供 `deploy/two_node/Dockerfile`，从此工程的 vendored 源码构建完整原生扩展，使用 CUDA 12.9.1 devel base、Python 3.10（Ubuntu 22.04 默认）、torch 2.9.1 cu129；限定 RTX 4090 的 SM 8.9，`MAX_JOBS=1`、`NVCC_THREADS=1`。本地版本标签 `0.13.0+ppd.local` 沿用项目 setup 的 0.13.0 标记，仅是本地包版本，不代表官方发布版本/ABI。

**Dockerfile 尚未实际构建**，因此下载、构建依赖解析、编译资源占用及最终运行兼容性仍需下一阶段验证。源码构建可能较慢，限制并发也不保证 32 GB 主机一定不会 OOM。成功后保留构建日志和镜像内 `runtime-freeze.txt`，只构建一次，再将同一镜像传到另一台；不要在两端分别解析一套浮动依赖。

构建示例（下一阶段执行）：

```bash
# 在有 Docker 的宿主机、以项目根目录为 build context 执行。
docker build --network host \
  --build-arg PPD_MAX_JOBS=1 --build-arg PPD_NVCC_THREADS=1 \
  --build-arg PPD_CUDA_ARCH_LIST=8.9 \
  -f deploy/two_node/Dockerfile -t ppd-qwen25:source-cu129 .
```

构建需要可访问 CUDA base、PyPI/PyTorch 及 CMake 外部源码。已知 dcn10 的本机代理 127.0.0.1:7897 可连；dcn9 宿主机同端口曾拒绝连接，不能假定两端代理服务都在运行。镜像拉取使用 Docker daemon 网络/代理，RUN 下载使用 build 网络/代理，分别验证，不修改系统配置来绕过问题。模型推理本身使用本地权重和 offline 模式。

## 运行入口

新增 `scripts/server/two_node_container.py`。默认仅输出 JSON 和 Docker 命令，只有显式 `--run` 才启动本机容器。例如：

```bash
# dcn9 宿主机：默认只预览。
python3 /home/dell/xwd/vllm-ppd/scripts/server/two_node_container.py \
  --node dcn9 --role proxy --routing 1P_1D
python3 /home/dell/xwd/vllm-ppd/scripts/server/two_node_container.py \
  --node dcn9 --role prefill

# dcn10 宿主机：默认只预览。
python3 /home/dell/xwd/vllm-ppd/scripts/server/two_node_container.py \
  --node dcn10 --role decode
```

容器使用 host 网络、2 GiB `/dev/shm`、unlimited memlock；P/D 只映射 GPU 0，Proxy 无需 GPU。模型挂载只读、节点配置挂载只读，结果写到宿主机 `/home/dell/xwd/ppd-results`。为避免挂载目录遮住镜像内已编译的 `.so`，**不把整个源码目录 bind 到 `/opt/ppd` 或其 `vllm-source`**。镜像内保留完整源代码与 editable 安装。修改运行代码后需要重建镜像。

初始 Socket 模式不映射 RDMA 设备。显式 `--transport rdma` 才给 P/D 添加 `/dev/infiniband`；它只生成运行条件，不能证明 RDMA/GDR 可用。启动器仍禁用 GDR。

## 本阶段验证范围与暂停点

在 dcn10 CLI 创建隔离的 `/workspace/ppd-validation-env`，仅安装 aiohttp、Quart、NumPy、pyzmq、msgpack 及其依赖用于代码测试。没有把它当作推理环境，没有安装 torch/vLLM，也没有更改全局 Python。

- 24 项测试通过：原双节点/benchmark 的 20 项，加容器边界、角色、显存设备映射与默认预览行为的 4 项。
- 两端模型 11 个顶层文件 SHA-256 一致。
- Python/shell 语法检查与三个角色的启动命令预览通过。
- 最终同步后比较两端源码文件清单及 SHA-256，结果存放在工程外的阶段证据目录。

原始测试日志、模型校验清单、容器命令预览和源码同步结果位于 dcn10 的 `/workspace/vllm-ppd/artifacts/two-node-phase1-evidence/`，在 dcn9 对应为 `/home/dell/xwd/vllm-ppd/artifacts/two-node-phase1-evidence/`。

下一阶段：验证下载路径并构建单一运行镜像，复制到两端；分别运行本地 runtime 预检，通过后再启动 Socket PD、静态 PPD 和实际 KV/cache 验证。此前的测试都是代码测试，不能当作推理、RDMA 或性能验证。

目录整理：两端统一采用 `vllm-ppd/artifacts/` 保存源码压缩包、校验文件及阶段证据。历史压缩包和原始日志保持原样，其中记录的旧绝对路径对应整理前的布局。打包脚本默认写入该目录，并将整个目录排除出源码包。
