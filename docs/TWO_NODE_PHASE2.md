# 双机阶段二：统一源码运行环境

本阶段目标：在 dcn10 构建本项目 vendored vLLM 的原生扩展，复制同一镜像至 dcn9，再逐节点验收依赖、GPU、小型原生算子与本地启动器预检。阶段结束后暂停，不启动 Proxy/P/D，不加载完整模型权重。

## 当前状态

**阶段二完成。** 同一源码运行镜像已在 dcn10 构建并传到 dcn9，两端镜像 ID 一致。两端的 runtime/GPU 算子、FA2 GQA、tokenizer 和 CLI 检查通过；dcn10 的 D 预检以及 dcn9 的 P、Proxy 预检通过。`pip check` 和镜像内 24 项代码测试通过。记录见 `artifacts/two-node-phase2-evidence/`。

阶段一的模型 SHA-256 核对结果保持有效，权重未修改。该阶段只确认四个模型分片存在并以只读方式挂载，没有重新校验分片内容。

## 构建选择与隔离

- JITServe 的 Python 3.11 / torch 2.5.0 cu124 / vLLM 0.6.3 不匹配本项目，保持原环境不变。
- 宿主机现有 torch 2.9.1 cu128 / vLLM 0.14.1 环境也未更改。
- 复用两端已有 `vllm/vllm-openai:v0.28.0-cu129` 的系统工具链层，其原始镜像 ID 为 `sha256:249ed60fdd67b96db472e16f945af5aaba565b20159d192ba378035b6d136a1c`。
- 在 `/opt/ppd-runtime` 独立 venv 中安装 Python 3.12 对应 torch 2.9.1 cu129，源码位于 `/opt/ppd/vllm-source`。基础镜像自带的 torch/vLLM 不是本项目运行依赖。
- 从项目完整源码构建，`VLLM_USE_PRECOMPILED=0`；本地版本 `0.13.0+ppd.local` 只是项目标签。
- 限定 SM 8.9，当前 `MAX_JOBS=4`、`NVCC_THREADS=1`；Docker build 内存上限 18 GiB、memory+swap 上限 20 GiB。
- 固定 pip 25.3、NumPy 2.2.6、CMake 3.31.6；镜像内 `pip check` 已通过。依赖完整清单保存在 `/opt/ppd/runtime-freeze.txt`，并已导出到证据目录。

缓存工具链缺少无版本 NVRTC 链接名和部分数学库头文件，Dockerfile 显式使用 torch CUDA wheel 的 NVRTC 路径和 NVIDIA include 目录。下载完成的源码和 torch 层保留为本地构建恢复快照；快照不用于推理。

## 证据与后续验收

全部原始记录位于 `artifacts/two-node-phase2-evidence/`，包括构建尝试、退出码、既有环境核对和原镜像 GPU 检查。人工停止产生的退出 137 已在 README 中注明，不能据此判定 OOM。

镜像完成后使用 `deploy/two_node/check_runtime.py` 分别验证两端 torch/CUDA、源码导入路径、PPD connector 补丁、FlashAttention/native 扩展、BF16 矩阵乘、RMSNorm、FA2 GQA 和本地 tokenizer，再执行 dcn9 Proxy/P 与 dcn10 D 的 `--check`。

这些预检不验证模型加载、实际生成、跨机注册、KV 传输、prefix cache、RDMA 或性能；这些属于下一阶段。

构建恢复：setup.py 支持可选 `VLLM_BUILD_DIR`，设置后保留 CMake/Ninja 编译目录，未设置时沿用原行为。当前保存前一次编译的绝对目录供镜像快照恢复；最终参数以证据目录的构建脚本为准。

4090 构建默认 `PPD_SKIP_FA3_BUILD=1`：FA3 仅支持 Hopper，本部署编译 FA2。setup.py 的可选 `VLLM_SKIP_FA3_BUILD=1` 只移除 FA3 扩展目标，不改变普通源码构建默认行为；其他硬件需按目标选择构建参数。

## 镜像标识

标签：`ppd-qwen25:source-cu129`。两端镜像 ID：`sha256:ae90422b157e0517caafcb7a4a4756bd2eb845eec7963d7b2f01b166cafce0ad`，未压缩大小 43,883,473,360 字节。镜像归档位于 `artifacts/images/`，传输前 SHA-256 校验通过。

归档 SHA-256：`8b67689427e6a0f9d7a1d77579285b129c3009a912d921ab45425e9fbcfc78c0`（`ppd-qwen25-source-cu129.tar.gz`）。

## 验收结果与边界

- dcn9 `dcn9-runtime.log`、`dcn9-cli-help.log`、`dcn9-prefill-preflight.log`、`dcn9-proxy-preflight.log` 均通过；dcn10 对应 runtime、CLI 和 D 预检也均通过。
- GPU 检查实际运行了 BF16 矩阵乘、原生 RMSNorm 和 FA2 GQA；本地 tokenizer 可离线加载。
- 启动预检确认网卡/IP 可绑定、计划端口空闲、模型四个权重分片存在。预检没有校验模型文件哈希。
- 没有启动 Proxy/P/D 服务，没有加载完整模型权重或执行生成；跨节点注册、NCCL/KV 传输、prefix cache、RDMA 和性能仍未验证。
- dcn9 预检之后没有遗留验证容器；这些容器均使用 `--rm`。
