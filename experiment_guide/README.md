# 手动启动与实验操作目录

适用环境：dcn9、dcn10 各一张 RTX 4090，模型 Qwen2.5-7B-Instruct，已有镜像 `ppd-qwen25:source-cu129`。

本目录提供操作说明和独立启动脚本；整理本目录时没有启动/停止服务，也没有开展新实验。每篇说明包含自己的路径、命令和停止方法，不要求按编号依次执行。

| 想做什么 | 阅读文件 |
|---|---|
| 一台服务器、一个容器直接推理 | [01_SINGLE_NODE.md](01_SINGLE_NODE.md) |
| 两台服务器，Proxy/P/D，Socket 或 RoCE | [02_TWO_NODE.md](02_TWO_NODE.md) |
| 自定义单轮/多轮请求，保存每请求 TTFT/TPOT | [03_CUSTOM_EXPERIMENTS.md](03_CUSTOM_EXPERIMENTS.md) |
| 重跑阶段5/7的双轮静态对照 | [04_TWO_NODE_BASELINE.md](04_TWO_NODE_BASELINE.md) |
| 复跑 Figure 2 的小 batch 干扰实验 | [05_FIGURE2.md](05_FIGURE2.md) |
| 参数、指标、修改代码和重新构建 | [06_PARAMETERS.md](06_PARAMETERS.md) |

## 命令在哪里执行

所有 `docker` 命令在对应服务器的**宿主机终端**执行。

- 宿主机目录：`/home/dell/xwd/vllm-ppd/experiment_guide`。
- 当前 Codex CLI 容器对应目录：`/workspace/vllm-ppd/experiment_guide`，这里不一定有 Docker CLI/socket。
- dcn9 SSH：`ssh -F /home/dell/xwd/ssh/config -o BatchMode=yes dcn9`。
- 两端模型目录：`/home/dell/xwd/models/Qwen2.5-7B-Instruct`。
- 使用已导入的镜像，不需要已清理的镜像/源码压缩包。

## 本目录的脚本

四个启动脚本相互独立，不需要 `source` 公共环境文件；默认只打印完整 Docker 命令，添加 `--run` 才执行。

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
bash start_single.sh                         # 单卡启动命令预览
TRANSPORT=rdma ROUTING=1P_1pD bash start_proxy.sh  # Proxy 命令预览
TRANSPORT=rdma bash start_prefill.sh          # P 命令预览
TRANSPORT=rdma bash start_decode.sh           # D 命令预览
```

脚本不会自动删除或替换现有容器，同名容器存在时 Docker 会报错。配置使用本目录 `configs/` 的快照，不修改原 `configs/two_node/`；如改变地址等参数，需要两端同步。本目录与镜像中的启动器一起使用，并非一份可独立替代整个项目的源码。

单卡服务与双机 D 都使用 dcn10 的 GPU 0，应按相应说明先停止另一个 GPU 服务。记录现有容器状态，实验后按原状态恢复；不要一键启动所有历史实验容器。

普通实验结果写入新的 `artifacts/manual-.../` 目录，原有阶段证据不覆盖。启动器挂载的 `/opt/ppd/results` 只是可写结果目录，不会自动保存 Docker stdout；须另用 `docker logs` 导出。
