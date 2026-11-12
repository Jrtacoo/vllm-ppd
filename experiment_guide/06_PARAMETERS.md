# 参数、结果和修改代码

本页可单独阅读，适用于两端已有的 `ppd-qwen25:source-cu129` 镜像。

## 宿主机、镜像和容器

| 内容 | 位置/作用 |
|---|---|
| 可编辑项目源码 | 宿主机 `/home/dell/xwd/vllm-ppd`，CLI中 `/workspace/vllm-ppd` |
| 已有运行镜像 | `ppd-qwen25:source-cu129`，不因宿主机改代码自动更新 |
| 镜像内独立Python环境 | `/opt/ppd-runtime` |
| 镜像内项目源码 | `/opt/ppd` |
| 镜像内编译的vLLM源码/原生扩展 | `/opt/ppd/vllm-source` |
| 模型 | 宿主机 `/home/dell/xwd/models/Qwen2.5-7B-Instruct`，只读挂载进服务 |
| 本操作目录的双机配置 | `experiment_guide/configs/dcn9.json`、`dcn10.json` |
| 手动实验结果 | 每次选择一个新 `artifacts/manual-.../` 目录 |

本目录配置是原双机配置的快照。启动脚本通过 `CONFIG` 可指定另一个绝对路径；两端需要兼容配置。不要把宿主机整个源码目录挂载到镜像 `/opt/ppd` 或 `/opt/ppd/vllm-source`，这会遮住已编译的扩展。

## 服务参数

| 参数 | 默认/当前意义 |
|---|---|
| `--host` / `host_ip` | 服务监听地址，必须属于当前宿主机可见接口 |
| HTTP端口 | 客户端/Proxy请求入口：单卡8300、Proxy10001、P8100、D8200 |
| KV端口14579/14580 | P/D的KV连接控制；不是全部RoCE数据固定经过的TCP端口 |
| Proxy ZMQ30001 | 后端注册/控制消息 |
| `--served-model-name` | API请求中的模型名称 `Qwen2.5-7B-Instruct` |
| `--max-model-len` | 默认2048，完整历史+本次输入+输出的上下文预算 |
| `--max-num-seqs` | 默认4，引擎最多同时处理的序列数；不是客户端QPS |
| `--max-num-batched-tokens` | 默认2048，每次调度的token预算；不是总KV缓存容量 |
| `--gpu-memory-utilization` | 默认0.75，vLLM的GPU内存规划比例，不是固定利用率 |
| `--enable-prefix-caching` | 允许复用完整KV块的前缀缓存 |
| `--block-size` | 16，前缀缓存复用受完整块限制 |
| `--enforce-eager` | 不使用CUDA Graph，沿用已验证部署方式 |
| `--dtype bfloat16` | BF16权重/计算配置 |
| `--tensor-parallel-size 1` / `--pipeline-parallel-size 1` | 每个服务完整模型独立运行，不切分权重 |
| `kv_buffer_size` | 当前512MiB的connector缓冲阈值，不是整个GPU显存的硬上限 |
| `mem_pool_size_gb` | 当前1GiB主机内存池配置 |

单卡脚本用 `MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`MAX_BATCHED_TOKENS`、`GPU_MEMORY_UTILIZATION` 等环境变量覆盖。双机参数编辑本目录的两个JSON，或 `CONFIG=/绝对路径/节点.json bash start_*.sh --run`。P的 `kv_role=kv_producer`、D为consumer。两端修改后重新创建相关容器。

## 路由和传输

`ROUTING=1P_1D/1P_1pD` 决定请求是否在后续轮次直接交给pD；`TRANSPORT=socket/rdma` 决定P→D的KV网络传输。它们是独立维度。

本目录脚本默认 Socket+PD，始终显式设置更清楚。RDMA脚本额外传 GID3、IPv4、RoCEv2；镜像内启动器设置 `NCCL_NET=IB`、`NCCL_IB_DISABLE=0`、精确网卡/HCA。Socket则为 `NCCL_NET=Socket`、`NCCL_IB_DISABLE=1`。当前4090环境禁用GDR，经过主机内存中转。

Proxy接收请求/转发响应；KV由P、D直接交换，不通过Proxy容器。开启动态策略还需要本硬件的离线校准表、启动参数与回退/旁路统计，本操作目录没有替你启用它。

## 查看运行中的内容

在相应Docker宿主机执行，例如：

```bash
docker inspect ppd-dcn10-decode
docker logs ppd-dcn10-decode
docker exec ppd-dcn10-decode python -c 'import torch,vllm; print(torch.__version__); print(vllm.__file__)'
docker exec -it ppd-dcn10-decode bash
```

如果没有运行中的容器，也能创建一个不使用GPU的临时检查容器：

```bash
docker run --rm --entrypoint python ppd-qwen25:source-cu129 \
  -c 'import torch,vllm; print(torch.__version__); print(vllm.__file__)'
```

## 修改源码后怎么办

- 修改本目录挂载的JSON：不需要重建镜像，但应重新创建服务来应用完整配置。
- 修改单卡Docker参数：保存日志、停止/删除该容器，再按新参数创建。
- 修改宿主机 `ppd/*.py` 或镜像内vLLM源码：现有容器使用镜像内的副本，不会自动更新。正式部署需要重建镜像并让两端使用同一镜像。
- 修改通过独立路径挂载的客户端或Figure2观测器：下次运行读取宿主机文件，通常不需重建镜像。

需要完整重建时，在dcn10宿主机项目根目录执行以下入口。它是源码编译，可能耗时并占用大量内存；当前无需为阅读本目录执行。

```bash
cd /home/dell/xwd/vllm-ppd
docker build --network host \
  -f deploy/two_node/Dockerfile -t ppd-qwen25:source-cu129 .
```

先阅读 Dockerfile 和 `docs/TWO_NODE_PHASE2.md`；该命令重新构建，不能保证复用历史手工恢复的编译缓存。两端部署前做运行环境预检，再用 `docker save`/`docker load` 传递同一镜像；镜像ID在两端应一致。旧容器仍引用旧镜像ID，打同名标签不会更新它，须重建容器。

源码包不包含可运行的Python/CUDA环境，源码传输和Docker镜像传输是两件事。本目录的操作不要求再生成历史压缩包。

## 读数边界

TTFT包含客户端到Proxy/P/D的排队、计算和网络时间；TPOT是平均值，不是每个token的抖动。对比时先核对口径、成功率、输入长度、并发与缓存条件。

P/D的local/external cache token指标说明调度器的缓存使用情况，不能直接换算真实网络KV字节量。PD本身也能命中D本地缓存；静态PPD的机制是后续请求绕过P/跨机KV路径并在D计算追加部分。
