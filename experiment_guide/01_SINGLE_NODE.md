# 单服务器、单 GPU、单容器推理

本页可单独使用。在 **dcn10 宿主机**执行，使用已有镜像和模型；不需要 Proxy、P、D 或跨机 KV connector。这是普通单卡推理，不是单机四卡论文拓扑。

## 1. 检查现有环境

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
docker image inspect ppd-qwen25:source-cu129 --format '{{.Id}}'
ls /home/dell/xwd/models/Qwen2.5-7B-Instruct/config.json
docker ps -a --format 'table {{.Names}}\t{{.Status}}'
nvidia-smi
```

记录哪些容器原来在运行。如果双机 D `ppd-dcn10-decode` 正在运行，先暂停向 Proxy 发请求，再停止它：

```bash
docker stop ppd-dcn10-decode
```

## 2. 创建单卡服务

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
bash start_single.sh             # 先检查完整命令
bash start_single.sh --run       # 创建 ppd-manual-single
docker logs -f ppd-manual-single
```

日志跟随用 Ctrl-C 退出，不会停止容器。另一个终端检查：

```bash
curl --noproxy '*' -f http://192.168.1.11:8300/health
curl --noproxy '*' -f http://192.168.1.11:8300/v1/models
```

`/health` 返回 HTTP 200 后再发送请求。加载模型可能需要数分钟；`docker run` 返回容器 ID 不代表服务已就绪。

## 3. 直接发流式请求

```bash
curl --noproxy '*' -N --fail-with-body \
  http://192.168.1.11:8300/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-7B-Instruct","prompt":"Explain prefix caching in simple terms.","max_tokens":64,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

需要自动记录每请求 TTFT/TPOT、完整 SSE 和响应时，使用本目录 `03_CUSTOM_EXPERIMENTS.md` 的独立客户端，把 URL 设为 `http://192.168.1.11:8300`。

## 4. 自定义容量

下面只是修改配置的示例，并非已经验证的性能最优参数。单张 4090 的容量有限，先从默认值开始。

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
MAX_MODEL_LEN=4096 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=4096 \
  GPU_MEMORY_UTILIZATION=0.80 bash start_single.sh
```

检查预览后，在同一命令末尾增加 `--run`。也可覆盖 `IMAGE`、`MODEL_DIR`、`HOST_IP`、`PORT`、`NAME`。`MODEL_DIR` 是宿主机绝对路径。

已有容器的参数不会被环境变量改变；保存日志、停止并删除该容器，再以新参数创建。修改 `NAME` 也不能解决同一 GPU 或端口的资源冲突。

## 5. 保存日志、停止和恢复

在 dcn10 宿主机执行，输出路径换成自己的新目录：

```bash
mkdir -p /home/dell/xwd/vllm-ppd/artifacts/manual-single-logs
docker logs ppd-manual-single > /home/dell/xwd/vllm-ppd/artifacts/manual-single-logs/service.log 2>&1
docker inspect ppd-manual-single > /home/dell/xwd/vllm-ppd/artifacts/manual-single-logs/container.json
docker stop ppd-manual-single
docker rm ppd-manual-single
```

若实验前双机 D 在运行，结束后执行 `docker start ppd-dcn10-decode`，并检查两端后端健康与 Proxy `/status` 注册。若 D 原本停止，保留停止状态。恢复不了时先检查日志，不要连续创建多个 GPU 服务。
