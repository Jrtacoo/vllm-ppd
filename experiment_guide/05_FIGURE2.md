# Figure 2：单卡 prefill / decode 干扰实验

本页独立列出小batch复跑步骤。所有命令都在 **dcn10 Docker宿主机**执行。使用同一运行镜像，独立服务容器 `ppd-manual-figure2`，不需要Proxy/P或跨机KV传输。

这是 H100/Llama 原论文 Figure 2 在 RTX4090/Qwen2.5-7B 上的适配。三个场景：纯decode、加入一次1024新token的full-prefill、加入一次“2048缓存历史+1024新token”的append-prefill。客户端并发不直接当作实际batch，须通过scheduler trace验收。

## 1. 记录旧服务并释放GPU

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}'
nvidia-smi
```

若原D或单卡服务正在运行，先停止客户端，再只停止相应容器，如 `docker stop ppd-dcn10-decode` 或 `docker stop ppd-manual-single`。记录原状态，实验后恢复。不使用全局pkill。

## 2. 启动带观测器的独立服务

选择一个新的 OUT；后面的每一步都使用相同值。不要复用旧证据目录。观测器和客户端使用当前宿主机 `scripts/benchmark/` 文件，不依赖镜像里是否包含阶段八脚本。

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-001
mkdir "$OUT"
docker run --detach --name ppd-manual-figure2 --init --stop-timeout 60 \
  --network host --gpus device=0 --shm-size 2g --ulimit memlock=-1:-1 \
  --mount type=bind,source=/home/dell/xwd/models/Qwen2.5-7B-Instruct,target=/workspace/models/Qwen2.5-7B-Instruct,readonly \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/opt/figure2-tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONPATH=/opt/figure2-tools:/opt/ppd:/opt/ppd/vllm-source \
  --env FIGURE2_TRACE=/evidence/scheduler-trace.jsonl \
  --env VLLM_HOST_IP=192.168.1.11 --env VLLM_LOGGING_LEVEL=INFO \
  --env no_proxy=localhost,127.0.0.1,192.168.1.11 \
  --env NO_PROXY=localhost,127.0.0.1,192.168.1.11 \
  --workdir /opt/ppd --entrypoint python ppd-qwen25:source-cu129 \
  -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct \
  --host 192.168.1.11 --port 8300 --dtype bfloat16 --seed 42 \
  --max-model-len 4096 --max-num-seqs 33 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --block-size 16 --enable-prefix-caching --enforce-eager \
  --generation-config vllm --enable-prompt-tokens-details \
  --scheduler-cls figure2_scheduler.TraceScheduler
```

查看 `docker logs -f ppd-manual-figure2`，等待 `curl --noproxy '*' -f http://192.168.1.11:8300/health` 成功。

## 3. 先跑小规模先导

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  ppd-qwen25:source-cu129 /tools/figure2_interference.py \
  --server http://192.168.1.11:8300 --output /evidence/pilot \
  --trial-prefix manual-pilot-001 --batches 1 16 --repeats 1 --defer-artifacts
```

只有客户端正常结束、没有其他推理请求在途时才触发导出。下面等待最多60秒，超时后停止并检查服务日志，不能在没有导出确认的情况下继续：

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-001
printf '%s' 'pilot-001' > "$OUT/export-request"
timeout 60 bash -c 'until [ "$(cat "$1/scheduler-trace.exported" 2>/dev/null)" = "$2" ]; do sleep 1; done' _ "$OUT" pilot-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  ppd-qwen25:source-cu129 /tools/figure2_analyze.py --evidence /evidence --run pilot
```

确认 `pilot/verification.json` 中 `passed=true`。分析命令非零退出或检查失败时先检查原始结果，不直接扩大负载。

## 4. 正式小batch实验

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  ppd-qwen25:source-cu129 /tools/figure2_interference.py \
  --server http://192.168.1.11:8300 --output /evidence/formal \
  --trial-prefix manual-formal-001 --batches 1 2 4 8 16 --repeats 5 --defer-artifacts
```

全部请求完成后，以**新的导出标识**导出并验收：

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-001
printf '%s' 'formal-001' > "$OUT/export-request"
timeout 60 bash -c 'until [ "$(cat "$1/scheduler-trace.exported" 2>/dev/null)" = "$2" ]; do sleep 1; done' _ "$OUT" formal-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  ppd-qwen25:source-cu129 /tools/figure2_analyze.py --evidence /evidence --run formal
```

主读 `formal/verification.json`、`formal/summary.csv`、`formal/decode-requests.csv`。主指标 `token_tpot_ms` 从首个到最后一个token-ID事件计时，与阶段7截止SSE结束的 `tpot_ms` 有区别；两者均保存。原始请求/SSE、失败与调度trace都保留，未通过验收不能只拿汇总表当作成功。

## 5. 保存配置与日志并停止

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-001
docker inspect ppd-manual-figure2 > "$OUT/container.json"
docker stop ppd-manual-figure2
docker logs ppd-manual-figure2 > "$OUT/service.log" 2>&1
docker rm ppd-manual-figure2
```

按实验前记录恢复原有GPU服务，并检查健康与Proxy注册。原来停止的服务继续保持停止。

不要直接把batch改成250并沿用容量。共享历史与独立历史是不同实验；大batch方案另见 `docs/TWO_NODE_PHASE8_LARGE_BATCH.md`。本页仅复跑小batch方法，不会重写已有发布图片或旧证据。
