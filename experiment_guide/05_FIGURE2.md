# Figure 2：单卡 prefill / decode 干扰实验

本页列出最新 16K 同上下文实验的复跑步骤。所有命令都在 **dcn10 Docker宿主机**执行。使用同一运行镜像，独立服务容器 `ppd-manual-figure2`，不需要Proxy/P或跨机KV传输。

这是 H100/Llama 原论文 Figure 2 在 RTX4090/Qwen2.5-7B 上的适配。三个场景：纯 decode、加入一次 Full prefill、加入一次 Append prefill。后两者提交相同的“16,384 历史 + 1,024 新输入”：Full 重算 17,408 token；Append 命中 16,384 历史，仅算 1,024。背景各有独立 128-token 历史、4 个新 token，输出 128 token。客户端并发不直接当作实际batch，须通过scheduler trace验收。

16K 正式结果及统计口径见 [Phase 8 实验报告](../docs/TWO_NODE_PHASE8_CONTEXT16K.md)。后续新增的[历史长度／调度预算诊断](../docs/TWO_NODE_PHASE8_DIAGNOSTIC.md)使用多块 prefill 和跨配置一致的随机输入，见[结果图索引](../results/figures/phase8/README.md)。本页命令用于另建一次 16K 实验，不覆盖正式结果。

## 1. 记录旧服务并释放GPU

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}'
nvidia-smi
```

若原D或单卡服务正在运行，先停止客户端，再只停止相应容器，如 `docker stop ppd-dcn10-decode` 或 `docker stop ppd-manual-single`。记录原状态，实验后恢复。不使用全局pkill。

## 2. 启动带观测器的独立服务

选择一个新的 OUT；后面的每一步都使用相同值。不要复用旧证据目录。观测器和客户端使用当前宿主机 `scripts/benchmark/` 文件，不依赖镜像里是否包含阶段八脚本。

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-context16k-001
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
  --max-model-len 18432 --max-num-seqs 201 --max-num-batched-tokens 17664 \
  --gpu-memory-utilization 0.85 --block-size 16 --enable-prefix-caching --enforce-eager \
  --generation-config vllm --enable-prompt-tokens-details \
  --scheduler-cls figure2_scheduler.TraceScheduler
```

查看 `docker logs -f ppd-manual-figure2`，等待 `curl --noproxy '*' -f http://192.168.1.11:8300/health` 成功。

## 3. 先跑小规模先导

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-context16k-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  ppd-qwen25:source-cu129 /tools/figure2_interference.py \
  --server http://192.168.1.11:8300 --output /evidence/pilot \
  --trial-prefix manual-context16k-pilot-001 \
  --prefix-mode independent --decode-history-tokens 128 \
  --prefill-history-tokens 16384 --new-input-tokens 1024 \
  --full-prefill-mode same-context --output-tokens 128 --max-prefill-chunks 1 \
  --batches 0 10 200 --repeats 1 --defer-artifacts
```

只有客户端正常结束、没有其他推理请求在途时才触发导出。下面等待最多60秒，超时后停止并检查服务日志，不能在没有导出确认的情况下继续：

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-context16k-001
printf '%s' 'pilot-001' > "$OUT/export-request"
timeout 60 bash -c 'until [ "$(cat "$1/scheduler-trace.exported" 2>/dev/null)" = "$2" ]; do sleep 1; done' _ "$OUT" pilot-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  ppd-qwen25:source-cu129 /tools/figure2_analyze.py --evidence /evidence --run pilot
```

确认 `pilot/verification.json` 中 `passed=true`。分析命令非零退出或检查失败时先检查原始结果，不直接扩大负载。

## 4. 正式 batch 0～200 实验

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-context16k-001
docker run --rm --network host --entrypoint python \
  --mount type=bind,source=/home/dell/xwd/vllm-ppd/scripts/benchmark,target=/tools,readonly \
  --mount "type=bind,source=$OUT,target=/evidence" \
  ppd-qwen25:source-cu129 /tools/figure2_interference.py \
  --server http://192.168.1.11:8300 --output /evidence/formal \
  --trial-prefix manual-context16k-formal-001 \
  --prefix-mode independent --decode-history-tokens 128 \
  --prefill-history-tokens 16384 --new-input-tokens 1024 \
  --full-prefill-mode same-context --output-tokens 128 --max-prefill-chunks 1 \
  --batches 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 \
  --repeats 5 --defer-artifacts
```

全部请求完成后，以**新的导出标识**导出并验收：

```bash
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-context16k-001
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
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-figure2-context16k-001
docker inspect ppd-manual-figure2 > "$OUT/container.json"
docker stop ppd-manual-figure2
docker logs ppd-manual-figure2 > "$OUT/service.log" 2>&1
docker rm ppd-manual-figure2
```

按实验前记录恢复原有GPU服务，并检查健康与Proxy注册。原来停止的服务继续保持停止。

B0 仅测独立 prefill，没有背景 TPOT。正式参数下实际 KV 容量为 62,800 token；batch 200 已通过无抢占验收，但仍须对新运行重新核实完整 batch、Full/Append 的计算量与缓存命中，不沿用历史验收代替本次检查。背景输出至少 8 token 后才注入干扰，干扰只输出 1 token；不要随意改变这些参数后继续套用容量结论。
