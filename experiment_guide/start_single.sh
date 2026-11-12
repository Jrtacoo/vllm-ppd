#!/usr/bin/env bash
# Independent single-GPU service; default prints the command only.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --run && "$1" != --preview ) ]]; then
  echo 'Usage: bash start_single.sh [--preview|--run]' >&2; exit 2
fi
IMAGE=${IMAGE:-ppd-qwen25:source-cu129}
MODEL_DIR=${MODEL_DIR:-/home/dell/xwd/models/Qwen2.5-7B-Instruct}
NAME=${NAME:-ppd-manual-single}
HOST_IP=${HOST_IP:-192.168.1.11}
PORT=${PORT:-8300}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
command=(docker run --detach --name "$NAME" --init --stop-timeout 60
  --network host --gpus device=0 --shm-size 2g --ulimit memlock=-1:-1
  --mount "type=bind,source=$MODEL_DIR,target=/workspace/models/Qwen2.5-7B-Instruct,readonly"
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env "VLLM_HOST_IP=$HOST_IP" --env VLLM_LOGGING_LEVEL=INFO
  --env "no_proxy=localhost,127.0.0.1,$HOST_IP" --env "NO_PROXY=localhost,127.0.0.1,$HOST_IP"
  --workdir /opt/ppd --entrypoint python "$IMAGE"
  -m vllm.entrypoints.openai.api_server
  --model /workspace/models/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct
  --host "$HOST_IP" --port "$PORT" --tensor-parallel-size 1 --pipeline-parallel-size 1
  --dtype bfloat16 --seed 42 --block-size 16 --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enable-prefix-caching --enforce-eager
  --generation-config vllm)
printf '%q ' "${command[@]}"; printf '\n'
if [[ ${1:---preview} == --run ]]; then
  command -v docker >/dev/null
  [[ -f "$MODEL_DIR/config.json" ]] || { echo 'Model config.json missing' >&2; exit 1; }
  "${command[@]}"
fi
