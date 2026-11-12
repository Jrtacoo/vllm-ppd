#!/usr/bin/env bash
# Run on dcn10 Docker host. No other guide script is sourced.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --run && "$1" != --preview ) ]]; then
  echo 'Usage: bash start_decode.sh [--preview|--run]' >&2; exit 2
fi
GUIDE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-ppd-qwen25:source-cu129}
MODEL_DIR=${MODEL_DIR:-/home/dell/xwd/models/Qwen2.5-7B-Instruct}
CONFIG=${CONFIG:-$GUIDE_DIR/configs/dcn10.json}
RESULTS_DIR=${RESULTS_DIR:-/home/dell/xwd/vllm-ppd/artifacts/manual-service}
NAME=${NAME:-ppd-dcn10-decode}
TRANSPORT=${TRANSPORT:-socket}
[[ $TRANSPORT == socket || $TRANSPORT == rdma ]] || { echo 'Invalid TRANSPORT' >&2; exit 2; }
command=(docker run --detach --name "$NAME" --init --stop-timeout 60
  --network host --gpus device=0 --shm-size 2g --ulimit memlock=-1:-1
  --mount "type=bind,source=$CONFIG,target=/node-config.json,readonly"
  --mount "type=bind,source=$MODEL_DIR,target=/workspace/models/Qwen2.5-7B-Instruct,readonly"
  --mount "type=bind,source=$RESULTS_DIR,target=/opt/ppd/results"
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONUNBUFFERED=1
  --env VLLM_LOGGING_LEVEL=DEBUG --env NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV
  --env no_proxy=localhost,127.0.0.1,dcn9,dcn10,192.168.1.9,192.168.1.11
  --env NO_PROXY=localhost,127.0.0.1,dcn9,dcn10,192.168.1.9,192.168.1.11)
if [[ $TRANSPORT == rdma ]]; then
  command+=(--device /dev/infiniband --env NCCL_IB_GID_INDEX=3
    --env NCCL_IB_ADDR_FAMILY=AF_INET --env NCCL_IB_ROCE_VERSION_NUM=2)
fi
command+=(--workdir /opt/ppd --entrypoint python "$IMAGE"
  ppd/dcn10.py --config /node-config.json --transport "$TRANSPORT"
  --model-path /workspace/models/Qwen2.5-7B-Instruct)
printf '%q ' "${command[@]}"; printf '\n'
if [[ ${1:---preview} == --run ]]; then
  [[ $(hostname -s) == dcn10 ]] || { echo 'Run on dcn10 host' >&2; exit 1; }
  command -v docker >/dev/null
  [[ -f $CONFIG && -f "$MODEL_DIR/config.json" ]] || { echo 'Config/model missing' >&2; exit 1; }
  mkdir -p -- "$RESULTS_DIR"
  "${command[@]}"
fi
