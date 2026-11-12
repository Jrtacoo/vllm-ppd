#!/usr/bin/env bash
# Run on dcn9 Docker host. No other guide script is sourced.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --run && "$1" != --preview ) ]]; then
  echo 'Usage: bash start_prefill.sh [--preview|--run]' >&2; exit 2
fi
GUIDE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-ppd-qwen25:source-cu129}
MODEL_DIR=${MODEL_DIR:-/home/dell/xwd/models/Qwen2.5-7B-Instruct}
CONFIG=${CONFIG:-$GUIDE_DIR/configs/dcn9.json}
RESULTS_DIR=${RESULTS_DIR:-/home/dell/xwd/vllm-ppd/artifacts/manual-service}
NAME=${NAME:-ppd-dcn9-prefill}
ROUTING=${ROUTING:-1P_1D}
TRANSPORT=${TRANSPORT:-socket}
[[ $ROUTING == 1P_1D || $ROUTING == 1P_1pD ]] || { echo 'Invalid ROUTING' >&2; exit 2; }
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
  ppd/dcn9.py --role prefill --routing "$ROUTING" --config /node-config.json
  --transport "$TRANSPORT" --model-path /workspace/models/Qwen2.5-7B-Instruct)
printf '%q ' "${command[@]}"; printf '\n'
if [[ ${1:---preview} == --run ]]; then
  [[ $(hostname -s) == dcn9 ]] || { echo 'Run on dcn9 host' >&2; exit 1; }
  command -v docker >/dev/null
  [[ -f $CONFIG && -f "$MODEL_DIR/config.json" ]] || { echo 'Config/model missing' >&2; exit 1; }
  mkdir -p -- "$RESULTS_DIR"
  "${command[@]}"
fi
