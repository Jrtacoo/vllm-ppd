#!/usr/bin/env bash
# Run on dcn9 Docker host. No other guide script is sourced.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --run && "$1" != --preview ) ]]; then
  echo 'Usage: bash start_proxy.sh [--preview|--run]' >&2; exit 2
fi
GUIDE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-ppd-qwen25:source-cu129}
NAME=${NAME:-ppd-dcn9-proxy}
CONFIG=${CONFIG:-$GUIDE_DIR/configs/dcn9.json}
RESULTS_DIR=${RESULTS_DIR:-/home/dell/xwd/vllm-ppd/artifacts/manual-service}
ROUTING=${ROUTING:-1P_1D}
TRANSPORT=${TRANSPORT:-socket}
[[ $ROUTING == 1P_1D || $ROUTING == 1P_1pD ]] || { echo 'Invalid ROUTING' >&2; exit 2; }
[[ $TRANSPORT == socket || $TRANSPORT == rdma ]] || { echo 'Invalid TRANSPORT' >&2; exit 2; }
command=(docker run --detach --name "$NAME" --init --stop-timeout 60
  --network host --shm-size 2g --ulimit memlock=-1:-1
  --mount "type=bind,source=$CONFIG,target=/node-config.json,readonly"
  --mount "type=bind,source=$RESULTS_DIR,target=/opt/ppd/results"
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONUNBUFFERED=1
  --env no_proxy=localhost,127.0.0.1,dcn9,dcn10,192.168.1.9,192.168.1.11
  --env NO_PROXY=localhost,127.0.0.1,dcn9,dcn10,192.168.1.9,192.168.1.11
  --workdir /opt/ppd --entrypoint python "$IMAGE"
  ppd/dcn9.py --role proxy --routing "$ROUTING" --config /node-config.json --transport "$TRANSPORT")
printf '%q ' "${command[@]}"; printf '\n'
if [[ ${1:---preview} == --run ]]; then
  [[ $(hostname -s) == dcn9 ]] || { echo 'Run on dcn9 host' >&2; exit 1; }
  command -v docker >/dev/null
  [[ -f $CONFIG ]] || { echo 'Config missing' >&2; exit 1; }
  mkdir -p -- "$RESULTS_DIR"
  "${command[@]}"
fi
