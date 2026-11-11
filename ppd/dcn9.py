"""Start one local dcn9 Proxy or P in the foreground, without SSH or cleanup."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shlex
import sys

# Also support invocation through the absolute file path in the shell wrapper.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from ppd.dcn10 import build_launch, load_config, preflight  # noqa: E402

DEFAULT_CONFIG = PROJECT_DIR / "configs/two_node/dcn9.json"


def build_proxy_launch(config, routing="1P_1D", http_port=10001, inherited_env=None):
    if config.kv_role != "kv_producer" or config.host_ip != config.proxy_ip:
        raise ValueError("dcn9 requires a local proxy and kv_producer")
    if routing not in ("1P_1D", "1P_1pD"):
        raise ValueError("routing must be 1P_1D or 1P_1pD")
    if type(http_port) is not int or not 1 <= http_port <= 65535:
        raise ValueError("proxy HTTP port must be an integer in 1..65535")
    if len({http_port, config.http_port, config.kv_port, config.proxy_zmq_port}) != 4:
        raise ValueError("P HTTP/KV and Proxy HTTP/ZMQ ports must all differ")
    _, env = build_launch(config, inherited_env)
    command = [sys.executable, str(PROJECT_DIR / "ppd/comprehensive_proxy.py"),
               "--config", routing, "--host", config.proxy_ip,
               "--http-port", str(http_port), "--zmq-port", str(config.proxy_zmq_port)]
    return command, env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("proxy", "prefill"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--routing", choices=("1P_1D", "1P_1pD"), default="1P_1D")
    parser.add_argument("--proxy-http-port", type=int, default=10001)
    parser.add_argument("--model-path")
    parser.add_argument("--transport", choices=("socket", "rdma"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.model_path:
            config.model_path = args.model_path
        if args.transport:
            config.transport = args.transport
        command, env = build_proxy_launch(config, args.routing, args.proxy_http_port)
        if args.role == "prefill":
            command, env = build_launch(config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        selected = {k: v for k, v in env.items() if k.startswith("NCCL_") or k in (
            "CUDA_VISIBLE_DEVICES", "VLLM_HOST_IP", "SIMULATED_KV_BW_GBps", "no_proxy", "NO_PROXY")}
        print(json.dumps({"role": args.role, "config": asdict(config),
                          "environment": selected, "command": command,
                          "shell_command": shlex.join(command)}, indent=2))
        return 0
    check_config = (replace(config, http_port=args.proxy_http_port,
                            kv_port=config.proxy_zmq_port)
                    if args.role == "proxy" else config)
    if not preflight(check_config, env, proxy_only=args.role == "proxy"):
        return 1
    if args.check:
        return 0
    print(f"Starting local dcn9 {args.role} in foreground", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    sys.exit(main())
