"""Local decode-node launcher. Planning requires only the Python standard library.

This launcher never starts a proxy or prefill node, connects over SSH, or kills
existing workers. D and pD use the same consumer process; the proxy selects the
routing mode. Run with --dry-run before --check or an actual foreground launch.
"""

import argparse
import importlib
import ipaddress
import json
import math
import os
from pathlib import Path
import shlex
import socket
import sys
from dataclasses import asdict, dataclass


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "two_node" / "dcn10.json"


@dataclass
class DecodeConfig:
    node_name: str
    host_ip: str
    proxy_ip: str
    network_interface: str
    rdma_hca: str
    transport: str
    gpu_id: int
    http_port: int
    kv_port: int
    proxy_zmq_port: int
    model_path: str
    served_model_name: str
    dtype: str
    seed: int
    block_size: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    kv_buffer_size: int
    mem_pool_size_gb: float
    enforce_eager: bool
    kv_role: str = "kv_consumer"

    def validate(self):
        # The existing connector formats addresses as IPv4:port.
        for key in ("host_ip", "proxy_ip"):
            address = ipaddress.IPv4Address(getattr(self, key))
            if address.is_unspecified or address.is_loopback:
                raise ValueError(f"{key} must be a reachable non-loopback IPv4 address")
        if self.kv_role not in ("kv_consumer", "kv_producer"):
            raise ValueError("kv_role must be kv_consumer or kv_producer")
        if self.kv_role == "kv_consumer" and self.host_ip == self.proxy_ip:
            raise ValueError("dcn10 host_ip must differ from the dcn9 proxy_ip")
        for key in ("http_port", "kv_port", "proxy_zmq_port"):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError(f"{key} must be an integer in 1..65535")
        if self.http_port == self.kv_port:
            raise ValueError("HTTP and KV ports must differ")
        for key in ("block_size", "max_model_len", "max_num_seqs",
                    "max_num_batched_tokens", "kv_buffer_size"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.gpu_id) is not int or self.gpu_id < 0:
            raise ValueError("gpu_id must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if not math.isfinite(self.mem_pool_size_gb) or not 0 < self.mem_pool_size_gb <= 4:
            raise ValueError("mem_pool_size_gb must be in (0, 4] on these 32 GB hosts")
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError("max_num_batched_tokens must be at least block_size")
        if self.transport not in ("socket", "rdma"):
            raise ValueError("transport must be socket or rdma")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError("dtype must be bfloat16 or float16")
        for key in ("node_name", "network_interface", "rdma_hca",
                    "model_path", "served_model_name"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty string")
        if type(self.enforce_eager) is not bool:
            raise ValueError("enforce_eager must be a boolean")


def load_config(path):
    with Path(path).open() as handle:
        return DecodeConfig(**json.load(handle))


def build_launch(config, inherited_env=None):
    config.validate()
    env = dict(os.environ if inherited_env is None else inherited_env)
    # Drop the original single-host NVLink/GDR settings, retaining unrelated
    # environment variables, including both existing HTTP proxy definitions.
    removed_keys = ("NCCL_P2P_LEVEL", "NCCL_NET_GDR_READ", "NCCL_IB_HCA")
    for key in removed_keys:
        env.pop(key, None)
    required_hosts = (
        "localhost", "127.0.0.1", "::1", "dcn9", "dcn10",
        config.host_ip, config.proxy_ip, "10.10.60.25", "10.10.60.26",
    )
    bypass = []
    for item in (env.get("no_proxy", "") + "," + env.get("NO_PROXY", "")).split(","):
        if item.strip() and item.strip() not in bypass:
            bypass.append(item.strip())
    for item in required_hosts:
        if item not in bypass:
            bypass.append(item)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(config.gpu_id),
        "VLLM_HOST_IP": config.host_ip,
        "NCCL_SOCKET_IFNAME": "=" + config.network_interface,
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_NET": "Socket" if config.transport == "socket" else "IB",
        "NCCL_IB_DISABLE": "1" if config.transport == "socket" else "0",
        # Host-staged RDMA can be tested separately; do not assume GDR on 4090.
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_DEBUG": "INFO",
        "SIMULATED_KV_BW_GBps": "0",
        "PYTHONUNBUFFERED": "1",
        "no_proxy": ",".join(bypass),
        "NO_PROXY": ",".join(bypass),
    })
    if config.transport == "rdma":
        env["NCCL_IB_HCA"] = "=" + config.rdma_hca
    kv_config = {
        "kv_connector": "P2pNcclConnector",
        "kv_role": config.kv_role,
        "kv_buffer_size": config.kv_buffer_size,
        "kv_port": config.kv_port,
        "kv_connector_extra_config": {
            # P2pNcclEngine concatenates these three fields as strings.
            "proxy_ip": config.proxy_ip,
            "proxy_port": str(config.proxy_zmq_port),
            "http_port": str(config.http_port),
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": config.mem_pool_size_gb,
        },
    }
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", config.model_path,
        "--host", config.host_ip, "--port", str(config.http_port),
        "--served-model-name", config.served_model_name,
        "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
        "--dtype", config.dtype, "--seed", str(config.seed),
        "--block-size", str(config.block_size),
        "--max-model-len", str(config.max_model_len),
        "--max-num-seqs", str(config.max_num_seqs),
        "--max-num-batched-tokens", str(config.max_num_batched_tokens),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--enable-prefix-caching", "--disable-log-requests",
        "--kv-transfer-config", json.dumps(kv_config, separators=(",", ":")),
    ]
    if config.enforce_eager:
        command.append("--enforce-eager")
    return command, env


def preflight(config, env, proxy_only=False):
    """Local checks only; no remote connection or process management.

    For proxy_only, http_port/kv_port describe the proxy HTTP/ZMQ listeners.
    """
    errors = []

    def check(label, function):
        try:
            detail = function()
            print(f"PASS {label}: {detail}", flush=True)
        except Exception as exc:
            errors.append(label)
            print(f"FAIL {label}: {type(exc).__name__}: {exc}", flush=True)

    def local_network():
        if config.network_interface not in dict(socket.if_nameindex()).values():
            raise RuntimeError(f"interface {config.network_interface} is not visible")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((config.host_ip, 0))
        return f"{config.network_interface}, {config.host_ip} can bind locally"

    def local_ports():
        sockets = []
        try:
            for port in (config.http_port, config.kv_port):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.bind((config.host_ip, port))
        finally:
            for sock in sockets:
                sock.close()
        return f"HTTP {config.http_port}, KV {config.kv_port} available"

    def local_model():
        path = Path(config.model_path)
        model_config = json.loads((path / "config.json").read_text())
        if model_config.get("model_type") != "qwen2":
            raise ValueError("expected Qwen2.5 model_type=qwen2")
        if not (path / "tokenizer_config.json").is_file():
            raise FileNotFoundError("tokenizer_config.json is missing")
        index = path / "model.safetensors.index.json"
        shards = (set(json.loads(index.read_text())["weight_map"].values())
                  if index.is_file() else {"model.safetensors"})
        missing = [name for name in shards if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing weights: {missing}")
        return f"{path}, {len(shards)} weight file(s) present (not checksummed)"

    def runtime():
        torch = importlib.import_module("torch")
        if torch.__version__.split("+", 1)[0] != "2.9.1":
            raise RuntimeError("vendored vLLM requires torch==2.9.1; use the same runtime on both nodes")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        connector = importlib.import_module(
            "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector")
        expected = PROJECT_DIR / "vllm-source" / "vllm"
        if not Path(connector.__file__).resolve().is_relative_to(expected.resolve()):
            raise RuntimeError("install this archive's vendored vLLM with pip -e vllm-source")
        if not hasattr(connector.P2pNcclConnector, "is_pd_request"):
            raise RuntimeError("the PPD D-direct connector patch is missing")
        return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
                f"{torch.cuda.get_device_name(0)}, {connector.__file__}")

    def rdma_device():
        device = config.rdma_hca.split(":", 1)[0]
        if not (Path("/sys/class/infiniband") / device).exists():
            raise RuntimeError(f"RDMA device {device} is not visible")
        if not Path("/dev/infiniband").exists():
            raise RuntimeError("/dev/infiniband is not visible")
        return device + " visible; cross-node RDMA has not been tested"

    def proxy_runtime():
        for name in ("aiohttp", "msgpack", "zmq", "quart"):
            importlib.import_module(name)
        return "aiohttp, msgpack, zmq, quart import successfully"

    # Apply environment only within this launcher process before CUDA imports.
    os.environ.clear()
    os.environ.update(env)
    check("network", local_network)
    check("ports", local_ports)
    if proxy_only:
        check("proxy runtime", proxy_runtime)
    else:
        check("model", local_model)
        check("runtime", runtime)
    if config.transport == "rdma" and not proxy_only:
        check("rdma", rdma_device)
    print("Local checks only; cross-node registration, NCCL transfer and inference are unverified.")
    return not errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", help="Override the local Qwen2.5 model directory")
    parser.add_argument("--transport", choices=("socket", "rdma"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print JSON; no dependency imports or launch")
    mode.add_argument("--check", action="store_true", help="Check local prerequisites without launching")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.model_path:
            config.model_path = args.model_path
        if args.transport:
            config.transport = args.transport
        if config.kv_role != "kv_consumer":
            raise ValueError("dcn10 launcher requires kv_consumer")
        command, env = build_launch(config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        # Never print inherited proxy URLs, keys, or other process credentials.
        selected = {key: value for key, value in env.items()
                    if key.startswith("NCCL_") or key in (
                        "CUDA_VISIBLE_DEVICES", "VLLM_HOST_IP", "SIMULATED_KV_BW_GBps",
                        "no_proxy", "NO_PROXY")}
        print(json.dumps({"config": asdict(config), "environment": selected,
                          "removed_environment_keys": ["NCCL_P2P_LEVEL", "NCCL_NET_GDR_READ"],
                          "command": command, "shell_command": shlex.join(command)},
                         ensure_ascii=False, indent=2))
        return 0
    if not preflight(config, env):
        return 1
    if args.check:
        return 0
    print(f"Starting local kv_consumer at {config.host_ip}:{config.http_port}; "
          f"proxy registration at {config.proxy_ip}:{config.proxy_zmq_port}", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    sys.exit(main())
