#!/usr/bin/env python3
"""Print a container command by default; --run starts just one local service.

Execute on the selected Docker host, not via the CLI container's filesystem.
Built source stays inside the image so bind mounts cannot hide native binaries.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]


def build_command(node, role, project_dir, model_dir, results_dir,
                  image="ppd-qwen25:source-cu129", routing="1P_1D", transport="socket"):
    allowed = {"dcn9": ("proxy", "prefill"), "dcn10": ("decode",)}
    if role not in allowed.get(node, ()):
        raise ValueError("dcn9 supports proxy/prefill; dcn10 supports decode")
    if routing not in ("1P_1D", "1P_1pD"):
        raise ValueError("unsupported routing configuration")
    if transport not in ("socket", "rdma"):
        raise ValueError("transport must be socket or rdma")
    if not image or image.startswith("-") or any(c.isspace() for c in image):
        raise ValueError("invalid image name")
    project_dir = Path(project_dir).resolve()
    model_dir = Path(model_dir).resolve()
    results_dir = Path(results_dir).resolve()
    if any("," in str(p) for p in (project_dir, model_dir, results_dir)):
        raise ValueError("Docker --mount paths cannot contain commas")
    config_path = project_dir / "configs/two_node" / (node + ".json")
    bypass = "localhost,127.0.0.1,::1,dcn9,dcn10,192.168.1.9,192.168.1.11,10.10.60.25,10.10.60.26"
    command = [
        "docker", "run", "--detach", "--name", "ppd-" + node + "-" + role,
        "--init", "--stop-timeout", "60", "--network", "host",
        "--shm-size", "2g", "--ulimit", "memlock=-1:-1",
        "--mount", "type=bind,source=" + str(config_path) + ",target=/node-config.json,readonly",
        "--mount", "type=bind,source=" + str(results_dir) + ",target=/opt/ppd/results",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "no_proxy=" + bypass, "--env", "NO_PROXY=" + bypass,
        "--workdir", "/opt/ppd", "--entrypoint", "python",
    ]
    if role != "proxy":
        command += ["--gpus", "device=0", "--mount",
                    "type=bind,source=" + str(model_dir)
                    + ",target=/workspace/models/Qwen2.5-7B-Instruct,readonly"]
        if transport == "rdma":
            command += ["--device", "/dev/infiniband"]
    command += [image]
    if node == "dcn9":
        command += ["ppd/dcn9.py", "--role", role, "--routing", routing]
    else:
        command += ["ppd/dcn10.py"]
    command += ["--config", "/node-config.json", "--transport", transport]
    if role != "proxy":
        command += ["--model-path", "/workspace/models/Qwen2.5-7B-Instruct"]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, choices=("dcn9", "dcn10"))
    parser.add_argument("--role", required=True, choices=("proxy", "prefill", "decode"))
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/home/dell/xwd/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/home/dell/xwd/ppd-results"))
    parser.add_argument("--image", default="ppd-qwen25:source-cu129")
    parser.add_argument("--routing", choices=("1P_1D", "1P_1pD"), default="1P_1D")
    parser.add_argument("--transport", choices=("socket", "rdma"), default="socket")
    parser.add_argument("--run", action="store_true", help="Start a local container; default only prints JSON")
    args = parser.parse_args(argv)
    try:
        command = build_command(args.node, args.role, args.project_dir, args.model_dir,
                                args.results_dir, args.image, args.routing, args.transport)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.run:
        print(json.dumps({"node": args.node, "role": args.role,
                          "command": command, "shell_command": shlex.join(command)}, indent=2))
        return 0
    if socket.gethostname().split(".", 1)[0] != args.node:
        parser.error("execute --run on the selected node")
    if not shutil.which("docker"):
        parser.error("Docker CLI is required on the host; preview does not require Docker")
    config_path = args.project_dir / "configs/two_node" / (args.node + ".json")
    if not config_path.is_file():
        parser.error("node configuration file is missing")
    if args.role != "proxy" and not (args.model_dir / "config.json").is_file():
        parser.error("model directory must contain config.json")
    subprocess.run(["docker", "image", "inspect", args.image], check=True, stdout=subprocess.DEVNULL)
    # mkdir only after the explicitly requested launch has passed basic checks.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
