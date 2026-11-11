"""Shared benchmark client options and two-node registry checks (stdlib only)."""

import os
from pathlib import Path
from urllib.parse import urlsplit

TWO_NODE_CONFIGURATIONS = ("1P_1D", "1P_1pD")


def add_client_arguments(parser):
    parser.add_argument("--proxy-url", default=os.environ.get("PROXY_URL", "http://localhost:10001"),
                        help="Running proxy HTTP URL")
    parser.add_argument("--api-model-name", "--model-name", dest="api_model_name",
                        default=os.environ.get("API_MODEL_NAME"),
                        help="Served API model name, independent of local weights/tokenizer path")
    parser.add_argument("--tokenizer-path", help="Local tokenizer directory for exact context accounting")
    parser.add_argument("--max-context-tokens", type=int,
                        help="Prompt plus output limit; defaults to 2048 for two-node configs")


def load_context_tokenizer(parser, args, configs):
    limit = args.max_context_tokens
    if limit is None and any(c in TWO_NODE_CONFIGURATIONS for c in configs):
        limit = 2048
    if limit is not None and limit <= 0:
        parser.error("--max-context-tokens must be positive")
    if not limit:
        return None, None
    path = args.tokenizer_path or str(Path(__file__).resolve().parents[1] / "configs/two_node/dcn9.json")
    if not args.tokenizer_path:
        import json
        path = json.loads(Path(path).read_text())["model_path"]
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:
        parser.error(f"local tokenizer unavailable ({type(exc).__name__}); set --tokenizer-path")
    return tokenizer, limit


def check_context(prompt, output_tokens, tokenizer, limit):
    if tokenizer is not None and limit:
        tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
        if tokens + output_tokens > limit:
            raise ValueError(f"Context limit: prompt {tokens} + output {output_tokens} > {limit}")


def resolve_client_options(parser, args, configs, fallback_model):
    url = urlsplit(args.proxy_url)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username
            or url.password or url.query or url.fragment or url.path not in ("", "/")):
        parser.error("--proxy-url must be an HTTP(S) origin without credentials, path or query")
    try:
        url.port
    except ValueError as exc:
        parser.error(str(exc))
    if args.skip_startup and len(configs) != 1:
        parser.error("external services require one explicit --config; switch proxy routing manually")
    if any(c in TWO_NODE_CONFIGURATIONS for c in configs) and not args.skip_startup:
        parser.error("two-node configurations require --skip-startup; launch each node explicitly")
    model = args.api_model_name or (
        "Qwen2.5-7B-Instruct" if configs[0] in TWO_NODE_CONFIGURATIONS else fallback_model)
    if not model.strip():
        parser.error("--api-model-name must not be empty")
    return args.proxy_url.rstrip("/"), model


def registry_error(data, expected_config=None):
    """Validate both roles and their addresses; registration is not inference proof."""
    if expected_config and data.get("config") != expected_config:
        return f"Proxy config {data.get('config')!r} does not match {expected_config!r}"
    instances = data.get("instances", {})
    if expected_config in TWO_NODE_CONFIGURATIONS:
        role = "decode" if expected_config == "1P_1D" else "ppd_decode"
        expected = {"prefill": ["192.168.1.9:8100"], role: ["192.168.1.11:8200"]}
        for name in ("prefill", "decode", "ppd_decode", "replica"):
            if sorted(instances.get(name, [])) != expected.get(name, []):
                return f"Unexpected {name} registry: {instances.get(name, [])}; expected {expected.get(name, [])}"
        # These configurations capture static PD/PPD baselines.
        if data.get("ppd_mode_enabled"):
            return "Static two-node baselines require dynamic PPD mode disabled"
    elif not any(instances.values()):
        return "No backend instances registered"
    return None
