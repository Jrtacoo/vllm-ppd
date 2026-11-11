"""Local deployment contracts; these tests do not run inference or RDMA."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from ppd import dcn9, dcn10
from ppd.benchmark_options import (add_client_arguments, check_context,
                                   registry_error, resolve_client_options)


class TwoNodeDeploymentTests(unittest.TestCase):
    def test_producer_consumer_layout_and_memory_agree(self):
        producer = dcn10.load_config(dcn9.DEFAULT_CONFIG)
        consumer = dcn10.load_config(dcn10.DEFAULT_CONFIG)
        for field in ("served_model_name", "dtype", "seed", "block_size", "max_model_len",
                      "max_num_seqs", "max_num_batched_tokens", "enforce_eager"):
            self.assertEqual(getattr(producer, field), getattr(consumer, field), field)
        command, env = dcn10.build_launch(producer, {})
        kv = json.loads(command[command.index("--kv-transfer-config") + 1])
        self.assertEqual(kv["kv_role"], "kv_producer")
        self.assertEqual(kv["kv_connector_extra_config"]["mem_pool_size_gb"], 1)
        self.assertEqual(kv["kv_connector_extra_config"]["http_port"], "8100")
        self.assertEqual(env["NCCL_NET"], "Socket")
        self.assertEqual(env["VLLM_HOST_IP"], "192.168.1.9")

    def test_proxy_plan_and_port_conflicts(self):
        config = dcn10.load_config(dcn9.DEFAULT_CONFIG)
        command, _ = dcn9.build_proxy_launch(config, "1P_1pD", inherited_env={})
        self.assertEqual(command[command.index("--config") + 1], "1P_1pD")
        self.assertEqual(command[command.index("--host") + 1], "192.168.1.9")
        for port in (8100, 14579, 30001, 0):
            with self.subTest(port=port), self.assertRaises(ValueError):
                dcn9.build_proxy_launch(config, http_port=port)
        with self.assertRaises(ValueError):
            dcn9.build_proxy_launch(replace(config, kv_role="kv_consumer"))

    def test_dcn9_failed_preflight_never_starts_process(self):
        for role in ("proxy", "prefill"):
            with self.subTest(role=role), mock.patch.object(dcn9, "preflight", return_value=False), \
                    mock.patch.object(dcn9.os, "execvpe") as execute:
                self.assertEqual(dcn9.main(["--role", role]), 1)
                execute.assert_not_called()

    def test_wrappers_preview_from_unrelated_directory(self):
        import os
        for role in ("proxy", "prefill"):
            result = subprocess.run(["bash", str(PROJECT_DIR / "scripts/server/start_dcn9.sh"),
                                     "--role", role, "--dry-run", "--transport", "rdma"],
                                    env=dict(os.environ, PYTHON_BIN=sys.executable), cwd="/tmp",
                                    capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["environment"]["NCCL_NET"], "IB")
            self.assertEqual(plan["environment"]["NCCL_NET_GDR_LEVEL"], "LOC")

    def test_two_node_clients_require_external_mode_and_match_api_name(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip-startup", action="store_true")
        add_client_arguments(parser)
        args = parser.parse_args(["--skip-startup", "--proxy-url", "http://192.168.1.9:10001/"])
        self.assertEqual(resolve_client_options(parser, args, ["1P_1D"], "llama"),
                         ("http://192.168.1.9:10001", "Qwen2.5-7B-Instruct"))
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            resolve_client_options(parser, parser.parse_args([]), ["1P_1D"], "llama")
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            resolve_client_options(parser, args, ["1P_1D", "1P_1pD"], "llama")

    def test_registry_rejects_partial_wrong_or_dynamic_baselines(self):
        for config, role in (("1P_1D", "decode"), ("1P_1pD", "ppd_decode")):
            status = {"config": config, "instances": {
                "prefill": ["192.168.1.9:8100"], role: ["192.168.1.11:8200"]}}
            self.assertIsNone(registry_error(status, config))
            self.assertIsNotNone(registry_error(status, "wrong"))
            self.assertIsNotNone(registry_error(dict(status, ppd_mode_enabled=True), config))
            status["instances"]["prefill"] = []
            self.assertIsNotNone(registry_error(status, config))

    def test_context_budget_includes_full_history_and_output(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = list(range(2000))
        with self.assertRaises(ValueError):
            check_context("full history", 64, tokenizer, 2048)
        tokenizer.encode.assert_called_once_with("full history", add_special_tokens=True)


if __name__ == "__main__":
    unittest.main()
