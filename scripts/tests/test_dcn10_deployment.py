"""Deployment contracts that can run without PyTorch, vLLM, or a remote node."""

import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from ppd import dcn10  # noqa: E402


class DecodeDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.config = dcn10.load_config(dcn10.DEFAULT_CONFIG)

    def test_consumer_contract_and_existing_proxy_environment(self):
        inherited = {
            "http_proxy": "http://127.0.0.1:7897",
            "HTTPS_PROXY": "http://127.0.0.1:7897",
            "no_proxy": "localhost,existing.internal",
            "NO_PROXY": "127.0.0.1,another.internal",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_NET_GDR_READ": "1",
            "NCCL_NET_GDR_LEVEL": "SYS",
            "SIMULATED_KV_BW_GBps": "10",
        }
        original = dict(inherited)
        command, env = dcn10.build_launch(self.config, inherited)
        self.assertEqual(inherited, original)
        self.assertEqual(env["http_proxy"], inherited["http_proxy"])
        self.assertEqual(env["HTTPS_PROXY"], inherited["HTTPS_PROXY"])
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])
        for host in ("existing.internal", "another.internal", "192.168.1.9", "192.168.1.11"):
            self.assertIn(host, env["no_proxy"].split(","))
        self.assertNotIn("NCCL_P2P_LEVEL", env)
        self.assertNotIn("NCCL_NET_GDR_READ", env)
        self.assertEqual(env["SIMULATED_KV_BW_GBps"], "0")
        self.assertEqual(env["VLLM_HOST_IP"], "192.168.1.11")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "=ens7f0np0")
        self.assertEqual(command[command.index("--host") + 1], env["VLLM_HOST_IP"])
        self.assertIn("--enable-prefix-caching", command)
        kv = json.loads(command[command.index("--kv-transfer-config") + 1])
        self.assertEqual(kv["kv_role"], "kv_consumer")
        self.assertEqual(kv["kv_buffer_size"], 512 * 1024**2)
        extra = kv["kv_connector_extra_config"]
        self.assertEqual(extra["mem_pool_size_gb"], 1)
        self.assertEqual(extra["proxy_ip"], "192.168.1.9")
        self.assertIsInstance(extra["proxy_port"], str)
        self.assertEqual(str(self.config.http_port), extra["http_port"])

    def test_transport_switch_does_not_assume_gpudirect(self):
        for transport, net, disabled in (("socket", "Socket", "1"), ("rdma", "IB", "0")):
            with self.subTest(transport=transport):
                _, env = dcn10.build_launch(
                    replace(self.config, transport=transport),
                    {"NCCL_NET": "wrong", "NCCL_IB_HCA": "wrong", "NCCL_NET_GDR_LEVEL": "SYS"},
                )
                self.assertEqual(env["NCCL_NET"], net)
                self.assertEqual(env["NCCL_IB_DISABLE"], disabled)
                self.assertEqual(env["NCCL_NET_GDR_LEVEL"], "LOC")
                if transport == "rdma":
                    self.assertEqual(env["NCCL_IB_HCA"], "=rocep202s0f0:1")
                else:
                    self.assertNotIn("NCCL_IB_HCA", env)

    def test_dry_run_wrapper_is_portable_and_does_not_expose_proxy_urls(self):
        env = dict(os.environ, PYTHON_BIN=sys.executable,
                   https_proxy="http://dummy-user:dummy-secret@127.0.0.1:7897",
                   PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as unrelated_dir:
            result = subprocess.run(
                ["bash", str(PROJECT_DIR / "scripts/server/start_dcn10.sh"),
                 "--dry-run", "--model-path", "/models/model path with spaces"],
                cwd=unrelated_dir, env=env, capture_output=True, text=True, check=True,
            )
            self.assertEqual(list(Path(unrelated_dir).iterdir()), [])
        plan = json.loads(result.stdout)
        self.assertIn("/models/model path with spaces", plan["command"])
        self.assertNotIn("dummy-secret", result.stdout)
        self.assertNotIn("https_proxy", plan["environment"])

    def test_invalid_memory_and_network_settings_are_rejected_before_launch(self):
        for changes in ({"mem_pool_size_gb": 32}, {"host_ip": "0.0.0.0"},
                        {"host_ip": "192.168.1.9"}, {"kv_port": 8200}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    dcn10.build_launch(replace(self.config, **changes), {})

    def test_failed_preflight_never_executes_a_server(self):
        with mock.patch.object(dcn10, "preflight", return_value=False), \
             mock.patch.object(dcn10.os, "execvpe") as execute:
            self.assertEqual(dcn10.main([]), 1)
            execute.assert_not_called()

    def test_dry_run_never_imports_runtime_or_runs_preflight(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
             mock.patch.object(dcn10, "preflight") as check, \
             mock.patch.object(dcn10.os, "execvpe") as execute:
            self.assertEqual(dcn10.main(["--dry-run"]), 0)
        self.assertEqual(json.loads(output.getvalue())["config"]["node_name"], "dcn10")
        check.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
