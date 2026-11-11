"""Verify container boundaries without building images or starting services."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.server.two_node_container import build_command


class TwoNodeContainerTests(unittest.TestCase):
    def test_runtime_does_not_hide_compiled_source_and_uses_readonly_model(self):
        command = build_command("dcn10", "decode", "/project with spaces", "/models/qwen", "/results")
        mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
        self.assertTrue(any("target=/workspace/models/Qwen2.5-7B-Instruct,readonly" in m for m in mounts))
        self.assertFalse(any("target=/opt/ppd," in m or "target=/opt/ppd/vllm-source" in m for m in mounts))
        self.assertIn("memlock=-1:-1", command)
        self.assertIn("host", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--device", command)

    def test_proxy_needs_no_gpu_or_model_and_rdma_is_explicit(self):
        command = build_command("dcn9", "proxy", "/p", "/m", "/r")
        self.assertNotIn("--gpus", command)
        self.assertFalse(any("target=/workspace/models" in arg for arg in command))
        rdma = build_command("dcn9", "prefill", "/p", "/m", "/r", transport="rdma")
        self.assertEqual(rdma[rdma.index("--device") + 1], "/dev/infiniband")
        self.assertIn("--gpus", rdma)

    def test_role_and_bind_mount_validation(self):
        for node, role in (("dcn9", "decode"), ("dcn10", "prefill"), ("dcn10", "proxy")):
            with self.assertRaises(ValueError):
                build_command(node, role, "/p", "/m", "/r")
        with self.assertRaises(ValueError):
            build_command("dcn10", "decode", "/p,other", "/m", "/r")

    def test_default_preview_is_side_effect_free_without_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "not-created"
            result = subprocess.run(
                [sys.executable, "-B", str(PROJECT_DIR / "scripts/server/two_node_container.py"),
                 "--node", "dcn10", "--role", "decode", "--results-dir", str(output_dir)],
                check=True, capture_output=True, text=True,
            )
            plan = json.loads(result.stdout)
            self.assertEqual(plan["role"], "decode")
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
