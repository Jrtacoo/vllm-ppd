"""Client/route tests with fake responses; no GPU, model, NCCL or RDMA tests."""

import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.benchmark import comprehensive_benchmark as comprehensive
from scripts.benchmark import sharegpt_benchmark as sharegpt
from ppd import comprehensive_proxy as proxy


class ExternalBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_comprehensive_stops_at_each_external_failure_without_process_control(self):
        for failure in ("initial_health", "warmup", "test_point", "between_workloads"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                health = [(True, None), (False, "worker lost")]
                if failure == "initial_health":
                    health = [(False, "consumer missing")]
                stack.enter_context(mock.patch.object(comprehensive, "check_server_health", mock.AsyncMock(side_effect=health)))
                stack.enter_context(mock.patch.object(comprehensive, "warmup_servers", mock.AsyncMock(return_value=failure != "warmup")))
                result = SimpleNamespace(success_rate=100, failed_requests=0,
                                         turn1=SimpleNamespace(avg_e2e_ms=1), turn2=SimpleNamespace(avg_e2e_ms=1))
                stack.enter_context(mock.patch.object(comprehensive, "run_benchmark_point", mock.AsyncMock(
                    return_value=(result, failure != "test_point"))))
                stack.enter_context(mock.patch.object(comprehensive, "save_result"))
                stack.enter_context(mock.patch.object(comprehensive.asyncio, "sleep", mock.AsyncMock()))
                control = [stack.enter_context(mock.patch.object(comprehensive, name))
                           for name in ("start_config", "run_cleanup", "stop_config", "restart_server")]
                with self.assertRaises(RuntimeError):
                    await comprehensive.run_config_benchmark("1P_1D", ["small_tiny", "small_small_bal"],
                        [0.5], Path(directory), Path(directory) / "checkpoint.json", skip_startup=True)
                for function in control:
                    function.assert_not_called()

    async def test_sharegpt_external_failure_does_not_manage_services(self):
        for failure in ("warmup", "test_point", "health_after_point"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                loader = stack.enter_context(mock.patch.object(sharegpt, "ShareGPTDataLoader"))
                loader.return_value.load_data.return_value = [object()]
                loader.return_value.sample_conversations.return_value = [object()]
                stack.enter_context(mock.patch.object(sharegpt, "check_server_health", mock.AsyncMock(
                    side_effect=[(True, None), (False, "consumer lost")])))
                stack.enter_context(mock.patch.object(sharegpt, "warmup_servers", mock.AsyncMock(return_value=failure != "warmup")))
                stack.enter_context(mock.patch.object(sharegpt, "MAX_CONSECUTIVE_FAILURES", 1))
                run = mock.AsyncMock(return_value=([], 1))
                if failure == "test_point":
                    run.side_effect = RuntimeError("fake server failure")
                stack.enter_context(mock.patch.object(sharegpt, "run_benchmark", run))
                stats = {"turn2plus": {"avg_ttft_ms": 1, "avg_e2e_ms": 1},
                         "counts": {"turn2plus_count": 1}, "success_rate": {"turns": 100}}
                stack.enter_context(mock.patch.object(sharegpt, "compute_statistics", return_value=stats))
                control = [stack.enter_context(mock.patch.object(sharegpt, name))
                           for name in ("start_config", "stop_config", "restart_server")]
                with self.assertRaises(RuntimeError):
                    await sharegpt.run_full_benchmark("1P_1pD", [0.5], 1, False,
                                                     Path(directory), skip_startup=True)
                for function in control:
                    function.assert_not_called()

    async def test_management_entrypoints_refuse_external_mode(self):
        for module in (comprehensive, sharegpt):
            with mock.patch.object(module, "EXTERNAL_SERVICES", True), \
                    mock.patch.object(module.subprocess, "run") as run, \
                    mock.patch.object(module.subprocess, "Popen") as popen:
                self.assertFalse(module.start_config("1P_1D"))
                self.assertFalse(await module.restart_server("1P_1D"))
                if module is comprehensive:
                    self.assertFalse(module.run_cleanup())
                    self.assertFalse(module.stop_config("1P_1D"))
                else:
                    self.assertFalse(module.stop_config())
                run.assert_not_called()
                popen.assert_not_called()

    async def test_http_200_error_body_is_not_successful_generation(self):
        import aiohttp
        from aiohttp import web
        async def error_response(request):
            return web.json_response({"error": {"message": "fake backend error"}})
        app = web.Application()
        app.router.add_post("/v1/completions", error_response)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                for module in (comprehensive, sharegpt):
                    with mock.patch.object(module, "PROXY_URL", f"http://127.0.0.1:{port}"), \
                            mock.patch.object(module, "CONTEXT_TOKENIZER", None):
                        args = (session, "test", 1, 16, 8, "") if module is comprehensive else (
                            session, "test", 1, "hello", 8, 0, "")
                        result, _ = await module.run_single_turn(*args)
                        self.assertFalse(result.success)
                        self.assertIn("No generated tokens", result.error)
        finally:
            await runner.cleanup()

    async def test_proxy_static_routes_and_expired_registry(self):
        for instances in (proxy.prefill_instances, proxy.decode_instances,
                          proxy.ppd_decode_instances, proxy.replica_instances):
            instances.clear()
        producer, consumer = "192.168.1.9:8100", "192.168.1.11:8200"
        for config, target, expected in (("1P_1D", proxy.decode_instances, "pd"),
                                         ("1P_1pD", proxy.ppd_decode_instances, "ppd_direct")):
            proxy.decode_instances.clear()
            proxy.ppd_decode_instances.clear()
            proxy.prefill_instances[producer] = ("192.168.1.9:14579", time.time() + 60)
            target[consumer] = ("192.168.1.11:14580", time.time() + 60)
            with mock.patch.object(proxy, "CONFIG_NAME", config):
                self.assertEqual(proxy._is_ppd_port(8200), config == "1P_1pD")
                first = proxy.select_servers("test", 1, {})
                self.assertEqual(first[3], "pd" if config == "1P_1D" else "ppd")
                second = proxy.select_servers("test", 2, {"prefill": producer, "decode": consumer})
                self.assertEqual(second[3], expected)
                self.assertEqual(second[1], consumer)
                target[consumer] = ("expired", time.time() - 1)
                self.assertIsNone(proxy.select_servers("test", 2, {"decode": consumer})[1])
                status = await proxy.get_status()
                self.assertFalse(status["instances"]["decode"])
                self.assertFalse(status["instances"]["ppd_decode"])
        proxy.prefill_instances.clear()

    async def test_two_turn_forwarding_uses_full_history_and_skips_p_only_in_ppd(self):
        producer, consumer = "192.168.1.9:8100", "192.168.1.11:8200"
        first_prompt = comprehensive.generate_prompt(128, prefix="unique", seed="turn1")
        second_prompt = first_prompt + "first answer\n" + comprehensive.generate_prompt(16, seed="turn2")
        self.assertEqual(proxy.get_conversation_hash({"prompt": first_prompt}),
                         proxy.get_conversation_hash({"prompt": second_prompt}))
        for config in ("1P_1D", "1P_1pD"):
            for instances in (proxy.prefill_instances, proxy.decode_instances,
                              proxy.ppd_decode_instances, proxy.replica_instances):
                instances.clear()
            proxy.conversation_state.clear()
            proxy.prefill_instances[producer] = ("192.168.1.9:14579", time.time() + 60)
            target = proxy.decode_instances if config == "1P_1D" else proxy.ppd_decode_instances
            target[consumer] = ("192.168.1.11:14580", time.time() + 60)
            calls = []
            async def fake_forward(url, data, request_id):
                calls.append((url, dict(data), request_id))
                yield json.dumps({"choices": [{"text": "fake answer"}]}).encode()
            with mock.patch.object(proxy, "CONFIG_NAME", config), \
                    mock.patch.object(proxy, "ENABLE_PPD_MODE", False), \
                    mock.patch.object(proxy, "forward_request", fake_forward):
                client = proxy.app.test_client()
                for prompt in (first_prompt, second_prompt):
                    response = await client.post("/v1/completions", json={
                        "model": "Qwen2.5-7B-Instruct", "prompt": prompt, "max_tokens": 8})
                    self.assertEqual(response.status_code, 200)
            hosts = [call[0].split("/")[2] for call in calls]
            self.assertEqual(hosts, [producer, consumer, producer, consumer]
                             if config == "1P_1D" else [producer, consumer, consumer])
            self.assertEqual(calls[-1][1]["prompt"], second_prompt)
        proxy.prefill_instances.clear()
        proxy.ppd_decode_instances.clear()
        proxy.conversation_state.clear()

    async def test_smoke_client_with_fake_services_reports_limited_evidence(self):
        from aiohttp import web
        for config in ("1P_1D", "1P_1pD"):
            stats = {k: 0 for k in ("total_requests", "turn1_requests", "turn2plus_requests",
                                    "pd_requests", "ppd_requests", "ppd_direct_requests")}
            requests = []
            async def status(request):
                return web.json_response({"config": config, "stats": stats, "instances": {
                    "prefill": ["192.168.1.9:8100"],
                    "decode" if config == "1P_1D" else "ppd_decode": ["192.168.1.11:8200"]}})
            async def chat(request):
                data = await request.json()
                requests.append(data)
                subsequent = len(data["messages"]) > 1
                stats["total_requests"] += 1
                stats["turn2plus_requests" if subsequent else "turn1_requests"] += 1
                stats["pd_requests" if config == "1P_1D" else (
                    "ppd_direct_requests" if subsequent else "ppd_requests")] += 1
                return web.json_response({"choices": [{"message": {"content": "blue"}}],
                                          "usage": {"completion_tokens": 1}})
            app = web.Application()
            app.router.add_get("/status", status)
            app.router.add_post("/v1/chat/completions", chat)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "evidence.json"
                    proc = await asyncio.create_subprocess_exec(sys.executable,
                        str(PROJECT_DIR / "scripts/tests/two_node_smoke.py"), "--config", config,
                        "--proxy-url", f"http://127.0.0.1:{port}", "--output", str(output),
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                    self.assertEqual(proc.returncode, 0, stderr.decode() + stdout.decode())
                    evidence = json.loads(output.read_text())
                    self.assertEqual(evidence["response_and_routing_checks"], "passed")
                    self.assertIn("unverified", evidence["kv_transfer"])
                    self.assertIn("unverified", evidence["prefix_cache_hit"])
                    self.assertEqual(len(requests[-1]["messages"]), 3)
                    self.assertEqual(requests[-1]["model"], "Qwen2.5-7B-Instruct")
            finally:
                await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
