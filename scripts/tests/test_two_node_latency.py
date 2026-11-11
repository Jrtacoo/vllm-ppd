"""Exercise real fragmented HTTP/SSE transport without GPUs."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark.two_node_latency import stream_request


class StreamingLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, chunks, status=200, content_type="text/event-stream"):
        async def respond(request):
            response = web.StreamResponse(status=status, headers={"Content-Type": content_type})
            await response.prepare(request)
            for chunk in chunks:
                await asyncio.sleep(.005)
                await response.write(chunk)
            return response
        app = web.Application()
        app.router.add_post("/v1/completions", respond)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stem = Path(tmp) / "test"
                async with aiohttp.ClientSession(trust_env=False) as session:
                    result = await stream_request(session, f"http://127.0.0.1:{port}",
                                                  {"max_tokens": 2}, stem)
                raw = stem.with_suffix(".sse").read_bytes()
                events = json.loads(stem.with_suffix(".events.json").read_text())
                saved = json.loads(stem.with_suffix(".response.json").read_text())
                self.assertEqual(result, saved)
                return result, raw, events
        finally:
            await runner.cleanup()

    @staticmethod
    def event(value):
        return ("data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n").encode()

    def valid_events(self):
        return [self.event({"id": "sample", "choices": [{"delta": {"role": "assistant"}}]}),
                self.event({"id": "sample", "choices": [{"text": "你好"}]}),
                self.event({"choices": [{"text": "!", "finish_reason": "length"}]}),
                self.event({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}),
                b"data: [DONE]\r\n\r\n"]

    async def test_unicode_fragmentation_role_event_and_tail_timing(self):
        events = self.valid_events()
        split = events[1].index("你".encode()) + 1
        chunks = [events[0], events[1][:split], events[1][split:], *events[2:]]
        result, raw, parsed = await self.request(chunks)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["text"], "你好!")
        self.assertEqual(raw, b"".join(chunks))
        self.assertGreater(result["ttft_ms"], parsed[0]["elapsed_ms"])
        self.assertEqual(result["ttft_ms"], parsed[1]["elapsed_ms"])
        self.assertAlmostEqual(result["tpot_ms"], result["e2e_ms"] - result["ttft_ms"])
        self.assertGreater(result["tpot_ms"], result["content_tpot_ms"])

    async def test_truncation_missing_usage_and_malformed_json_fail(self):
        valid = self.valid_events()
        for chunks in (valid[:-1], valid[:3] + valid[-1:], [b'data: {bad}\n\n'],
                       valid[:3] + [b'data: [DONE]']):
            result, _, _ = await self.request(chunks)
            self.assertFalse(result["success"])
            self.assertIsNone(result["tpot_ms"])
            self.assertTrue(result["error"])

    async def test_http_200_sse_error_and_non_sse_fail(self):
        result, _, _ = await self.request([self.event({"error": {"message": "out of memory"}})])
        self.assertFalse(result["success"])
        self.assertIn("out of memory", result["error"])
        result, _, _ = await self.request([b'{"error":"failure"}'], content_type="application/json")
        self.assertFalse(result["success"])
        self.assertIsNone(result["ttft_ms"])
        result, _, _ = await self.request([b'failure'], status=503)
        self.assertFalse(result["success"])
        self.assertEqual(result["http_status"], 503)

    async def test_wrong_output_length_fails(self):
        chunks = self.valid_events()
        chunks[-2] = self.event({"usage": {"prompt_tokens": 4, "completion_tokens": 1}})
        result, _, _ = await self.request(chunks)
        self.assertFalse(result["success"])
        self.assertIsNone(result["tpot_ms"])


if __name__ == "__main__":
    unittest.main()
