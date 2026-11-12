# 自己发送请求并保存 TTFT / TPOT

本页可单独使用，前提是服务已经启动。客户端不启动、不停止、不切换服务。

| 服务类型 | 客户端 URL |
|---|---|
| dcn10 独立单卡服务 | http://192.168.1.11:8300 |
| 双机 Proxy（PD 或静态 PPD） | http://192.168.1.9:10001 |

从任一可访问服务的 Docker 宿主机运行下面的客户端。它复用已有推理镜像中的 Python/aiohttp，**不映射 GPU、不加载模型**，结束后自动删除客户端容器。也可直接使用已安装 aiohttp 的 Python 执行本目录 `request_client.py`，不需要 torch/vLLM。

## 1. 单卡单轮，自定义问题

在 dcn10 宿主机执行。每次把 `OUT` 换成一个新的绝对路径；结果目录 `run/` 已存在时客户端会拒绝覆盖。

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-single-request-001
mkdir -p "$OUT"
docker run --rm --network host --entrypoint python \
  --mount "type=bind,source=$PWD,target=/manual,readonly" \
  --mount "type=bind,source=$OUT,target=/out" \
  ppd-qwen25:source-cu129 /manual/request_client.py \
  --url http://192.168.1.11:8300 \
  --prompt 'Explain how prefix caching reduces inference latency.' \
  --max-tokens 64 --conversations 1 --concurrency 1 --output /out/run
```

客户端固定 `temperature=0, seed=42, ignore_eos=true`，用于可控的定长生成。输入会包装为 `User: Session <唯一ID>. 问题\nAssistant:`，使 Proxy 可以识别会话，同时减少不同会话间的意外共享前缀。输入是文本，不保证某个精确 token 数；实际 token 数以响应 usage 为准。

## 2. 双机两轮，4个并发会话

确认 P/D 健康、Proxy `/status` 为你想要的路由。以下在任一有项目目录且能访问数据网的 Docker 宿主机执行：

```bash
cd /home/dell/xwd/vllm-ppd/experiment_guide
OUT=/home/dell/xwd/vllm-ppd/artifacts/manual-proxy-two-turn-001
mkdir -p "$OUT"
docker run --rm --network host --entrypoint python \
  --mount "type=bind,source=$PWD,target=/manual,readonly" \
  --mount "type=bind,source=$OUT,target=/out" \
  ppd-qwen25:source-cu129 /manual/request_client.py \
  --url http://192.168.1.9:10001 \
  --prompt 'Explain KV caching in LLM inference.' \
  --append 'Continue by explaining when cached prefixes can be reused.' \
  --max-tokens 64 --conversations 12 --concurrency 4 --output /out/run
```

总共12个会话，每个两轮，正常应记录24个请求。每个会话的第二轮等待第一轮结束，并发送“原提示+第一轮实际回答+新增问题”。单个会话失败后跳过其后续轮次，其余会话继续完成，失败响应和部分 SSE 保留，不自动重试。程序退出非零或 `summary.json` 的 `passed=false` 时不能当作完整成功实验。

这个客户端是**闭环并发**：并发上限是同时活动的会话数，不是每秒请求数；没有固定 QPS，没有自动预热，也不证明请求在 GPU 上属于同一个 batch。每请求输出到终端和写盘会产生客户端开销，不把它当作原论文的大规模饱和负载生成器。

## 3. 输入来自自己的文本文件

把 UTF-8 文本放在本目录，例如 `my_prompt.txt`，将前面命令的 `--prompt '...'` 换成：

```bash
--prompt-file /manual/my_prompt.txt
```

需要第二轮时加 `--append '新增问题'`。所有历史和输出合计必须满足服务上下文限制；默认2048。超限会保存失败，不自动截断。

## 4. 看结果

结果在所设置 `OUT` 的 `run/` 下：

- `run.json`：URL、输入、并发、生成设置、客户端 SHA256 和唯一会话命名空间。
- `requests.csv`：最方便阅读的逐请求 TTFT、TPOT、E2E、token 数、响应ID和错误。
- `requests.jsonl`：逐请求完整记录，含生成正文与 usage。
- `summary.json`：成功数、是否全部完成，以及按轮次的均值/中位数。
- `raw/cXXXX-tN.request.json`：实际发送的完整请求。
- `raw/cXXXX-tN.sse`：原始流；`.events.json`：带客户端时间的事件；`.response.json`：正文、usage和计时。

`ttft_ms` 从 POST 开始到第一个非空文本事件；`tpot_ms=(SSE结束−首文本)/(输出token数−1)`，包含末尾 usage/连接结束时间。`content_tpot_ms` 截止最后一个非空文本事件。一个 SSE 事件可能包含多个 token，不能把事件间隔直接当作逐 token ITL。输出只有1 token时 TPOT 留空。

这个简单客户端不自动核验后台路由或 KV/cache 命中；双机严谨对照使用本目录 `04_TWO_NODE_BASELINE.md`，并结合服务日志和前后指标。

## 5. 做可比较的实验

每轮保留新目录、完整请求、原始 SSE、失败和服务配置。同一轮只运行一种负载，避免其他客户端干扰。比较 PD/PPD 或 Socket/RoCE 时尽量固定模型、服务容量、输入、输出、预热和缓存起点，并多次重复。

本客户端每次生成新的 Session ID，适合功能与自定义负载测试，跨次运行不是逐字节相同 payload。需要阶段5/7式可配对基线时使用共享 manifest，并对第二轮 payload hash 再配对；第一轮生成内容也可能改变第二轮输入。
