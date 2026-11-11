# 双机阶段五：Socket PD / 静态 PPD 逐请求性能基线

2026-11-11 UTC，本阶段完成。正式采集 192 条请求，全部成功；每个请求的客户端 TTFT、TPOT、输入/输出 token 数和原始 SSE 均已保存。P/D/Proxy 保持运行，最终 Proxy 为静态 `1P_1pD`；本阶段结束后暂停，未进入 RDMA。

## 结果

下表均为中位数，单位 ms；每个负载/并发/轮次有 12 个正式样本。均值、最小值和最大值见每种模式的 `summary.json`。

| 负载 | 并发对话 | 模式 | T1 TTFT | T2 TTFT | T1 TPOT | T2 TPOT |
|---|---|---|---:|---:|---:|---:|
| short | 1 | 1P_1D | 108.14 | 111.83 | 17.40 | 17.41 |
| short | 1 | 1P_1pD | 102.19 | 44.63 | 17.39 | 17.41 |
| short | 4 | 1P_1D | 112.88 | 112.12 | 18.13 | 18.13 |
| short | 4 | 1P_1pD | 121.07 | 54.16 | 18.06 | 18.10 |
| long | 1 | 1P_1D | 157.17 | 146.79 | 17.38 | 17.42 |
| long | 1 | 1P_1pD | 154.00 | 66.79 | 17.33 | 17.38 |
| long | 4 | 1P_1D | 162.25 | 140.76 | 18.21 | 18.32 |
| long | 4 | 1P_1pD | 195.24 | 68.31 | 18.47 | 18.59 |

第二轮中有 39 对 payload 完全相同，下表仅使用该子集，仍为中位数/ms。其余 9 对保留在全量结果中并明确标记，不进入这个子集。该子集也显示本次负载下 PPD 第二轮 TTFT 较低，TPOT 大体接近；long/并发 4 的 PPD TPOT 略高。

| 负载 | 并发 | 相同输入对数 | PD TTFT | PPD TTFT | PD TPOT | PPD TPOT |
|---|---|---:|---:|---:|---:|---:|
| short | 1 | 12 | 111.83 | 44.63 | 17.41 | 17.41 |
| short | 4 | 8 | 112.94 | 52.40 | 18.13 | 18.04 |
| long | 1 | 12 | 146.79 | 66.79 | 17.42 | 17.38 |
| long | 4 | 7 | 142.51 | 68.64 | 18.32 | 18.60 |

完整数值见 `matched-second-turn-comparison.json`。

## 负载和测量方法

使用 Qwen2.5-7B-Instruct、两台各一张 RTX 4090、100G 接口 `ens7f0np0`。保持阶段三/四的同一源码运行镜像 `ppd-qwen25:source-cu129`，BF16、block size 16、上下文 2048、max seqs 4、max batched tokens 2048、GPU utilization 0.75、eager、prefix caching、Socket、无延迟模拟。NCCL/vLLM 日志级别沿用 INFO/DEBUG。

新增 `scripts/benchmark/two_node_latency.py` 为独立外部服务客户端，不改论文原有 benchmark、不管理服务、不改变推理源码或依赖。客户端在 dcn10 CLI 的 `ppd-validation-env` 运行，结果直接写入宿主机对应的阶段五目录；没有新建镜像，也没有把客户端放入已有运行镜像。

每种模式测试 short/long 两类合成负载，并发为 1 和 4，每单元 12 组两轮对话，共 48 组、96 个正式请求。short 第一轮精确 256 输入 token，每轮固定输出 64；long 第一轮精确 768 输入 token，每轮固定输出 128。追加文本分别含 32/256 个 `detail` 填充 token，此外还有指令和轮次分隔符，所以实际追加输入长于 32/256。第二轮完整输入包括第一轮 prompt、实际生成文本和新追加文本。short 第二轮完整历史输入 368–369 tokens；long 第二轮完整历史输入 1167–1169 tokens。

两种模式共享同一 `workload.json`。`temperature=0, seed=42, ignore_eos=true`，所有正式请求均以 `length` 结束并达到指定输出 token 数。通过 D 的 `/tokenize` 在计时前核对每个 prompt，并检查上下文预算；API usage 与独立分词长度一致。

使用闭环并发：最多同时运行 1/4 组对话，每组第一轮结束后提交带完整历史的第二轮，无人为 think time；轮次之间包含客户端分词验证和文件写入。它不等价于开放式固定 QPS 或服务器峰值吞吐测试。

当前镜像未开放 `/reset_prefix_cache`。因此在每种模式开始前停止并重新启动原 P/D 容器以清除进程内缓存，重建 Proxy 选择对应路由，检查健康与两端注册后开始。每单元另跑一组独立两轮预热，预热原始响应保留但不计入正式请求表。不同单元之间不重启 P/D；每个正式对话在开头含唯一标识，避免跨对话复用完整 16-token block。第二轮保留本对话历史缓存，这是要测量的行为。

## 指标口径与文件

- `ttft_ms`：客户端开始 POST 到第一段非空正文，包含 Proxy、P、KV 传输、D 和网络。角色事件和 usage 事件不算首 token。
- `tpot_ms = (e2e_ms - ttft_ms) / (completion_tokens - 1)`，沿用项目原客户端口径。`e2e_ms` 到 SSE 流 EOF，因此 TPOT 包含正文结束后的 usage/结束标记尾部时间。
- 额外保存 `last_content_ms`、`done_ms`、`content_tpot_ms`，便于核查尾部时间。SSE 可能合并多个 token，未把网络事件间隔当成逐 token ITL。
- `prompt_tokens`、`completion_tokens` 来自 API usage；记录包含请求 ID、finish reason、错误、payload SHA-256、回答文本、原始文件位置。正式结果逐请求 flush 保存，失败也会保留证据并阻止本阶段通过。

实际文件在两端 `/home/dell/xwd/vllm-ppd/artifacts/two-node-phase5-evidence/`：

- [`all-requests.csv`](../artifacts/two-node-phase5-evidence/all-requests.csv)：192 条正式请求合并表，适合直接查看 TTFT/TPOT。
- [`all-requests.jsonl`](../artifacts/two-node-phase5-evidence/all-requests.jsonl)：保留完整文本、usage 的合并记录。
- `1P_1D/`、`1P_1pD/`：各自的请求 CSV/JSONL、汇总、逐单元 status/metrics/路由检查及 `raw/` 下的请求 JSON、SSE、带相对时间的事件和解析响应。
- [`verification.json`](../artifacts/two-node-phase5-evidence/verification.json)：逐请求、原始证据、路由、P/D 计数和配对核查。
- [`README.md`](../artifacts/two-node-phase5-evidence/README.md)：其余字段和证据文件说明。

这些汇总和验收 JSON 由本次新增脚本计算；原始 SSE/模型回答、API usage 和 Prometheus 指标来自真实服务。阶段三预留的容器 `results/` 挂载不是本次客户端结果位置。

## 路由与配对核查

所有单元的路由增量均符合预期。正式请求中，PD 的 P/D 各处理 96 次；静态 PPD 的 P 处理 48 次、D 处理 96 次。两种模式的第一轮均经 P，静态 PPD 第二轮均为 `ppd_direct_` 响应；后端请求数、prompt token 数、输出 token 数与逐请求表一致。D 本地 prefix cache 和外部 KV 命中计数均有实际增量。预热请求单独排除于这些计数。

跨模式 96 对请求中，payload 相同 87 对，回答相同 61 对，输入 token 数相同 94 对。48 对第一轮 payload 全部相同；配对明细见 `paired-requests.json`。若第一轮回答在不同批处理/计算路径下不同，第二轮包含各自实际回答，会形成不同历史；因此不自动把所有第二轮视为严格相同输入。

运行日志额外确认：PD 共收到 104 次 28 层 KV（含 8 次预热）；静态 PPD 收到 52 次 28 层 KV、出现 52 次 direct（各含 4 次预热）。两种模式均观察到 NCCL Socket，所扫描的 P/D 运行日志未出现 ERROR、Traceback、CUDA OOM 或 KV missing/mismatch。最终三个 HTTP 检查均通过，两端推理镜像 SHA 完全相同。详见 `runtime-verification.json` 和 `final-health.json`。

客户端与部署相关 28 项本地测试通过，其中新增 4 项使用真实模拟 HTTP/SSE，检查 UTF-8/TCP 分片、首正文计时、结束尾部、HTTP 200 错误、缺少 usage/DONE、截断与错误输出长度。真实推理结果由另行保存的 192 条记录和后端指标验证。

## 过程记录与边界

首次运维检查误用了 Proxy 不存在的 `/health`，改为 `/status` 后继续，没有发送 benchmark 推理。随后首次采集完成 48 条短输入请求，长输入预热第二轮因预热文本过长被客户端上下文检查拦截。已改为复用准备好的输入并替换 session 标识，核对预热预算后，从 P/D 重启重新运行完整两种模式。P/D 重启过程中还出现端口预检 `Address already in use`，均在确认 HTTP/KV 端口可独占绑定后，重新启动同一已退出容器；没有重启运行中的容器或重试推理请求。相应预检/重试日志及 `repair-port-start.py` 保留在证据目录。历史尝试保留在 `startup-attempt1/`、`benchmark-attempt2/`，不进入最终统计。

这是单次、先 PD 后 PPD 的小样本合成负载基线。未随机交错测试模式，未做长时间重复、自然对话质量评测、缓存驱逐后的恢复、高 QPS 压测、RDMA、动态 PPD 或论文性能复现。不要用这些结果推断所有工作负载的加速比。服务健康、日志和镜像/容器状态记录保存在同目录，后续应基于这些基线单独制定网络及更完整负载测试。
