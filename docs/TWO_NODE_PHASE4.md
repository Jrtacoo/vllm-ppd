# 双机阶段四：静态 1P_1pD 与本地 prefix cache

2026-11-11 UTC，本阶段完成。保留 dcn9 的 P 和 dcn10 的 D 容器，仅重建 dcn9 Proxy，将路由从 `1P_1D` 切换为静态 `1P_1pD`。三个服务保持运行，暂停进一步实验。

## 实测结果

使用一组新的两轮完整历史对话，第一轮输入含唯一 session 标识和较长参考文本，避免把共同模板的少量缓存命中当成历史复用。两轮均正确回答 `turquoise`。

| 证据 | 第一轮 | 第二轮 |
|---|---|---|
| 输入 token | 458 | 486（包含完整第一轮历史） |
| Proxy 路由计数增量 | ppd_requests +1 | ppd_direct_requests +1 |
| P 成功请求数增量 | +1 | 0 |
| P prompt token 增量 | +458 | 0 |
| D 成功请求数增量 | +1 | +1 |
| D 本地 prefix cache token 命中增量 | +16 | +448 |
| D 外部 KV cache token 命中增量 | +441 | 0 |
| D connector 新增 KV 接收 | 1 次，28 层，29 blocks | 0 次 |

第二轮响应 ID 含 `ppd_direct_`，D 日志出现 `D-direct mode ... local prefix cache only`。结合 P 的请求/token 指标均不变、D 无新增 KV 接收、本地缓存实际命中 448 tokens，确认该请求绕过 P 并复用了本地历史缓存。该结论不只依赖 Proxy 路由名称。

最终 P/D/Proxy 健康接口均返回 200；P/D 的容器 ID、镜像 ID、StartedAt 在切换前后相同，模型进程没有重启。测试期 P/D 日志未发现 ERROR、Traceback、KV 缺失/布局不匹配或 OOM。

## 环境与服务状态

继续使用 `ppd-qwen25:source-cu129`，镜像 ID `sha256:ae90422b157e0517caafcb7a4a4756bd2eb845eec7963d7b2f01b166cafce0ad`。未修改推理源码、模型、CUDA 扩展、依赖或 P/D 参数，没有构建新镜像。

- dcn9：`ppd-dcn9-proxy`，HTTP `192.168.1.9:10001`，路由 `1P_1pD`，动态 PPD 关闭。
- dcn9：`ppd-dcn9-prefill`，HTTP `192.168.1.9:8100`，KV `14579`。
- dcn10：`ppd-dcn10-decode`，HTTP `192.168.1.11:8200`，KV `14580`。名称仍含 decode，作为 pD 的能力由 Proxy 后续轮直达路由使用。

沿用 Socket、prefix caching、BF16、block size 16、上下文 2048、max seqs 4、GPU utilization 0.75、eager 模式。延迟模拟关闭，没有启用 RDMA。

Proxy 首次紧接旧容器退出后启动，端口预检报 `Address already in use` 并退出，尚未处理任何请求；稍后检查没有监听占用，重试同一个 Proxy 容器成功。本次保留失败记录并重试，没有更改端口预检源码。P/D 均未受此重试影响。

## 文件与验收边界

两端宿主机的实际验证文件位于 `vllm-ppd/artifacts/two-node-phase4-evidence/` 顶层：

- `static-ppd-verification.json`：按轮汇总的指标、路由、KV 和最终健康检查，`passed=true`。
- `turn1-request.json`、`turn2-request.json`：精确请求，第二轮包含完整历史。
- `turn1-response.json`、`turn2-response.json`：原始模型回答。
- `before-*`、`after-turn1-*`、`after-turn2-*`：按轮 metrics、Proxy status 和 D 日志快照。
- `proxy.log`、`prefill-test-period.log`：服务日志；Proxy 日志保留首次预检失败记录。
- `proxy-command.json`：当前 Proxy 容器的准确启动命令。

当前 Proxy 继续挂载阶段三预留的 `results/` 到 `/opt/ppd/results`，本次客户端将结果直接保存到阶段四证据目录，预留目录没有生成结果文件。

此阶段只有一组两轮对话，不是精度评测、长时间稳定性测试、压力测试或性能比较；没有测试缓存驱逐后的恢复。RDMA、动态决策表及吞吐/延迟 benchmark 均未验证。

下一阶段建议先在 Socket 下以相同负载采集 `1P_1D` 与 `1P_1pD` 的性能基线，再单独检查 RDMA 条件与传输性能。
