# 双机阶段三：Socket 1P_1D 真实推理与 KV 传输

2026-11-11 UTC，本阶段完成。dcn9 运行 Proxy/P，dcn10 运行 D；Qwen2.5-7B-Instruct BF16，两端均为单张 RTX 4090。阶段结束后保留三个服务运行，暂停进一步实验。

## 验证结果

- 启动前对最新阶段二源码归档中的 3,657 个文件核对 SHA-256：dcn10 全部一致；dcn9 只有两份阶段说明落后，备份并同步后也全部一致。
- 两端使用 `ppd-qwen25:source-cu129`，镜像 ID 均为 `sha256:ae90422b157e0517caafcb7a4a4756bd2eb845eec7963d7b2f01b166cafce0ad`，没有重建镜像或修改推理源码。
- 两端完整模型加载成功，使用 FLASH_ATTN 后端；P/D 健康接口返回 200。
- Proxy `/status` 注册了 P `192.168.1.9:8100` 与 D `192.168.1.11:8200`，配置为 `1P_1D`，动态 PPD 关闭。
- 一次预热加一组两轮完整历史请求均成功。第二轮回答 `The test color you gave me is blue.`，与第一轮指定颜色一致。
- 两轮测试的 Proxy 计数增量为 total=2、turn1=1、turn2plus=1、pd=2、ppd_direct=0，第二轮仍经过 P。
- 两端 NCCL 日志显示 `NET/Socket`、`Using network Socket`，绑定 `ens7f0np0` 和各自数据 IP，并成功建立两节点 NCCL communicator。
- D connector 对预热及两轮请求分别记录了 28 层 KV 接收（4、5、7 个 block）；与实际请求/生成完成情况相互印证，确认跨机 KV 传输路径已运行。
- 保存的 P/D 日志未发现 Traceback、ERROR、KV 缺失、布局不匹配或 OOM。

## 运行布局

| 节点 | 容器 | 地址 |
|---|---|---|
| dcn9 | `ppd-dcn9-proxy` | HTTP `192.168.1.9:10001`，注册 ZMQ `30001` |
| dcn9 | `ppd-dcn9-prefill` | HTTP `192.168.1.9:8100`，KV `14579` |
| dcn10 | `ppd-dcn10-decode` | HTTP `192.168.1.11:8200`，KV `14580` |

沿用上下文 2048、max seqs 4、block size 16、GPU utilization 0.75、KV 传输缓存 512 MiB、CPU pinned pool 1 GiB、enforce eager 和 prefix caching。Socket 模式关闭 RDMA/GDR，网络延迟模拟为 0。

启动命令依据 `scripts/server/two_node_container.py` 生成，为收集 KV 证据增加 `VLLM_LOGGING_LEVEL=DEBUG`。准确命令保存为证据目录的 `*-command.json`。模型和节点配置挂载只读，未覆盖镜像内的源码/原生扩展。宿主机 `vllm-ppd/artifacts/two-node-phase3-evidence/results/` 是容器预留写入目录，本阶段为空；实际客户端验证结果保存在其上一级证据目录。

当前每张 GPU 占用约 18 GiB，服务继续运行且未设置自动重启策略。本次没有更改其他项目容器。

## 证据与范围

两端 `vllm-ppd/artifacts/two-node-phase3-evidence/` 保存原始响应、路由计数、P/D metrics 前后快照、服务日志和启动命令。综合验收结果为 `socket-1p1d-verification.json`。原始 `socket-1p1d-smoke.json` 只断言响应和路由，KV 结论由综合验收结合 NCCL/connector 日志给出。

本阶段只有一次预热和一组两轮请求，不能作为完整模型精度验证、稳定性压测或性能 benchmark。虽然记录了 PD 模式的缓存指标，仍未验证静态 `1P_1pD` 的第二轮直达和缓存复用。RDMA、动态决策表、吞吐/延迟测试也未执行。

下一阶段：固定相同模型与参数，将 Proxy 切换到静态 `1P_1pD`；按轮采集 P 请求量、D prefix-cache 指标和 KV 接收日志，验证第二轮绕过 P、复用历史缓存并生成正确回答。
