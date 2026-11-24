# Phase9 扩展：P热缓存、D冷缓存的32K KV传输

当前状态：2026-11-24实验测量、审计与绘图完成。用户明确授权终止dcn10干扰任务后，向已核验命令的PID3132832发送SIGTERM，确认GPU计算进程列表为空，再开始测量。该任务为`perturbation_view_count_experiment.py`，结束记录保存在`authorized-interference-stop.json`。

本实验用于减少原Phase9中完整Prefill计算对传输测量的影响。它测量的是应用中的KV传输路径及单token请求时延，不能把端到端时延解释成纯网络时间。

## 请求与缓存

- 精确32,768-token输入，输出固定1token、temperature=0、seed=42、ignore_eos=true。
- 所有预热、正式请求及两协议使用同一正文，由真实模型`/tokenize`校准并保存SHA256。
- P端启用前缀缓存。每次部署先做一次同输入冷预热，再做一次热预热；预热后不清P缓存。
- D端明确设置`--no-enable-prefix-caching`，每次都从P加载远端KV，避免重复输入在D本地命中后跳过加载。
- 按当前block_size=16预期：P冷预热计算32,768tokens，热请求计算末尾16tokens；D每次计算末尾1token。最终以调度记录验证，不以预期代替实测。
- 每次应发送并接收28层、1,879,048,192bytes，即1,792MiB/1.75GiB。字节数由真实tensor形状和dtype统计。

## 部署与测量

仍使用既有镜像`ppd-qwen25:source-cu129`和Qwen2.5-7B-Instruct BF16。dcn9=P，dcn10=D，纯`1P_1D`、串行。

使用独立容器`ppd-p9hot-proxy/prefill/decode`，Proxy HTTP10101、ZMQ30101。保留已有`ppd-dcn9-proxy`。P/D HTTP8100/8200、KV14579/14580；端口占用时直接停止实验启动，不结束其他服务。

模型参数：max_model_len=32769、max_num_batched_tokens=2048、max_num_seqs=1、gpu_memory_utilization=0.85、block_size=16、eager、KV传输缓冲区2GiB、host pool2GiB。冷预热允许分块计算。第一块实际P端KV池为101,568tokens，32K冷/热预热均已通过。

模型配置的原生上下文上限为32768。API校验需要为32768输入加1输出预留长度，因此临时实验容器设置`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`并仅将max_model_len设为32769。真实前向的输入位置为0..32767；输出只生成1token，不再将输出送入下一轮前向，不计算超出原生范围的位置。没有修改模型配置文件或启用额外RoPE缩放。

Socket使用`NCCL_NET=Socket`、`NCCL_IB_DISABLE=1`；RoCE使用`NCCL_NET=IB`、`NCCL_IB_DISABLE=0`、HCA`rocep202s0f0:1`、IPv4 GID3、RoCEv2；两者均走ens7f0np0。当前GDR关闭，RoCE经主机内存中转。

四个独立部署块顺序为Socket→RoCE→RoCE→Socket，每块5次正式请求。每协议10次，共20正式请求；另8次冷/热预热单独保留，不并入正式统计。

每次保留原请求、完整SSE、事件时间戳、响应/失败、调度器实际处理token数、真实TX/RX层数及字节。每块保留日志、metrics、GPU状态、网卡和RoCE累计计数。HTTP返回后还需核验所有异步TX/RX遥测已完成，再导出调度记录并清理。

## 指标边界

- 客户端TTFT/E2E：包含入口、调度、P端少量重算、KV提取/传输/装入、D端首token计算和响应。
- TX/RX跨度：从首层相应传输事件到最后一层同步调用完成，包含控制消息、分配、层间间隔等软件开销。
- NCCL同步调用累计：原28次send/recv及其已有同步等待的累计，接收端也可能等待发送端。
- E2E减RX跨度及其比例仅作同请求描述，不作为互斥阶段分解；不跨节点相减绝对时间戳。

沿用Phase9只增加计时的单Python模块，不新增CUDA同步；逐层模型接口不变。仅向新建停止状态容器复制该单模块，不挂载覆盖镜像编译目录，不修改镜像或仓库vLLM源码。调度观察器只在内存保存记录，测量完成后通过导出信号写文件。

## 文件与运行

- 原始证据：`artifacts/two-node-phase9-kv-transport-evidence/extension-hot32k/`。
- 操作入口：`scripts/benchmark/phase9_hot32k.py`；`prepare`冻结计划和源码，`run`执行完整矩阵，`inventory`只读查看，`cleanup`仅清理带本实验标签的容器。
- 运行观察器：`scripts/benchmark/phase9_hot32k_runtime.py`。
- 验收/绘图：`scripts/benchmark/phase9_hot32k_analyze.py`。全部验收通过后才生成GitHub结果快照与PNG/SVG/PDF。

使用既有`/workspace/ppd-validation-env/bin/python`执行操作脚本，`/workspace/ppd-paper-tools/bin/python`执行验收/绘图。正式运行前必须确认两端GPU可用；未经用户明确允许不使用`--allow-shared-gpu`。

## 启动修正记录

1. 初始33792长度被模型配置校验拒绝，尚未发送请求。记录保存在`startup-attempt1-context-limit/`；按上述仅增加1输出预留修正后重建。
2. 第一块5次Socket正式请求完成后，切换部署时严格端口绑定检测被旧连接回收状态阻挡；未发送第二块请求。记录保存在`startup-attempt2-port-rebind/`。增加SO_REUSEADDR和有界等待后继续未完成的部署块，第一块有效数据保留，没有重跑或筛选。
3. 原始运行源码在`source-code/`，续跑操作脚本在`resume-source-code/`，各自保存SHA256；容器内计时及调度模块没有因续跑而改变。

## 最终结果与验收

20/20正式请求、8/8冷/热预热成功，228/228项审计通过。每条均匹配完整API ID与engine的`-0`后缀、真实28层TX/RX计数和调度记录。P正式请求每次计算16tokens，D每次计算1token；D的computed_before为32767，不计算原生范围之外的位置。

| 指标（ms，中位数） | Socket/TCP | RoCE/RDMA | RoCE降低 |
|---|---:|---:|---:|
| 客户端TTFT | 876.07 | 471.05 | 46.2% |
| 客户端E2E | 877.37 | 472.36 | 46.2% |
| 发送跨度 | 576.20 | 177.36 | 69.2% |
| 接收跨度 | 581.17 | 176.44 | 69.6% |
| NCCL发送调用累计 | 420.06 | 161.67 | 61.5% |
| NCCL接收调用累计 | 578.30 | 171.32 | 70.4% |

RoCE端到端中位数降低46.2%，接收跨度降低69.6%。两协议每请求E2E减接收跨度的中位数分别约297/296ms，说明即使P已热，客户端时延也不等于纯传输时间。差值仅为同请求时长的描述，不能据此归因为某个互斥阶段。

四块每端GPU KV池均为101,568tokens。正式请求后的显存快照为P端23,335MiB、D端22,745MiB；这是采样值，不声称覆盖瞬时峰值。全部测量无抢占、OOM或NCCL WARN。调度记录最大观察开销26.258微秒，没有新增CUDA同步。

结束时两端GPU计算进程列表为空，三个临时实验容器均已删除。原有`ppd-dcn9-proxy`的ID、StartedAt、Config与实际开跑前一致。

结果图、独立重画脚本及精简数据在[结果目录](../results/figures/phase9/extension-hot32k/README.md)。正式绘图审计源码另冻结于`postprocessing-source/`，其SHA256及原始/续跑源码均保留。原始数据没有按期望收益筛选；两次启动故障均发生在相应部署的请求之前，保留留档。

本扩展阶段完成后暂停，不启动后续实验。
