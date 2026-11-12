# Figure 2：同一完整上下文的 Full / Append 对照

本轮按用户确认的定义重新采集：full 和 append 提交完全相同的 16,384-token 历史与 1,024-token 新输入，full 的历史 KV 不可复用，append 复用预热历史。背景为每条请求独立的 128-token 历史。这里使用精确 token ID 构造的合成历史，没有先运行真实多轮问答；历史、完整请求、输出、原始 SSE 和调度记录全部保留。

## 负载与执行

| 角色 | 历史 | 新输入 | 输出 | 本次计算的 prompt token |
|---|---:|---:|---:|---:|
| 背景 decode | 128，独立 KV | 4 | 128 | 初始 4，然后逐 token decode |
| Full 干扰 | 16,384，无可复用缓存 | 1,024 | 1 | 17,408 |
| Append 干扰 | 16,384，预热命中 | 1,024 | 1 | 1,024 |

每次 trial 预热 B 条背景历史及一个独立的干扰历史；三种场景准备过程相同。Full 使用独立 `cache_salt`，阻止复用已预热的历史；append 使用其预热盐。跨 trial 的盐隔离，同 batch/重复的背景输入和 full/append 完整输入相同。预热输出一个 token，其输出不拼入随后请求的合成历史。

B 从 0 到 200，间隔 10；每点重复五次。正 batch 比较 decode-only/full/append，batch 0 仅比较 standalone full/append，decode TPOT 无定义。每次重复随机排列 batch 与场景。背景请求均输出至少 8 token 后才注入干扰，trace 验证真正参与混合步骤的 decode 数量，不以 HTTP 并发数代替 batch。

每个请求使用新连接、连接池容量至少为最大 batch+16，推理期间把原始证据缓存在内存，所有在途请求完成后才写盘；失败不自动重试。观测器测量期间仅记录到内存，run 完成后再导出 trace。

使用既有 `ppd-qwen25:source-cu129`，单 dcn10 RTX 4090、Qwen2.5-7B-Instruct BF16、FA2、eager、prefix cache，block=16、seed=42、GPU utilization=0.85。独立容器 `ppd-figure2-context16k-dcn10`，context limit=18,432、max sequences=201、max batched tokens=17,664，足够让 17,408-token full 与 200 decode 在一个步骤内执行。没有挂载覆盖镜像编译目录。

## 容量验证

服务启动后实际 KV 容量为 62,800 token / 3.35 GiB，低于旧配置的 80,224 token。新的大 prefill token 预算增加了临时显存需求。

200 条背景生成到末尾的保守 block 容量为 `200 × ceil((128+4+128)/16) × 16 = 54,400` token。单个完整干扰请求保守需求为 `ceil((16384+1024+1)/16) × 16 = 17,424` token，两者相加 71,824 是假设它们同时保持完整最终长度的保守上界。实际干扰在背景生成前段到来且只输出 1 token，完成后其不再被活动请求引用的 KV 可被缓存管理器回收；因此这个上界超过容量并不等于实际运行不可行。

恢复后的先导覆盖 B=0/10/200，8 个 trial、1,274 个请求全部通过。B200 full 与 append 均与真实 200 条 decode 混合执行，无抢占；full computed=17,408/cached=0，append computed=1,024/cached=16,384。完整输入哈希一致。容量结论仍以正式全量验收为准，不以预算代替实际证据。

首次先导在用户要求暂停时中断，已完成五条 trial 记录，保留在 `pilot/`；随后使用独立请求命名空间完整重跑 `pilot-resumed/`，不将被中断的数据拼入先导或正式结果。

## 统计口径

输出沿用 event-window 方法：插入前后逐 iteration 曲线、1/5/16/32/64 步固定窗口、整段 127-gap TPOT 和带正负号的累计配对额外延迟。混合步骤定位为 offset=0。按同组、batch、重复、请求序号及输出 token 序号匹配纯 decode；完整 batch 对照不足的步骤列入排除清单。

三个时间尺度分别保存：`model_ms` 为调度返回到模型输出交回的墙钟耗时，`cycle_ms` 为连续 scheduler update 返回时间差，`client_ms` 为该步骤输出 token 对应的客户端 SSE 间隔。后二者包含更多主机、调度和客户端处理开销，均不能称为纯 GPU kernel 时间。

先对同 trial 的背景请求平均，再对五次重复取均值与样本标准差。恢复阈值与 3%/5%/10% 敏感性继续保存，但不用于筛选曲线。所有失败和尖峰保留，不预设应该达到论文的某个比例。完整输入变更后，本组与旧的 full 仅处理 1024-token 实验不能混成同一条曲线。

## 产物

原始证据：`artifacts/two-node-phase8-figure2-context16k-evidence/`。

事件窗口派生结果：`artifacts/phase8-figure2-context16k-window-analysis/`。

结果图及可独立重画快照：[context16k 图目录](../results/figures/phase8/context16k/README.md)。

正式采集、汇总、服务清理及最终验收完成后补充实测结果。
