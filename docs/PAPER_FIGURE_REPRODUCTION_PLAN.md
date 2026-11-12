# 论文 Figure 复现评估与建议计划

评估日期：2026-11-11。依据项目根目录 `PPD.pdf`（arXiv:2603.13358v2，19 页）、当前源码和阶段五至七证据。本文件是阅读结论与待实施计划，不代表新增实验已完成。用户尚未指定目标 Figure 编号。

## 结论与复现范围

阶段七足以证明双机 Socket/RoCE 真实 KV 传输、静态 PD/全 AP-to-D 两轮路由与缓存复用可用，并提供这两个静态端点的小样本延迟基线。现有数据不足以复现论文任何一张完整实验图，也不足以验证动态策略的收益。

论文主实验为 4×H100 80GB、NVLink、Llama-3.1-8B；当前为两台各一张 RTX 4090 24GB、100G RoCE、Qwen2.5-7B-Instruct。因此当前可做同方法下的机制与趋势复现，报告必须标注模型、硬件和拓扑变化。不能承诺相同绝对数值、加速比例、容量或 QPS。

如果重点是核心 P/pD 动态选择，建议将 Figure 6 作为首个动态系统复现目标，先用 Figure 2 的小规模干扰实验与静态负载扫描提供校准，再做 Figure 4/11 的系统比较。若用户选择其他图，按下表重新收敛实验范围。

## Figure 对照

| Figure / PDF 页码 | 原图内容与主要条件 | 现有证据和缺口 |
|---|---|---|
| 1 / p3 | P99 TTFT–TPS Pareto，10k 输入、100 输出、5 轮，QPS 0.5/1/1.5，多种四卡配置 | 当前只有两个静态端点、2 轮、context 2048；缺长上下文、多配置、吞吐和尾延迟样本 |
| 2 / p4 | 单 H100/Llama-8B，decode-only、加 1 次 full-prefill、加 1 次 append-prefill；横轴 decode batch size，纵轴平均 decode TPOT；caption 指两类 prefill 均处理 1024 token | Phase7 未隔离同卡干扰，也未控制 decode batch；适合作为第一项机制实验，需核对实际缓存命中、未命中 token 和执行重叠 |
| 3 / p5 | Replica、PD、动态 PPD 架构示意 | 不是数值实验；当前已验证前两种静态路径，动态尚未启用 |
| 4 / p8 | ShareGPT/WildChat，三个 P:D 比例，PD 与动态 PPD 的平均 query latency–QPS；SR<95% 标记退化 | 缺真实数据、开环 QPS、多轮、动态引擎和多种四卡比例；可先做双卡 1P_1D 的对应实验 |
| 5 / p9 | 1P_3D、QPS=1、500 WildChat 对话；在 NVLink 上注入延迟模拟四档网络，比较 T2+ TTFT/TPOT、E2E | Phase7 比的是 Socket/RoCE 两种后端，并未构造不同带宽；没有动态或 WildChat；不能等同此图 |
| 6 / p9 | 1P_3D，500 条 prefill-heavy 多轮会话，来自 ShareGPT/WildChat；QPS 8/16；静态 PD 与 wtpot=6/3/1 的 TTFT–TPOT 权衡及 D-local 比例 | 缺覆盖权衡区间的本机离线表、动态路由、权重扫描、真实混合负载及开环 QPS。最直接检验核心动态选择 |
| 7–8 / p16–17 | 4 个并发 prefill 的干扰、2K–64K 上下文敏感性 | 属于 Figure 2 的扩展，当前上下文与并发参数不覆盖 |
| 9 / p18 | 三种负载类别 × 三档 QPS 的 P99 TTFT–TPS Pareto | 缺负载与配置网格、P99 样本和吞吐测量 |
| 10 / p18 | 2/4/8/16 轮及 8B/14B/30B 模型扩展，含 Replica | 当前只测 2 轮、单一 7B 模型，缺 Replica；需先评估单卡 BF16 显存约束 |
| 11 / p19 | WildChat 上静态 PD、静态 x=1、动态 PPD 的 E2E–QPS 三方对比，配合 Table 3 分解 TTFT/TPOT/SR | 当前只有前两种模式，且为合成闭环请求；缺动态、真实多轮和受控到达率 |

## 为什么 Phase7 不能直接画成论文图

1. 两种静态端点没有动态决策。论文 Algorithm 1 先测本机 x=0/x=1，再离线构表，在线按 workload/context/QPS 查表。
2. 闭环对话并发 1/4 不等于固定 QPS。Phase7 请求完成后才发后续请求，不能作为 QPS 8/16、拥塞或持续负载的证据。
3. 只有 short/long 两类，第一轮输入 256/768、输出 64/128；不足以覆盖论文 decode-heavy、balanced、prefill-heavy 的决策边界。每条件每轮只有 12 个正式样本，尤其不足以稳定描述 P99。
4. 本地缓存命中在 PD 与静态 PPD 下接近，不能替代干扰、路由决策或真实网络字节测量。connector 的 external hit 是调度器统计的外部补足 token，不是实际线上流量。
5. Figure 5 的模拟网络档位以 GB/s 为单位，实际网卡是 Gbit/s；同一 100G NIC 上切换 Socket/RoCE 不等于 NVLink/NDR/HDR/100GbE 带宽扫描。若在当前 RoCE 上做模拟，只能模拟比实测链路更慢的目标，不能声称模拟出了更快的 NVLink。

## 已发现的源码适配点

- `ppd/dcn9.py` 的双机 Proxy 启动器尚未暴露动态启用、离线表路径和权重参数。核心 Proxy 有 `--enable-ppd-mode`，不能只在静态 `1P_1pD` 配置上加开关就认为动态已生效。
- 动态模式应在 `1P_1D` 上启用，使 D 接受每请求 P/pD 决策；静态 `1P_1pD` 仍单独保留为 x=1 对照。
- 决策引擎支持 `1P_1D` 到 `1P_1pD` 的名称回退，但现有离线加载器需要 comprehensive 格式、workload/QPS 文件；Phase7 的 CSV/summary 不能直接当作校准表。
- 短输入 bypass 默认 512；Proxy 用字符数/4 估计新输入。Phase7 的 short/long 第二轮估算分别约 75/467，均落入 bypass，直接开启动态很可能全部选 pD。必须记录实际 token、估计 token、命中表格、score 和 decision reason。
- 缺数据时引擎默认选 PPD，不能把 fallback 当成有效学习到的决策。需统计表格覆盖率，区分 lookup/bypass/fallback，并保留两种分支的真实执行证据。
- 当前 Proxy 的 QPS 特征是启动以来累计请求数/时间；它与实验的对话到达率并不天然一致，预热和空闲也影响此值。需明确对话/请求 QPS 定义，隔离预热，核对实际查表所用 QPS。
- 原 `interference_benchmark.py` 可作为起点，但 header 对 append 输入写 128、实际配置为 1024；并发客户端请求数不能直接视为引擎实际 decode batch。需保存调度/批处理证据。它目前主要保存批次汇总，不能满足逐请求原始 SSE 保留要求。
- 原单卡启动脚本包含全局 `pkill`，计划使用限定名称的独立容器执行干扰实验，不照搬其清理行为。

## 建议的分阶段实施

### A. Figure 2 的本机干扰预实验

使用现有运行镜像与 Qwen 模型，在一张卡上运行独立、无 connector 的单实例服务。完整保留现有服务启动命令，按计划暂停该卡模型服务后执行，结束恢复。

先验证模型及 KV 内存预算，context 提升到能容纳约 2048 cached + 1024 append + 输出的范围（例如 4096，最终以分词和显存检查为准）。比较纯 decode、加入一个 cold full-prefill、加入一个有缓存 append-prefill。按原文/现有脚本明确新计算 token 和缓存 token，禁止用实际更短的 append 冒充 caption 的同规模输入。

先测小批量候选 1/2/4/8/16，依据真实调度与显存扩大；不预设 4090 能复现 H100 的 batch=200。确保 prefill 到达时 decode 已经在生成，记录实际重叠。每点至少 5 轮独立重复，随机/交错场景顺序，保存每请求 TTFT/TPOT、带时间 SSE、usage、缓存命中与实际 batch 信息。出均值、误差范围及相对纯 decode 的退化比例，不预设应得到论文 48%/2%。

交付单卡对应图、原始数据与设置说明，完成后暂停汇报。

### B. 建立双机动态选择的本机离线表

固定当前 RoCE，测试 PD 与静态 x=1。在显存/上下文允许范围内，覆盖短/长历史和 decode-heavy、balanced、prefill-heavy 负载，追加输入必须同时覆盖 bypass 阈值上下两侧。先用小范围开环到达率扫描确定可运行区域，再选低、中、接近饱和的负载点；不能直接照搬 H100 的 QPS 8/16。

同一个测试点共用相同输入构造、到达时间和随机种子。输出长度控制与历史构造规则预先声明。采集均值、P50/P95/P99、SR、实际吞吐与请求数；每点的样本量由试运行确定，正式点目标至少数百条请求并做独立重复，避免以 12 个样本描述尾延迟。保存失败与排队，不能只保留成功请求。

训练校准与评估使用独立会话/随机种子。离线表仅使用本机配置、实际吞吐区域和已验证的 token/到达率口径；保存每格 x=0/x=1 的 TTFT/TPOT、样本数、方差和 score。

交付数据覆盖与权衡分析，判断是否存在动态选择空间后暂停。若所有测点都偏好 pD，应如实报告并扩展有依据的负载范围，不能人为制造获胜结果。

### C. Figure 6 的双机对应实验

为双机启动器增加动态参数，验证首轮固定 P；第二轮及以后确实存在 lookup 决策。用校准后的 prefill-heavy 多轮集，固定 wttft=1，先按论文 wtpot=1/3/6，加上静态 x=0 和 x=1 两个端点。

QPS 选本机扫描确定的两档压力，样本预期按论文 500 会话规模，在 pilot 后明确。逐请求保存特征、score、选择、原因、延迟、SSE、缓存及实际 P/D 调用。绘制 T2+ 平均 TTFT–平均 TPOT，标注实测 D-local 比例、重复误差和吞吐/SR。不能硬编码论文的 20%/50%/95% 本地比例；如 1/3/6 不形成边界，可另加明确标注的本机权重扫描。

### D. Figure 4/11 的系统验证

在独立 ShareGPT/WildChat 多轮样本上比较静态 PD、静态 x=1、动态 PPD，扫描本机开环 QPS。统一超时与成功率口径（Figure 11 使用 30s 超时，现有客户端默认 180s 不可混用），同时报告各轮及整体 E2E、TTFT、TPOT、P99、吞吐、失败原因、D-local 比例。

两卡只报告 1P_1D 对应面板；原图的三种四卡 P:D 配比需要更多独立 GPU，不能通过同卡多进程宣称等价复现。失败或不同趋势均保留。

## 本轮产物

- 论文：[PPD.pdf](../PPD.pdf)。
- 文本提取及部分 Figure 所在页渲染：`artifacts/paper-review/`。
- PDF 阅读器使用独立 `/workspace/ppd-paper-tools` 环境；运行镜像、模型服务和阶段七证据未改动。
