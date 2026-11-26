# Phase11：GPU 资源时间线

单张 RTX 4090、Qwen2.5-7B BF16；模型权重驻留显存。背景请求独立 256 历史 + 32 新增，输出 128；干扰 2048 历史 + 128 新增，输出 1；每次只插入一次干扰。六种 batch、三场景、各五次，共 90 个正式 trial。

[资源与时延汇总](resource-summary.png)：全部 5 次重复的均值 ± 样本标准差。普通 decode GPU step、整段背景 TPOT、干扰期间 SM 活跃度和显存读带宽分别展示。整段 TPOT 包含每个背景请求全部 127 个输出间隔，不能代替事件窗口延迟。

| 背景 batch | 整段资源时间线 | 前向放大图 |
|---|---|---|
| 10 | [PNG](resource-timeline-b10.png) | [PNG](forward-zoom-b10.png) |
| 50 | [PNG](resource-timeline-b50.png) | [PNG](forward-zoom-b50.png) |
| 100 | [PNG](resource-timeline-b100.png) | [PNG](forward-zoom-b100.png) |
| 150 | [PNG](resource-timeline-b150.png) | [PNG](forward-zoom-b150.png) |
| 200 | [PNG](resource-timeline-b200.png) | [PNG](forward-zoom-b200.png) |
| 250 | [PNG](resource-timeline-b250.png) | [PNG](forward-zoom-b250.png) |

时间线和放大图固定展示第 3 次重复，不按结果挑选，不将错位的峰值平均。每张图都提供同名 PNG、SVG 和 PDF。

- 整段图：三行分别是 Decode only、Append、Full；左侧 SM Active，右侧显存读取/写入带宽。横轴从各场景首个背景 GPU kernel 起算。100 μs 原始采样以 5 ms 时长加权分箱展示，背景历史预热和磁盘模型加载不计入。阴影标记实际干扰 prefill 的 GPU step。
- 放大图：保留 100 μs 原始采样，每行使用自己的横轴尺度。虚线界定实际 execute kernel 边界，顶部彩条是模型主体边界；Full 的两个 prefill 分块一并显示。普通 decode 取同次重复、接近 Append 事件背景输出位置的一步；Full 事件可能稍后被调度，未将两者视为完全同时发生。
- 纵轴都是 Nsight 定义的百分比。SM Active 不是 occupancy 或 FLOPS 利用率；DRAM read/write 是 GDDR6X 带宽，不是显存占用，也不是 RoCE。曲线无法直接区分权重流量与 KV 流量。

`plot-data.json` 保存 90 个 trial 的标量统计和所选 18 条时间线的完整前向元数据；`snapshots/` 保存这 18 条时间线的原始 100 μs 采样。全部 90 次原始曲线、Nsight 报告、SQLite、逐请求 request/SSE/response、调度 trace 与验收数据均在项目 artifacts/phase11-resource-timeline-evidence/，不放入 Git。

复画：在本目录运行 `python generate.py`（需要 NumPy、Matplotlib）。报告和先导开销说明见 [Phase11 文档](../../../docs/TWO_NODE_PHASE11.md)，采样完整性和计数见 [verification.json](verification.json)。Nsight trace 会带来测量开销，先导每条件仅一次；本图属于开启采样的观测值，未扣除假定固定开销。

B=10 的 Append 第 1 次重复包含约 35.16 ms 的 GPU kernel 间隙：该步首尾跨度 60.43 ms，kernel 执行区间并集约 23.40 ms；第 3 次重复相应为 24.78 / 23.86 ms。具体间隙来源未隔离。该点保留在全部 5 次的均值和标准差中，因此小 batch 的大误差条不能直接归因于 Append 计算本身。
