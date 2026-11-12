# 各阶段实验结果图

从 Phase 7 开始，按阶段保存可上传到 GitHub 的结果图及说明：

```text
results/figures/
├── README.md
├── phase7/
│   ├── README.md
│   ├── generate.py
│   ├── plot-data.json
│   ├── requirements.txt
│   ├── all-requests-latency.{png,svg,pdf}
│   ├── matched-turn2-comparison.{png,svg,pdf}
│   └── socket-roce-comparison.{png,svg,pdf}
└── phase8/
    ├── README.md
    ├── generate.py
    ├── plot-data.json
    ├── requirements.txt
    └── figure2-interference.{png,svg,pdf}
```

后续阶段使用 `phase9/` 等同级目录。每个阶段的 README 记录图的用途、数据来源、统计口径、生成方法和结论边界。

优先提供 PNG 供 GitHub 直接展示，可同时保存 SVG 或 PDF。文件名使用能表达内容的英文小写名称。Phase 7 的三张汇总图、预览与复现方法见 [phase7/README.md](phase7/README.md)。

Figure 2 的单 RTX 4090 / Qwen2.5-7B 干扰适配实验见 [phase8/README.md](phase8/README.md)，包括一张双面板汇总图及独立重画所需的数据快照。

扩大 decode batch 到 250 的共享历史适配曲线，以及独立历史 batch 32 对照，见 [phase8/large-batch/README.md](phase8/large-batch/README.md)。

原始请求、SSE、服务日志及完整实验数据继续保存在项目 `artifacts/` 下。此目录中的图片及说明可由 Git 跟踪，其他运行结果仍按现有规则忽略。

插入前后逐步耗时、不同平均窗口与累计额外延迟的后分析，见 [phase8/event-window/README.md](phase8/event-window/README.md)。

Full/append 提交相同 16K 历史加 1K 新输入、独立背景 KV 到 batch 200 的新实验见 [phase8/context16k/](phase8/context16k/README.md)。
