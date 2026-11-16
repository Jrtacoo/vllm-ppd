# 各阶段实验结果图

从 Phase 7 开始，按阶段保存可上传到 GitHub 的结果图、数据快照和重画代码：

```text
results/figures/
├── README.md
├── phase7/
│   ├── README.md
│   ├── generate.py
│   ├── plot-data.json
│   ├── requirements.txt
│   └── 三组结果图（PNG / SVG / PDF）
└── phase8/
    ├── README.md
    ├── requirements.txt
    ├── history-budget-diagnostic/  # 2×2 因子诊断与图注字面条件对照
    └── context16k/
        ├── README.md
        ├── generate.py
        ├── generate-decode-tpot.py
        ├── generate-amortized.py
        ├── plot-data.json
        └── 六组结果图及补充数据
```

- [Phase 7](phase7/README.md)：真实 KV 经 RoCE 传输、PD / 静态 PPD 的请求延迟对照。
- [Phase 8](phase8/README.md)：16K 历史的 Figure 2 适配实验，以及新增的历史长度／调度预算诊断。比较背景平均 TPOT、瞬时峰值与累计干扰，保留原始统计口径。

后续阶段使用 `phase9/` 等同级目录。每个阶段的 README 记录数据来源、统计口径、重画方法与结论边界。PNG 用于 GitHub 预览，SVG / PDF 用于缩放和导出。

原始请求、SSE、服务日志及完整实验数据保存在项目 `artifacts/` 下。此处的结果图、数据快照和说明可由 Git 跟踪。
