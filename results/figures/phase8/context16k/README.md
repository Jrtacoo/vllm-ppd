# Figure 2：16K 历史的同上下文 Full / Append 实验

单 RTX 4090 / Qwen2.5-7B。Full 与 append 提交相同的 **16,384-token 历史 + 1,024-token 新输入**；full 冷缓存重算全部 17,408 token，append 复用历史，只计算新增的 1,024 token。背景请求拥有独立的 **128-token 历史 + 4-token 新输入**，各生成 128 token。历史为合成 token 序列，非先前真实问答的记录。

B=0/10/20/.../200，每点五次重复。B0 没有背景 decode，仅测独立 prefill，不赋予 decode TPOT 或 slowdown 数值。单次干扰在所有背景请求至少生成 8 token 后插入，actual batch、缓存命中和计算 token 数由 scheduler trace 验收。

方法、容量与异常记录见[实验报告](../../../../docs/TWO_NODE_PHASE8_CONTEXT16K.md)。历史语义说明与旧实验分开，不改写旧图。

## 图表

- `prefill-cost`：从 B0 到 B200 的 prefill／混合步骤模型墙钟，以及 B0 独立请求 TTFT。
- `event-timeline`：B10/100/200 插入前后逐步耗时，分别展示完整引擎 iteration 与客户端 token 间隔。
- `window-dilution`：B10～200 的混合步骤单间隔、前五个间隔及整段 127 个间隔的相对变化，各面板纵轴不同。
- `cumulative-delay`：从插入步骤开始累加带正负号的配对额外延迟，保留后续尖峰。

每类提供 PNG/SVG/PDF，误差为五次 trial 的样本标准差，不是置信区间。请求、token 和 iteration 不视作独立重复。

## 重画

```bash
pip install -r results/figures/phase8/requirements.txt
python results/figures/phase8/context16k/generate.py --from-snapshot
```

使用 `--focus-batches 20 80 160 --output-dir /tmp/context16k-other-batches` 可以重画其他 batch 的详细逐步图；概览图仍覆盖全范围。

原始请求/SSE、服务日志和 trace 保存在 `artifacts/two-node-phase8-figure2-context16k-evidence/`；逐 trial/步骤 CSV、恢复阈值、排除清单及输入哈希在 `artifacts/phase8-figure2-context16k-window-analysis/`。GitHub 仅需本目录快照和绘图代码即可重画，不需要传全部原始证据。

正式验收后补充结果与图片。
