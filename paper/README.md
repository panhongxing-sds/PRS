# ASE 实验论文材料

本目录集中存放实验汇报用的 LaTeX 报告与可再生的结果表格（TokUR 风格：每个数据集三列 **AUROC | AUPRC | ACC***）。

## 目录结构

```
paper/
├── README.md                 # 本说明
├── report/
│   └── ASE_experiment_report.tex   # 主报告（\input 引用 tables/）
└── tables/
    ├── table_main.md         # 主结果：TokUR EU + T/W/TW-ASE
    ├── table_main.tex
    ├── table_extended.md     # 扩展：含 *_num_clusters 变体
    ├── table_extended.tex
    └── results.json          # 机器可读指标
```

## 指标说明

| 指标 | 含义 |
|------|------|
| **AUROC** | 错误检测 ROC-AUC（分数越高越可疑） |
| **AUPRC** | PR-AUC |
| **ACC*** | Top-50% accuracy：按不确定性分数降序取前 50% 样本的正确率（TokUR 协议） |

所有指标在 **Clean label** 上计算（`label_wrong_clean`），实现见 `prs.metrics_tokur.compute_detection_metrics`。

## 再生表格

```bash
cd /home/phx/PRS
export PYTHONPATH=src

python3 -m prs.ase.analyze_paper_tables \
  --out-dir /HDDDATA/phx/PRS/outputs/ase_full \
  --paper-dir /home/phx/PRS/paper \
  --datasets minerva,math500,gsm8k,deepscaler
```

依赖各数据集目录下的 `features.jsonl`（或等价 enriched 源）与 `tokur_baseline.jsonl`。全量三数据集约需 20–25 分钟。

## 编译报告

```bash
cd /home/phx/PRS/paper/report
xelatex ASE_experiment_report.tex
```

报告通过 `\input{../tables/table_main.tex}` 与 `\input{../tables/table_extended.tex}` 嵌入表格。

## 数据路径

- 实验输出根目录：`/HDDDATA/phx/PRS/outputs/ase_full/`
- 可选符号链接：`outputs/ase_full/paper` → 本目录

## 主结果速览（AUROC / AUPRC / ACC*，clean labels）

| Method | Minerva | MATH-500 | GSM8K | DeepScaler |
|--------|---------|----------|-------|------------|
| TokUR EU | 0.736 / 0.836 / 0.531 | **0.755** / 0.616 / 0.840 | 0.857 / 0.401 / 0.990 | （扩样后刷新） |
| TW-ASE | **0.808** / **0.867** / 0.612 | **0.875** / **0.790** / 0.910 | **0.967** / **0.864** / 1.000 | （扩样后刷新） |

> MATH-500 TokUR AUROC 以 **0.755** 为准（非旧缓存 ~0.580）。索引见 `tables/TABLES_summary.md`。

完整数字见 `tables/table_main.md` 或 `tables/results.json`。
