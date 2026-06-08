# 论文表格索引（PRS ASE，无 AIME24）

数据根目录：`/HDDDATA/phx/PRS/outputs/ase_full/`  
模型：TFB-Qwen2.5-3B-Instruct（`ase_full`）

## 主结果（200 样本规模，待扩至 400）

| 数据集 | N (raw) | 目标 N | TokUR EU AUROC | TW-ASE AUROC |
|--------|--------:|-------:|---------------:|-------------:|
| Minerva | 200 | 272 | 0.736 | 0.808 |
| MATH-500 | 200 | 400 | **0.755** | 0.875 |
| GSM8K | 200 | 400 | 0.857 | 0.967 |
| DeepScaler | 200 | 400 | （见 `table_extended.md` / 扩样后刷新） | — |

> **TokUR MATH-500**：旧稿/缓存曾误用 **~0.580**（未对齐 clean label / 全量 tokur baseline）；当前 `table_main.md` 以 **0.755** 为准（`recompute_metrics` + `tokur_baseline.jsonl`）。

## 表格文件

| 文件 | 内容 |
|------|------|
| `table_main.md` / `.tex` | TokUR EU + T/W/TW-ASE（三集；扩样后含 DeepScaler） |
| `table_extended.md` / `.tex` | 含 num_clusters 变体 |
| `table_process_dynamics.md` | 过程动力学（含 deepscaler） |
| `table_prefix_trajectory.md` | 前缀轨迹 A_k / T_k |
| `table_overfitting_validation.md` | 交叉验证 / 置换 / Bootstrap |
| `table_representation_attack.md` | 表示攻击 |
| `table_altmass_*.md` | AltMass 分解四表 |
| `table_local_repair.md` | 局部修复（proxy；API pilot 见 `outputs/ase_full/local_repair/`） |
| `results.json` | 机器可读指标 |

## 再生命令

```bash
cd /home/phx/PRS
export PYTHONPATH=src TMPDIR=/HDDDATA/phx/tmp

# 清洗 variants（ASE 前）
bash scripts/clean_api_variants.sh

# ASE 扩至 400（8 GPU，后台）
nohup bash scripts/launch_ase_expand_400.sh \
  > /HDDDATA/phx/PRS/outputs/ase_full/logs/expand_400_master.log 2>&1 &

# 指标与论文表
python3 -m prs.ase.recompute_metrics \
  --out-dir /HDDDATA/phx/PRS/outputs/ase_full \
  --datasets minerva,math500,gsm8k,deepscaler
python3 -m prs.ase.analyze_paper_tables \
  --out-dir /HDDDATA/phx/PRS/outputs/ase_full \
  --datasets minerva,math500,gsm8k,deepscaler
```

## 多模型（Llama 3.2 1B）

- 输出：`/HDDDATA/phx/PRS/outputs/ase_llama32_1b/`
- 队列：`nohup bash scripts/queue_ase_extra_models.sh > /HDDDATA/phx/PRS/outputs/ase_model_queue.log 2>&1 &`
- `MAX_SAMPLES=400`，`MAX_SAMPLES_MINERVA=272`，含 **deepscaler**

## 局部修复 API pilot（MATH-500 × 50）

```bash
export DEEPSEEK_API_KEY=...   # 或 OPENAI_API_KEY + OPENAI_BASE_URL
python3 -m prs.ase.run_local_repair --mode api \
  --pilot-dataset math500 --pilot-n-wrong 25 --pilot-n-high-t 25 \
  --strategies baseline_a0,random_local,Tmax_local,full_SC
```

策略：Prompt 2 前缀续写（非整段重写）；指标：correction yield、damage rate、net gain、token cost。
