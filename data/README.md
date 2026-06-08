# Data

| File | In git | Notes |
|------|--------|-------|
| `examples.jsonl` | yes | Tiny smoke-test prompts |
| Benchmark JSONL (math500, gsm8k, …) | no | Download via TokUR — see below |

After clone, fetch benchmark datasets:

```bash
bash scripts/download_tokur_datasets.sh
```

Datasets are stored under `third_party/TokUR/datasets/` (tracked in git when small enough).
