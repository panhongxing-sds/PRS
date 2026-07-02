# 磁盘清理报告

**扫描时间**: 2026-07-02  
**主机路径**: `/root/autodl-tmp`（数据盘约 200G，已用 148G，可用 53G）  
**系统盘** (`/` overlay 30G): 已用 22G，可用 **8.2G** — pip 缓存与 `/tmp` 占用系统盘，优先清理可缓解根分区压力。

## 总体占用摘要

| 路径 | 大小 | 说明 |
|------|------|------|
| `/root/autodl-tmp/panda-outputs` | 88G | 实验主输出 |
| `/root/autodl-tmp/panda-models` | 40G | 本地模型权重 |
| `/root/autodl-tmp/archive` | 12G | 2026-06-15 全量备份 tar.zst |
| `/root/autodl-tmp/PANDA` | 7.9G | 代码仓（其中 `.tokur_venv` 7.5G） |
| `/root/.cache/pip` | 8.9G | pip HTTP 缓存（在系统盘） |
| `/tmp` (pip-unpack-*) | 12G | pip 安装临时目录（在系统盘） |
| `/root/miniconda3` | 8.2G | Conda 基础环境 |
| `/root/.cursor-server` | 995M | Cursor 远程服务 |

---

## 大项清单（>1GB 或显著可回收）

| 路径 | 大小 | 可完全删除？ | 原因 | 删除命令 |
|------|------|--------------|------|----------|
| `/tmp/pip-unpack-*`（约 396 个目录） | **~12G** | **是（强烈推荐）** | pip 安装失败后遗留的解压临时目录，非项目产物；位于系统盘，删除后不影响 PANDA/panda-outputs/panda-models | `find /tmp -maxdepth 1 -type d -name 'pip-unpack-*' -exec rm -rf {} +` |
| `/root/.cache/pip` | **~8.9G** | **是（推荐）** | 仅加速重复 `pip install`；删除后下次安装会重新下载 wheel/索引页 | `pip cache purge` 或 `rm -rf /root/.cache/pip/http-v2 /root/.cache/pip/http` |
| `/root/autodl-tmp/archive/panda_full_20260615.tar.zst` | **~12G** | **条件性** | MANIFEST 表明内容为 `PANDA/` + `panda-outputs/` + `logs/` 的 2026-06-15 快照；当前本机仍有完整 `panda-outputs`（88G）与 PANDA 仓。若已确认 OSS/异地备份校验通过（见 `SHA256SUMS_20260615.txt`），该 tar 为冗余副本 | `rm -f /root/autodl-tmp/archive/panda_full_20260615.tar.zst`（保留 MANIFEST/SHA256 小文件） |
| `/root/autodl-tmp/PANDA/.tokur_venv` | **~7.5G** | **否（默认）** | TokUR/GPU 复现用虚拟环境；删除后需按文档重装（耗时长、需网络） | 仅当确定不再跑 TokUR：`rm -rf /root/autodl-tmp/PANDA/.tokur_venv` |
| `/root/miniconda3/pkgs` | **~524M** | **部分** | Conda 包缓存，非运行必需 | `conda clean -a -y` |
| `/root/autodl-tmp/PANDA/.tokur_venv/**/__pycache__` | **~372M** | **是** | Python 字节码缓存，可自动再生 | `find /root/autodl-tmp/PANDA/.tokur_venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null` |
| `/root/autodl-tmp/panda-outputs` | **88G** | **否** | 主表实验 raw runs / 聚合输入；论文与复现核心数据 | **禁止删除**（见下文） |
| `/root/autodl-tmp/panda-models/*` | **40G** | **否** | TFB 与 DeBERTa 等推理权重 | **禁止删除** |
| `/root/autodl-tmp/PANDA/experiments/spurious_consensus` | **~342M** | **否** | 仓库内实验数据（对话上下文要求保留） | **禁止删除** |
| `/root/autodl-tmp/panda-outputs/_oom_probe_llama31_8b` | **~18M** | **是** | OOM 探测小规模输出，非主表 | `rm -rf /root/autodl-tmp/panda-outputs/_oom_probe_llama31_8b` |
| `/root/autodl-tmp/panda-outputs/qaac_api_bench` | **~8.6M** | **条件性** | API bench 产物；若论文不再引用可删 | `rm -rf /root/autodl-tmp/panda-outputs/qaac_api_bench` |
| `/root/.cache/huggingface` | **~13M** | **是** | 体量很小，清理收益极低 | `rm -rf /root/.cache/huggingface/*` |
| `/root/autodl-tmp/PANDA/third_party/TokUR/results` | **~0** | N/A | 仅有空目录结构，无大文件 | 可选：`find .../results -type d -empty -delete` |
| `/root/autodl-tmp/logs/*` | **~368K** | **是** | 历史运行日志，体积小 | `rm -f /root/autodl-tmp/logs/*.log` |
| `/root/.cursor-server` | **~995M** | **否** | 删除会导致 Cursor 远程 IDE 需重新拉取 | 不建议 |
| `/root/autodl-tmp/PANDA/.vllm_venv` | **~23M** | **否** | vLLM 环境很小，保留即可 | — |
| `/root/autodl-tmp/.specstory` | **~28M** | **条件性** | Cursor/SpecStory 本地历史，非 PANDA 产物 | `rm -rf /root/autodl-tmp/.specstory` |

---

## 推荐删除顺序（由安全到激进）

1. **清理 `/tmp` 的 `pip-unpack-*`**（~12G，系统盘，零业务风险）
2. **`pip cache purge`**（~8.9G，系统盘，仅影响后续安装速度）
3. **（可选）删除 venv 内 `__pycache__`**（~372M）
4. **`conda clean -a -y`**（~0.5G）
5. **（可选）删除 `panda-outputs/_oom_probe_*` 等小探测目录**（~18M）
6. **（需人工确认备份）删除 `archive/panda_full_20260615.tar.zst`**（~12G）— 确认 SHA256 与 OSS 一致且不再需要离线 tar 后再做
7. **（仅当放弃 TokUR）删除 `.tokur_venv`**（~7.5G）— 不在默认推荐路径内

---

## 可回收空间估算

| 场景 | 估算可回收 |
|------|------------|
| **保守（强烈推荐）**：步骤 1–2 | **~21G**（主要在系统盘 `/`） |
| **+ venv pycache + conda clean** | **~22G** |
| **+ 已验证冗余的 archive tar** | **~34G** |
| **+ 放弃 TokUR 并重装可接受** | **~41G** |

说明：`panda-outputs`（88G）与 `panda-models`（40G）**不应**计入可回收；若误删需从备份/OSS 恢复，成本极高。

---

## 切勿删除（NEVER DELETE）

| 路径 | 大小 | 原因 |
|------|------|------|
| `/root/autodl-tmp/panda-outputs/` | 88G | 主表多模型多种子实验输出；论文与 PANDA/PANDA 复现的唯一本地完整集 |
| `/root/autodl-tmp/panda-models/` | 40G | 本地 TFB/Qwen/Llama/DeBERTa 权重；重新下载耗时长、占带宽 |
| `/root/autodl-tmp/PANDA/`（源码、configs、`experiments/spurious_consensus`、`.git`） | ~400M+（不含 venv） | 项目本体与版本历史 |
| `/root/autodl-tmp/PANDA/.tokur_venv` | 7.5G | **在未明确放弃 TokUR 前视为关键依赖** |
| 任意 `panda-outputs/maintable_*` 子树 | 各 4–39G | 已完成的 maintable 运行数据 |

---

## 一键保守清理（复制前请确认）

```bash
# 1) 系统盘：pip 临时目录
find /tmp -maxdepth 1 -type d -name 'pip-unpack-*' -exec rm -rf {} +

# 2) 系统盘：pip 缓存
pip cache purge

# 3) 可选：venv 字节码
find /root/autodl-tmp/PANDA/.tokur_venv -type d -name __pycache__ -print -exec rm -rf {} + 2>/dev/null

# 4) 可选：conda 包缓存
conda clean -a -y
```

---

## 扫描方法备注

- `du -h --max-depth=1/2` 于 `/root/autodl-tmp`, `/root`, `/tmp`, `/root/.cache`
- `find ... -size +1G` 于 `/root/autodl-tmp`
- TokUR `results/` 当前几乎为空（失败/未写入大 artifact）
- Docker：未检测到可用 `docker` 占用
- 根分区与数据盘分离：优先清理 `/tmp` 与 `~/.cache/pip` 可释放 **系统盘** 空间，避免 overlay 满导致系统异常

