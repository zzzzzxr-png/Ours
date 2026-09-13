# 260903 反推三个 physics 红星（Gamma）

**目的：** 把图上三颗 physics 红星的训练/验证协议从 7 月记录对齐回可跑代码。  
**代码快照：** `/data/zhouxirou/Ours_260903`（不改 `Ours_core`）。  
**环境：** `conda activate ph_model`（`/home/qyb/.conda/envs/ph_model/bin/python`）。  
**GPU：** 只用 0–3。不要用 4；5–7 上有别人的任务。  
**数据：** 245×245，`α=5000`，`β=1600`，clip_high=`32767`。

`/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245/`

新训必须 `--no_resume`。对盘上旧 ckpt 用 `--eval_ckpt`，`--pth_dir` 指到本仓库 `experiments/eval_stars/`，不要写回 7 月目录。

---

## 0. 三颗星

| Hz | 训练协议 | 星上 SNR | ckpt |
|---|---|---|---|
| 0.1 | `260722` dual-context 学 κ，height_mask 5%，md=2，20-group 穷举 val | post **17.4772**（prior 17.3787，groups=20） | `E_47_Iter_6992.pth` |
| 1 | `260719` 固定 κ=100，temporal_mask 1%，md=3，无 mask 一次前向 val | **18.6735** | `E_50_Iter_6992.pth` |
| 10 | 同上固定 κ=100 | **21.4150** | `E_45_Iter_6992.pth` |

旧目录：

- 0.1：`Ours_core/experiments/260722_dual_context_sweep/dual_0.1Hz_height_mask/...`
- 1 / 10：`Ours_core/experiments/prev/260719_gamma_sweep/gamma_{1,10}Hz_kappa100_temporal_mask/...`

1/10 的 `val_metrics.md` 表头是 `SNR_denoised / SNR_noisy / κ`，κ 整列 = 100。  
0.1 的表头是 `SNR_prior / SNR_post / groups`。

---

## 1. 代码对齐（相对当前 Ours_core 的两处分叉）

现码对 `temporal_mask + nll` **一律** dual，即使 `--kappa_mode fixed`。1/10 星因此进不去 7.19 训练循环。

本快照：

| | 0.1 星 | 1 / 10 星 |
|---|---|---|
| 开关 | `kappa_mode=learned_map` → `_use_dual_context=True` | `kappa_mode=fixed` → **不要 dual** |
| 训练 | `dual_axis_context_prior` + masked Gamma-NB NLL | `make_directional_mask_pair` + `gamma_nb_nll_single_target(..., kappa=100)` |
| Val | 20-group 穷举 dual，写 prior/post/groups | 无 mask 一次前向 μ → 固定 κ Gamma-NB posterior mean，**无 flux rescale** |
| 网络 | 2 通道 `LearnedKappaHead` | 1 通道，不能 load 进 learned 头 |

入口：`scripts/train_and_val_posterior_gamma.py` → `training_class_srdtrans_gamma`。

---

## 2. 共享锁（三条 job 相同，只换下行）

| 项 | 值 |
|---|---|
| backbone | `srdtrans_v2` |
| STB | 旧 checkpoint 评估为 `ts`；新训练为 `st` |
| patch | T128 × H128 × W128 |
| overlap | train **0.75**，val 0.5 |
| batch / epoch | 1 / 6000 patch，**50 epoch** |
| Adam | `lr=5e-5`，`β=(0.5, 0.999)` |
| seed | 1024 |
| 物理 | `α=5000`，`β=1600`，`o=0`，`kmax=32`，quant=1.0，clip_high=32767 |
| Headline | `cal_snr_srdtrans`，窗 `[50:350]`，**不用强度缩放**，不用 GT 选盘 |

| fs | sampling | mask | κ | val |
|---|---|---|---|---|
| 0.1 Hz | `height_mask` | 5%，md=2 | learned_map | 20-group 穷举，`--val_patch_batch 2` |
| 1 Hz | `temporal_mask` | 1%，md=3 | fixed 100 | 无 mask 一次前向 |
| 10 Hz | `temporal_mask` | 1%，md=3 | fixed 100 | 无 mask 一次前向 |

---

## 3. 命令

```bash
conda activate ph_model
# 先对三个旧 ckpt 跑一次 val（写到 experiments/eval_stars/，不碰 7 月目录）
bash /data/zhouxirou/Ours_260903/launch_stars.sh eval
# 从零重训三颗星（--no_resume，写到 experiments/stars/）
bash /data/zhouxirou/Ours_260903/launch_stars.sh train
```

`eval` 保持 `ts` 以匹配 7 月旧 checkpoint；`train` 和入口默认值使用 `st`。

对盘结果（本快照 `--eval_ckpt`，2026-09-03，与 7 月 `val_metrics.md` **逐位一致**）：

| 星 | 重跑 | 原记录 |
|---|---|---|
| 0.1 post | 17.4772（prior 17.3787） | 17.4772 / 17.3787 |
| 1 | 18.6735 | 18.6735 |
| 10 | 21.4150 | 21.4150 |

`prior/` 是指向 `Ours_core/prior` 的 symlink。
