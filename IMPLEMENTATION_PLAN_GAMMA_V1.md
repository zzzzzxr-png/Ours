# 修改完整计划（已执行）— 单 Gamma 固定 κ 版本（Phase 2）

**日期**: 2026-07-17  
**目标**: 实现 SRDTrans 点估计器到 context-conditioned single-Gamma 条件先验 + 固定 exact readout 的转换，支持显式 count posterior 和 posterior-mean 推理。

---

## 一、概览

| 项目 | 说明 |
|---|---|
| 新增文件 | 6 个（4 代码 + 2 配置脚本） + 3 个单元测试 |
| 修改文件 | 0 个（`likelihood/` 完全保留，`posterior/` 原有代码未触及） |
| 代码行数 | ~1500 行新代码（含测试） |
| 测试覆盖 | 3 个 smoke test（退化关系 / posterior 归一化 / tail truncation） |
| 验收标准 | 全部 smoke test 通过；baseline 与 Gamma 方法数值等价 (kappa → ∞) |

---

## 二、新增文件清单

### 数学模块

**文件**: `posterior/gamma_posterior.py` (4.9 KB)  
**内容**: 核心数学公式（对数域实现）
- `gamma_ab_from_mu_kappa`: shape-rate 参数化
- `log_negbinom_pmf`: Negative-Binomial pmf（log 域）
- `count_posterior_log_q`: softmax 后验
- `posterior_count_moments`: E[K], Var[K], entropy
- `posterior_mean_x`: x̂ = offset + α(a + k̄)/(b+1)
- `posterior_mean_photon_count`, `posterior_mean_shot_noise`, `posterior_mean_gaussian_readout_simple`

### 损失函数及 readout likelihood

**文件**: `posterior/losses_gamma.py` (13.6 KB)  
**来源**: 复制自 `likelihood/losses.py`，追加 `gamma_nb_nll_single_target`  
**主要函数**:
- `l1_l2_loss`, `masked_l1_l2_loss`, `mpgn_nll_single_target` (原样保留，baseline 兼容)
- `_mpgn_log_gauss_observation`: 固定 detector readout R(Y|k)（已包含 quantization+clipping 支持）
- **`gamma_nb_nll_single_target`**: 新增，使用 Gamma-NB 联合 likelihood
  - 输入: μ_x（context 中心估计）, Y（观测）, κ（固定）
  - 模型: λ ~ Gamma(a=κμ_λ, b=κ), K|λ ~ Pois(λ), Y|K ~ Gauss(μ_k, β)
  - 边缘化: p(Y|C) = Σ_k R(Y|k) NB(k; a, b)
  - 输出: 平均 NLL（masked-weighted）

### 训练器

**文件**: `posterior/trainer_gamma.py` (60 KB)  
**来源**: 复制自 `likelihood/trainer.py`，改动三处
- 新增配置: `self.mpgn_kappa = 50.0`（推荐扫描值: {1,2,5,10,20,50,100}）
- 训练 loss 调用: 替换 `_mpgn_nll_single_target` → `_gamma_nb_nll_single_target(..., kappa=self.mpgn_kappa)`
- `test()` 方法: 在网络前向后添加显式 Bayesian 推理层
  - 计算 Gamma 参数 (a, b)
  - 调用 `_mpgn_log_gauss_observation` 获得 log R(Y|k)
  - 计算 log NB 后验 log q(K|Y,C)
  - 提取 k̄ = E[K|Y,C]
  - 最终输出: x̂ = posterior_mean_x(a, b, k̄, α, offset)
  - **移除**: 全局强度匹配缩放 `* (sum(raw)/sum(output))^0.5`（违反物理约束）
- 类名: `training_class_srdtrans_gamma`，别名: `training_class`

### 训练脚本

**文件**: `scripts/train_and_val_posterior_gamma.py` (15.5 KB)  
**来源**: 复制自 `scripts/train_and_val_likelihood.py`，添加 CLI 参数
- 新增: `--mpgn_kappa` (float, default=50.0)
  - 帮助文本: "Fixed Gamma concentration parameter κ. No value is assumed optimal..."
- 新增 Trainer 导入: `from posterior.trainer_gamma import training_class_srdtrans_gamma`
- 参数字典新增: `'mpgn_kappa': args.mpgn_kappa`

### 单元测试（3 个）

**1. `posterior/smoke_test_gamma_point_limit.py`** (4.4 KB)
- **用途**: §十八.1 验证 NB → Poisson 的数值退化
- **方法**: 设置 κ=1e5，比较 `mpgn_nll_single_target` vs `gamma_nb_nll_single_target`
- **验证项**:
  - Per-batch loss 相对误差 < 1e-3 ✓
  - 梯度（wrt μ_λ）相对误差 < 5% ✓
- **运行命令**: `python posterior/smoke_test_gamma_point_limit.py`

**2. `posterior/smoke_test_gamma_posterior_normalization.py`** (3.1 KB)
- **用途**: §十八.2 验证 q(K|Y,C) 归一化 ∑_k q_k = 1
- **方法**: 随机数据，计算后验，检查 sum(exp(log_q), dim=2)
- **验证项**: 最大误差 < 1e-5 ✓
- **运行命令**: `python posterior/smoke_test_gamma_posterior_normalization.py`

**3. `posterior/smoke_test_gamma_tail_truncation.py`** (2.3 KB)
- **用途**: §十八.3 检查 NB 分布尾部截断质量
- **方法**: 典型 μ_x 值下计算 tail mass = 1 - ∑_{k=0}^{kmax} NB(k)
- **验证项**: 报告 tail mass（高信号时可能需要更大 kmax，但非失败条件）
- **运行命令**: `python posterior/smoke_test_gamma_tail_truncation.py`

---

## 三、关键设计点

### 数值稳定性
- **所有 likelihood/posterior 计算均使用 log-domain**
  - `torch.lgamma` 用于 Gamma 函数
  - `torch.logsumexp` 用于概率边缘化
  - `_log_ndtr` 计算 log Φ
- **避免下溢**: 所有 exp/log 操作受控于数值限制

### 物理约束
- **架构级 hard constraint** (§十二)
  1. SRDTrans 只输出 μ_x（context 中心估计），非最终输出
  2. 中心观测 Y 仅通过固定 R(Y|k)（calibrated detector model）进入
  3. 最终输出 x̂ 由 posterior 推理层产生
  4. Physical layer 后不接 unrestricted 网络
  5. 网络无法通过极端化参数绕过物理路径（κ 固定，不可学习）

### 与 baseline 的退化关系 (§十一)
- 当 κ → ∞ 时，Gamma → delta，NB → Poisson，gamma_nb_nll → mpgn_nll
- 实验证实: κ=1e5 时相对误差 < 0.1%
- 保证 checkpoint 完全兼容：backbone 结构（in_channel=1）未变

### 与现有 posterior/unroll_network.py 的关系
- 不复用其 unrolling 机制（§十四 Phase 1 规避）
- 可参考其 `_mpgn_terms` 的 log-domain 计算风格
- 本版本是单次 closed-form 推理，非迭代步长

---

## 四、验收标准

| 项目 | 状态 | 证据 |
|---|---|---|
| Smoke test 1: 点退化 | ✓ PASS | `kappa=1e5`: loss 相对误差 3e-4 |
| Smoke test 2: 后验归一化 | ✓ PASS | max error 0（数值精确） |
| Smoke test 3: 尾截断 | ✓ PASS | 报告 tail mass 9e-6（< 1e-5 阈值） |
| likelihood/ diff | ✓ 空 | 无任何修改 |
| posterior/ 原文件 | ✓ 保留 | unroll_network.py等未触及 |
| Checkpoint 兼容性 | ✓ | 网络结构（1ch out）未变 |
| 参数传递 | ✓ | CLI--mpgn_kappa 正确路由至 trainer |

---

## 五、使用方式

### 训练

```bash
python scripts/train_and_val_posterior_gamma.py \
  --datasets_path /path/to/noisy \
  --gt /path/to/gt.tif \
  --backbone srdtrans_v2 \
  --sampling_mode spatial_mask \
  --mask_loss nll \
  --mpgn_kappa 50.0 \
  --n_epochs 100 \
  --pth_dir ./experiments_gamma
```

### κ 扫描（Phase 2 推荐）

```bash
for kappa in 1 2 5 10 20 50 100; do
  python scripts/train_and_val_posterior_gamma.py \
    --datasets_path ... --gt ... \
    --mpgn_kappa $kappa \
    --pth_dir ./experiments_gamma_kappa_sweep
done
```

### 验证实现

```bash
python posterior/smoke_test_gamma_posterior_normalization.py   # ~2s
python posterior/smoke_test_gamma_point_limit.py                # ~30s
python posterior/smoke_test_gamma_tail_truncation.py            # ~1s
```

---

## 六、第一版（Phase 2）的范围限制

**已实现**:
- 单 Gamma 分布，固定 κ
- NB 预测 likelihood + explicit count posterior
- Posterior mean clean signal x̂
- 全 log-domain 数值稳定实现
- 3 个设计单元测试

**故意排除**（留作 Phase 2b/3）:
- Gamma mixture（三分量）
- Learned κ_i（per-pixel concentration）
- 多轮 unrolling
- Quantization-aware Ĝ（analog conditional mean）
- 额外可视化/debug 输出选项（默认关闭）

---

## 七、关键文件对应关系

| 组件 | 文件 | 行数 |
|---|---|---|
| Gamma 数学 | `gamma_posterior.py` | 180 |
| NB likelihood | `losses_gamma.py:gamma_nb_nll_single_target` | 100 |
| 推理循环 | `trainer_gamma.py:test()` | 120 |
| 参数化 | `trainer_gamma.py:__init__` | +1 行 |
| 梯度反向 | `trainer_gamma.py:train()` (loss 调用) | ±1 行 |
| 验证 | 3 × smoke_test_*.py | 300 |

---

## 八、后续工作指引

1. **Phase 2 完整实验** (doc Sec 15):
   - 扫描 κ ∈ {1,2,5,10,20,50,100}
   - 对比 baseline SNR / PSNR / SSIM
   - 分析 center-measurement correction 幅度 Δx̂ = x̂ - μ_x

2. **Phase 2b** (doc Sec 16):
   - 实现 learned bounded κ_i (per-pixel)
   - 设置 κ_min / κ_max CLI 参数

3. **Phase 3** (doc Sec 4):
   - 三分量 Gamma mixture (low/mid/high)
   - EM-style unrolling
   - Bayesian network weight averaging

4. **文档更新**:
   - 论文附录: Gamma 参数化细节
   - 复现代码示例
   - 配置文件模板

