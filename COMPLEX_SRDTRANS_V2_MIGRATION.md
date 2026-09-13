# SRDTrans_v2 → Complex SRDTrans_v2 迁移审计

## 结论

保持 SRDTrans_v2 topology、时空分支、窗口、receptive field 和训练策略不变是可行的，但“只替换 convolution / normalization / activation，不改 attention”的 Version 1 在本代码中不可行。complex feature 会在现有 `nn.Linear`、`nn.LayerNorm`、GELU、Dropout 和 Softmax 处失败；在 attention 前取 `.real()` 或 `.abs()` 虽可运行，却会丢 phase。

因此最小有效版本必须让 encoder、temporal attention、shifted-window spatial attention、FFN 和 decoder 全程保持 complex dtype。它不是新网络，而是把原拓扑中的 real operator 逐一换成已有 complex operator。

## 环境与依据

```bash
conda activate physics
```

当前环境已安装 `complextorch==2.1.1`。搜索审计表明，真正开始实现前应升级到
`complextorch==2.2.0`：2.2.0 才加入完整 attention mask API，并修复
`PhaseSoftMax` 将负 mask 加到复数 score 后取模、反而放大被 mask 位置的问题。

- Complex convolution、normalization 和初始化基础：[Deep Complex Networks](https://arxiv.org/abs/1705.09792)
- Complex layer、Wirtinger calculus 和 normalization：[Theory and Implementation of Complex-Valued Neural Networks](https://arxiv.org/abs/2302.08286)
- modReLU：[Unitary Evolution Recurrent Neural Networks](https://arxiv.org/abs/1511.06464)
- 采用的公开实现：[ComplexTorch](https://github.com/josiahwsmith10/complextorch)
- 备选实现：[complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)。它缺少当前迁移所需的完整 3D、LayerNorm 和 transformer 路径，因此不作为主库。

## Complex Swin 搜索与源码审计（2026-09-10）

| 工作 | 是否有代码 | 审计结论 |
|---|---|---|
| [CVSwinFreq](https://arxiv.org/abs/2309.09352) / [源码](https://github.com/josiahwsmith10/spectral-super-resolution-swin) | 有，GPL-3.0 | 最直接的 complex shifted-window 先例：complex QKV、交替窗口、relative bias、mask、complex MLP/LN 均存在。但实现仅为 1D，使用旧 ComplexTorch API，不能直接移植。 |
| [Building Blocks for a Complex-Valued Transformer](https://arxiv.org/abs/2306.09827) / [公开 PyTorch 实现](https://github.com/lucidrains/complex-valued-transformer) | 有，MIT | 给出 Hermitian similarity、取其实部后 real softmax 的 CAtt，并明确 mask 应在 complex-to-real similarity 之后施加；这是本项目 attention arithmetic 的主要依据。 |
| [ComplexTorch](https://github.com/josiahwsmith10/complextorch) 2.2.0 | 有，Apache-2.0 | `ScaledDotProductAttention(softmax_on="real")` 已实现 Hermitian product、real softmax 和 bool/additive mask；可以复用 attention core，但不能整块替换 SRDTrans window wrapper。 |
| [DCHT](https://arxiv.org/abs/2310.19602) | 未找到作者公开代码 | 论文声称 complex W-MSA/SW-MSA，但没有给出足够精确的 attention/softmax/mask 实现，不能作为代码迁移来源。 |
| [Complex Swin for SMWI](https://arxiv.org/abs/2512.22202) | 未找到公开代码 | magnitude/phase 分支分别提特征后 concat 到共享 Swin，不是全程 complex tensor，违反本项目约束。 |

### CVSwinFreq 不能直接复制的原因

源码逐行检查发现：

1. 使用 `Q @ K.transpose(...)`，不是 Hermitian product `Q @ K.conj().transpose(...)`；论文也明确采用 $QK^T$。
2. 先把 `-100` mask 加到 complex score，再做 `PhaseSoftMax(|score|)`。本地最小验证中，两个相同 score 加 `[0,-100]` 后，被 mask 项的权重约为 `1.0`，未 mask 项约为 `2.7e-43`，语义完全反转。ComplexTorch 2.2.0 已改为把 mask 加在 magnitude logits 上。
3. shifted block 的 reverse-shift 分支对旧变量 `x` 执行 roll，没有使用 `attn_windows` 合并得到的 `shifted_x`，导致 shifted-attention 输出被丢弃。
4. 代码依赖已经删除的 `CVTensor`、`CVLinear`、`cvtorch.roll` 等旧 API，与当前环境不兼容。

因此它只用于证明“complex shifted-window topology 已有公开先例”，不复制其实现；具体 attention arithmetic 使用 Eilers–Jiang CAtt 与 ComplexTorch 2.2.0 的公开实现。

## 当前真实计算图

配置：`T=H=W=128`、`f_maps=[8,16,32,64]`、`embedding_dim=128`、`hidden_dim=512`、8 heads、1 个 ST block。

```text
[B,1,128,128,128]
  encoder: 1→8→16→32→64，四次只沿 T stride-2
[B,64,8,128,128]
  bottleneck projection: 64→128
  temporal transformer: [(B·H·W), T, C]
  spatial Swin transformer: [(B·T), H·W, C]
  bottleneck projection: 128→64
  decoder: 64→32→16→8→1，四次只沿 T upsample，逐级 skip-add
[B,1,128,128,128]
```

当前实例包含 22 个 Conv3d、4 个 ConvTranspose3d、16 个 Linear、9 个 LayerNorm、18 个 LeakyReLU、4 个 GELU、17 个 Dropout、2 层 temporal attention、2 层 spatial window attention。网络没有 BatchNorm，不能为了“complex 化”额外添加 ComplexBatchNorm。

## 逐模块替换

| 原位置 | 原功能 | 替换 | 来源/实现 | wrapper | 物理影响 |
|---|---|---|---|---|---|
| `MainFrame.py:29,39,49,60,157,202` | encoder/decoder/local projection Conv3d | `complextorch.nn.Conv3d` | Deep Complex Networks / ComplexTorch | 否 | 复权重联合混合实部与虚部 |
| `MainFrame.py:181` | temporal upsampling | `complextorch.nn.ConvTranspose3d` | ComplexTorch | 否 | stride、kernel、padding 不变 |
| `MainFrame.py:35,45,56,61,204` | convolution activation | `complextorch.nn.modReLU` | Arjovsky et al. / ComplexTorch | 否 | 调制 magnitude，保留非零响应的 phase |
| `TemporalTrans.py:81,83,143,146` | temporal QKV/projection/FFN | `complextorch.nn.Linear` | Deep Complex Networks / ComplexTorch | 否 | complex feature mixing |
| `SpatioiTrans.py:120,122,175,177` | spatial QKV/projection/FFN | `complextorch.nn.Linear` | 同上 | 否 | window topology 不变 |
| `TemporalTrans.py:121,131`、`SpatioiTrans.py:225,231`、ST wrapper | pre/post normalization | `complextorch.nn.LayerNorm` | Barrachina et al. / ComplexTorch | 否 | joint real/imag whitening，不采用 split norm |
| temporal/spatial FFN activation | GELU | `complextorch.nn.modReLU` | Arjovsky et al. / ComplexTorch | 否 | 不对 real/imag 分别门控 |
| 所有 feature Dropout | real Dropout | 实虚共享的最小 `ComplexDropout` wrapper | 标准 Dropout unit-mask 语义 / ComplexTorch shared-mask 实现模式 | 是 | 一个实数 mask 乘整个 complex unit，保持 surviving feature phase |
| skip、residual、reshape、view、permute、rearrange、cat、roll、pad | topology/data layout | 原样保留 | PyTorch native complex | 否 | 不改变信息和 receptive field |

`MSConvBeforeTrans` 当前配置关闭；若以后启用，只需对其 4 个 Conv3d 和 4 个 activation 使用同一替换，不改变三个分支。

## Attention 的最小合法迁移

### 采用的 attention 定义

采用 Eilers–Jiang 在 ICASSP 2023 给出的 CAtt：

$$
A=\operatorname{softmax}\!\left(\frac{\operatorname{Re}(QK^H)}{\sqrt d}+B+M\right),
\qquad Y=AV.
$$

它不是把 feature 转成 real：$Q,K,V,Y$ 始终为 complex；只有用于形成概率分布的
similarity 是 real。$\operatorname{Re}(QK^H)$ 含有
$|q||k|\cos(\phi_q-\phi_k)$，所以 phase difference 仍参与 attention；real 权重再作用于
complex $V$。论文也明确要求 mask 在 similarity 映射到 real 之后施加。

不采用 CVSwinFreq 的 $QK^T+$ `PhaseSoftMax` 路线，因为它缺少共同相位旋转不变性，且其公开 shifted-mask 实现存在上述错误。

### Temporal attention

保留现有 QKV 合并投影、head 数、reshape、residual 和 FFN，只替换：

1. QKV / output projection 为 ComplexTorch Linear；
2. score 改用 Hermitian product $QK^H/\sqrt{d}$；
3. 调用 ComplexTorch 2.2.0 `ScaledDotProductAttention(softmax_on="real")`：

$$
A=\operatorname{softmax}(\operatorname{Re}(QK^H)/\sqrt d),\qquad Y=AV.
$$

$A$ 是 real attention probability，$V$ 和输出仍为 complex，因此不会把 feature 转成 real，也没有 complex softmax 的归一化歧义。

### Shifted-window spatial attention

不能直接整块替换为通用 MultiheadAttention，因为原实现还包含 shifted windows、real relative-position bias 和 real attention mask。应保留原 `WindowAttention` 外壳，仅把内部 QKV/projection 换成 ComplexTorch Linear，并调用 2.2.0 的 `ScaledDotProductAttention` core。

原有 real relative-position bias 保持为 real；它与 real shift mask 合并为 additive bias：

$$
S=\operatorname{Re}(QK^H)/\sqrt d+B_{rel}+M,\qquad A=\operatorname{softmax}(S),\qquad Y=AV.
$$

对 shifted block，先保持原代码的 window batch 维，再把 score 前缀组织为
`[batch, nW, heads, N, N]`，使 $B_{rel}$ 的
`[1, 1, heads, N, N]` 和 $M$ 的 `[1, nW, 1, N, N]` 正确广播。
薄适配只负责注入 SRDTrans 原有的 $B_{rel}$ 和 $M$，没有定义新的 attention。

不采用 ComplexTorch 默认的 split `CVSoftMax`，因为其对 real/imag 分别 softmax，会显著改变 phase。ComplexTorch 2.2.0 已能安全 mask magnitude/phase softmax，但它们属于 Eilers–Jiang 的 AAtt/APAtt 备选定义；本项目固定采用该论文有对称性、共同相位旋转不变性且实验结果最好的 CAtt，不额外引入 attention 变体。

也不直接使用 `ComplexTorch.MultiheadAttention`：它采用分离的 Q/K/V projection，且可自带 residual/LayerNorm，会改变当前 SRDTrans 的合并 QKV、pre-norm 和 residual 位置。这里只复用无拓扑意见的 `Linear`、`LayerNorm` 和 `ScaledDotProductAttention`。

### Dropout 审计

`complextorch.nn.Dropout` 对 real/imag 使用两个独立 Bernoulli mask，会改变 surviving feature 的 phase，不能用于 feature、projection 或 FFN dropout。迁移时使用一个极薄的 shared-mask wrapper：对 `x.real` 形状生成一次标准 dropout mask，再直接乘 complex `x`。这是 established complex dropout arithmetic，不是新网络模块。

ComplexTorch attention core 内部的 dropout 可以保留：在 `softmax_on="real"` 路径中被 dropout 的 attention weights 原本就是 real，随后才与 complex $V$ 相乘。

### Position encoding 审计

Temporal `LearnedPositionalEncoding` 保持现有 real parameter；real position 加到 complex feature 不会改变 dtype。Eilers–Jiang 也明确采用 real positional encoding，因为序列位置本身是 real。

Spatial `relative_position_bias_table` 同样保持 real，使其与 real CAtt logits 和 shift mask 在同一域相加。CVSwinFreq 使用 complex relative bias 是另一种已发表设计，但会增加 imaginary parameters 并改变原 SRDTrans 参数化，因此不采用。

## 输入与输出决策门

在以下选择确定前，不应开始代码迁移，否则必然引入未经定义的 representation adapter。

| Representation | 与 `[B,C,T,H,W]` 兼容性 | 问题 |
|---|---|---|
| Identity lift：$Z=X+0i$ | 同形状，可直接运行 | 是纯 complex architecture baseline，但不是 physics transform |
| 3D FFT | 同形状、可逆 | 卷积改在 frequency coordinates 上局部作用，改变原时空 receptive field 的物理含义 |
| 2D DTCWT | 不同尺度、6 方向、不同分辨率 | 必须新增 packing/fusion adapter，违反“只替换 operator” |
| Riesz quaternion | 4 个实分量 | 单个 PyTorch complex 只能容纳 2 个实分量；配成两个 complex channels 是新的设计选择 |

输出也必须预先固定：`.real`、magnitude 或 inverse physics transform 会对应不同模型。对于 image restoration，loss 应继续在 real image domain 计算。

## 验收标准

迁移完成后至少检查：

1. 每个 encoder、STB 和 decoder hook 的输入/输出均为 `torch.complex64`，只允许最终 readout 转 real；
2. 源码中间路径不存在 `.real`、`.imag`、`.abs()` 或 `.float()`，attention 的 `score.real` 是唯一例外，且 complex value path 保留；
3. 输入输出 shape、窗口划分、`ts/st` 顺序、四次 temporal down/up sampling 和 skip 数量与 real SRDTrans_v2 完全一致；
4. 一个最小 forward/backward check 覆盖 complex dtype、finite output、finite gradient 和最终 real loss；
5. 与 real baseline 使用相同数据、mask、loss、优化器和训练 schedule，不能同时改训练策略。
