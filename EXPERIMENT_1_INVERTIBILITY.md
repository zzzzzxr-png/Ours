# Experiment 1：Candidate physics transform 可逆性验证

本实验完全独立于网络，验证 candidate transform 是否只改变表示空间，而不破坏原始 fluorescence information。

## 环境

```bash
conda activate physics
```

依赖：PyTorch、tifffile、pytorch_wavelets。DTCWT 必须使用 `DTCWTForward` / `DTCWTInverse`；`DWTForward` / `DWTInverse` 是实值 DWT，不能代替 complex DTCWT。

## 运行

```bash
cd /data/zhouxirou/Ours_260903
python scripts/experiment_1_invertibility.py /path/to/fluorescence_stack.tif \
  --output experiments/transform_invertibility.md
```

输入必须是 `[T,H,W]` TIFF stack。默认使用 float32、3-level DTCWT 和 CUDA（CUDA 不可用时自动使用 CPU）。快速检查可加 `--max-frames 32`。

## 定义与判定

统一指标：

$$
E_{rec}=\frac{\lVert X-X_{rec}\rVert_2}{\lVert X\rVert_2}.
$$

- Identity：$X_{rec}=X$。
- FFT：对完整 `[T,H,W]` volume 执行 `ifftn(fftn(X))`。
- DTCWT：`pytorch_wavelets` 提供二维 DTCWT，因此逐帧处理空间维度。使用滤波器组的 `symmetric` 完美重建边界；奇数空间尺寸由库内部补齐，inverse 后裁回原尺寸。
- Riesz quaternion：保留完整 $(X,R_t,R_y,R_x)$；inverse 读取 scalar component $X$。Riesz filter 在 Fourier domain 中定义，DC 点的三个 Riesz multiplier 设为 0。若丢弃 $X$，DC/intensity 不可恢复，因此不属于本实验认可的可逆表示。

float32 的默认验收阈值为 $E_{rec}\le10^{-5}$。脚本只要发现一个 transform 超出阈值，就以非零状态退出，表示该 candidate 应淘汰。

## 输出

完整 0.1 Hz GT stack（`[1000,245,245]`、float32、CUDA）的实测结果已保存到 `experiments/transform_invertibility.md`：

| Transform | Physics meaning | Invertibility error | Implementation |
|---|---|---:|---|
| Identity | reference representation | 0.000e+00 | PyTorch identity |
| FFT | global frequency amplitude/phase | 2.213e-07 | torch.fft |
| DTCWT | local scale/orientation/phase | 1.843e-07 | pytorch_wavelets |
| Riesz | local geometry/quadrature | 0.000e+00 | torch.fft |

四项均通过 $10^{-5}$ 阈值，本轮没有 candidate 因可逆性被淘汰。Riesz 的零误差成立于完整 quaternion 显式保留 $X$；不能据此认为只保留三个 Riesz 分量仍然可逆。

## 可视化

```bash
python scripts/visualize_experiment_1.py /path/to/fluorescence_stack.tif \
  --output-dir experiments/transform_visualizations
```

默认展示中间帧；可用 `--frame N` 指定帧。输出包括：

- `identity.png`：原始帧、中心 $x-t$ 切片和中心 ROI fluorescence 曲线。
- `fft.png`：全局空间频谱、时间—空间频谱和低幅值 mask 后的 phase。
- `dtcwt_amplitude.png` / `dtcwt_phase.png`：每个尺度的 6 个方向系数；同一尺度的 amplitude 共用显示范围。
- `riesz.png`：$X,R_t,R_y,R_x$、local amplitude 和 local phase。
- `reconstruction_residuals.png`：四种表示在同一范围内的绝对重建残差。
