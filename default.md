# Default model and training settings

This file documents the effective settings of the six-frequency pure-network experiment:
`experiments/260925_steerable_fourier_add_interleaved_width24_36_48_64_head8_hidden384_patch64_pure_l1l2_lr1e-4_v2`.
The saved per-frequency `para.yaml` files are the source of truth for data paths and automatically measured Fourier channel scales.

## Shared settings

| Area | Setting |
|---|---|
| Frequencies | `0.1, 0.3, 1, 3, 10, 30 Hz` |
| Training input / GT | `dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245/<Hz>/<Hz>_Noisy`; validation GT is `clean_<Hz>_1000frames.tif` |
| Pure network | Enabled (`--pure-network`); masked `L1 + L2` objective; no Gamma posterior / MPGN correction in training or validation output |
| Patch `(T,H,W)` | `(128,64,64)` |
| Training | 100 epochs; 6,000 patches/epoch; batch size 1; 4 data-loader workers |
| Optimizer | Adam; learning rate `1e-4`; betas `(0.9,0.999)` |
| Seed | `1024` |
| Train patch overlap | `0.75` |
| Validation | 400 frames; validation patch overlap `0.5`; evaluate each epoch; validation inference uses 400 frames |
| Backbone | `srdtrans_v2`; feature maps `[24,36,48,64]`; four encoder/decoder levels |
| Compression | Temporal (the default in this experiment): stride `(2,1,1)` at each encoder level; `T:128→8`, spatial size unchanged; decoder uses mirrored transpose convolutions |
| Transformer | Embedding 128; 8 heads; hidden dimension 384; 1 block; window size 7; order `st`; interleaved transformer enabled |
| Attention / skips | Spatial attention `swin`; additive skip fusion |
| Representation | `steerable_fourier`; 2D, 3 levels; legacy 20-channel Fourier adapter enabled |
| Fourier normalization | Per-channel normalization enabled; scales are measured per frequency and saved in each run's `para.yaml` |
| Dropout | Attention dropout `0.1`; input and spatial dropout `0` |
| Gradient checkpointing | Disabled |
| Initialization | Trained from scratch; no initial checkpoint |

## Frequency-specific settings

| Frequency | Sampling mode | Notes |
|---|---|---|
| 0.1 Hz | `spatial_mask` | Other listed training settings match the shared configuration |
| 0.3 Hz | `spatial_mask` | Other listed training settings match the shared configuration |
| 1 Hz | `temporal_mask` | Other listed training settings match the shared configuration |
| 3 Hz | `temporal_mask` | `orientation_heads=4` and structured-interaction keys are recorded but inactive for this representation |
| 10 Hz | `temporal_mask` | `orientation_heads=4` and structured-interaction keys are recorded but inactive for this representation |
| 30 Hz | `temporal_mask` | Other listed training settings match the shared configuration |

All runs use mask ratio `0.05`, minimum mask distance `2`, random patch coordinates, random slice axis, and random lattice phase. `0.1 Hz` and `0.3 Hz` are the only runs using spatial replacement masks; the remaining four use temporal replacement masks.

## Later-added modules: whether this experiment used them

| Module / option | Used? | Reason |
|---|---|---|
| Structured Fourier cross-scale interaction | **No** | The experiment selected `representation=steerable_fourier`, which uses the legacy Fourier adapter path, not `steerable_fourier_structured`. |
| Structured Fourier orientation interaction | **No** | Same reason. The `attention` keys saved for 3 Hz and 10 Hz do not activate these modules on the selected representation path. |
| Gamma prior / Poisson–Gaussian posterior correction | **No** | `--pure-network` selects direct denoiser training and validation. MPGN parameter values in `para.yaml` are configuration metadata, not an active correction step in this run. |
| Physical gain/MAP correction layer | **No** | No such post-network correction is in this pure-network pipeline. |
| DTCWT representation | **No** | `dtcwt_dim=2` and `dtcwt_levels=3` are retained CLI/config fields; the selected representation is Fourier. |
| Interleaved Transformer | **Yes** | Explicitly enabled; Transformer order is `st`. |
| Swin spatial attention | **Yes** | Effective setting is `swin`; omitted from a few older `para.yaml` files because it was the default. |

## Other recorded compatibility / inactive settings

`adaptive_grad_clip=false`; gradient clipping warmup settings are recorded but inactive. `freq_aware=false`, `ftvsr_enc1=false`, `use_msconv_before_trans=false`, `space_post_norm=false`, `temporal_strides=null`, `last_squeeze_op=conv`, and `upsample_mode=convt`. The legacy `fmap=16` field is not used by the Transformer backbone. MPGN metadata is `alpha=5000`, `beta=1600`, `offset=0`, `kmax=512`, NLL chunk length `8`, quantization step `1`, no clipping bounds, and boundary tolerance `1e-6`; these do not imply that a posterior correction was used.

Training image checkpoints were evaluated/saved each epoch (`save_test_images_per_epoch=true`); `eval_every_iters=0`, `diagnostic_interval=0`, and `snr_margin=50`.
