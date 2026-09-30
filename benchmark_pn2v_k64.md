# PN2V K=64 benchmark

Configuration: 0925 patch `(128,64,64)`, batch 1, Fourier legacy adapter,
`f_maps=[24,36,48,64]`, embedding 128, 8 heads, hidden 384, one transformer
block. CUDA RTX 5090, eager mode, 20 warmup + 100 measured iterations.

| mode | mean iteration (s) | median (s) | max allocated (GiB) | max reserved (GiB) | output |
|---|---:|---:|---:|---:|---|
| l1l2 | 0.23317 | 0.23314 | 10.2143 | 11.1309 | `[1,1,128,64,64]` |
| pn2v_pg K=64 | 0.23926 | 0.23936 | 10.8178 | 11.5039 | `[1,64,128,64,64]` |

The smoke benchmark used the adaptive photon-count likelihood on masked
positions and did not allocate a `[B,64,T,H,W,kmax]` tensor.
