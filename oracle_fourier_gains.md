# 0925 E50 steerable-Fourier gain oracle

Input: frozen 0925 E50 TIFF, frames 50:350 of the matching GT. `K=1` is
the global reconstructed signal, `K=5` is highpass/three scales/lowpass, and
`K=20` is highpass + 18 orientations + lowpass. Full and even/odd cross-frame
fits are reported.

| Hz | prior | 1-gain | 5-gain | 20-gain | 20-gain cross-frame |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 17.73808 | 17.74144 | 17.78541 | 17.79246 | 17.79201 |
| 0.3 | 17.74983 | 17.75067 | 17.76888 | 17.77804 | 17.77662 |
| 1 | 18.57549 | 18.57550 | 18.61434 | 18.61828 | 18.60844 |
| 3 | 19.97262 | 19.97267 | 19.99461 | 19.99896 | 19.99891 |
| 10 | 21.13182 | 21.13341 | 21.17292 | 21.18771 | 21.18768 |
| 30 | 22.62938 | 22.63433 | 22.66400 | 22.66920 | 22.66906 |

The gain vectors and raw logs are in `/tmp/oracle_fourier_*Hz.log` from the
CUDA 5/6/7 run.

## Centered subband gains

Each reconstructed subband was centered by its mean over the 300-frame
evaluation volume before fitting the gain; the prediction mean was preserved.

| Hz | prior | 1-gain | 5-gain | 20-gain |
|---:|---:|---:|---:|---:|
| 0.1 | 17.73808 | 17.74719 | 17.78863 | 17.79568 |
| 0.3 | 17.74983 | 17.75381 | 17.77046 | 17.77962 |
| 1 | 18.57549 | 18.59651 | 18.61707 | 18.62099 |
| 3 | 19.97262 | 19.99107 | 19.99921 | 20.00356 |
| 10 | 21.13182 | 21.13615 | 21.17773 | 21.19251 |
| 30 | 22.62938 | 22.63433 | 22.67378 | 22.67901 |
