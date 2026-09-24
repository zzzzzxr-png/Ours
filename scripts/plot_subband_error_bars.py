#!/usr/bin/env python3
"""Plot normalized subband magnitude errors: Noisy=100%, GT=0%."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path('/data/zhouxirou/Ours_260911/experiments/representation_export/'
           '30Hz_subband_comparison_245/full300_magnitude_stats.json')
OUT = SRC.parent / 'full300_normalized_subband_error_bars.png'

data = json.loads(SRC.read_text())
keys = list(data)
labels = ['HP'] + [f'S{s}/O{o}' for s in range(3) for o in range(6)] + ['LP']
noisy = np.array([data[k]['Noisy']['mag_err_mean_percent'] for k in keys])
srd = np.array([data[k]['SRDTrans']['mag_err_mean_percent'] for k in keys]) / noisy * 100
ours = np.array([data[k]['Ours']['mag_err_mean_percent'] for k in keys]) / noisy * 100

x = np.arange(len(keys))
w = 0.36
fig, ax = plt.subplots(figsize=(15, 5.5))
ax.bar(x - w/2, srd, w, label='SRDTrans', color='#4C78A8')
ax.bar(x + w/2, ours, w, label='Ours', color='#E45756')
ax.axhline(100, color='black', linestyle='--', linewidth=1, label='Noisy = 100%')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x, labels, rotation=45, ha='right')
ax.set_ylabel('Normalized magnitude error (%)')
ax.set_ylim(0, max(110, float(max(srd.max(), ours.max()) * 1.12)))
ax.set_title('30 Hz Fourier subband magnitude error, 300-frame mean\nNoisy = 100%, GT = 0%')
ax.grid(axis='y', alpha=0.25)
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(OUT, dpi=220)
fig.savefig(OUT.with_suffix('.pdf'))
print(OUT)
