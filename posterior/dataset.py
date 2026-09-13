"""Dataset classes for the posterior (unroll-transformer) pipeline."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import random_transform_3

class trainset_temporal_2targets(Dataset):
    """Training dataset for temporal 2-target sampling.

    For each input frame at time τ, uses the frame at τ-1 and τ+1 as targets,
    giving two independent noisy observations of the same underlying signal.

    Coordinates use the same 2*patch_t window as the original temporal mode.
    Only the first patch_t+2 frames of that window are consumed:
      inp  = noise_img[init_s+1 : init_s+1+patch_t]   (middle frames)
      tgt1 = noise_img[init_s   : init_s+patch_t]      (one step behind)
      tgt2 = noise_img[init_s+2 : init_s+2+patch_t]    (one step ahead)

    Expects noise_img_all to already be stack-mean-centered (SRDTrans process).
    Returns (inp, tgt1, tgt2, stack_mean), each image [1, patch_t, H, W].
    """

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index, patch_t,
                 stack_means=None):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index
        self.patch_t = patch_t
        self.stack_means = stack_means

    def __getitem__(self, index):
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        coord = self.coordinate_list[self.name_list[index]]

        init_h = coord['init_h']
        end_h = coord['end_h']
        init_w = coord['init_w']
        end_w = coord['end_w']
        init_s = coord['init_s']

        inp = noise_img[init_s + 1:init_s + 1 + self.patch_t, init_h:end_h, init_w:end_w]
        tgt1 = noise_img[init_s:init_s + self.patch_t, init_h:end_h, init_w:end_w]
        tgt2 = noise_img[init_s + 2:init_s + 2 + self.patch_t, init_h:end_h, init_w:end_w]

        if self.stack_means is not None:
            stack_mean = np.float32(self.stack_means[stack_index])
        else:
            stack_mean = np.float32(0.0)

        inp, tgt1, tgt2 = random_transform_3(inp, tgt1, tgt2)

        inp = torch.from_numpy(np.expand_dims(inp, 0).copy()).float()
        tgt1 = torch.from_numpy(np.expand_dims(tgt1, 0).copy()).float()
        tgt2 = torch.from_numpy(np.expand_dims(tgt2, 0).copy()).float()

        stack_mean = torch.tensor([[[[stack_mean]]]], dtype=torch.float32)

        return inp, tgt1, tgt2, stack_mean

    def __len__(self):
        return len(self.name_list)

