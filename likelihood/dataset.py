"""Data loading and patch tiling aligned with upstream SRDTrans/data_process.py."""

import math
import os
import random

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


def random_transform_srdtrans(input_arr):
    """Same 8-way H/W augmentation as SRDTrans random_transform."""
    p_trans = random.randrange(8)
    if p_trans == 0:
        return input_arr
    if p_trans == 1:
        return np.rot90(input_arr, k=1, axes=(1, 2))
    if p_trans == 2:
        return np.rot90(input_arr, k=2, axes=(1, 2))
    if p_trans == 3:
        return np.rot90(input_arr, k=3, axes=(1, 2))
    if p_trans == 4:
        return input_arr[:, :, ::-1]
    if p_trans == 5:
        return np.rot90(input_arr[:, :, ::-1], k=1, axes=(1, 2))
    if p_trans == 6:
        return np.rot90(input_arr[:, :, ::-1], k=2, axes=(1, 2))
    return np.rot90(input_arr[:, :, ::-1], k=3, axes=(1, 2))


def random_transform_pair_srdtrans(input_arr, target_arr):
    """Apply the same SRDTrans H/W augmentation to a training pair."""
    p_trans = random.randrange(8)

    def apply(x):
        if p_trans == 0:
            return x
        if p_trans == 1:
            return np.rot90(x, k=1, axes=(1, 2))
        if p_trans == 2:
            return np.rot90(x, k=2, axes=(1, 2))
        if p_trans == 3:
            return np.rot90(x, k=3, axes=(1, 2))
        if p_trans == 4:
            return x[:, :, ::-1]
        if p_trans == 5:
            return np.rot90(x[:, :, ::-1], k=1, axes=(1, 2))
        if p_trans == 6:
            return np.rot90(x[:, :, ::-1], k=2, axes=(1, 2))
        return np.rot90(x[:, :, ::-1], k=3, axes=(1, 2))

    return apply(input_arr), apply(target_arr)


def _random_coordinate(noise_img, coordinate, seed):
    """Draw a reproducible crop of the same size anywhere in one stack."""
    rng = random.Random(int(seed))
    size_t = coordinate['end_s'] - coordinate['init_s']
    size_h = coordinate['end_h'] - coordinate['init_h']
    size_w = coordinate['end_w'] - coordinate['init_w']
    if size_t > noise_img.shape[0] or size_h > noise_img.shape[1] or size_w > noise_img.shape[2]:
        raise ValueError('patch size exceeds stack shape')
    init_s = rng.randint(0, noise_img.shape[0] - size_t)
    init_h = rng.randint(0, noise_img.shape[1] - size_h)
    init_w = rng.randint(0, noise_img.shape[2] - size_w)
    return {
        'init_s': init_s, 'end_s': init_s + size_t,
        'init_h': init_h, 'end_h': init_h + size_h,
        'init_w': init_w, 'end_w': init_w + size_w,
    }


def _im_folder(args):
  folder = getattr(args, 'datasets_folder', None) or args.datasets_path
  return os.path.abspath(folder)


def get_gap_t_srdtrans(args, img, stack_num, window_t=None):
    whole_x = img.shape[2]
    whole_y = img.shape[1]
    whole_t = img.shape[0]
    window_t = int(window_t if window_t is not None else args.patch_t)
    mode = getattr(args, 'sampling_mode', 'spatial')

    window_h = _train_window_h(args)
    w_num = math.floor((whole_x - args.patch_x) / args.gap_x) + 1
    h_num = math.floor((whole_y - window_h) / args.gap_y) + 1
    s_num = math.ceil(args.train_datasets_size / w_num / h_num / stack_num)

    if s_num <= 1:
        if mode == 'spatial':
            s_num = 2
            gap_t = max(1, math.floor((whole_t - window_t) / (s_num - 1)))
        else:
            gap_t = max(1, int(args.patch_t * (1 - args.overlap_factor)))
    else:
        gap_t = max(1, math.floor((whole_t - window_t) / (s_num - 1)))
    return gap_t


def _train_window_t(args):
    if getattr(args, 'sampling_mode', 'spatial') == 'temporal':
        return int(args.patch_t) * 2
    return int(args.patch_t)


def _train_window_h(args):
    if getattr(args, 'sampling_mode', 'spatial') == 'height':
        return int(args.patch_y) * 2
    return int(args.patch_y)


def train_preprocess_lessMemoryMulStacks_srdtrans(args):
    """Port of SRDTrans train_preprocess_lessMemoryMulStacks (+ temporal window)."""
    patch_y = _train_window_h(args)
    patch_x = args.patch_x
    window_t = _train_window_t(args)
    gap_y = args.gap_y
    gap_x = args.gap_x
    im_folder = _im_folder(args)
    folder_tag = os.path.basename(os.path.normpath(im_folder)) or im_folder

    name_list = []
    coordinate_list = {}
    stack_index = []
    noise_im_all = []
    noise_im_means = []
    ind = 0
    print('\033[1;31mImage list for training (SRDTrans protocol) -----> \033[0m')
    print('sampling_mode -----> ', getattr(args, 'sampling_mode', 'spatial'))
    print('All files are in -----> ', im_folder)
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    print('Total stack number -----> ', stack_num)

    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        im_dir = os.path.join(im_folder, im_name)
        noise_im = tiff.imread(im_dir)
        print(im_name, ' -----> the shape is', noise_im.shape)
        if noise_im.shape[0] > args.select_img_num:
            noise_im = noise_im[0:args.select_img_num, :, :]
        gap_t = get_gap_t_srdtrans(args, noise_im, stack_num, window_t=window_t)

        noise_im = noise_im.astype(np.float32)
        # Mean in scaled units, BEFORE centering. Needed by MPGN NLL to
        # recover physical units via _to_phys_units: x_centered + mean.
        stack_mean = float(noise_im.mean())
        noise_im = noise_im - stack_mean
        noise_im_all.append(noise_im)
        noise_im_means.append(stack_mean)

        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]
        for x in range(0, int((whole_y - patch_y + gap_y) / gap_y)):
            for y in range(0, int((whole_x - patch_x + gap_x) / gap_x)):
                for z in range(0, int((whole_t - window_t + gap_t) / gap_t)):
                    single_coordinate = {
                        'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0,
                    }
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                    init_s = gap_t * z
                    end_s = gap_t * z + window_t
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    patch_name = (
                        folder_tag + '_' + im_name.replace('.tif', '')
                        + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                    )
                    name_list.append(patch_name)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind += 1

    return name_list, noise_im_all, coordinate_list, stack_index, noise_im_means


class trainset_srdtrans(Dataset):
    """Port of SRDTrans trainset (spatial-neighbor masking input)."""

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index,
                 return_stack_mean: bool = False, stack_means=None,
                 coordinate_seed=None):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index
        self.return_stack_mean = return_stack_mean
        self.stack_means = stack_means
        self.coordinate_seed = coordinate_seed

    def __getitem__(self, index):
        noise_img = self.noise_img_all[self.stack_index[index]]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        if self.coordinate_seed is not None:
            single_coordinate = _random_coordinate(
                noise_img, single_coordinate, int(self.coordinate_seed) + index)
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        patch = random_transform_srdtrans(patch)
        patch = torch.from_numpy(np.expand_dims(patch, 0).copy()).float()
        if self.return_stack_mean:
            # noise_img is already mean-centered at load time, so its mean is ~0.
            # Use the stored pre-centering stack mean (scaled units) so MPGN NLL
            # can recover physical units via (x_centered + mean).
            if self.stack_means is not None:
                stack_mean_val = float(self.stack_means[self.stack_index[index]])
            else:
                stack_mean_val = float(noise_img.mean())
            stack_global_mean = torch.tensor(stack_mean_val, dtype=torch.float32)
            return patch, stack_global_mean
        return patch

    def __len__(self):
        return len(self.name_list)


class trainset_temporal_srdtrans(Dataset):
    """DeepCAD temporal interleaving with SRDTrans stack-mean + augmentation.

    Set return_stack_mean=True to also return the pre-centering stack mean
    (scaled units). Unroll / NLL use this to recover physical units; the
    returned patches themselves remain stack-mean-centered.
    """

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index,
                 return_stack_mean: bool = False, stack_means=None):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index
        self.return_stack_mean = return_stack_mean
        self.stack_means = stack_means

    def __getitem__(self, index):
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        inp = noise_img[init_s:end_s:2, init_h:end_h, init_w:end_w]
        tgt = noise_img[init_s + 1:end_s:2, init_h:end_h, init_w:end_w]
        if random.random() >= 0.5:
            inp, tgt = tgt, inp

        inp, tgt = random_transform_pair_srdtrans(inp, tgt)
        inp = torch.from_numpy(np.expand_dims(inp, 0).copy()).float()
        tgt = torch.from_numpy(np.expand_dims(tgt, 0).copy()).float()
        if self.return_stack_mean:
            if self.stack_means is not None:
                stack_mean_val = float(self.stack_means[stack_index])
            else:
                stack_mean_val = float(noise_img.mean())
            stack_global_mean = torch.tensor(stack_mean_val, dtype=torch.float32)
            return inp, tgt, stack_global_mean
        return inp, tgt

    def __len__(self):
        return len(self.name_list)


class trainset_height_srdtrans(Dataset):
    """H-axis interleaving aligned with trainset_temporal_srdtrans.

    Crop a 2× height window, then take adjacent even/odd H slices as the pair.
    Spatial aug is applied after the split, same transform on both views.
    """

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index,
                 return_stack_mean: bool = False, stack_means=None):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index
        self.return_stack_mean = return_stack_mean
        self.stack_means = stack_means

    def __getitem__(self, index):
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        inp = noise_img[init_s:end_s, init_h:end_h:2, init_w:end_w]
        tgt = noise_img[init_s:end_s, init_h + 1:end_h:2, init_w:end_w]
        if random.random() >= 0.5:
            inp, tgt = tgt, inp

        inp, tgt = random_transform_pair_srdtrans(inp, tgt)
        inp = torch.from_numpy(np.expand_dims(inp, 0).copy()).float()
        tgt = torch.from_numpy(np.expand_dims(tgt, 0).copy()).float()
        if self.return_stack_mean:
            if self.stack_means is not None:
                stack_mean_val = float(self.stack_means[stack_index])
            else:
                stack_mean_val = float(noise_img.mean())
            stack_global_mean = torch.tensor(stack_mean_val, dtype=torch.float32)
            return inp, tgt, stack_global_mean
        return inp, tgt

    def __len__(self):
        return len(self.name_list)


class testset_srdtrans(Dataset):
  """Port of SRDTrans testset."""

  def __init__(self, name_list, coordinate_list, noise_img):
    self.name_list = name_list
    self.coordinate_list = coordinate_list
    self.noise_img = noise_img

  def __getitem__(self, index):
    single_coordinate = self.coordinate_list[self.name_list[index]]
    init_h = single_coordinate['init_h']
    end_h = single_coordinate['end_h']
    init_w = single_coordinate['init_w']
    end_w = single_coordinate['end_w']
    init_s = single_coordinate['init_s']
    end_s = single_coordinate['end_s']
    noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
    noise_patch = torch.from_numpy(np.expand_dims(noise_patch, 0)).float()
    return noise_patch, single_coordinate

  def __len__(self):
    return len(self.name_list)


def singlebatch_test_save_srdtrans(single_coordinate, output_image, raw_image):
  stack_start_w = int(single_coordinate['stack_start_w'])
  stack_end_w = int(single_coordinate['stack_end_w'])
  patch_start_w = int(single_coordinate['patch_start_w'])
  patch_end_w = int(single_coordinate['patch_end_w'])

  stack_start_h = int(single_coordinate['stack_start_h'])
  stack_end_h = int(single_coordinate['stack_end_h'])
  patch_start_h = int(single_coordinate['patch_start_h'])
  patch_end_h = int(single_coordinate['patch_end_h'])

  stack_start_s = int(single_coordinate['stack_start_s'])
  stack_end_s = int(single_coordinate['stack_end_s'])
  patch_start_s = int(single_coordinate['patch_start_s'])
  patch_end_s = int(single_coordinate['patch_end_s'])

  aaaa = output_image[
      patch_start_s:patch_end_s,
      patch_start_h:patch_end_h,
      patch_start_w:patch_end_w,
  ]
  bbbb = raw_image[
      patch_start_s:patch_end_s,
      patch_start_h:patch_end_h,
      patch_start_w:patch_end_w,
  ]
  return aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save_srdtrans(single_coordinate, batch_id, output_image, raw_image):
  stack_start_w = int(single_coordinate['stack_start_w'].numpy()[batch_id])
  stack_end_w = int(single_coordinate['stack_end_w'].numpy()[batch_id])
  patch_start_w = int(single_coordinate['patch_start_w'].numpy()[batch_id])
  patch_end_w = int(single_coordinate['patch_end_w'].numpy()[batch_id])

  stack_start_h = int(single_coordinate['stack_start_h'].numpy()[batch_id])
  stack_end_h = int(single_coordinate['stack_end_h'].numpy()[batch_id])
  patch_start_h = int(single_coordinate['patch_start_h'].numpy()[batch_id])
  patch_end_h = int(single_coordinate['patch_end_h'].numpy()[batch_id])

  stack_start_s = int(single_coordinate['stack_start_s'].numpy()[batch_id])
  stack_end_s = int(single_coordinate['stack_end_s'].numpy()[batch_id])
  patch_start_s = int(single_coordinate['patch_start_s'].numpy()[batch_id])
  patch_end_s = int(single_coordinate['patch_end_s'].numpy()[batch_id])

  output_image_id = output_image[batch_id]
  raw_image_id = raw_image[batch_id]
  aaaa = output_image_id[
      patch_start_s:patch_end_s,
      patch_start_h:patch_end_h,
      patch_start_w:patch_end_w,
  ]
  bbbb = raw_image_id[
      patch_start_s:patch_end_s,
      patch_start_h:patch_end_h,
      patch_start_w:patch_end_w,
  ]
  return aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def test_preprocess_lessMemoryNoTail_chooseOne_srdtrans(args, stack_index=0):
  """Port of SRDTrans test_preprocess_lessMemoryNoTail_chooseOne."""
  patch_y = args.patch_y
  patch_x = args.patch_x
  patch_t2 = args.patch_t
  gap_y = args.gap_y
  gap_x = args.gap_x
  gap_t2 = args.gap_t
  cut_w = (patch_x - gap_x) / 2
  cut_h = (patch_y - gap_y) / 2
  cut_s = (patch_t2 - gap_t2) / 2

  assert cut_w >= 0 and cut_h >= 0 and cut_s >= 0, 'test cut size is negative!'
  im_folder = _im_folder(args)
  folder_tag = os.path.basename(os.path.normpath(im_folder)) or im_folder

  name_list = []
  coordinate_list = {}
  img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
  img_list.sort()
  im_name = img_list[stack_index]

  im_dir = os.path.join(im_folder, im_name)
  noise_im = tiff.imread(im_dir)

  input_data_type = noise_im.dtype
  img_mean = noise_im.mean()

  if noise_im.shape[0] > args.test_datasize:
    noise_im = noise_im[0:args.test_datasize, :, :]
  noise_im = noise_im.astype(np.float32)
  noise_im = noise_im - img_mean

  whole_x = noise_im.shape[2]
  whole_y = noise_im.shape[1]
  whole_t = noise_im.shape[0]

  num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
  num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
  num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)

  for z in range(0, num_s):
    for x in range(0, num_h):
      for y in range(0, num_w):
        single_coordinate = {
          'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0,
        }
        if x != (num_h - 1):
          init_h = gap_y * x
          end_h = gap_y * x + patch_y
        else:
          init_h = whole_y - patch_y
          end_h = whole_y

        if y != (num_w - 1):
          init_w = gap_x * y
          end_w = gap_x * y + patch_x
        else:
          init_w = whole_x - patch_x
          end_w = whole_x

        if z != (num_s - 1):
          init_s = gap_t2 * z
          end_s = gap_t2 * z + patch_t2
        else:
          init_s = whole_t - patch_t2
          end_s = whole_t

        single_coordinate['init_h'] = init_h
        single_coordinate['end_h'] = end_h
        single_coordinate['init_w'] = init_w
        single_coordinate['end_w'] = end_w
        single_coordinate['init_s'] = init_s
        single_coordinate['end_s'] = end_s

        if y == 0:
          single_coordinate['stack_start_w'] = y * gap_x
          single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
          single_coordinate['patch_start_w'] = 0
          single_coordinate['patch_end_w'] = patch_x - cut_w
        elif y == num_w - 1:
          single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
          single_coordinate['stack_end_w'] = whole_x
          single_coordinate['patch_start_w'] = cut_w
          single_coordinate['patch_end_w'] = patch_x
        else:
          single_coordinate['stack_start_w'] = y * gap_x + cut_w
          single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
          single_coordinate['patch_start_w'] = cut_w
          single_coordinate['patch_end_w'] = patch_x - cut_w

        if x == 0:
          single_coordinate['stack_start_h'] = x * gap_y
          single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
          single_coordinate['patch_start_h'] = 0
          single_coordinate['patch_end_h'] = patch_y - cut_h
        elif x == num_h - 1:
          single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
          single_coordinate['stack_end_h'] = whole_y
          single_coordinate['patch_start_h'] = cut_h
          single_coordinate['patch_end_h'] = patch_y
        else:
          single_coordinate['stack_start_h'] = x * gap_y + cut_h
          single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
          single_coordinate['patch_start_h'] = cut_h
          single_coordinate['patch_end_h'] = patch_y - cut_h

        if z == 0:
          single_coordinate['stack_start_s'] = z * gap_t2
          single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
          single_coordinate['patch_start_s'] = 0
          single_coordinate['patch_end_s'] = patch_t2 - cut_s
        elif z == num_s - 1:
          single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
          single_coordinate['stack_end_s'] = whole_t
          single_coordinate['patch_start_s'] = cut_s
          single_coordinate['patch_end_s'] = patch_t2
        else:
          single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
          single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
          single_coordinate['patch_start_s'] = cut_s
          single_coordinate['patch_end_s'] = patch_t2 - cut_s

        patch_name = folder_tag + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
        name_list.append(patch_name)
        coordinate_list[patch_name] = single_coordinate

  return name_list, noise_im, coordinate_list, img_mean, input_data_type
