"""SRDTrans-protocol trainer (aligned with upstream SRDTrans train.py / test.py).

Differences from posterior.trainer (unroll-transformer pipeline):
  - Stack-level mean subtraction before patch extraction
  - Patch-sized crop (no 2x H/W expansion)
  - Dual-target spatial-neighbor L1+L2, or DeepCAD temporal interleaving L1+L2
  - SRDTrans random_transform augmentation
  - Validation inference matches SRDTrans test.py (no per-patch mean centering)
  - Backbone selectable: 3D U-Net or SRDTrans transformer
"""

import datetime
import csv
import glob
import math
import os
import random
import re
import shutil
import time

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import yaml
from skimage import io
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from representation import DTCWT2D, FourierPyramid2D

from .dataset import (
    multibatch_test_save_srdtrans,
    singlebatch_test_save_srdtrans,
    test_preprocess_lessMemoryNoTail_chooseOne_srdtrans,
    testset_srdtrans,
    train_preprocess_lessMemoryMulStacks_srdtrans,
    trainset_srdtrans,
    trainset_temporal_srdtrans,
)
from .backbone_factory import (
    DEFAULT_SRDTRANS_ROOT,
    DTCWTComplexBackbone,
    FourierComplexBackbone,
    build_denoise_network_srdtrans,
)
from .sampling import generate_mask_pair, generate_subimages
from .losses import (
    l1_l2_loss,
    masked_l1_l2_loss,
    mpgn_nll_single_target as _mpgn_nll_single_target,
)
from .masks import (
    _DIRECTIONAL_MASK_MEAN_MODES,
    _DIRECTIONAL_MASK_MODES,
    _SLICE_MASK_MODES,
    make_directional_mask_mean_pair,
    make_directional_mask_pair,
    make_n2v_mask_pair,
    make_slice_mask_pair,
)


def set_random_seed(seed: int) -> None:
    """Fix Python / NumPy / PyTorch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _estimate_fixed_dtcwt_scales(stacks, levels, chunk_frames=32):
    """One fixed RMS per DTCWT scale from the centered training stacks."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    transform = DTCWT2D(levels=levels)
    square_sums = [0.0] * (levels + 1)
    counts = [0] * (levels + 1)
    divisor = 2 ** levels
    with torch.no_grad():
        for stack in stacks:
            height = stack.shape[-2] // divisor * divisor
            width = stack.shape[-1] // divisor * divisor
            top = (stack.shape[-2] - height) // 2
            left = (stack.shape[-1] - width) // 2
            for start in range(0, stack.shape[0], chunk_frames):
                block = np.ascontiguousarray(
                    stack[start:start + chunk_frames, top:top + height, left:left + width]
                )
                value = torch.from_numpy(block)[None, None]
                coefficients = transform(value)
                branches = (coefficients.low,) + coefficients.highs
                for index, branch in enumerate(branches):
                    square_sums[index] += float(branch.abs().square().sum())
                    counts[index] += branch.numel()
    scales = [math.sqrt(total / count) for total, count in zip(square_sums, counts)]
    if not all(math.isfinite(scale) and scale > 0 for scale in scales):
        raise RuntimeError('failed to estimate finite positive DTCWT scales')
    torch.set_num_threads(previous_threads)
    return scales


def _estimate_fixed_fourier_channel_scales(stacks, image_size, chunk_frames=32):
    """One fixed RMS per Fourier coefficient channel from centered stacks."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    transform = FourierPyramid2D(image_size=int(image_size), height=3, order=5)
    square_sums = [0.0] * 20
    counts = [0] * 20
    try:
        with torch.no_grad():
            for stack in stacks:
                top = (stack.shape[-2] - image_size) // 2
                left = (stack.shape[-1] - image_size) // 2
                for start in range(0, stack.shape[0], chunk_frames):
                    block = np.ascontiguousarray(
                        stack[start:start + chunk_frames,
                              top:top + image_size, left:left + image_size]
                    )
                    value = torch.from_numpy(block)[None, None]
                    coefficients = transform(value)
                    branches = (
                        coefficients.highpass,
                        *coefficients.bands,
                        coefficients.lowpass,
                    )
                    offset = 0
                    for branch in branches:
                        for channel in range(branch.shape[1]):
                            item = branch[:, channel]
                            square_sums[offset] += float(item.abs().square().sum())
                            counts[offset] += item.numel()
                            offset += 1
        scales = [math.sqrt(total / count) for total, count in zip(square_sums, counts)]
        if not all(math.isfinite(scale) and scale > 0 for scale in scales):
            raise RuntimeError('failed to estimate finite positive Fourier channel scales')
        return scales
    finally:
        torch.set_num_threads(previous_threads)


def _adaptive_clip_grad_(parameters, warmup_norms, threshold, warmup_iters,
                         percentile, warmup_cap, median_multiplier):
    """Clip safely while calibrating, then use the fixed warmup percentile."""
    cap = warmup_cap if threshold is None else threshold
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=cap, error_if_nonfinite=True)
    if threshold is None:
        warmup_norms.append(float(total_norm.detach()))
        if len(warmup_norms) == warmup_iters:
            threshold = min(
                float(np.percentile(warmup_norms, percentile)),
                float(median_multiplier) * float(np.median(warmup_norms)),
            )
    return total_norm, threshold


def _bind_visible_gpu(gpu: str) -> None:
    """Bind process to physical GPU(s) before any CUDA init (must run before cuda calls)."""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def _diagnostic_parameter_group(name):
    """Collapse SRDTrans parameters into stages useful for instability diagnosis."""
    parts = name.split('.')
    if len(parts) >= 3 and parts[0] == 'backbone':
        if parts[1] in ('encoders', 'decoders', 'layers'):
            return '{}.{}'.format(parts[1], parts[2])
        if parts[1] in ('branch_encoders', 'branch_decoders'):
            return '{}.{}.{}'.format(parts[1], parts[2], parts[3])
        if parts[1] in ('conv_before_trans', 'conv_after_trans'):
            return parts[1]
    return parts[0]


def _tensor_rms(value):
    if isinstance(value, (tuple, list)):
        square_sum = sum(item.detach().abs().square().sum() for item in value)
        count = sum(item.numel() for item in value)
        return float((square_sum / count).sqrt().cpu())
    return float(value.detach().abs().square().mean().sqrt().cpu())


class _TrainingDiagnostics:
    """Sparse stage activations and per-stage optimizer-update diagnostics."""

    def __init__(self, model, path, interval):
        self.model = model.module if isinstance(
            model, (nn.DataParallel, DistributedDataParallel)
        ) else model
        self.path = path
        self.interval = int(interval)
        self.enabled = False
        self.activations = {}
        self.handles = []
        self._register_hooks()
        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'iter', 'kind', 'name', 'value', 'loss', 'param_norm',
                'grad_norm', 'update_norm', 'update_ratio',
            ])

    def _save_activation(self, name, value):
        if self.enabled:
            self.activations[name] = _tensor_rms(value)

    def _save_encoder(self, index, output):
        self._save_activation('encoders.{}.skip'.format(index), output[0])
        self._save_activation('encoders.{}.down'.format(index), output[1])

    def _register_hooks(self):
        if not isinstance(self.model, (DTCWTComplexBackbone, FourierComplexBackbone)):
            raise TypeError('training diagnostics require a complex representation backbone')
        core = self.model.backbone
        self.handles.append(core.register_forward_pre_hook(
            lambda _, inputs: self._save_activation('dtcwt_input', inputs[0])))
        for index, module in enumerate(core.encoders):
            self.handles.append(module.register_forward_hook(
                lambda _, __, output, i=index: self._save_encoder(i, output)))
        for name in ('conv_before_trans', 'conv_after_trans'):
            module = getattr(core, name)
            self.handles.append(module.register_forward_hook(
                lambda _, __, output, n=name: self._save_activation(n, output)))
        for index, module in enumerate(core.layers):
            self.handles.append(module.register_forward_hook(
                lambda _, __, output, i=index: self._save_activation(
                    'layers.{}'.format(i), output)))
        for index, module in enumerate(core.decoders):
            self.handles.append(module.register_forward_hook(
                lambda _, __, output, i=index: self._save_activation(
                    'decoders.{}'.format(i), output)))
        self.handles.append(core.register_forward_hook(
            lambda _, __, output: self._save_activation('dtcwt_output', output)))
        self.handles.append(self.model.register_forward_hook(
            lambda _, __, output: self._save_activation('image_output', output)))

    def begin(self, iteration):
        self.enabled = iteration == 1 or iteration % self.interval == 0
        if self.enabled:
            self.activations.clear()
            return {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
        return None

    @staticmethod
    def _norms_by_group(named_values):
        sums = {}
        for name, value in named_values:
            if value is None:
                continue
            group = _diagnostic_parameter_group(name)
            sums[group] = sums.get(group, 0.0) + float(
                value.detach().abs().square().sum().cpu())
        return {name: math.sqrt(value) for name, value in sums.items()}

    def finish(self, iteration, loss, before):
        parameters = list(self.model.named_parameters())
        param_norms = self._norms_by_group(parameters)
        grad_norms = self._norms_by_group(
            (name, parameter.grad) for name, parameter in parameters)
        update_norms = self._norms_by_group(
            (name, parameter.detach() - before[name])
            for name, parameter in parameters
        )
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(
                [iteration, 'activation_rms', name, value, loss, '', '', '', '']
                for name, value in sorted(self.activations.items())
            )
            for name in sorted(param_norms):
                param_norm = param_norms[name]
                update_norm = update_norms.get(name, 0.0)
                writer.writerow([
                    iteration, 'parameter', name, '', loss, param_norm,
                    grad_norms.get(name, 0.0), update_norm, update_norm / max(param_norm, 1e-30),
                ])
        self.enabled = False

    def save_failure(self, iteration):
        torch.save(
            self.model.state_dict(),
            os.path.join(os.path.dirname(self.path),
                         'diagnostic_last_finite_iter_{:04d}.pth'.format(iteration - 1)),
        )

def cal_snr_srdtrans(noisy_img: np.ndarray, clean_img: np.ndarray) -> float:
    noise_signal_2 = (noisy_img.astype(np.float32) - clean_img.astype(np.float32)) ** 2
    clean_signal_2 = clean_img.astype(np.float32) ** 2
    sum1 = float(clean_signal_2.sum())
    sum2 = float(noise_signal_2.sum())
    if sum2 <= 0:
        return float('inf')
    return 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))



class training_class_srdtrans:
    """SRDTrans-protocol trainer with optional GT SNR validation."""

    def __init__(self, params_dict):
        self.distributed = bool(int(os.environ.get('WORLD_SIZE', '1')) > 1)
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.overlap_factor = 0.5
        self.val_overlap_factor = 0.5
        self.datasets_path = ''
        self.gt_path = ''
        self.n_epochs = 30
        self.fmap = 16
        self.output_dir = './results'
        self.pth_dir = './experiments_srdtrans_protocol'
        self.batch_size = None
        self.patch_t = 128
        self.patch_x = 128
        self.patch_y = 128
        self.gap_y = 64
        self.gap_x = 64
        self.gap_t = 64
        self.lr = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.GPU = '0'
        self.ngpu = 1
        self.train_datasets_size = 6000
        self.select_img_num = 100000
        self.num_workers = 4
        self.backbone = 'srdtrans_v2'
        self.representation = 'dtcwt'
        self.dtcwt_dim = 2
        self.dtcwt_levels = 3
        self.dtcwt_channel_normalize = False
        self.dtcwt_channel_scales = None
        self.fourier_channel_normalize = True
        self.legacy_fourier_adapter = False
        self.fourier_channel_scales = None
        self.sampling_mode = 'spatial'
        # Full-resolution directional masked self-supervision.
        # spatial_mask / temporal_mask keep input shape [B, C, T, H, W] unchanged
        # and compute loss only on sparse masked voxels.
        self.mask_ratio = 0.01
        self.mask_min_dist = 3
        self.lattice_random_phase = True
        self.random_patch_coordinates = False
        self.slice_axis = 'random'
        self.srdtrans_root = DEFAULT_SRDTRANS_ROOT
        self.embedding_dim = 128
        self.num_heads = 8
        self.hidden_dim = 512
        self.window_size = 7
        self.num_transBlock = 1
        self.attn_dropout_rate = 0.1
        self.input_dropout_rate = 0.0
        self.srdtrans_f_maps = [8, 16, 32, 64]
        self.skip_fusion = 'add'
        self.interleaved_transformer = False
        self.space_attention = 'swin'
        self.temporal_strides = None
        self.last_squeeze_op = 'conv'
        self.freq_aware = False
        self.ftvsr_enc1 = False
        self.enc_d2 = [1, 1, 1]
        self.init_ckpt = ''
        self.trans_order = 'ts'
        self.upsample_mode = 'convt'
        self.space_post_norm = False
        self.space_dropout_rate = 0.0
        self.use_msconv_before_trans = False
        self.gradient_checkpointing = True
        self.mask_loss = 'l1l2'
        self.mpgn_alpha = 5000.0
        self.mpgn_beta = 1600.0
        self.mpgn_offset = 0.0
        self.mpgn_kmax = 32
        self.mpgn_nll_chunk_t = 8
        self.mpgn_quant_step = 1.0
        self.mpgn_clip_low = None
        self.mpgn_clip_high = None
        self.mpgn_boundary_atol = 1e-6
        self.no_resume = False
        self.eval_val_per_epoch = True
        self.eval_every_iters = 0
        self.val_process_frames = 400
        self.snr_margin = 50
        self.save_test_images_per_epoch = True
        self.visualize_images_per_epoch = False
        self.seed = 1024
        self.adaptive_grad_clip = False
        self.grad_clip_warmup_iters = 1000
        self.grad_clip_percentile = 95.0
        self.grad_clip_warmup_cap = 1e8
        self.grad_clip_median_multiplier = 2.0
        self.diagnostic_interval = 0
        self._eval_cache_ready = False
        self.set_params(params_dict)

    def set_params(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.patch_y = self.patch_x
        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))
        # Inference may need >= patch_t frames for tiling, but SNR always uses
        # the fixed protocol window [snr_margin : val_process_frames - snr_margin].
        self.val_infer_frames = max(int(self.val_process_frames), int(self.patch_t))
        if self.distributed:
            self.ngpu = self.world_size
            self.GPU = str(self.local_rank)
        else:
            self.ngpu = str(self.GPU).count(',') + 1
        if self.batch_size is None:
            self.batch_size = self.ngpu
        elif int(self.batch_size) < 1:
            raise ValueError('batch_size must be positive, got {}'.format(self.batch_size))
        else:
            self.batch_size = int(self.batch_size)
        if not self.distributed or self.rank == 0:
            print('\033[1;31mSRDTrans protocol training parameters -----> \033[0m')
            print(self.__dict__)

    def prepare_file(self):
        parts = self.datasets_path.rstrip('/').split('/')
        self.datasets_name = parts[-1] if parts[-1] else parts[-2]
        backbone_tag = 'unet' if self.backbone in ('unet', '3dunet', '3DUNet') else self.backbone
        mode_tag = self.sampling_mode
        if self.sampling_mode == 'n2v':
            mode_tag = 'n2v'
        if self.sampling_mode in _SLICE_MASK_MODES:
            axis = (getattr(self, 'slice_axis', 'random') or 'random').lower()
            if self.sampling_mode == 'temporal_mask_slice':
                axis = 't'
            elif self.sampling_mode == 'spatial_mask_slice' and axis == 'random':
                axis = 'hw'
            if axis in ('t', 'h', 'w'):
                mode_tag = 'slice_mask_{}'.format(axis)
        self.pth_path = os.path.join(
            self.pth_dir,
            '{}_srdtrans_{}_{}'.format(self.datasets_name, mode_tag, backbone_tag),
        )

        if os.path.exists(self.pth_path):
            if self.no_resume:
                shutil.rmtree(self.pth_path)
                os.makedirs(self.pth_path)
            # else: keep existing folder and resume from latest .pth if present
        else:
            os.makedirs(self.pth_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def _find_latest_checkpoint(self):
        """Return (path, epoch_1idx, iter_1idx) for newest E_XX_Iter_YYYY.pth."""
        pattern = os.path.join(self.pth_path, 'E_*_Iter_*.pth')
        best_path = None
        best_key = (-1, -1)
        for path in glob.glob(pattern):
            match = re.search(r'E_(\d+)_Iter_(\d+)\.pth$', os.path.basename(path))
            if not match:
                continue
            key = (int(match.group(1)), int(match.group(2)))
            if key > best_key:
                best_key = key
                best_path = path
        if best_path is None:
            return None
        return best_path, best_key[0], best_key[1]

    def _try_resume_checkpoint(self):
        """Load latest model/optimizer state; continue from the next epoch."""
        self._resume_start_epoch = 0
        self._resume_global_iter = 0
        self._resume_optimizer_state = None

        latest_path = os.path.join(self.pth_path, 'latest.pth')
        if os.path.isfile(latest_path):
            state = torch.load(latest_path, map_location='cpu', weights_only=False)
            model_state = state.get('model_state_dict')
            optimizer_state = state.get('optimizer_state_dict')
            if model_state is not None and optimizer_state is not None:
                target = self.local_model.module if isinstance(
                    self.local_model, (nn.DataParallel, DistributedDataParallel)
                ) else self.local_model
                target.load_state_dict(model_state)
                self._resume_optimizer_state = optimizer_state
                self._resume_start_epoch = int(state['next_epoch'])
                self._resume_global_iter = int(state.get('global_iter', 0))
                print(
                    '\033[1;31mResume latest checkpoint -----> \033[0m{} '
                    '(next epoch {}, global_iter {})'.format(
                        latest_path, self._resume_start_epoch + 1,
                        self._resume_global_iter,
                    )
                )
                return True

        found = self._find_latest_checkpoint()
        if found is None:
            return False

        ckpt_path, epoch_1idx, iter_1idx = found
        state = torch.load(ckpt_path, map_location='cpu')
        target = self.local_model.module if isinstance(
            self.local_model, (nn.DataParallel, DistributedDataParallel)
        ) else self.local_model
        target.load_state_dict(state)

        # E_36 was saved at end of 0-based epoch 35 -> resume at epoch index 36.
        self._resume_start_epoch = epoch_1idx
        print(
            '\033[1;31mResume checkpoint -----> \033[0m{} (epoch {}, iter {})'.format(
                ckpt_path, epoch_1idx, iter_1idx,
            )
        )
        print(
            'Continuing training from epoch {}/{}'.format(
                epoch_1idx + 1, self.n_epochs,
            )
        )
        return True

    def _args_proxy(self):
        class Args:
            pass

        args = Args()
        args.datasets_path = self.datasets_path
        args.datasets_folder = self.datasets_path
        args.patch_x = self.patch_x
        args.patch_y = self.patch_y
        args.patch_t = self.patch_t
        args.gap_x = self.gap_x
        args.gap_y = self.gap_y
        args.gap_t = self.gap_t
        args.select_img_num = self.select_img_num
        args.train_datasets_size = self.train_datasets_size
        args.test_datasize = self.val_infer_frames
        args.overlap_factor = self.val_overlap_factor
        args.sampling_mode = self.sampling_mode
        return args

    def save_yaml_train(self):
        yaml_name = os.path.join(self.pth_path, 'para.yaml')
        para = {key: getattr(self, key) for key in (
            'n_epochs', 'datasets_path', 'gt_path', 'output_dir', 'pth_path', 'GPU',
            'batch_size', 'patch_x', 'patch_y', 'patch_t', 'gap_y', 'gap_x', 'gap_t',
            'lr', 'b1', 'b2', 'fmap', 'select_img_num',
            'train_datasets_size', 'overlap_factor', 'val_overlap_factor',
            'val_process_frames', 'val_infer_frames', 'snr_margin', 'eval_every_iters', 'backbone',
            'representation', 'dtcwt_dim', 'dtcwt_levels', 'dtcwt_channel_normalize',
            'legacy_fourier_adapter',
            'dtcwt_channel_scales', 'fourier_channel_normalize', 'fourier_channel_scales',
            'srdtrans_root', 'embedding_dim', 'num_heads', 'hidden_dim', 'window_size',
            'num_transBlock', 'attn_dropout_rate',             'srdtrans_f_maps', 'input_dropout_rate',
            'skip_fusion', 'interleaved_transformer', 'space_attention',
            'temporal_strides', 'last_squeeze_op', 'freq_aware', 'ftvsr_enc1', 'enc_d2', 'upsample_mode', 'init_ckpt',
            'sampling_mode',
            'mask_ratio', 'mask_min_dist', 'lattice_random_phase',
            'random_patch_coordinates', 'slice_axis', 'seed',
            'trans_order', 'space_post_norm', 'space_dropout_rate',
            'use_msconv_before_trans', 'gradient_checkpointing', 'mask_loss',
            'adaptive_grad_clip', 'grad_clip_warmup_iters',
            'grad_clip_percentile', 'grad_clip_warmup_cap',
            'grad_clip_median_multiplier',
            'diagnostic_interval',
            'mpgn_alpha', 'mpgn_beta', 'mpgn_offset', 'mpgn_kmax', 'mpgn_nll_chunk_t',
            'mpgn_quant_step', 'mpgn_clip_low', 'mpgn_clip_high', 'mpgn_boundary_atol',
        )}
        para['protocol'] = 'srdtrans'
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def initialize_network(self):
        self.local_model = build_denoise_network_srdtrans(self)

    def _load_init_ckpt(self, path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError('init_ckpt not found: {}'.format(path))
        state = torch.load(path, map_location='cpu')
        target = self.local_model.module if isinstance(self.local_model, nn.DataParallel) else self.local_model
        mode = getattr(self, 'upsample_mode', 'convt') or 'convt'
        incompatible = target.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if mode == 'convt':
            if missing or unexpected:
                raise RuntimeError(
                    'init_ckpt mismatch for convt upsample: missing={} unexpected={}'.format(
                        missing, unexpected))
        else:
            bad_missing = missing
            bad_unexpected = [k for k in unexpected if '.up_sample.' not in k]
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    'init_ckpt mismatch for upsample_mode={}: missing={} unexpected={}'.format(
                        mode, missing, unexpected))
            print('\033[1;31mSkipped ConvTranspose init keys -----> \033[0m{}'.format(
                len(unexpected)))
        print('\033[1;31mLoaded shared init -----> \033[0m{}'.format(path))

    def distribute_GPU(self):
        _bind_visible_gpu(str(self.GPU))
        if torch.cuda.is_available():
            if self.distributed:
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend='nccl', init_method='env://')
                torch.cuda.set_device(self.local_rank)
                self.local_model = self.local_model.cuda(self.local_rank)
                self.local_model = DistributedDataParallel(
                    self.local_model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=False,
                )
                if self.rank == 0:
                    print('\033[1;31mUsing DDP on {} GPU(s) -----> \033[0m'.format(self.world_size))
            else:
                self.local_model = self.local_model.cuda()
                self.local_model = nn.DataParallel(self.local_model, device_ids=range(self.ngpu))
                print('\033[1;31mUsing {} GPU(s) -----> \033[0m'.format(torch.cuda.device_count()))

    def _prepare_eval_cache(self):
        if self._eval_cache_ready:
            return
        if not self.gt_path:
            raise ValueError('gt_path is required for validation SNR metrics.')
        if not os.path.exists(self.gt_path):
            raise ValueError('gt_path does not exist: {}'.format(self.gt_path))

        val_args = self._args_proxy()
        val_args.gap_x = int(self.patch_x * (1 - self.val_overlap_factor))
        val_args.gap_y = int(self.patch_y * (1 - self.val_overlap_factor))
        val_args.gap_t = int(self.patch_t * (1 - self.val_overlap_factor))

        name_list, noise_img, coordinate_list, img_mean, input_data_type = \
            test_preprocess_lessMemoryNoTail_chooseOne_srdtrans(val_args, 0)

        gt = tiff.imread(self.gt_path).astype(np.float32)
        if gt.shape[0] > self.val_infer_frames:
            gt = gt[:self.val_infer_frames]

        self._eval_name_list = name_list
        self._eval_noise_img = noise_img
        self._eval_coordinate_list = coordinate_list
        self._eval_img_mean = img_mean
        self._eval_input_data_type = input_data_type
        self._eval_ref_img = gt
        self._eval_cache_ready = True

    def train(self):
        if self.adaptive_grad_clip:
            if int(self.grad_clip_warmup_iters) <= 0:
                raise ValueError('grad_clip_warmup_iters must be positive')
            if not 0.0 < float(self.grad_clip_percentile) <= 100.0:
                raise ValueError('grad_clip_percentile must be in (0, 100]')
            if float(self.grad_clip_warmup_cap) <= 0.0:
                raise ValueError('grad_clip_warmup_cap must be positive')
            if float(self.grad_clip_median_multiplier) <= 0.0:
                raise ValueError('grad_clip_median_multiplier must be positive')
        optimizer_G = torch.optim.Adam(
            self.local_model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        if getattr(self, '_resume_optimizer_state', None) is not None:
            optimizer_G.load_state_dict(self._resume_optimizer_state)
            # Keep the requested run learning rate when resuming Adam state.
            for group in optimizer_G.param_groups:
                group['lr'] = float(self.lr)
            print('Resume Adam optimizer state -----> loaded')
        diagnostics = None
        if int(getattr(self, 'diagnostic_interval', 0)) > 0 and (
                not self.distributed or self.rank == 0):
            diagnostics = _TrainingDiagnostics(
                self.local_model,
                os.path.join(self.pth_path, 'training_diagnostics.csv'),
                int(self.diagnostic_interval),
            )
        warmup_grad_norms = []
        grad_clip_threshold = None
        cuda = torch.cuda.is_available()

        prev_time = time.time()
        time_start = time.time()
        global_iter = 0
        start_epoch = getattr(self, '_resume_start_epoch', 0)
        train_args = self._args_proxy()
        loader_generator = None
        worker_init_fn = None
        if getattr(self, 'seed', None) is not None:
            loader_generator = torch.Generator()
            loader_generator.manual_seed(int(self.seed))
            base_seed = int(self.seed)

            def worker_init_fn(worker_id):
                s = base_seed + worker_id
                random.seed(s)
                np.random.seed(s)

        for epoch in range(start_epoch, self.n_epochs):
            self.local_model.train()
            if self.sampling_mode == 'temporal':
                train_data = trainset_temporal_srdtrans(
                    self.train_name_list,
                    self.train_coordinate_list,
                    self.train_noise_img,
                    self.train_stack_index,
                )
            else:
                # spatial / *_mask / *_mask_slice: full-resolution raw patch.
                use_nll = (
                    getattr(self, 'mask_loss', 'l1l2') == 'nll'
                    and self.sampling_mode in _DIRECTIONAL_MASK_MODES
                )
                train_data = trainset_srdtrans(
                    self.train_name_list,
                    self.train_coordinate_list,
                    self.train_noise_img,
                    self.train_stack_index,
                    return_stack_mean=use_nll,
                    stack_means=getattr(self, 'train_stack_means', None),
                    coordinate_seed=(
                        int(self.seed or 0) + epoch * len(self.train_name_list)
                        if self.random_patch_coordinates else None
                    ),
                )
            sampler = DistributedSampler(
                train_data,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=int(self.seed or 0),
            ) if self.distributed else None
            if sampler is not None:
                sampler.set_epoch(epoch)
            trainloader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=self.num_workers,
                generator=loader_generator,
                worker_init_fn=worker_init_fn,
            )
            if epoch == start_epoch and start_epoch > 0:
                global_iter = getattr(
                    self, '_resume_global_iter', start_epoch * len(trainloader)
                )

            for iteration, batch in enumerate(trainloader):
                diagnostic_before = (
                    diagnostics.begin(global_iter + 1) if diagnostics else None
                )
                if self.sampling_mode == 'temporal':
                    inp, tgt = batch
                    if cuda:
                        inp = inp.cuda()
                        tgt = tgt.cuda()
                    noisy_output = self.local_model(inp)
                    total_loss = l1_l2_loss(noisy_output, tgt)
                elif self.sampling_mode in _DIRECTIONAL_MASK_MODES:
                    stack_global_mean = None
                    if getattr(self, 'mask_loss', 'l1l2') == 'nll':
                        noisy, stack_global_mean = batch
                    else:
                        noisy = batch
                    if cuda:
                        noisy = noisy.cuda()
                        if stack_global_mean is not None:
                            stack_global_mean = stack_global_mean.cuda()

                    if self.sampling_mode == 'n2v':
                        masked_input, masked_target, loss_mask = make_n2v_mask_pair(
                            noisy,
                        )
                    elif self.sampling_mode in _SLICE_MASK_MODES:
                        masked_input, masked_target, loss_mask = make_slice_mask_pair(
                            noisy,
                            mode=self.sampling_mode,
                            mask_ratio=self.mask_ratio,
                            min_dist=self.mask_min_dist,
                            slice_axis=getattr(self, 'slice_axis', 'random'),
                        )
                    elif self.sampling_mode in _DIRECTIONAL_MASK_MEAN_MODES:
                        masked_input, masked_target, loss_mask = (
                            make_directional_mask_mean_pair(
                                noisy,
                                mode=self.sampling_mode,
                                mask_ratio=self.mask_ratio,
                                min_dist=self.mask_min_dist,
                                lattice_random_phase=bool(
                                    getattr(self, 'lattice_random_phase', True)
                                ),
                            )
                        )
                    else:
                        masked_input, masked_target, loss_mask = make_directional_mask_pair(
                            noisy,
                            mode=self.sampling_mode,
                            mask_ratio=self.mask_ratio,
                            min_dist=self.mask_min_dist,
                            lattice_random_phase=bool(
                                getattr(self, 'lattice_random_phase', True)
                            ),
                        )

                    noisy_output = self.local_model(masked_input)
                    if getattr(self, 'mask_loss', 'l1l2') == 'nll':
                        patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
                        total_loss = _mpgn_nll_single_target(
                            noisy_output,
                            masked_target,
                            loss_mask,
                            pred_img_mean=patch_mean,
                            target_img_mean=patch_mean,
                            alpha=self.mpgn_alpha,
                            beta=self.mpgn_beta,
                            offset=self.mpgn_offset,
                            kmax=int(self.mpgn_kmax),
                            chunk_t=int(self.mpgn_nll_chunk_t),
                            quant_step=getattr(self, 'mpgn_quant_step', 1.0),
                            clip_low=getattr(self, 'mpgn_clip_low', None),
                            clip_high=getattr(self, 'mpgn_clip_high', None),
                            boundary_atol=float(
                                getattr(self, 'mpgn_boundary_atol', 1e-6)
                            ),
                        )
                    else:
                        total_loss = masked_l1_l2_loss(
                            noisy_output,
                            masked_target,
                            loss_mask,
                            l1_weight=0.5,
                            l2_weight=0.5,
                        )
                else:
                    noisy = batch
                    if cuda:
                        noisy = noisy.cuda()

                    mask1, mask2, mask3 = generate_mask_pair(noisy)
                    noisy_sub1 = generate_subimages(noisy, mask1)
                    noisy_sub2 = generate_subimages(noisy, mask2)
                    noisy_sub3 = generate_subimages(noisy, mask3)

                    noisy_output = self.local_model(noisy_sub1)

                    loss_a = l1_l2_loss(noisy_output, noisy_sub2)
                    loss_b = l1_l2_loss(noisy_output, noisy_sub3)
                    total_loss = 0.5 * loss_a + 0.5 * loss_b

                if diagnostics and not bool(torch.isfinite(total_loss).item()):
                    diagnostics.save_failure(global_iter + 1)
                    raise FloatingPointError(
                        'non-finite loss at global iteration {}'.format(global_iter + 1))

                optimizer_G.zero_grad(set_to_none=True)
                total_loss.backward()
                if diagnostics:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            self.local_model.parameters(),
                            max_norm=float('inf'),
                            error_if_nonfinite=True,
                        )
                    except RuntimeError:
                        diagnostics.save_failure(global_iter + 1)
                        raise
                if self.adaptive_grad_clip:
                    previous_threshold = grad_clip_threshold
                    grad_norm, grad_clip_threshold = _adaptive_clip_grad_(
                        self.local_model.parameters(),
                        warmup_grad_norms,
                        grad_clip_threshold,
                        int(self.grad_clip_warmup_iters),
                        float(self.grad_clip_percentile),
                        float(self.grad_clip_warmup_cap),
                        float(self.grad_clip_median_multiplier),
                    )
                    if previous_threshold is None and grad_clip_threshold is not None:
                        calibration = {
                            'warmup_iters': len(warmup_grad_norms),
                            'percentile': float(self.grad_clip_percentile),
                            'percentile_value': float(np.percentile(
                                warmup_grad_norms, self.grad_clip_percentile)),
                            'median_multiplier': float(
                                self.grad_clip_median_multiplier),
                            'threshold': grad_clip_threshold,
                            'median': float(np.median(warmup_grad_norms)),
                            'maximum': float(np.max(warmup_grad_norms)),
                        }
                        if not self.distributed or self.rank == 0:
                            with open(os.path.join(
                                    self.pth_path, 'grad_clip_calibration.yaml'), 'w') as f:
                                yaml.safe_dump(calibration, f, sort_keys=False)
                            print(
                                '\nAdaptive gradient clip calibrated: '
                                'P{:.1f}={:.6g}, median={:.6g}, max={:.6g}'.format(
                                    float(self.grad_clip_percentile),
                                    grad_clip_threshold,
                                    calibration['median'],
                                    calibration['maximum'],
                                )
                            )
                optimizer_G.step()
                global_iter += 1
                if diagnostic_before is not None:
                    diagnostics.finish(
                        global_iter, float(total_loss.detach().cpu()), diagnostic_before)

                batches_left = self.n_epochs * len(trainloader) - (
                    epoch * len(trainloader) + iteration)
                time_left = datetime.timedelta(
                    seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()

                if self.distributed and self.rank != 0:
                    continue
                print(
                    '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f] [ETA: %s] [Time cost: %.0d s] '
                    % (
                        epoch + 1,
                        self.n_epochs,
                        iteration + 1,
                        len(trainloader),
                        total_loss.item(),
                        time_left,
                        time.time() - time_start,
                    ),
                    end=' ',
                )

                if (getattr(self, 'eval_val_per_epoch', False)
                        and int(getattr(self, 'eval_every_iters', 0)) > 0
                        and global_iter % int(self.eval_every_iters) == 0):
                    print('\nValidation (global_iter={}) ----->'.format(global_iter))
                    self.test(epoch, iteration)
                    self.local_model.train()

                if (iteration + 1) % len(trainloader) == 0:
                    print('\n', end=' ')
                    self.save_model(epoch, iteration, optimizer_G, global_iter)
                    if getattr(self, 'eval_val_per_epoch', False) and int(
                            getattr(self, 'eval_every_iters', 0)) <= 0:
                        print('Validation ----->')
                        self.test(epoch, iteration)
                        self.local_model.train()
                    print('\n', end=' ')
            if self.distributed:
                torch.distributed.barrier()

    def save_model(self, epoch, iteration, optimizer=None, global_iter=None):
        if self.distributed and self.rank != 0:
            return
        os.makedirs(self.pth_path, exist_ok=True)
        model_save_name = os.path.join(
            self.pth_path,
            'E_{}_Iter_{}.pth'.format(str(epoch + 1).zfill(2), str(iteration + 1).zfill(4)),
        )
        if isinstance(self.local_model, (nn.DataParallel, DistributedDataParallel)):
            model_state = self.local_model.module.state_dict()
        else:
            model_state = self.local_model.state_dict()
        torch.save(model_state, model_save_name)

        if optimizer is not None:
            latest_path = os.path.join(self.pth_path, 'latest.pth')
            latest_tmp = latest_path + '.tmp'
            torch.save({
                'format_version': 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'next_epoch': epoch + 1,
                'global_iter': int(global_iter if global_iter is not None else 0),
            }, latest_tmp)
            os.replace(latest_tmp, latest_path)

    def test(self, train_epoch, train_iteration):
        """SRDTrans test.py stitching + DeepCAD GT SNR metrics."""
        self._prepare_eval_cache()

        name_list = self._eval_name_list
        noise_img = self._eval_noise_img
        coordinate_list = self._eval_coordinate_list
        img_mean = self._eval_img_mean
        input_data_type = self._eval_input_data_type
        ref_img = self._eval_ref_img

        prev_time = time.time()
        time_start = time.time()
        denoise_img = np.zeros(noise_img.shape)
        denoise_before_match = np.zeros(noise_img.shape)
        input_img = np.zeros(noise_img.shape)

        test_data = testset_srdtrans(name_list, coordinate_list, noise_img)
        testloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        cuda = torch.cuda.is_available()
        self.local_model.eval()
        with torch.no_grad():
            for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                if cuda:
                    noise_patch = noise_patch.cuda()

                fake_B = self.local_model(noise_patch)
                output_image = np.squeeze(fake_B.cpu().detach().numpy())
                raw_image = np.squeeze(noise_patch.cpu().detach().numpy())

                batches_left = len(testloader) - iteration
                prev_time = time.time()
                print(
                    '\r [Patch %d/%d] [Time Cost: %.0d s] [ETA: %.0d s]     '
                    % (
                        iteration + 1,
                        len(testloader),
                        time.time() - time_start,
                        int(batches_left * (time.time() - prev_time)),
                    ),
                    end=' ',
                )

                if output_image.ndim == 3:
                    postprocess_turn = 1
                else:
                    postprocess_turn = output_image.shape[0]

                if postprocess_turn > 1:
                    for batch_id in range(postprocess_turn):
                        output_patch, raw_patch, stack_start_w, stack_end_w, \
                            stack_start_h, stack_end_h, stack_start_s, stack_end_s = \
                            multibatch_test_save_srdtrans(
                                single_coordinate, batch_id, output_image, raw_image)
                        output_patch = output_patch + img_mean
                        raw_patch = raw_patch + img_mean
                        denoise_before_match[
                            stack_start_s:stack_end_s,
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = output_patch
                        denoise_img[
                            stack_start_s:stack_end_s,
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                        input_img[
                            stack_start_s:stack_end_s,
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = raw_patch
                else:
                    output_patch, raw_patch, stack_start_w, stack_end_w, \
                        stack_start_h, stack_end_h, stack_start_s, stack_end_s = \
                        singlebatch_test_save_srdtrans(
                            single_coordinate, output_image, raw_image)
                    output_patch = output_patch + img_mean
                    raw_patch = raw_patch + img_mean
                    denoise_before_match[
                        stack_start_s:stack_end_s,
                        stack_start_h:stack_end_h,
                        stack_start_w:stack_end_w,
                    ] = output_patch
                    denoise_img[
                        stack_start_s:stack_end_s,
                        stack_start_h:stack_end_h,
                        stack_start_w:stack_end_w,
                    ] = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                    input_img[
                        stack_start_s:stack_end_s,
                        stack_start_h:stack_end_h,
                        stack_start_w:stack_end_w,
                    ] = raw_patch

        print('\n', end=' ')

        output_pre = denoise_before_match.squeeze().astype(np.float32)
        output_post = denoise_img.squeeze().astype(np.float32)
        noisy_full = input_img.squeeze().astype(np.float32)

        T = min(output_pre.shape[0], output_post.shape[0], noisy_full.shape[0], ref_img.shape[0])
        # Align metric window to prior protocol (default 400 frames, margin 50),
        # independent of how many frames were tiled for inference (may be patch_t).
        eval_t = min(int(self.val_process_frames), T)
        s = self.snr_margin
        e = eval_t - self.snr_margin
        if e <= s:
            s, e = 0, eval_t
        snr_pre = cal_snr_srdtrans(output_pre[s:e], ref_img[s:e])
        snr_post = cal_snr_srdtrans(output_post[s:e], ref_img[s:e])
        snr_noisy = cal_snr_srdtrans(noisy_full[s:e], ref_img[s:e])
        print(
            'SNR (frames {:d}:{:d} of first {:d}; infer_T={:d}; '
            'denoised_no_scale / denoised / noisy vs GT) '
            '-----> {:.4f} dB / {:.4f} dB / {:.4f} dB'.format(
                s, e, eval_t, T, snr_pre, snr_post, snr_noisy))

        metrics_path = os.path.join(self.pth_path, 'val_metrics.md')
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                f.write('| Epoch | Iteration | SNR_no_scale (dB) | SNR_denoised (dB) | SNR_noisy (dB) |\n')
                f.write('| ----- | --------- | ----------------- | ----------------- | -------------- |\n')
        with open(metrics_path, 'a') as f:
            f.write('| {} | {} | {:.4f} | {:.4f} | {:.4f} |\n'.format(
                train_epoch + 1, train_iteration + 1, snr_pre, snr_post, snr_noisy))

        if self.save_test_images_per_epoch:
            save_img = output_post[s:e]
            if input_data_type == 'uint16':
                save_img = np.clip(save_img, 0, 65535).astype('uint16')
            elif input_data_type == 'int16':
                save_img = np.clip(save_img, -32767, 32767).astype('int16')
            else:
                save_img = save_img.astype('int32')

            img_list = list(os.walk(self.datasets_path, topdown=False))[-1][-1]
            img_list.sort()
            test_im_name = img_list[0] if img_list else 'unknown.tif'
            result_name = os.path.join(
                self.pth_path,
                test_im_name.replace('.tif', '')
                + '_E_{}_Iter_{}.tif'.format(
                    str(train_epoch + 1).zfill(2),
                    str(train_iteration + 1).zfill(4),
                ),
            )
            io.imsave(result_name, save_img, check_contrast=False)

    def run(self):
        _bind_visible_gpu(str(self.GPU))
        if self.distributed:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
        if getattr(self, 'seed', None) is not None:
            set_random_seed(int(self.seed))
            print('Random seed fixed: {}'.format(self.seed))
        self.prepare_file()
        train_args = self._args_proxy()
        train_args.sampling_mode = self.sampling_mode
        (
            self.train_name_list,
            self.train_noise_img,
            self.train_coordinate_list,
            self.train_stack_index,
            self.train_stack_means,
        ) = train_preprocess_lessMemoryMulStacks_srdtrans(train_args)
        if self.dtcwt_channel_normalize:
            self.dtcwt_channel_scales = _estimate_fixed_dtcwt_scales(
                self.train_noise_img, int(self.dtcwt_levels)
            )
            print('Fixed global DTCWT scales -----> {}'.format(
                ', '.join('{:.6f}'.format(value)
                          for value in self.dtcwt_channel_scales)
            ))
        if self.representation == 'steerable_fourier' and self.fourier_channel_normalize:
            image_size = min(
                int(self.patch_x),
                min(int(stack.shape[-2]) for stack in self.train_noise_img),
                min(int(stack.shape[-1]) for stack in self.train_noise_img),
            )
            image_size = image_size // 16 * 16
            if image_size < 16:
                raise ValueError('Fourier normalization requires spatial size >= 16')
            self.fourier_channel_scales = _estimate_fixed_fourier_channel_scales(
                self.train_noise_img, image_size
            )
            print('Fixed Fourier channel scales -----> {}'.format(
                ', '.join('{:.6f}'.format(value)
                          for value in self.fourier_channel_scales)
            ))
        if not self.distributed or self.rank == 0:
            self.save_yaml_train()
        self.initialize_network()
        self.distribute_GPU()
        if getattr(self, 'init_ckpt', ''):
            self._load_init_ckpt(self.init_ckpt)
        if not self.no_resume:
            self._try_resume_checkpoint()
        self.train()


# Backward-compatible alias used by train_and_val_srdtrans.py
training_class = training_class_srdtrans
