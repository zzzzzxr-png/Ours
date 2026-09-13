"""Gamma-Poisson posterior trainer: single-Gamma context prior with fixed kappa."""

import datetime
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
import torch.distributed as dist
import yaml
from skimage import io
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _safe_mu_lambda(raw_mu_lambda):
    """Identity-preserving Gamma mean mapping with an extreme-value guard."""
    return raw_mu_lambda.clamp(min=1e-12, max=1e12)

from likelihood.dataset import (
    multibatch_test_save_srdtrans,
    singlebatch_test_save_srdtrans,
    test_preprocess_lessMemoryNoTail_chooseOne_srdtrans,
    testset_srdtrans,
    train_preprocess_lessMemoryMulStacks_srdtrans,
    trainset_srdtrans,
    trainset_temporal_srdtrans,
)
from likelihood.backbone_factory import DEFAULT_SRDTRANS_ROOT, build_denoise_network_srdtrans
from likelihood.sampling import generate_mask_pair, generate_subimages
from likelihood.masks import (
    _DIRECTIONAL_MASK_MEAN_MODES,
    _DIRECTIONAL_MASK_MODES,
    _DUAL_CONTEXT_MASK_MODES,
    _SLICE_MASK_MODES,
    make_directional_mask_mean_pair,
    make_directional_mask_pair,
    make_exhaustive_mask_groups,
    make_n2v_mask_pair,
    make_slice_mask_pair,
    make_training_mask_group,
)
from posterior.losses_gamma import (
    l1_l2_loss,
    masked_l1_l2_loss,
    gamma_nb_nll_single_target as _gamma_nb_nll_single_target,
    gamma_nb_predictive_and_posterior,
    gamma_mixture_nb_predictive_and_posterior,
    gamma_mixture_nb_nll_from_ab,
)
from posterior.gamma_posterior import (
    gamma_ab_from_mu_kappa,
    log_negbinom_pmf,
    count_posterior_log_q,
    posterior_count_moments,
    posterior_mean_x,
)
from posterior.dual_context_prior import (
    dual_axis_context_prior,
    dual_axis_from_sampling_mode,
)
from posterior.analytic_representation import quadrature_contribution


_PARALLEL_TYPES = (nn.DataParallel, DistributedDataParallel)


def _unwrap_model(model):
    return model.module if isinstance(model, _PARALLEL_TYPES) else model


def set_random_seed(seed: int) -> None:
    """Fix Python / NumPy / PyTorch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bind_visible_gpu(gpu: str) -> None:
    """Bind process to physical GPU(s) before any CUDA init (must run before cuda calls)."""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

def cal_snr_srdtrans(noisy_img: np.ndarray, clean_img: np.ndarray) -> float:
    noise_signal_2 = (noisy_img.astype(np.float32) - clean_img.astype(np.float32)) ** 2
    clean_signal_2 = clean_img.astype(np.float32) ** 2
    sum1 = float(clean_signal_2.sum())
    sum2 = float(noise_signal_2.sum())
    if sum2 <= 0:
        return float('inf')
    return 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))



class training_class_srdtrans_gamma:
    """SRDTrans-protocol trainer with optional GT SNR validation."""

    def __init__(self, params_dict):
        self.overlap_factor = 0.5
        self.val_overlap_factor = 0.5
        self.datasets_path = ''
        self.gt_path = ''
        self.n_epochs = 30
        self.fmap = 16
        self.output_dir = './results'
        self.pth_dir = './experiments_srdtrans_protocol'
        self.batch_size = 1
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
        self.sampling_mode = 'spatial'
        # Full-resolution directional masked self-supervision.
        # spatial_mask / temporal_mask keep input shape [B, C, T, H, W] unchanged
        # and compute loss only on sparse masked voxels.
        self.mask_ratio = 0.05
        self.mask_min_dist = 2
        self.lattice_random_phase = True
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
        self.temporal_strides = None
        self.last_squeeze_op = 'conv'
        self.trans_order = 'st'
        self.space_post_norm = False
        self.space_dropout_rate = 0.0
        self.use_msconv_before_trans = False
        self.mask_loss = 'nll'
        self.kappa_mode = 'learned_map'
        self.mpgn_alpha = 5000.0
        self.mpgn_beta = 1600.0
        self.mpgn_offset = 0.0
        self.mpgn_kmax = 512
        self.mpgn_nll_chunk_t = 8
        self.eval_ckpt = None
        self.mpgn_k_tail_tol = 1e-8
        self.mpgn_kappa = 50.0
        self.mpgn_kappa_init = 50.0
        self.mpgn_kappa_min = 1e-4
        self.mpgn_prior_var_min = 1e-12
        self.val_patch_batch = 1
        self.smoke_test_val_batch = False
        self.smoke_test_multigpu = False
        self.smoke_test_batch_candidates = '1,2,4,8,16,32'
        self.smoke_test_memory_fraction = 0.85
        self.save_debug_posterior = False
        self.checkpoint_every_epochs = 5
        self.validation_every_epochs = 1
        self.no_resume = False
        self.eval_val_per_epoch = True
        self.eval_every_iters = 0
        self.val_process_frames = 400
        self.snr_margin = 50
        self.save_test_images_per_epoch = True
        self.visualize_images_per_epoch = False
        self.seed = 1024
        self._eval_cache_ready = False
        self.set_params(params_dict)

        if self.sampling_mode == 'spatial_mask':
            raise ValueError(
                'spatial_mask is split into height_mask and width_mask; '
                'set sampling_mode accordingly.'
            )
        if self.kappa_mode == 'learned_map' and self.mask_loss != 'nll':
            raise ValueError('kappa_mode=learned_map requires mask_loss=nll')
        if int(self.checkpoint_every_epochs) < 1:
            raise ValueError('checkpoint_every_epochs must be positive')
        if int(self.validation_every_epochs) < 1:
            raise ValueError('validation_every_epochs must be positive')
        # 260722 dual-context = learned_map. 260719 fixed-κ = single
        # directional mask + gamma_nb_nll_single_target (else-branch below).
        if (
            self.sampling_mode in _DUAL_CONTEXT_MASK_MODES
            and self.mask_loss == 'nll'
            and self.kappa_mode == 'learned_map'
        ):
            self._use_dual_context = True
        else:
            self._use_dual_context = False

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
        self.ngpu = str(self.GPU).count(',') + 1
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        self.batch_size = 1 if self.distributed else self.ngpu
        if self.is_main_process:
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
            '{}_srdtrans_gamma_{}_{}'.format(self.datasets_name, mode_tag, backbone_tag),
        )

        if not self.is_main_process:
            return

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

    def _load_eval_ckpt(self, ckpt_path):
        """Load a named checkpoint for one-shot validation. Returns 1-based (epoch, iter)."""
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError('eval_ckpt not found: {}'.format(ckpt_path))
        state = torch.load(ckpt_path, map_location='cpu')
        _unwrap_model(self.local_model).load_state_dict(state)
        match = re.search(r'E_(\d+)_Iter_(\d+)\.pth$', os.path.basename(ckpt_path))
        if match:
            epoch_1idx, iter_1idx = int(match.group(1)), int(match.group(2))
        else:
            epoch_1idx, iter_1idx = 1, 1
        print(
            '\033[1;31mEval checkpoint -----> \033[0m{} (epoch {}, iter {})'.format(
                ckpt_path, epoch_1idx, iter_1idx,
            )
        )
        return epoch_1idx, iter_1idx

    def _try_resume_checkpoint(self):
        """Load latest model weights; continue from the next epoch index."""
        self._resume_start_epoch = 0
        found = self._find_latest_checkpoint()
        if found is None:
            return False

        ckpt_path, epoch_1idx, iter_1idx = found
        state = torch.load(ckpt_path, map_location='cpu')
        _unwrap_model(self.local_model).load_state_dict(state)

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
            'checkpoint_every_epochs', 'validation_every_epochs',
            'srdtrans_root', 'embedding_dim', 'num_heads', 'hidden_dim', 'window_size',
            'num_transBlock', 'attn_dropout_rate', 'srdtrans_f_maps', 'input_dropout_rate',
            'sampling_mode',
            'mask_ratio', 'mask_min_dist', 'lattice_random_phase', 'slice_axis', 'seed',
            'trans_order', 'space_post_norm', 'space_dropout_rate',
            'use_msconv_before_trans', 'mask_loss', 'kappa_mode',
            'mpgn_alpha', 'mpgn_beta', 'mpgn_offset', 'mpgn_kmax', 'mpgn_nll_chunk_t',
            'mpgn_k_tail_tol',
            'mpgn_kappa', 'mpgn_kappa_init', 'mpgn_kappa_min', 'mpgn_prior_var_min',
            'val_patch_batch',
        )}
        para['protocol'] = 'srdtrans'
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def initialize_network(self):
        self.local_model = build_denoise_network_srdtrans(self)

    def distribute_GPU(self):
        _bind_visible_gpu(str(self.GPU))
        if not torch.cuda.is_available():
            return
        if self.distributed:
            if self.world_size != self.ngpu:
                raise ValueError(
                    'torchrun world size {} does not match --gpu {!r}'.format(
                        self.world_size, self.GPU
                    )
                )
            torch.cuda.set_device(self.local_rank)
            self.local_model = self.local_model.cuda(self.local_rank)
            self.local_model = DistributedDataParallel(
                self.local_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        elif self.ngpu == 1:
            self.local_model = self.local_model.cuda()
        else:
            raise RuntimeError(
                'multiple GPUs require the DDP launcher; use '
                'scripts/run_multigpu_posterior_gamma.sh {}'.format(self.GPU)
            )
        if self.is_main_process:
            print('\033[1;31mUsing {} GPU(s) with {} -----> \033[0m'.format(
                self.world_size if self.distributed else 1,
                'DDP' if self.distributed else 'single GPU',
            ))

    def smoke_test_distributed(self):
        """One full complex/Gamma DDP step; all ranks must produce one gradient."""
        if not self.distributed:
            raise RuntimeError('--smoke-test-multigpu requires at least two GPUs')
        device = torch.device('cuda', self.local_rank)
        x = torch.randn(
            1, 1, self.patch_t, self.patch_y, self.patch_x, device=device
        ) + 0.1 * self.rank
        candidates = self.local_model(x)
        mu_lambda = _safe_mu_lambda((candidates + 5000.0) / 5000.0)
        a, b = gamma_ab_from_mu_kappa(
            mu_lambda, torch.full_like(mu_lambda, float(self.mpgn_kappa))
        )
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, ::2, ::2, ::2] = True
        loss = gamma_mixture_nb_nll_from_ab(
            a,
            b,
            x + 5000.0,
            mask,
            alpha=5000.0,
            beta=1600.0,
            kmax=int(self.mpgn_kmax),
            chunk_t=int(self.mpgn_nll_chunk_t),
            tail_tol=float(self.mpgn_k_tail_tol),
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in self.local_model.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(
            gradient is None or not torch.isfinite(gradient).all()
            for gradient in gradients
        ):
            raise RuntimeError('DDP smoke test produced a missing/non-finite gradient')
        checksum = torch.stack([gradient.abs().norm() for gradient in gradients]).sum()
        checksums = [torch.zeros_like(checksum) for _ in range(self.world_size)]
        dist.all_gather(checksums, checksum)
        if not torch.allclose(
            torch.stack(checksums), checksums[0].expand(self.world_size), rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError('DDP ranks received different reduced gradients')
        dist.barrier()
        if self.is_main_process:
            with torch.inference_mode():
                validation_output = _unwrap_model(self.local_model)(x)
            if validation_output.shape != candidates.shape:
                raise RuntimeError('rank-0 validation output shape changed')
        dist.barrier()
        if self.is_main_process:
            print(
                'DDP smoke test passed: world_size={} output={} loss={:.6f}'.format(
                    self.world_size, tuple(candidates.shape), loss.detach().item()
                )
            )

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
        optimizer_G = torch.optim.Adam(
            self.local_model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
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
                )
            train_sampler = (
                DistributedSampler(
                    train_data,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=int(self.seed or 0),
                )
                if self.distributed else None
            )
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            trainloader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=self.num_workers,
                generator=loader_generator,
                worker_init_fn=worker_init_fn,
            )
            # Diagnostic for the clamp's zero-gradient region (per candidate).
            metric_device = next(self.local_model.parameters()).device
            clamp_hits = torch.zeros(3, device=metric_device, dtype=torch.float64)
            clamp_total = torch.zeros(3, device=metric_device, dtype=torch.float64)
            if epoch == start_epoch and start_epoch > 0:
                global_iter = start_epoch * len(trainloader)

            for iteration, batch in enumerate(trainloader):
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

                    if getattr(self, '_use_dual_context', False):
                        loss_mask, _group_idx, _num_groups = make_training_mask_group(
                            noisy,
                            mask_ratio=self.mask_ratio,
                            min_dist=self.mask_min_dist,
                        )
                        patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
                        axis = dual_axis_from_sampling_mode(self.sampling_mode)
                        dual_result = dual_axis_context_prior(
                            self.local_model,
                            noisy,
                            loss_mask,
                            img_mean=patch_mean,
                            alpha=self.mpgn_alpha,
                            offset=self.mpgn_offset,
                            axis=axis,
                            kappa_mode=self.kappa_mode,
                            fixed_kappa=self.mpgn_kappa,
                            kappa_min=self.mpgn_kappa_min,
                            min_lambda=1e-12,
                            min_variance=self.mpgn_prior_var_min,
                        )
                        fused = dual_result['fused_prior']
                        y_phys = noisy + patch_mean
                        prob_result = gamma_nb_predictive_and_posterior(
                            fused.a,
                            fused.b,
                            y_phys,
                            alpha=self.mpgn_alpha,
                            beta=self.mpgn_beta,
                            offset=self.mpgn_offset,
                            valid_mask=loss_mask,
                            kmax=int(self.mpgn_kmax),
                            tail_tol=float(self.mpgn_k_tail_tol),
                            chunk_t=int(self.mpgn_nll_chunk_t),
                        )
                        total_loss = prob_result['nll_mean']
                    else:
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
                            masked_input, masked_target, loss_mask = (
                                make_directional_mask_pair(
                                    noisy,
                                    mode=self.sampling_mode,
                                    mask_ratio=self.mask_ratio,
                                    min_dist=self.mask_min_dist,
                                    lattice_random_phase=bool(
                                        getattr(self, 'lattice_random_phase', True)
                                    ),
                                )
                            )

                        noisy_output = self.local_model(masked_input)
                        if getattr(self, 'mask_loss', 'l1l2') == 'nll':
                            patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
                            if noisy_output.shape[1] == 1:
                                total_loss = _gamma_nb_nll_single_target(
                                    noisy_output,
                                    masked_target,
                                    loss_mask,
                                    pred_img_mean=patch_mean,
                                    target_img_mean=patch_mean,
                                    alpha=self.mpgn_alpha,
                                    beta=self.mpgn_beta,
                                    kappa=self.mpgn_kappa,
                                    offset=self.mpgn_offset,
                                    kmax=int(self.mpgn_kmax),
                                    chunk_t=int(self.mpgn_nll_chunk_t),
                                    tail_tol=float(self.mpgn_k_tail_tol),
                                )
                                mixture = None
                            elif noisy_output.shape[1] != 3:
                                raise RuntimeError(
                                    'analytic complex SRDTrans must return three candidates, got {}'.format(
                                        tuple(noisy_output.shape)
                                    )
                                )
                            else:
                                mu_phys = noisy_output + patch_mean
                                clamp_hits += (mu_phys <= float(self.mpgn_offset)).sum(
                                    dim=(0, 2, 3, 4), dtype=torch.float64
                                ).detach()
                                clamp_total += float(mu_phys[:, 0].numel())
                                mu_lambda = _safe_mu_lambda(
                                    (mu_phys - float(self.mpgn_offset)) / float(self.mpgn_alpha)
                                )
                                kappa = torch.full_like(mu_lambda, float(self.mpgn_kappa))
                                a, b = gamma_ab_from_mu_kappa(mu_lambda, kappa)
                                total_loss = gamma_mixture_nb_nll_from_ab(
                                    a,
                                    b,
                                    masked_target + patch_mean,
                                    loss_mask,
                                    alpha=self.mpgn_alpha,
                                    beta=self.mpgn_beta,
                                    offset=self.mpgn_offset,
                                    kmax=int(self.mpgn_kmax),
                                    chunk_t=int(self.mpgn_nll_chunk_t),
                                    tail_tol=float(self.mpgn_k_tail_tol),
                                )
                        else:
                            if noisy_output.shape[1] != 1:
                                noisy_output = noisy_output[:, 0:1]
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

                optimizer_G.zero_grad()
                total_loss.backward()
                optimizer_G.step()
                global_iter += 1

                batches_left = self.n_epochs * len(trainloader) - (
                    epoch * len(trainloader) + iteration)
                time_left = datetime.timedelta(
                    seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()

                if self.is_main_process:
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
                    if self.distributed:
                        dist.barrier()
                    if self.is_main_process:
                        print('\nValidation (global_iter={}) ----->'.format(global_iter))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.test(epoch, iteration)
                    if self.distributed:
                        dist.barrier()
                    self.local_model.train()

                if (iteration + 1) % len(trainloader) == 0:
                    if self.distributed:
                        dist.barrier()
                    if self.is_main_process:
                        print('\n', end=' ')
                        if ((epoch + 1) % int(self.checkpoint_every_epochs) == 0
                                or epoch + 1 == self.n_epochs):
                            self.save_model(epoch, iteration)
                    validate_epoch = (
                        (epoch + 1) % int(self.validation_every_epochs) == 0
                        or epoch + 1 == self.n_epochs
                    )
                    if (self.is_main_process
                            and getattr(self, 'eval_val_per_epoch', False)
                            and int(getattr(self, 'eval_every_iters', 0)) <= 0
                            and validate_epoch):
                        print('Validation ----->')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.test(epoch, iteration)
                    if self.distributed:
                        dist.barrier()
                    if self.is_main_process:
                        self.local_model.train()
                        print('\n', end=' ')

            if self.distributed:
                dist.all_reduce(clamp_hits, op=dist.ReduceOp.SUM)
                dist.all_reduce(clamp_total, op=dist.ReduceOp.SUM)
            if self.is_main_process and clamp_total.sum().item() > 0:
                ratios = (clamp_hits / clamp_total.clamp_min(1)).tolist()
                print(
                    'Clamp ratio (mu_phys <= offset) x/y/t: '
                    '{:.3%}/{:.3%}/{:.3%}'.format(*ratios)
                )

    def save_model(self, epoch, iteration):
        os.makedirs(self.pth_path, exist_ok=True)
        model_save_name = os.path.join(
            self.pth_path,
            'E_{}_Iter_{}.pth'.format(str(epoch + 1).zfill(2), str(iteration + 1).zfill(4)),
        )
        torch.save(_unwrap_model(self.local_model).state_dict(), model_save_name)

    def _dual_context_kwargs(self):
        return dict(
            kappa_mode=self.kappa_mode,
            fixed_kappa=self.mpgn_kappa,
            kappa_min=self.mpgn_kappa_min,
            min_lambda=1e-12,
            min_variance=self.mpgn_prior_var_min,
        )

    def smoke_test_val_patch_batch(self, sample_patch, sample_mask):
        """Find largest safe val_patch_batch for dual-replacement forward."""
        import json

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for validation batch smoke test')

        candidates = [
            int(x.strip())
            for x in str(self.smoke_test_batch_candidates).split(',')
            if x.strip()
        ]
        model = _unwrap_model(self.local_model)
        model.eval()
        axis = dual_axis_from_sampling_mode(self.sampling_mode)
        img_mean = float(self._eval_img_mean) if self._eval_cache_ready else 0.0
        best = None
        rows = []

        for patch_batch in candidates:
            if 2 * patch_batch < self.ngpu:
                rows.append({
                    'val_patch_batch': patch_batch,
                    'skipped': True,
                    'reason': '2*val_patch_batch < ngpu',
                })
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                patch = sample_patch.repeat(patch_batch, 1, 1, 1, 1)
                mask = sample_mask.repeat(patch_batch, 1, 1, 1, 1)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    result = dual_axis_context_prior(
                        model,
                        patch,
                        mask,
                        img_mean=img_mean,
                        alpha=self.mpgn_alpha,
                        offset=self.mpgn_offset,
                        axis=axis,
                        **self._dual_context_kwargs(),
                    )
                    _ = (
                        result['fused_prior'].mu_lambda.mean()
                        + result['fused_prior'].var_lambda.mean()
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated()
                rows.append({
                    'val_patch_batch': patch_batch,
                    'actual_variant_batch': 2 * patch_batch,
                    'peak_memory_bytes': int(peak),
                    'elapsed_sec_one_group': float(elapsed),
                    'success': True,
                })
                best = patch_batch
                del patch, mask, result
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                rows.append({
                    'val_patch_batch': patch_batch,
                    'actual_variant_batch': 2 * patch_batch,
                    'success': False,
                    'error': 'CUDA out of memory',
                })
                torch.cuda.empty_cache()
                break

        if best is None:
            raise RuntimeError('Even val_patch_batch=1 failed the smoke test.')

        out_path = os.path.join(self.pth_path, 'val_batch_smoke_test.json')
        with open(out_path, 'w') as f:
            json.dump({'best': best, 'rows': rows}, f, indent=2)
        print('Validation batch smoke-test results:')
        for row in rows:
            print(row)
        print('\nSuggested: --val_patch_batch {}\nSaved: {}'.format(best, out_path))
        return best, rows

    def _stitch_volume_maps(
        self, volumes, np_maps, single_coordinate
    ):
        """np_maps values are [B,T,H,W]; volumes are full stitched arrays."""
        postprocess_turn = next(iter(np_maps.values())).shape[0]
        if postprocess_turn > 1:
            for batch_id in range(postprocess_turn):
                for key, volume in volumes.items():
                    output_patch, _, stack_start_w, stack_end_w, stack_start_h, \
                        stack_end_h, stack_start_s, stack_end_s = \
                        multibatch_test_save_srdtrans(
                            single_coordinate,
                            batch_id,
                            np_maps[key],
                            np_maps[key],
                        )
                    volume[
                        stack_start_s:stack_end_s,
                        stack_start_h:stack_end_h,
                        stack_start_w:stack_end_w,
                    ] = output_patch
        else:
            for key, volume in volumes.items():
                patch = np_maps[key][0]
                output_patch, _, stack_start_w, stack_end_w, stack_start_h, \
                    stack_end_h, stack_start_s, stack_end_s = \
                    singlebatch_test_save_srdtrans(
                        single_coordinate, patch, patch
                    )
                volume[
                    stack_start_s:stack_end_s,
                    stack_start_h:stack_end_h,
                    stack_start_w:stack_end_w,
                ] = output_patch

    def test(self, train_epoch, train_iteration):
        """Dispatch: 260722 exhaustive dual vs 260719 one-pass fixed-κ."""
        if getattr(self, '_use_dual_context', False):
            return self._test_dual_exhaustive(train_epoch, train_iteration)
        return self._test_fixed_kappa_onepass(train_epoch, train_iteration)

    def _test_fixed_kappa_onepass(self, train_epoch, train_iteration):
        """260719 val: no mask, one-pass μ, then Gamma-NB posterior mean.

        Metrics match the old table: SNR_denoised / SNR_noisy / κ.
        No global intensity rescale (GAMMA_QUICKSTART).
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._prepare_eval_cache()

        name_list = self._eval_name_list
        noise_img = self._eval_noise_img
        coordinate_list = self._eval_coordinate_list
        img_mean = self._eval_img_mean
        ref_img = self._eval_ref_img

        time_start = time.time()
        vol_post = np.zeros(noise_img.shape, dtype=np.float32)
        vol_map = np.zeros(noise_img.shape, dtype=np.float32)
        vol_map_boundary = np.zeros(noise_img.shape, dtype=np.float32)
        vol_noisy = np.zeros(noise_img.shape, dtype=np.float32)
        vol_candidates = {
            direction: np.zeros(noise_img.shape, dtype=np.float32)
            for direction in ('x', 'y', 't')
        }

        test_data = testset_srdtrans(name_list, coordinate_list, noise_img)
        testloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        cuda = torch.cuda.is_available()
        dtype = torch.float32
        device = torch.device('cuda' if cuda else 'cpu')
        kappa = float(self.mpgn_kappa)
        alpha_t = torch.as_tensor(self.mpgn_alpha, dtype=dtype, device=device)
        beta_t = torch.as_tensor(self.mpgn_beta, dtype=dtype, device=device).clamp(
            min=1e-12
        )
        offset_t = torch.as_tensor(self.mpgn_offset, dtype=dtype, device=device)
        img_mean_t = torch.as_tensor(img_mean, dtype=dtype, device=device)

        self.local_model.eval()
        q = {'x': [], 'y': [], 't': []}
        with torch.inference_mode():
            for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                if cuda:
                    noise_patch = noise_patch.cuda()
                local = _unwrap_model(self.local_model)
                if hasattr(local, 'forward_with_complex'):
                    mu_centered, complex_output = local(
                        noise_patch, return_complex=True
                    )
                    for index, direction in enumerate(('x', 'y', 't')):
                        q[direction].extend(
                            quadrature_contribution(
                                complex_output[:, index:index + 1], direction
                            ).cpu().tolist()
                        )
                else:
                    mu_centered = local(noise_patch)
                if mu_centered.shape[1] not in (1, 3):
                    raise RuntimeError(
                        'expected one prior or three analytic candidates, got {}'.format(
                            tuple(mu_centered.shape)
                        )
                    )
                y_phys = noise_patch + img_mean_t
                mu_phys = mu_centered + img_mean_t
                mu_lambda = (mu_phys - offset_t) / alpha_t
                mu_lambda = _safe_mu_lambda(mu_lambda)
                kappa_t = torch.full_like(mu_lambda, kappa)
                a, b = gamma_ab_from_mu_kappa(mu_lambda, kappa_t)
                posterior_kwargs = dict(
                    alpha=alpha_t, beta=beta_t, offset=offset_t,
                    kmax=int(self.mpgn_kmax),
                    tail_tol=float(self.mpgn_k_tail_tol),
                    chunk_t=int(self.mpgn_nll_chunk_t),
                )
                if mu_centered.shape[1] == 3:
                    posterior = gamma_mixture_nb_predictive_and_posterior(
                        a, b, y_phys, compute_map=True, **posterior_kwargs
                    )
                    x_map = posterior['x_map_phys']
                    map_boundary = posterior['map_boundary'].to(dtype=x_map.dtype)
                else:
                    posterior = gamma_nb_predictive_and_posterior(
                        a, b, y_phys, **posterior_kwargs
                    )
                    x_map = posterior['x_post_phys']
                    map_boundary = torch.zeros_like(x_map)
                x_post = posterior['x_post_phys']
                maps = {
                    'post': x_post,
                    'map': x_map,
                    'map_boundary': map_boundary,
                    'noisy': y_phys,
                }
                if mu_phys.shape[1] == 3:
                    maps.update({
                        'candidate_{}'.format(direction): mu_phys[:, index:index + 1]
                        for index, direction in enumerate(('x', 'y', 't'))
                    })
                np_maps = {}
                for k, v in maps.items():
                    arr = v.detach().cpu().numpy()
                    if arr.ndim == 5 and arr.shape[1] == 1:
                        arr = arr[:, 0]
                    np_maps[k] = arr
                volumes = {
                    'post': vol_post,
                    'map': vol_map,
                    'map_boundary': vol_map_boundary,
                    'noisy': vol_noisy,
                }
                if mu_phys.shape[1] == 3:
                    volumes.update({
                        'candidate_{}'.format(direction): volume
                        for direction, volume in vol_candidates.items()
                    })
                self._stitch_volume_maps(volumes, np_maps, single_coordinate)
                print(
                    '\r [Patch %d/%d]'
                    % (iteration + 1, len(testloader)),
                    end=' ',
                )

        print('\n', end=' ')
        T = min(vol_post.shape[0], vol_noisy.shape[0], ref_img.shape[0])
        eval_t = min(int(self.val_process_frames), T)
        s = self.snr_margin
        e = eval_t - self.snr_margin
        if e <= s:
            s, e = 0, eval_t
        snr_post = cal_snr_srdtrans(vol_post[s:e], ref_img[s:e])
        snr_map = cal_snr_srdtrans(vol_map[s:e], ref_img[s:e])
        snr_noisy = cal_snr_srdtrans(vol_noisy[s:e], ref_img[s:e])
        snr_candidates = {
            direction: (
                cal_snr_srdtrans(volume[s:e], ref_img[s:e])
                if q[direction] else float('nan')
            )
            for direction, volume in vol_candidates.items()
        }
        map_boundary_fraction = float(vol_map_boundary[s:e].mean())
        q_mean = {
            direction: float(np.mean(values)) if values else float('nan')
            for direction, values in q.items()
        }
        print(
            'SNR (frames {:d}:{:d}; posterior mean / MAP / noisy vs GT) '
            '-----> {:.4f} dB / {:.4f} dB / {:.4f} dB; '
            'candidates x/y/t {:.4f}/{:.4f}/{:.4f} dB; MAP@0 {:.2%}'.format(
                s, e, snr_post, snr_map, snr_noisy,
                snr_candidates['x'], snr_candidates['y'], snr_candidates['t'],
                map_boundary_fraction,
            )
        )
        metrics_path = os.path.join(self.pth_path, 'val_metrics.md')
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                f.write(
                    '| Epoch | Iteration | SNR_mean (dB) | SNR_MAP (dB) | SNR_noisy (dB) | MAP@0 | κ | SNR_x | SNR_y | SNR_t | qx | qy | qt |\n'
                )
                f.write(
                    '| ----- | --------- | ------------- | ------------ | -------------- | ----- | --- | ----- | ----- | ----- | -- | -- | -- |\n'
                )
        with open(metrics_path, 'a') as f:
            f.write(
                '| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.6f} | {:.1f} | {:.4f} | {:.4f} | {:.4f} | {:.6f} | {:.6f} | {:.6f} |\n'.format(
                    train_epoch + 1,
                    train_iteration + 1,
                    snr_post,
                    snr_map,
                    snr_noisy,
                    map_boundary_fraction,
                    kappa,
                    snr_candidates['x'],
                    snr_candidates['y'],
                    snr_candidates['t'],
                    q_mean['x'],
                    q_mean['y'],
                    q_mean['t'],
                )
            )
        elapsed = time.time() - time_start
        print('fixed-κ one-pass val {:.1f}s'.format(elapsed))

        if self.save_test_images_per_epoch:
            img_list = list(os.walk(self.datasets_path, topdown=False))[-1][-1]
            img_list.sort()
            stem = (img_list[0] if img_list else 'unknown.tif').replace('.tif', '')
            tag = 'E_{}_Iter_{}'.format(
                str(train_epoch + 1).zfill(2),
                str(train_iteration + 1).zfill(4),
            )
            input_data_type = self._eval_input_data_type
            for estimator, volume in (('posterior_mean', vol_post), ('posterior_map', vol_map)):
                out = volume[s:e]
                path = os.path.join(
                    self.pth_path, '{}_{}_{}.tif'.format(stem, tag, estimator)
                )
                if input_data_type == 'uint16':
                    out = np.clip(out, 0, 65535).astype('uint16')
                else:
                    out = out.astype(np.float32)
                io.imsave(path, out, check_contrast=False)

    def _test_dual_exhaustive(self, train_epoch, train_iteration):
        """Exhaustive dual-context prior + one-shot MPGN posterior correction."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._prepare_eval_cache()

        name_list = self._eval_name_list
        noise_img = self._eval_noise_img
        coordinate_list = self._eval_coordinate_list
        img_mean = self._eval_img_mean
        input_data_type = self._eval_input_data_type
        ref_img = self._eval_ref_img

        time_start = time.time()
        vol_prior = np.zeros(noise_img.shape, dtype=np.float32)
        vol_post = np.zeros(noise_img.shape, dtype=np.float32)
        vol_corr = np.zeros(noise_img.shape, dtype=np.float32)
        vol_kappa = np.zeros(noise_img.shape, dtype=np.float32)
        vol_noisy = np.zeros(noise_img.shape, dtype=np.float32)
        vol_prior_var = np.zeros(noise_img.shape, dtype=np.float32)
        vol_post_var = np.zeros(noise_img.shape, dtype=np.float32)
        vol_kbar = np.zeros(noise_img.shape, dtype=np.float32)
        vol_var_k = np.zeros(noise_img.shape, dtype=np.float32)
        vol_entropy = np.zeros(noise_img.shape, dtype=np.float32)

        val_bs = max(1, int(getattr(self, 'val_patch_batch', 1)))
        test_data = testset_srdtrans(name_list, coordinate_list, noise_img)
        testloader = DataLoader(
            test_data,
            batch_size=val_bs,
            shuffle=False,
            num_workers=self.num_workers,
        )

        cuda = torch.cuda.is_available()
        dtype = torch.float32
        device = torch.device('cuda' if cuda else 'cpu')
        self.local_model.eval()
        eval_model = _unwrap_model(self.local_model)

        # Build exhaustive groups for the patch shape used in this dataset.
        sample_noise, _ = test_data[0]
        sample_noise = sample_noise.unsqueeze(0)
        if cuda:
            sample_noise = sample_noise.cuda()
        exhaustive_groups = make_exhaustive_mask_groups(
            sample_noise,
            mask_ratio=self.mask_ratio,
            min_dist=self.mask_min_dist,
        )
        num_groups = len(exhaustive_groups)
        axis = dual_axis_from_sampling_mode(self.sampling_mode)

        if getattr(self, 'smoke_test_val_batch', False):
            self.smoke_test_val_patch_batch(
                sample_noise, exhaustive_groups[0]
            )
            return

        alpha_t = torch.as_tensor(self.mpgn_alpha, dtype=dtype, device=device)
        beta_t = torch.as_tensor(self.mpgn_beta, dtype=dtype, device=device).clamp(
            min=1e-12
        )
        offset_t = torch.as_tensor(self.mpgn_offset, dtype=dtype, device=device)
        img_mean_t = torch.as_tensor(img_mean, dtype=dtype, device=device)
        var_floor = float(self.mpgn_prior_var_min)

        n_patches_done = 0
        with torch.inference_mode():
            for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                if cuda:
                    noise_patch = noise_patch.cuda()
                B = noise_patch.shape[0]
                fused_mu = torch.empty_like(noise_patch)
                fused_var = torch.empty_like(noise_patch)
                coverage = torch.zeros(
                    (B, 1) + noise_patch.shape[2:],
                    dtype=torch.int16,
                    device=noise_patch.device,
                )

                for loss_mask in exhaustive_groups:
                    loss_mask_b = loss_mask[:1].expand(B, -1, -1, -1, -1).to(
                        device=noise_patch.device
                    )
                    dual = dual_axis_context_prior(
                        eval_model,
                        noise_patch,
                        loss_mask_b,
                        img_mean=img_mean_t,
                        alpha=self.mpgn_alpha,
                        offset=self.mpgn_offset,
                        axis=axis,
                        **self._dual_context_kwargs(),
                    )
                    prior = dual['fused_prior']
                    fused_mu = torch.where(loss_mask_b, prior.mu_lambda, fused_mu)
                    fused_var = torch.where(loss_mask_b, prior.var_lambda, fused_var)
                    coverage += loss_mask_b.to(torch.int16)

                if not torch.all(coverage == 1):
                    raise RuntimeError(
                        'Exhaustive validation masks did not cover each voxel '
                        'exactly once. coverage range = [{}, {}]'.format(
                            int(coverage.min().item()),
                            int(coverage.max().item()),
                        )
                    )

                a = fused_mu.square() / fused_var.clamp_min(var_floor)
                b = fused_mu / fused_var.clamp_min(var_floor)
                y_phys = noise_patch + img_mean_t
                posterior = gamma_nb_predictive_and_posterior(
                    a,
                    b,
                    y_phys,
                    alpha=alpha_t,
                    beta=beta_t,
                    offset=offset_t,
                    kmax=int(self.mpgn_kmax),
                    tail_tol=float(self.mpgn_k_tail_tol),
                    chunk_t=int(self.mpgn_nll_chunk_t),
                )

                x_prior = posterior['x_prior_phys']
                x_post = posterior['x_post_phys']
                correction = posterior['correction']
                kappa_map = b
                maps = {
                    'prior': x_prior,
                    'post': x_post,
                    'corr': correction,
                    'kappa': kappa_map,
                    'noisy': y_phys,
                    'prior_var': fused_var,
                    'post_var': posterior['posterior_var_x'],
                    'kbar': posterior['kbar'],
                    'var_k': posterior['var_k'],
                    'entropy': posterior['entropy'],
                }
                np_maps = {}
                for k, v in maps.items():
                    arr = v.detach().cpu().numpy()
                    if arr.ndim == 5 and arr.shape[1] == 1:
                        arr = arr[:, 0]
                    np_maps[k] = arr  # [B,T,H,W]

                vol_targets = {
                    'prior': vol_prior,
                    'post': vol_post,
                    'corr': vol_corr,
                    'kappa': vol_kappa,
                    'noisy': vol_noisy,
                    'prior_var': vol_prior_var,
                    'post_var': vol_post_var,
                    'kbar': vol_kbar,
                    'var_k': vol_var_k,
                    'entropy': vol_entropy,
                }
                self._stitch_volume_maps(vol_targets, np_maps, single_coordinate)
                n_patches_done += np_maps['post'].shape[0]
                print(
                    '\r [Patch batch %d/%d] groups=%d val_patch_batch=%d'
                    % (iteration + 1, len(testloader), num_groups, val_bs),
                    end=' ',
                )

        print('\n', end=' ')
        elapsed = time.time() - time_start
        pps = n_patches_done / max(elapsed, 1e-6)

        def _win(arr):
            T = min(arr.shape[0], ref_img.shape[0])
            eval_t = min(int(self.val_process_frames), T)
            s = self.snr_margin
            e = eval_t - self.snr_margin
            if e <= s:
                s, e = 0, eval_t
            return arr[s:e], ref_img[s:e], s, e, eval_t

        prior_w, gt_w, s, e, eval_t = _win(vol_prior)
        post_w, _, _, _, _ = _win(vol_post)
        noisy_w, _, _, _, _ = _win(vol_noisy)
        snr_prior = cal_snr_srdtrans(prior_w, gt_w)
        snr_post = cal_snr_srdtrans(post_w, gt_w)
        snr_noisy = cal_snr_srdtrans(noisy_w, gt_w)
        snr_delta = snr_post - snr_prior

        kappa_flat = vol_kappa[s:e].reshape(-1)
        metrics = {
            'SNR_noisy': snr_noisy,
            'SNR_prior': snr_prior,
            'SNR_posterior': snr_post,
            'SNR_posterior_minus_prior': snr_delta,
            'mean_abs_correction': float(np.abs(vol_corr[s:e]).mean()),
            'kappa_mean': float(kappa_flat.mean()),
            'kappa_median': float(np.median(kappa_flat)),
            'kappa_p05': float(np.percentile(kappa_flat, 5)),
            'kappa_p95': float(np.percentile(kappa_flat, 95)),
            'prior_variance_mean': float(vol_prior_var[s:e].mean()),
            'posterior_variance_mean': float(vol_post_var[s:e].mean()),
            'number_of_mask_groups': num_groups,
            'val_patch_batch': val_bs,
            'total_validation_seconds': elapsed,
            'patches_per_second': pps,
        }
        print(
            'SNR prior/post/noisy ({:d}:{:d}) -----> {:.4f} / {:.4f} / {:.4f} dB'.format(
                s, e, snr_prior, snr_post, snr_noisy
            )
        )

        metrics_path = os.path.join(self.pth_path, 'val_metrics.md')
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                f.write(
                    '| Epoch | Iteration | SNR_prior | SNR_post | SNR_noisy | '
                    'Δ(post-prior) | κ_mean | groups | val_bs | sec |\n'
                )
                f.write(
                    '| ----- | --------- | --------- | -------- | --------- | '
                    '------------- | ------ | ------ | ------ | --- |\n'
                )
        with open(metrics_path, 'a') as f:
            f.write(
                '| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {:.1f} |\n'.format(
                    train_epoch + 1,
                    train_iteration + 1,
                    snr_prior,
                    snr_post,
                    snr_noisy,
                    snr_delta,
                    metrics['kappa_mean'],
                    num_groups,
                    val_bs,
                    elapsed,
                )
            )

        if self.save_test_images_per_epoch:
            img_list = list(os.walk(self.datasets_path, topdown=False))[-1][-1]
            img_list.sort()
            stem = (img_list[0] if img_list else 'unknown.tif').replace('.tif', '')
            tag = 'E_{}_Iter_{}'.format(
                str(train_epoch + 1).zfill(2),
                str(train_iteration + 1).zfill(4),
            )

            def _save(name, arr):
                out = arr[s:e]
                path = os.path.join(
                    self.pth_path, '{}_{}_{}.tif'.format(stem, tag, name)
                )
                if input_data_type == 'uint16' and name in ('prior', 'posterior', 'noisy'):
                    out = np.clip(out, 0, 65535).astype('uint16')
                else:
                    out = out.astype(np.float32)
                io.imsave(path, out, check_contrast=False)

            _save('prior', vol_prior)
            _save('posterior', vol_post)
            _save('correction', vol_corr)
            _save('kappa', vol_kappa)
            if self.save_debug_posterior:
                _save('prior_variance', vol_prior_var)
                _save('posterior_variance', vol_post_var)
                _save('kbar', vol_kbar)
                _save('var_k', vol_var_k)
                _save('entropy', vol_entropy)
                _save('noisy', vol_noisy)

        print('Validation metrics:', metrics)

    def run(self):
        _bind_visible_gpu(str(self.GPU))
        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError('NCCL DDP requires CUDA')
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend='nccl', device_id=torch.device('cuda', self.local_rank)
            )
        if getattr(self, 'seed', None) is not None:
            set_random_seed(int(self.seed))
            if self.is_main_process:
                print('Random seed fixed: {}'.format(self.seed))
        self.prepare_file()
        if self.distributed:
            dist.barrier()

        if self.smoke_test_multigpu:
            self.initialize_network()
            self.distribute_GPU()
            self.smoke_test_distributed()
            dist.destroy_process_group()
            return

        eval_ckpt = getattr(self, 'eval_ckpt', None)
        if eval_ckpt:
            # One-shot val of a frozen ckpt: skip train cache and resume.
            if self.is_main_process:
                self.save_yaml_train()
            self.initialize_network()
            self.distribute_GPU()
            self.save_test_images_per_epoch = False
            epoch_1idx, iter_1idx = self._load_eval_ckpt(eval_ckpt)
            if self.distributed:
                dist.barrier()
            if self.is_main_process:
                self.test(epoch_1idx - 1, iter_1idx - 1)
            if self.distributed:
                dist.barrier()
                dist.destroy_process_group()
            return

        train_args = self._args_proxy()
        train_args.sampling_mode = self.sampling_mode
        (
            self.train_name_list,
            self.train_noise_img,
            self.train_coordinate_list,
            self.train_stack_index,
            self.train_stack_means,
        ) = train_preprocess_lessMemoryMulStacks_srdtrans(train_args)
        if self.is_main_process:
            self.save_yaml_train()
        self.initialize_network()
        self.distribute_GPU()
        if not self.no_resume:
            self._try_resume_checkpoint()

        if getattr(self, 'smoke_test_val_batch', False):
            self._prepare_eval_cache()
            sample_noise = torch.from_numpy(
                np.expand_dims(
                    self._eval_noise_img[
                        : self.patch_t,
                        : self.patch_y,
                        : self.patch_x,
                    ].astype(np.float32),
                    0,
                )
            ).float().unsqueeze(0)  # [1,1,T,H,W]
            # mean-center like dataloader patches
            sample_noise = sample_noise - float(self._eval_img_mean)
            if torch.cuda.is_available():
                sample_noise = sample_noise.cuda()
            from likelihood.masks import make_exhaustive_mask_groups
            groups = make_exhaustive_mask_groups(
                sample_noise,
                mask_ratio=self.mask_ratio,
                min_dist=self.mask_min_dist,
            )
            self.smoke_test_val_patch_batch(sample_noise, groups[0])
            print('Smoke test finished; exiting without training.')
            return

        self.train()
        if self.distributed:
            dist.destroy_process_group()


# Backward-compatible alias
training_class = training_class_srdtrans_gamma
