import os
import csv
import re
import shutil
import datetime
import math
import random
import time

import numpy as np
import yaml
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from .backbone_factory import (
    DEFAULT_SRDTRANS_ROOT,
    build_denoise_network,
    get_srdtrans_input_shape,
    import_srdtrans_v2_class,
    _build_transformer_unet,
)
from likelihood.dataset import (
    multibatch_test_save_srdtrans,
    singlebatch_test_save_srdtrans,
    test_preprocess_lessMemoryNoTail_chooseOne_srdtrans,
    testset_srdtrans,
    train_preprocess_lessMemoryMulStacks_srdtrans,
    trainset_srdtrans,
    trainset_temporal_srdtrans,
)
from likelihood.losses import mpgn_nll_single_target as _mpgn_nll_single_target
from skimage import io
try:
    from likelihood.movie_display import test_img_display, display_img
except ImportError:
    def test_img_display(*args, **kwargs): pass
    def display_img(*args, **kwargs): pass

from .diagnostics import (
    load_pretrained_prior,
    cal_snr,
    cal_snr_scaled,
    _accumulate_val_patch,
    _check_train_coordinate_consistency,
    _check_val_coordinate_consistency,
    _format_float_for_path,
)
from .dataset import trainset_temporal_2targets
from .masks import (
    _checkerboard_diag_h_pair,
    _checkerboard_diag_w_pair,
    _stripe_h_pair,
    _stripe_w_pair,
    _stripe_t_pair,
    _srd_generate_mask_pair,
    _srd_generate_subimages,
    _make_masked_replacement,
)


class training_class():
    """
    Class implementing training process
    """

    def __init__(self, params_dict):
        """
        Constructor class for training process

        Args:
           params_dict: dict
               The collection of training params set by users
        Returns:
           self

        """
        self.overlap_factor = 0.5
        self.datasets_path = ''
        self.n_epochs = 20
        self.fmap = 16
        self.output_dir = './results'
        self.pth_dir = './pth'
        self.batch_size = 1
        self.patch_t = 150
        self.patch_x = 150
        self.patch_y = 150
        self.gap_y = 115
        self.gap_x = 115
        self.gap_t = 115
        self.lr = 1e-4
        self.lr_decay_every = 40000
        self.lr_decay_gamma = 0.5
        self.lr_min = 3e-6
        self.b1 = 0.9
        self.b2 = 0.999
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.train_datasets_size = 2000
        self.select_img_num = 1000
        self.test_datasize = 400
        self.visualize_images_per_epoch = False
        self.save_test_images_per_epoch = False
        self.colab_display = False
        self.result_display = ''

        # Validation settings
        # Path to ground-truth tif used for SNR metrics.
        self.gt_path = ''
        # Number of frames (from the start) to run through inference during val.
        self.val_process_frames = 400
        # Frames to drop from each temporal end before computing SNR.
        # SNR is computed on [snr_margin : val_process_frames - snr_margin].
        self.snr_margin = 50
        # Overlap factor used during validation inference.
        # If None, falls back to overlap_factor.
        self.val_overlap_factor = None
        # Enable periodic validation (requires gt_path to be set).
        self.eval_val_per_epoch = False
        # Run validation every N training iterations (0 = epoch-end only).
        self.eval_every_iters = 0

        # Sampling mode: 'temporal' (original DeepCAD temporal interleaving),
        # 'spatial' (SRDTrans spatial-neighbor masking, 2x raw crop),
        # 'spatial_ori' (SRDTrans original: patch-sized crop, H/W halved by mask),
        # 'spatial_1target', 'spatial_checker', 'spatial_stripe',
        # 'temporal_checker', 'temporal_2targets'.
        self.sampling_mode = 'temporal'
        self.mask_ratio = 0.01
        self.mask_min_dist = 3

        # Denoise backbone: 'unet', 'unet_unroll', or 'srdtrans_v2'.
        self.backbone = 'unet'
        self.prior_backbone = 'srdtrans_v2'
        self.srdtrans_root = DEFAULT_SRDTRANS_ROOT
        self.embedding_dim = 128
        self.num_heads = 8
        self.hidden_dim = 128 * 4
        self.window_size = 7
        self.num_transBlock = 1
        self.attn_dropout_rate = 0.1
        self.srdtrans_f_maps = [8, 16, 32, 64]
        self.input_dropout_rate = 0.0

        # If True, delete an existing experiment folder and restart instead of
        # raising an error.
        self.no_resume = False

        # Loss mode: always exact MPGN likelihood for transformer unroll.
        self.use_likelihood = True
        self.mpgn_nll_chunk_t = 8
        # MPGN noise model parameters (physical acquisition units).
        # alpha: photon gain;  beta: read-noise variance (not std);
        # offset: baseline;  kmax: truncation of Poisson sum.
        self.mpgn_alpha = 8160.0
        self.mpgn_beta = 25.0
        self.mpgn_offset = 0.0
        self.mpgn_kmax = 32
        self.mpgn_min_signal = 1e-6
        self.mpgn_quant_step = 1.0
        self.mpgn_clip_low = None
        self.mpgn_clip_high = None
        self.mpgn_boundary_atol = 1e-6

        # Monotone-schedule exact-MPGN proximal unroll settings.
        self.unroll_steps = 2
        self.unroll_rho_min = 1e-6
        self.unroll_eps = 1e-8

        self.unroll_rho_sched_min = 1.0
        self.unroll_rho_sched_max = 10.0

        self.unroll_corr_n_iter = 2
        self.unroll_corr_chunk_t = 8
        self.unroll_corr_step_cap_sigma = 1.0

        self.unroll_gradient_checkpointing = False
        self.user_batch_size = None

        self.prior_pretrained_path = None
        self.prior_pretrained_root = (
            '/data/zhouxirou/DeepCAD-RT/DeepCAD_RT_pytorch/'
            '260622_experiments_srdprotocol_srdtrans_mask_T1000H245W245'
        )
        self.prior_pretrained_epoch = 25

        self.visualize_unroll = True
        self.visualize_every = 1
        self.visualize_t_index = -1

        # Global mean used to convert centered tensors back to physical units.
        # Computed automatically in train_preprocess_lessMemoryMulStacks.
        self.noisy_train_img_mean = 0.0
        # Per-stack means (one entry per loaded tif).
        self.noise_im_means = []

        # Checkerboard spatial sampling: separate H/W coordinate lists.
        self.name_list_h = []
        self.coordinate_list_h = {}
        self.stack_index_h = []

        self.name_list_w = []
        self.coordinate_list_w = {}
        self.stack_index_w = []

        # Eval cache (populated once by _prepare_eval_cache).
        self._eval_cache_ready = False
        self._eval_name_list = None
        self._eval_noise_img = None
        self._eval_coordinate_list = None
        self._eval_img_mean = None
        self._eval_input_data_type = None
        self._eval_ref_img = None

        self.set_params(params_dict)

    def _get_train_crop_xy(self):
        """Return raw training crop size before any sampling-mode transform.

        For spatial-neighbor sampling (spatial / spatial_1target), a 2x2 block
        is reduced to one pixel in each generated subimage, so the raw H/W crop
        must be doubled to keep the network input size equal to patch_y/patch_x.

        spatial_ori follows original SRDTrans: crop patch_x x patch_y directly,
        network input spatial size is patch/2 after mask downsampling.
        """
        mode = getattr(self, "sampling_mode", "temporal")

        if mode in ("spatial", "spatial_1target"):
            return int(self.patch_x) * 2, int(self.patch_y) * 2

        if mode in _FIXED_HW_SPATIAL_MODES:
            # Placeholder only. Real H/W coordinate lists are generated separately.
            return int(self.patch_x), int(self.patch_y)

        if mode in _FIXED_TW_TEMPORAL_MODES:
            # Placeholder only. Real T-H / T-W coordinate lists are generated separately.
            return int(self.patch_x), int(self.patch_y)

        return int(self.patch_x), int(self.patch_y)

    def _get_train_window_t(self):
        """Temporal length of the raw crop window used for coordinate generation."""
        mode = getattr(self, "sampling_mode", "temporal")
        if mode in ("spatial", "spatial_1target", "spatial_ori"):
            return int(self.patch_t)
        if mode in _SPARSE_MASK_MODES:
            return int(self.patch_t)
        return int(self.patch_t) * 2

    def run(self):
        """
        General function for training DeepCAD network.

        """
        self.prepare_file()
        self._print_checkpoint_init_summary('after prepare_file')
        self.train_preprocess_lessMemoryMulStacks()
        self.save_yaml_train()
        self.initialize_network()
        if not self.resume_checkpoint:
            self._load_prior_pretrained()
        self.distribute_GPU()
        if self.resume_checkpoint:
            state = torch.load(self.resume_checkpoint, map_location='cpu')
            if isinstance(self.local_model, nn.DataParallel):
                self.local_model.module.load_state_dict(state)
            else:
                self.local_model.load_state_dict(state)
            print('\033[1;31mLoaded resume weights from -----> {}\033[0m'.format(
                self.resume_checkpoint))
        self._print_checkpoint_init_summary('after weight load')
        if not self.resume_checkpoint:
            self._write_pretrained_ep_minus1_metrics()
        self._sanity_check_prior_net()
        if not self.resume_checkpoint:
            self._run_ep0_validation()
        self.train()

    @staticmethod
    def _find_latest_train_checkpoint(pth_path):
        pattern = re.compile(r'^E_(\d+)_Iter_(\d+)\.pth$')
        best = None
        best_name = None
        for name in os.listdir(pth_path):
            m = pattern.match(name)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)))
            if best is None or key > best:
                best = key
                best_name = name
        if best_name is None:
            return None
        return os.path.join(pth_path, best_name)

    @staticmethod
    def _find_train_checkpoint_at_epoch(pth_path, epoch):
        """Return E_XX_Iter_*.pth for a fixed epoch (e.g. ep25 SRDTrans pretrain)."""
        epoch_tag = 'E_{:02d}_'.format(int(epoch))
        matches = []
        if not os.path.isdir(pth_path):
            return None
        for name in os.listdir(pth_path):
            if not name.startswith(epoch_tag) or not name.endswith('.pth'):
                continue
            if not re.match(r'^E_\d+_Iter_\d+\.pth$', name):
                continue
            matches.append(name)
        if not matches:
            return None
        matches.sort()
        return os.path.join(pth_path, matches[-1])

    def prepare_file(self):
        """
        Make data folder to store training results
        Important Fields:
            self.datasets_name: the sub folder of the dataset
            self.pth_path: the folder for pth file storage

        """
        if self.datasets_path[-1] != '/':
            self.datasets_name = self.datasets_path.split("/")[-1]
        else:
            self.datasets_name = self.datasets_path.split("/")[-2]
        pth_name = self.datasets_name + '_' + self.sampling_mode
        if getattr(self, 'backbone', 'unet') not in (None, '', 'unet', '3dunet', '3DUNet'):
            pth_name += '_' + str(self.backbone)
        if getattr(self, 'backbone', 'unet') == 'srdtrans_unroll_transformer':
            pth_name += (
                '_transformer'
                '_prior{}'
                '_K{}'
                '_rho{}to{}'
                '_mask{}'
                '_exactMPGN'
                '_conv'
            ).format(
                str(self.prior_backbone),
                int(self.unroll_steps),
                _format_float_for_path(self.unroll_rho_sched_min),
                _format_float_for_path(self.unroll_rho_sched_max),
                _format_float_for_path(getattr(self, 'mask_ratio', 0.0)),
            )
        self.pth_path = self.pth_dir + '/' + pth_name
        self.resume_checkpoint = None
        self.start_epoch = 0
        self.resume_global_iter = 0

        if os.path.exists(self.pth_path):
            if getattr(self, 'no_resume', False):
                shutil.rmtree(self.pth_path)
            else:
                latest_ckpt = self._find_latest_train_checkpoint(self.pth_path)
                if latest_ckpt is None:
                    raise RuntimeError(
                        "Experiment folder already exists but no checkpoint found: '{}'. "
                        "Pass no_resume=True to delete it and restart.".format(self.pth_path)
                    )
                self.resume_checkpoint = latest_ckpt
                m = re.search(r'E_(\d+)_Iter_(\d+)\.pth$', os.path.basename(latest_ckpt))
                if m:
                    self.start_epoch = int(m.group(1))
                    self.resume_global_iter = self.start_epoch * int(m.group(2))
                print('\033[1;31mResume checkpoint -----> {}\033[0m'.format(latest_ckpt))
                print('\033[1;31mResume from epoch {} / global_iter {}\033[0m'.format(
                    self.start_epoch + 1, self.resume_global_iter))

        os.makedirs(self.pth_path, exist_ok=True)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def set_params(self, params_dict):
        """
        Set the params set by user to the training class object and calculate some default parameters for training

        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if getattr(self, 'val_overlap_factor', None) is None:
            self.val_overlap_factor = self.overlap_factor

        self.train_crop_x, self.train_crop_y = self._get_train_crop_xy()

        # For spatial mode, gap is computed on the raw crop size.
        # Example: patch_x=128 -> raw crop_x=256 -> subimage width=128.
        self.gap_x = int(self.train_crop_x * (1 - self.overlap_factor))  # raw crop gap in x
        self.gap_y = int(self.train_crop_y * (1 - self.overlap_factor))  # raw crop gap in y
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))  # patch gap in t
        self.ngpu = str(self.GPU).count(',') + 1
        if getattr(self, 'user_batch_size', None) is not None:
            self.batch_size = int(self.user_batch_size)
        else:
            self.batch_size = self.ngpu
        print('\033[1;31mTraining parameters -----> \033[0m')
        print(self.__dict__)

    def initialize_network(self):
        """
        Initialize denoise network (3D U-Net or SRDTrans).

        Important Fields:
           self.fmap: the number of the feature map in U-Net 3D network.
           self.local_model: the denoise network

        """
        self.local_model = build_denoise_network(self)

    def get_gap_t(self):
        """
        Calculate the patch gap in t according to the size of input data and the patch gap in x and y

        Important Fields:
           self.gap_t: the patch gap in t.

        """
        crop_x = getattr(self, 'train_crop_x', self.patch_x)
        crop_y = getattr(self, 'train_crop_y', self.patch_y)
        w_num = math.floor((self.whole_x - crop_x) / self.gap_x) + 1
        h_num = math.floor((self.whole_y - crop_y) / self.gap_y) + 1
        s_num = math.ceil(self.train_datasets_size / w_num / h_num / self.stack_num)
        window_t = self._get_train_window_t()
        if s_num <= 1:
            if getattr(self, 'sampling_mode', 'temporal') == 'spatial_ori':
                # Original SRDTrans forces at least two temporal positions.
                s_num = 2
                self.gap_t = max(1, math.floor((self.whole_t - window_t) / (s_num - 1)))
            else:
                self.gap_t = max(1, int(self.patch_t * (1 - self.overlap_factor)))
        else:
            self.gap_t = max(1, math.floor((self.whole_t - window_t) / (s_num - 1)))

    def _append_coordinates_for_crop(
        self,
        im_name,
        stack_id,
        crop_x,
        crop_y,
        name_list,
        coordinate_list,
        stack_index,
        tag,
        window_t=None,
    ):
        gap_x = max(1, int(crop_x * (1 - self.overlap_factor)))
        gap_y = max(1, int(crop_y * (1 - self.overlap_factor)))

        w_num = math.floor((self.whole_x - crop_x) / gap_x) + 1
        h_num = math.floor((self.whole_y - crop_y) / gap_y) + 1

        window_t = int(self.patch_t if window_t is None else window_t)
        s_num = math.ceil(self.train_datasets_size / max(1, w_num * h_num * self.stack_num))

        if s_num <= 1:
            gap_t = max(1, int(self.patch_t * (1 - self.overlap_factor)))
        else:
            gap_t = max(1, math.floor((self.whole_t - window_t) / (s_num - 1)))

        for ih in range(0, int((self.whole_y - crop_y + gap_y) / gap_y)):
            for iw in range(0, int((self.whole_x - crop_x + gap_x) / gap_x)):
                for iz in range(0, int((self.whole_t - window_t + gap_t) / gap_t)):
                    init_h = gap_y * ih
                    end_h = init_h + crop_y
                    init_w = gap_x * iw
                    end_w = init_w + crop_x
                    init_s = gap_t * iz
                    end_s = init_s + window_t

                    single_coordinate = {
                        'init_h': init_h,
                        'end_h': end_h,
                        'init_w': init_w,
                        'end_w': end_w,
                        'init_s': init_s,
                        'end_s': end_s,
                    }

                    patch_name = (
                        self.datasets_name
                        + '_' + im_name.replace('.tif', '')
                        + '_' + tag
                        + '_h' + str(ih)
                        + '_w' + str(iw)
                        + '_z' + str(iz)
                    )

                    name_list.append(patch_name)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(stack_id)

    def _make_data_args(self):
        """Build an args proxy for likelihood.dataset (SRDTrans process).

        patch_x/y use train_crop_* so spatial 2x crops stay consistent with
        set_params. sampling_mode is mapped to SRDTrans's temporal vs spatial
        window rule (2*patch_t vs patch_t).
        """
        class _Args:
            pass

        args = _Args()
        args.datasets_path = self.datasets_path
        args.datasets_folder = self.datasets_path
        args.patch_x = int(getattr(self, 'train_crop_x', self.patch_x))
        args.patch_y = int(getattr(self, 'train_crop_y', self.patch_y))
        args.patch_t = int(self.patch_t)
        args.gap_x = int(self.gap_x)
        args.gap_y = int(self.gap_y)
        args.gap_t = int(self.gap_t)
        args.select_img_num = self.select_img_num
        args.train_datasets_size = self.train_datasets_size
        args.overlap_factor = self.overlap_factor

        mode = getattr(self, 'sampling_mode', 'temporal')
        if (
            mode in ('temporal', 'temporal_2targets')
            or mode in _FIXED_TW_TEMPORAL_MODES
        ):
            args.sampling_mode = 'temporal'
        else:
            args.sampling_mode = 'spatial'
        return args

    def train_preprocess_lessMemoryMulStacks(self):
        """Load stacks via SRDTrans process (stack-mean centering + tiling).

        Special H/W checker/stripe modes still build dual coordinate lists on
        top of the already-centered stacks.
        """
        self.name_list = []
        self.coordinate_list = {}
        self.stack_index = []
        self.name_list_h = []
        self.coordinate_list_h = {}
        self.stack_index_h = []
        self.name_list_w = []
        self.coordinate_list_w = {}
        self.stack_index_w = []

        args = self._make_data_args()
        (
            name_list,
            noise_im_all,
            coordinate_list,
            stack_index,
            noise_im_means,
        ) = train_preprocess_lessMemoryMulStacks_srdtrans(args)

        self.noise_im_all = noise_im_all
        self.noise_im_means = noise_im_means
        self.stack_num = len(noise_im_all)

        if self.sampling_mode in _FIXED_HW_SPATIAL_MODES:
            tag_prefix = (
                'checker' if self.sampling_mode == 'spatial_checker' else 'stripe'
            )
            im_names = list(os.walk(self.datasets_path, topdown=False))[-1][-1]
            for ind, (im_name, noise_im) in enumerate(zip(im_names, self.noise_im_all)):
                self.whole_x = noise_im.shape[2]
                self.whole_y = noise_im.shape[1]
                self.whole_t = noise_im.shape[0]
                self._append_coordinates_for_crop(
                    im_name=im_name,
                    stack_id=ind,
                    crop_x=int(self.patch_x),
                    crop_y=int(self.patch_y) * 2,
                    name_list=self.name_list_h,
                    coordinate_list=self.coordinate_list_h,
                    stack_index=self.stack_index_h,
                    tag=tag_prefix + '_h',
                )
                self._append_coordinates_for_crop(
                    im_name=im_name,
                    stack_id=ind,
                    crop_x=int(self.patch_x) * 2,
                    crop_y=int(self.patch_y),
                    name_list=self.name_list_w,
                    coordinate_list=self.coordinate_list_w,
                    stack_index=self.stack_index_w,
                    tag=tag_prefix + '_w',
                )
        else:
            self.name_list = name_list
            self.coordinate_list = coordinate_list
            self.stack_index = stack_index
            if self.noise_im_all:
                noise_im0 = self.noise_im_all[0]
                self.whole_x = noise_im0.shape[2]
                self.whole_y = noise_im0.shape[1]
                self.whole_t = noise_im0.shape[0]

        self.noisy_train_img_mean = (
            float(np.mean(self.noise_im_means)) if self.noise_im_means else 0.0
        )
        print(
            'noisy_train_img_mean (for likelihood) -----> {:.4f}'.format(
                self.noisy_train_img_mean
            )
        )
    def save_yaml_train(self):
        """
        Save some essential params in para.yaml.

        """
        yaml_name = self.pth_path + '//para.yaml'
        para = {'n_epochs': 0, 'datasets_path': 0, 'overlap_factor': 0,
                'output_dir': 0, 'pth_path': 0, 'GPU': 0, 'batch_size': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'lr': 0, 'b1': 0, 'b2': 0, 'fmap': 0,
                'select_img_num': 0, 'train_datasets_size': 0,
                'sampling_mode': 0, 'val_process_frames': 0,
                'snr_margin': 0, 'val_overlap_factor': 0, 'eval_every_iters': 0,
                'use_likelihood': 0, 'mpgn_alpha': 0, 'mpgn_beta': 0,
                'mpgn_offset': 0, 'mpgn_kmax': 0, 'mpgn_min_signal': 0,
                'mpgn_nll_chunk_t': 0,
                'mpgn_quant_step': 0, 'mpgn_clip_low': 0, 'mpgn_clip_high': 0,
                'mpgn_boundary_atol': 0,
                'unroll_steps': 0,
                'unroll_rho_min': 0, 'unroll_eps': 0,
                'unroll_rho_sched_min': 0,
                'unroll_rho_sched_max': 0,
                'unroll_corr_n_iter': 0,
                'unroll_corr_chunk_t': 0,
                'unroll_corr_step_cap_sigma': 0.0,
                'mask_ratio': 0, 'mask_min_dist': 0,
                'unroll_gradient_checkpointing': 0,
                'lr_decay_every': 0, 'lr_decay_gamma': 0, 'lr_min': 0,
                'visualize_unroll': 0, 'visualize_every': 0, 'visualize_t_index': 0,
                'noisy_train_img_mean': 0, 'backbone': 0,
                'prior_backbone': 0, 'srdtrans_root': 0,
                'embedding_dim': 0, 'num_heads': 0, 'hidden_dim': 0,
                'window_size': 0, 'num_transBlock': 0, 'attn_dropout_rate': 0,
                'srdtrans_f_maps': 0, 'input_dropout_rate': 0}
        para["n_epochs"] = self.n_epochs
        para["datasets_path"] = self.datasets_path
        para["output_dir"] = self.output_dir
        para["pth_path"] = self.pth_path
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["lr"] = self.lr
        para["lr_decay_every"] = self.lr_decay_every
        para["lr_decay_gamma"] = self.lr_decay_gamma
        para["lr_min"] = self.lr_min
        para["b1"] = self.b1
        para["b2"] = self.b2
        para["fmap"] = self.fmap
        para["select_img_num"] = self.select_img_num
        para["train_datasets_size"] = self.train_datasets_size
        para["overlap_factor"] = self.overlap_factor
        para["sampling_mode"] = self.sampling_mode
        para["val_process_frames"] = self.val_process_frames
        para["snr_margin"] = self.snr_margin
        para["val_overlap_factor"] = self.val_overlap_factor
        para["eval_every_iters"] = self.eval_every_iters
        para["use_likelihood"] = self.use_likelihood
        para["mpgn_alpha"] = self.mpgn_alpha
        para["mpgn_beta"] = self.mpgn_beta
        para["mpgn_offset"] = self.mpgn_offset
        para["mpgn_kmax"] = self.mpgn_kmax
        para["mpgn_min_signal"] = self.mpgn_min_signal
        para["mpgn_nll_chunk_t"] = self.mpgn_nll_chunk_t
        para["mpgn_quant_step"] = self.mpgn_quant_step
        para["mpgn_clip_low"] = self.mpgn_clip_low
        para["mpgn_clip_high"] = self.mpgn_clip_high
        para["mpgn_boundary_atol"] = self.mpgn_boundary_atol
        para["unroll_steps"] = self.unroll_steps
        para["unroll_rho_min"] = self.unroll_rho_min
        para["unroll_eps"] = self.unroll_eps
        para["unroll_rho_sched_min"] = self.unroll_rho_sched_min
        para["unroll_rho_sched_max"] = self.unroll_rho_sched_max
        para["unroll_corr_n_iter"] = self.unroll_corr_n_iter
        para["unroll_corr_chunk_t"] = self.unroll_corr_chunk_t
        para["unroll_corr_step_cap_sigma"] = self.unroll_corr_step_cap_sigma
        para["mask_ratio"] = self.mask_ratio
        para["mask_min_dist"] = self.mask_min_dist
        para["unroll_gradient_checkpointing"] = (
            self.unroll_gradient_checkpointing
        )
        para["visualize_unroll"] = self.visualize_unroll
        para["visualize_every"] = self.visualize_every
        para["visualize_t_index"] = self.visualize_t_index
        para["prior_pretrained_path"] = getattr(
            self, 'prior_pretrained_path', None
        )
        para["prior_pretrained_root"] = getattr(
            self, 'prior_pretrained_root', None
        )
        para["prior_pretrained_epoch"] = int(
            getattr(self, 'prior_pretrained_epoch', 25)
        )
        para["noisy_train_img_mean"] = self.noisy_train_img_mean
        para["backbone"] = self.backbone
        para["prior_backbone"] = self.prior_backbone
        para["srdtrans_root"] = self.srdtrans_root
        para["embedding_dim"] = self.embedding_dim
        para["num_heads"] = self.num_heads
        para["hidden_dim"] = self.hidden_dim
        para["window_size"] = self.window_size
        para["num_transBlock"] = self.num_transBlock
        para["attn_dropout_rate"] = self.attn_dropout_rate
        para["srdtrans_f_maps"] = self.srdtrans_f_maps
        para["input_dropout_rate"] = self.input_dropout_rate
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def distribute_GPU(self):
        """
        Allocate the GPU for the training program.

        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
            self.local_model = nn.DataParallel(self.local_model, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for training -----> \033[0m'.format(torch.cuda.device_count()))

    def _is_unroll_model(self):
        return getattr(self, 'backbone', 'unet') in (
            'unet_unroll',
            '3dunet_unroll',
            '3DUNet_unroll',
            'srdtrans_unroll_transformer',
        )

    def _forward_model(self, x, patch_mean, y_obs=None):
        """
        Unified model forward.

        For unrolled MPGN model:
            local_model(y_masked=x, patch_mean=patch_mean, y_obs=y_obs)

        x and output are both in centered tensor units.
        """
        if self._is_unroll_model():
            return self.local_model(
                x,
                patch_mean=patch_mean,
                y_obs=y_obs,
            )
        return self.local_model(x)

    def _unwrap_local_model(self):
        if isinstance(self.local_model, nn.DataParallel):
            return self.local_model.module
        return self.local_model

    def _forward_unroll_data_and_prior(self, x, patch_mean):
        return self._unwrap_local_model().forward_data_and_prior(
            x,
            patch_mean=patch_mean,
        )

    @staticmethod
    def _coordinate_to_int(value):
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple, np.ndarray)):
            return int(np.asarray(value).reshape(-1)[0])
        return int(value)

    def _extract_gt_patch(
        self,
        ref_img,
        single_coordinate,
    ):
        init_h = self._coordinate_to_int(
            single_coordinate['init_h']
        )
        end_h = self._coordinate_to_int(
            single_coordinate['end_h']
        )
        init_w = self._coordinate_to_int(
            single_coordinate['init_w']
        )
        end_w = self._coordinate_to_int(
            single_coordinate['end_w']
        )
        init_s = self._coordinate_to_int(
            single_coordinate['init_s']
        )
        end_s = self._coordinate_to_int(
            single_coordinate['end_s']
        )

        gt_patch = ref_img[
            init_s:end_s,
            init_h:end_h,
            init_w:end_w,
        ]

        return gt_patch.astype(np.float32)

    def _save_unroll_debug_figure(
        self,
        debug,
        patch_mean,
        gt_patch_phys,
        save_dir,
        epoch,
    ):
        from .visualize import (
            save_unroll_stage_figure
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'unroll_epoch_{epoch:04d}.png')
        save_unroll_stage_figure(
            save_path=save_path,
            debug_dict=debug,
            patch_mean=patch_mean,
            gt_patch_phys=gt_patch_phys,
            t_index=self.visualize_t_index,
            sample_index=0,
            use_physical_units=True,
        )

    @staticmethod
    def _diag_scalar(value):
        if value is None:
            return float('nan')

        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            if value.numel() == 0:
                return float('nan')
            return float(value.mean().item())

        return float(value)

    @staticmethod
    def _tensor_stat(tensor, reducer='mean'):
        if not torch.is_tensor(tensor):
            return float('nan')

        tensor = tensor.detach().float().cpu()
        if tensor.numel() == 0:
            return float('nan')

        if reducer == 'mean':
            return float(tensor.mean().item())
        if reducer == 'std':
            return float(tensor.std(unbiased=False).item())
        raise ValueError('Unsupported reducer: {!r}'.format(reducer))

    @staticmethod
    def _clean_state_dict(state):
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        cleaned = {}
        for key, value in state.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            cleaned[key] = value
        return cleaned

    def _resolve_prior_pretrained_dir(self):
        explicit = getattr(self, 'prior_pretrained_path', None)
        if explicit:
            return os.path.dirname(os.path.abspath(str(explicit)))

        root = getattr(self, 'prior_pretrained_root', None)
        if not root:
            return None

        folder_name = '{}_srdtrans_{}_srdtrans'.format(
            self.datasets_name,
            self.sampling_mode,
        )
        return os.path.join(os.path.abspath(str(root)), folder_name)

    def _resolve_prior_pretrained_path(self):
        explicit = getattr(self, 'prior_pretrained_path', None)
        if explicit:
            return os.path.abspath(str(explicit))

        folder = self._resolve_prior_pretrained_dir()
        if folder is None:
            return None

        epoch = int(getattr(self, 'prior_pretrained_epoch', 25))
        ckpt_path = self._find_train_checkpoint_at_epoch(folder, epoch)
        if ckpt_path is not None:
            return ckpt_path

        print(
            '\033[1;33m[Prior init] E_{:02d} not found under {}; '
            'falling back to latest checkpoint.\033[0m'.format(
                epoch,
                folder,
            )
        )
        return self._find_latest_train_checkpoint(folder)

    def _load_prior_pretrained(self):
        if not self._is_unroll_model():
            return False

        ckpt_path = self._resolve_prior_pretrained_path()
        self.prior_pretrained_loaded_path = None
        if ckpt_path is None:
            return False
        if not os.path.isfile(ckpt_path):
            print(
                '\033[1;33m[Prior init] pretrained checkpoint not found: {}\033[0m'.format(
                    ckpt_path
                )
            )
            return False

        model = self.local_model
        if isinstance(model, nn.DataParallel):
            model = model.module
        if not hasattr(model, 'PriorNet'):
            print('\033[1;33m[Prior init] model has no PriorNet; skip.\033[0m')
            return False

        stats = load_pretrained_prior(model.PriorNet, ckpt_path, verbose=True)
        self.prior_pretrained_load_stats = stats
        self.prior_pretrained_loaded_path = ckpt_path

        print(
            '\033[1;31mLoaded PriorNet pretrained weights from -----> {}\033[0m'.format(
                ckpt_path
            )
        )
        if stats['loaded_tensors'] < stats['prior_tensors']:
            print(
                '\033[1;33m[Prior init] WARNING: only {}/{} PriorNet tensors loaded.\033[0m'.format(
                    stats['loaded_tensors'],
                    stats['prior_tensors'],
                )
            )
        return stats['loaded_tensors'] > 0

    def _print_checkpoint_init_summary(self, stage):
        pretrained_path = None
        if self._is_unroll_model():
            pretrained_path = self._resolve_prior_pretrained_path()

        print('\033[1;36m===== Checkpoint init ({}) =====\033[0m'.format(stage))
        print('resume_checkpoint =', getattr(self, 'resume_checkpoint', None))
        print('pretrained_path =', pretrained_path)
        print('prior_pretrained_loaded_path =', getattr(self, 'prior_pretrained_loaded_path', None))
        print('start_epoch =', getattr(self, 'start_epoch', None))
        print('no_resume =', getattr(self, 'no_resume', False))

        if self._is_unroll_model() and hasattr(self, 'local_model'):
            model = self._unwrap_local_model()
            if hasattr(model, 'PriorNet'):
                prior = model.PriorNet
                img_dim = getattr(prior, 'img_dim', None)
                img_time = getattr(prior, 'img_time', None)
                in_ch = getattr(prior, 'in_channel', getattr(prior, 'in_channels', None))
                print(
                    'PriorNet arch: img_dim={} img_time={} in_channel={} '
                    'embedding_dim={} num_heads={} hidden_dim={} window_size={} '
                    'num_transBlock={} f_maps={}'.format(
                        img_dim,
                        img_time,
                        in_ch,
                        getattr(prior, 'embedding_dim', None),
                        getattr(prior, 'num_heads', None),
                        getattr(prior, 'hidden_dim', None),
                        getattr(prior, 'window_size', None),
                        getattr(prior, 'num_transBlock', None),
                        getattr(prior, 'f_maps', None),
                    )
                )
                expected_dim, expected_time = get_srdtrans_input_shape(
                    getattr(self, 'sampling_mode', 'temporal'),
                    getattr(self, 'patch_x', None),
                    getattr(self, 'patch_y', None),
                    getattr(self, 'patch_t', None),
                )
                if (img_dim, img_time) != (expected_dim, expected_time):
                    print(
                        '\033[1;33m[Prior init] WARNING: PriorNet shape '
                        '({}, {}) != standalone SRDTrans expected ({}, {})\033[0m'.format(
                            img_dim,
                            img_time,
                            expected_dim,
                            expected_time,
                        )
                    )
        print('\033[1;36m================================\033[0m')

    def _fetch_one_sanity_batch(self):
        cuda = torch.cuda.is_available()
        if self.sampling_mode == 'temporal':
            train_data = trainset_temporal_srdtrans(
                self.name_list,
                self.coordinate_list,
                self.noise_im_all,
                self.stack_index,
                return_stack_mean=True,
                stack_means=self.noise_im_means,
            )
        elif self.sampling_mode == 'temporal_2targets':
            train_data = trainset_temporal_2targets(
                self.name_list,
                self.coordinate_list,
                self.noise_im_all,
                self.stack_index,
                self.patch_t,
                stack_means=self.noise_im_means,
            )
        elif self.sampling_mode in _FIXED_HW_SPATIAL_MODES:
            train_data = trainset_srdtrans(
                self.name_list_h,
                self.coordinate_list_h,
                self.noise_im_all,
                self.stack_index_h,
                return_stack_mean=True,
                stack_means=self.noise_im_means,
            )
        else:
            train_data = trainset_srdtrans(
                self.name_list,
                self.coordinate_list,
                self.noise_im_all,
                self.stack_index,
                return_stack_mean=True,
                stack_means=self.noise_im_means,
            )

        loader = DataLoader(
            train_data,
            batch_size=min(1, int(getattr(self, 'batch_size', 1))),
            shuffle=True,
            num_workers=0,
        )
        batch = next(iter(loader))

        if self.sampling_mode in _SPARSE_MASK_MODES:
            noisy_centered, stack_global_mean = batch
            if cuda:
                noisy_centered = noisy_centered.cuda()
            patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
            if cuda:
                patch_mean = patch_mean.cuda()
            y_prime_centered, _ = _make_masked_replacement(
                noisy_centered,
                mode=self.sampling_mode,
                mask_ratio=self.mask_ratio,
                mask_min_dist=self.mask_min_dist,
            )
            return y_prime_centered, patch_mean, noisy_centered

        if self.sampling_mode in (
            'spatial', 'spatial_1target', 'spatial_ori',
        ) or self.sampling_mode in _FIXED_SINGLE_TARGET_MODES:
            noisy_centered, stack_global_mean = batch
            if cuda:
                noisy_centered = noisy_centered.cuda()
            patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
            if cuda:
                patch_mean = patch_mean.cuda()
            return noisy_centered, patch_mean, None

        if self.sampling_mode == 'temporal_2targets':
            inp, _tgt1, _tgt2, patch_mean = batch
            if cuda:
                inp = inp.cuda()
                patch_mean = patch_mean.cuda()
            return inp, patch_mean, None

        inp, tgt, patch_mean = batch
        if cuda:
            inp = inp.cuda()
            patch_mean = patch_mean.cuda()
        patch_mean = patch_mean.view(-1, 1, 1, 1, 1)
        return inp, patch_mean, tgt

    def _tensor_stats_line(self, name, tensor):
        xx = tensor.detach().float()
        return (
            '{} mean/std/min/max = {:.6g} / {:.6g} / {:.6g} / {:.6g}'.format(
                name,
                float(xx.mean().item()),
                float(xx.std().item()),
                float(xx.min().item()),
                float(xx.max().item()),
            )
        )

    def _save_prior_sanity_figure(self, y_masked, x_hat, save_path, gt=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('[Prior sanity] matplotlib not available; skip figure.')
            return

        def _mid_slice(vol):
            arr = vol.detach().float().cpu().numpy()
            if arr.ndim == 5:
                arr = arr[0, 0, arr.shape[2] // 2]
            elif arr.ndim == 4:
                arr = arr[0, arr.shape[1] // 2]
            elif arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            return arr

        panels = [
            ('y_masked', _mid_slice(y_masked)),
            ('PriorNet(x_hat)', _mid_slice(x_hat)),
        ]
        if gt is not None:
            panels.append(('GT', _mid_slice(gt)))

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, img) in zip(axes, panels):
            vmin, vmax = np.percentile(img, [1, 99])
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis('off')
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print('[Prior sanity] saved figure ->', save_path)

    def _sanity_check_prior_net(self):
        if not self._is_unroll_model():
            return

        model = self._unwrap_local_model()
        if not hasattr(model, 'PriorNet'):
            return

        print('\033[1;36m===== PriorNet sanity check (pre-train) =====\033[0m')
        if getattr(self, 'resume_checkpoint', None):
            print(
                '[Prior sanity] resume_checkpoint is set; '
                'PriorNet output reflects resumed unroll weights, not fresh pretrained init.'
            )

        y_masked, patch_mean, gt = self._fetch_one_sanity_batch()
        model.eval()
        with torch.no_grad():
            x_hat = model.PriorNet(y_masked)

        print(self._tensor_stats_line('y_masked', y_masked))
        print(self._tensor_stats_line('x_hat', x_hat))

        fig_path = os.path.join(self.pth_path, 'prior_sanity_check.png')
        self._save_prior_sanity_figure(y_masked, x_hat, fig_path, gt=gt)

        ckpt_path = getattr(self, 'prior_pretrained_loaded_path', None)
        if ckpt_path is None:
            ckpt_path = self._resolve_prior_pretrained_path()
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print('[Prior sanity] no pretrained checkpoint path; skip baseline compare.')
            print('\033[1;36m============================================\033[0m')
            return

        try:
            SRDTrans_v2, srdtrans_root = import_srdtrans_v2_class(
                getattr(self, 'srdtrans_root', None)
            )
            baseline = _build_transformer_unet(
                self,
                SRDTrans_v2,
                'SRDTrans_v2-baseline',
                srdtrans_root,
            )
            baseline_stats = load_pretrained_prior(
                baseline,
                ckpt_path,
                verbose=False,
            )
            print(
                '[Prior sanity] baseline loaded tensors: {}/{}'.format(
                    baseline_stats['loaded_tensors'],
                    baseline_stats['prior_tensors'],
                )
            )
            if torch.cuda.is_available():
                baseline = baseline.cuda()

            baseline.eval()
            with torch.no_grad():
                out_baseline = baseline(y_masked)

            diff = (x_hat - out_baseline).abs()
            print(
                '[Prior sanity] |PriorNet - baseline| mean = {:.6g}, max = {:.6g}'.format(
                    float(diff.mean().item()),
                    float(diff.max().item()),
                )
            )
            if float(diff.max().item()) > 1e-4:
                print(
                    '\033[1;33m[Prior sanity] WARNING: PriorNet differs from '
                    'standalone SRDTrans baseline on the same input.\033[0m'
                )
        except Exception as exc:
            print('[Prior sanity] baseline compare failed:', exc)

        print('\033[1;36m============================================\033[0m')

    def _read_pretrained_snr_metrics(self, epoch=None):
        """Read SNR row from standalone SRDTrans val_metrics.md (pretrained PriorNet)."""
        if epoch is None:
            epoch = int(getattr(self, 'prior_pretrained_epoch', 25))
        folder = self._resolve_prior_pretrained_dir()
        if folder is None:
            return None

        metrics_path = os.path.join(folder, 'val_metrics.md')
        if not os.path.isfile(metrics_path):
            print(
                '\033[1;33m[val_metrics ep-1] pretrained val_metrics not found: {}\033[0m'.format(
                    metrics_path
                )
            )
            return None

        target = str(int(epoch))
        with open(metrics_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped.startswith('|'):
                    continue
                if stripped.startswith('| -----') or stripped.startswith('| Epoch'):
                    continue
                parts = [p.strip() for p in stripped.split('|')]
                parts = [p for p in parts if p]
                if len(parts) < 5:
                    continue
                if parts[0] != target:
                    continue
                return {
                    'epoch': int(parts[0]),
                    'iteration': int(parts[1]),
                    'snr_no_scale': float(parts[2]),
                    'snr_denoised': float(parts[3]),
                    'snr_noisy': float(parts[4]),
                    'source': metrics_path,
                    'pretrained_epoch': int(epoch),
                }

        print(
            '\033[1;33m[val_metrics ep-1] epoch {} not found in {}\033[0m'.format(
                epoch,
                metrics_path,
            )
        )
        return None

    def _pretrained_ep_minus1_metrics_line(self, row):
        """Format epoch -1 row from standalone SRDTrans val_metrics (pretrained baseline)."""
        snr_prior = row['snr_no_scale']
        return (
            '| -1 | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n'.format(
                row['iteration'],
                snr_prior,
                snr_prior,
                snr_prior,
                snr_prior,
                row['snr_noisy'],
            )
        )

    def _write_pretrained_ep_minus1_metrics(self):
        """Write epoch -1 SNR copied from standalone SRDTrans val_metrics.md."""
        if not self._is_unroll_model():
            return
        if self._val_metrics_has_epoch(-1):
            return

        row = self._read_pretrained_snr_metrics()
        if row is None:
            return

        metrics_path = os.path.join(self.pth_path, 'val_metrics.md')
        os.makedirs(self.pth_path, exist_ok=True)
        ep_minus1_line = self._pretrained_ep_minus1_metrics_line(row)

        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                f.write(self._unroll_val_metrics_header())
                f.write(ep_minus1_line)
        else:
            with open(metrics_path, 'r') as f:
                lines = f.readlines()
            insert_at = 2
            for idx, line in enumerate(lines):
                if line.strip().startswith('| -----'):
                    insert_at = idx + 1
                    break
            lines.insert(insert_at, ep_minus1_line)
            with open(metrics_path, 'w') as f:
                f.writelines(lines)

        print(
            '[val_metrics ep-1] copied standalone E{} SNR into {} '
            '(xK_prior={:.4f}, zK_exactMPGN={:.4f}, noisy={:.4f}; source={})'.format(
                row['pretrained_epoch'],
                metrics_path,
                row['snr_no_scale'],
                row['snr_no_scale'],
                row['snr_noisy'],
                row['source'],
            )
        )

    def _val_metrics_has_epoch(self, epoch):
        metrics_path = os.path.join(self.pth_path, 'val_metrics.md')
        if not os.path.isfile(metrics_path):
            return False
        pattern = re.compile(r'^\|\s*{}\s*\|'.format(int(epoch)))
        with open(metrics_path, 'r') as f:
            for line in f:
                if pattern.match(line.strip()):
                    return True
        return False

    def _unroll_val_metrics_header(self):
        return (
            '| Epoch | Iteration | SNR_xK_prior_no_scale (dB) | '
            'SNR_zK_exactMPGN_no_scale (dB) | SNR_xK_prior_scaled (dB) | '
            'SNR_zK_exactMPGN_scaled (dB) | SNR_noisy (dB) |\n'
            '| ----- | --------- | -------------------------- | '
            '------------------------------ | ------------------------- | '
            '------------------------------ | -------------- |\n'
        )

    def _run_ep0_validation(self):
        """Run full validation + unroll_viz at epoch 0 (pretrained init, before training)."""
        if not self._is_unroll_model():
            return
        if not getattr(self, 'gt_path', ''):
            print('[ep0] gt_path not set; skip epoch-0 validation.')
            return
        print(
            '\033[1;36m===== Epoch-0 validation + unroll_viz (pretrained init) =====\033[0m'
        )
        self.test(
            train_epoch=-1,
            train_iteration=6991,
            metrics_epoch=0,
            viz_epoch=0,
        )

    def _save_unroll_diagnostics(
        self,
        debug,
        save_dir,
        epoch,
    ):
        os.makedirs(save_dir, exist_ok=True)

        csv_path = os.path.join(
            save_dir,
            'unroll_stage_diagnostics.csv',
        )

        fieldnames = [
            'epoch',
            'stage',
            'rho',

            'x_prev_mean',
            'x_prev_std',
            'z_corr_mean',
            'z_corr_std',
            'x_pred_mean',
            'x_pred_std',
            'x_prior_mean',
            'x_prior_std',

            'delta_correction_std',
            'delta_prior_std',
            'delta_to_y_std',

            'nll_before',
            'nll_after',
            'prox_objective_before',
            'prox_objective_after',
            'step_abs_mean',
            'step_abs_p95',
            'delta_corr_mae',
            'delta_corr_std',
            'negative_y_fraction',
            'x_ref_floor_fraction',
            'z_floor_fraction',
            'v_ref_mean',
            'v_ref_std',
            'finite_fraction',
        ]

        write_header = not os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            if write_header:
                writer.writeheader()

            for stage in debug.get('stages', []):
                corr_diag = stage.get('corr_diag') or {}

                row = {
                    'epoch': int(epoch),
                    'stage': int(stage['stage_idx']) + 1,
                    'rho': self._diag_scalar(stage.get('rho')),

                    'x_prev_mean': self._tensor_stat(stage.get('x_prev'), 'mean'),
                    'x_prev_std': self._tensor_stat(stage.get('x_prev'), 'std'),

                    'z_corr_mean': self._tensor_stat(stage.get('z_corr'), 'mean'),
                    'z_corr_std': self._tensor_stat(stage.get('z_corr'), 'std'),

                    'x_pred_mean': self._tensor_stat(stage.get('x_pred'), 'mean'),
                    'x_pred_std': self._tensor_stat(stage.get('x_pred'), 'std'),

                    'x_prior_mean': self._tensor_stat(stage.get('x_prior'), 'mean'),
                    'x_prior_std': self._tensor_stat(stage.get('x_prior'), 'std'),

                    'delta_correction_std': self._tensor_stat(
                        stage.get('delta_correction'), 'std'
                    ),
                    'delta_prior_std': self._tensor_stat(
                        stage.get('delta_prior'), 'std'
                    ),
                    'delta_to_y_std': self._tensor_stat(
                        stage.get('delta_to_y'), 'std'
                    ),
                }

                for key in [
                    'nll_before',
                    'nll_after',
                    'prox_objective_before',
                    'prox_objective_after',
                    'step_abs_mean',
                    'step_abs_p95',
                    'delta_corr_mae',
                    'delta_corr_std',
                    'negative_y_fraction',
                    'x_ref_floor_fraction',
                    'z_floor_fraction',
                    'v_ref_mean',
                    'v_ref_std',
                    'finite_fraction',
                ]:
                    row[key] = self._diag_scalar(corr_diag.get(key))

                writer.writerow(row)

                print(
                    '\n[Unroll diagnostic] '
                    'stage={} rho={:.3e} '
                    'x_pred_std={:.3e} x_prior_std={:.3e} '
                    'delta_corr_std={:.3e} delta_prior_std={:.3e} '
                    'nll_before={:.3e} nll_after={:.3e} '
                    'finite={:.3%}'.format(
                        row['stage'],
                        row['rho'],
                        row['x_pred_std'],
                        row['x_prior_std'],
                        row['delta_correction_std'],
                        row['delta_prior_std'],
                        row['nll_before'],
                        row['nll_after'],
                        row['finite_fraction'],
                    )
                )

    def _prepare_eval_cache(self):
        """Pre-load and cache validation data (runs once)."""
        if self._eval_cache_ready:
            return

        if not getattr(self, 'gt_path', ''):
            raise ValueError("gt_path must be set to enable validation SNR metrics.")
        if not os.path.exists(self.gt_path):
            raise ValueError("gt_path does not exist: {}".format(self.gt_path))

        # Proxy args with validation-specific settings.
        class _ValArgs:
            pass

        val_args = _ValArgs()
        val_args.patch_x = self.patch_x
        val_args.patch_y = self.patch_y
        val_args.patch_t = self.patch_t
        val_args.gap_x = int(self.patch_x * (1 - self.val_overlap_factor))
        val_args.gap_y = int(self.patch_y * (1 - self.val_overlap_factor))
        val_args.gap_t = int(self.patch_t * (1 - self.val_overlap_factor))
        val_args.overlap_factor = self.val_overlap_factor
        val_args.datasets_path = self.datasets_path
        val_args.datasets_folder = self.datasets_path
        val_args.datasets_name = self.datasets_name
        val_args.test_datasize = self.val_process_frames
        val_args.print_img_name = True

        name_list, noise_img, coordinate_list, img_mean, input_data_type = \
            test_preprocess_lessMemoryNoTail_chooseOne_srdtrans(val_args, 0)

        gt = tiff.imread(self.gt_path).astype(np.float32)
        if gt.shape[0] > self.val_process_frames:
            gt = gt[:self.val_process_frames]

        self._eval_name_list = name_list
        self._eval_noise_img = noise_img
        self._eval_coordinate_list = coordinate_list
        self._eval_img_mean = img_mean
        self._eval_input_data_type = input_data_type
        self._eval_ref_img = gt
        self._eval_cache_ready = True

    def train(self):
        """
        Pytorch training workflow.

        Sampling modes (sampling_mode):
          'temporal'          : original DeepCAD temporal interleaving.
          'spatial'           : SRDTrans spatial-neighbor masking (2 targets, 2x raw crop).
          'spatial_ori'       : original SRDTrans spatial (2 targets, patch-sized crop).
          'spatial_1target'   : SRDTrans spatial-neighbor masking (1 target).
          'spatial_checker'   : fixed checkerboard diagonal sampling (H/W alternate).
          'spatial_stripe'    : fixed row/column stripe sampling (H/W alternate).
          'temporal_checker'  : odd-even frame sampling (T compression only).
          'temporal_2targets' : each input frame paired with its prev/next frames.

        Loss: masked exact MPGN NLL on final prior output x_K only.
        """
        optimizer_G = torch.optim.Adam(self.local_model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        cuda = torch.cuda.is_available()

        prev_time = time.time()
        time_start = time.time()
        global_iter = int(getattr(self, 'resume_global_iter', 0))
        start_epoch = int(getattr(self, 'start_epoch', 0))

        for epoch in range(start_epoch, self.n_epochs):
            hw_axis = None

            if self.sampling_mode in _FIXED_HW_SPATIAL_MODES:
                hw_axis = 'h' if (epoch % 2 == 0) else 'w'

                if hw_axis == 'h':
                    train_data = trainset_srdtrans(
                        self.name_list_h,
                        self.coordinate_list_h,
                        self.noise_im_all,
                        self.stack_index_h,
                        return_stack_mean=True,
                        stack_means=self.noise_im_means,
                    )
                else:
                    train_data = trainset_srdtrans(
                        self.name_list_w,
                        self.coordinate_list_w,
                        self.noise_im_all,
                        self.stack_index_w,
                        return_stack_mean=True,
                        stack_means=self.noise_im_means,
                    )

                round_id = epoch // 2 + 1
                phase = 'middle/H' if hw_axis == 'h' else 'end/W'
                print('\n[{}] Round {}, phase: {}, epoch: {}'.format(
                    self.sampling_mode, round_id, phase, epoch + 1
                ))

            elif self.sampling_mode == 'temporal_2targets':
                train_data = trainset_temporal_2targets(
                    self.name_list,
                    self.coordinate_list,
                    self.noise_im_all,
                    self.stack_index,
                    self.patch_t,
                    stack_means=self.noise_im_means,
                )

            elif self.sampling_mode == 'temporal':
                train_data = trainset_temporal_srdtrans(
                    self.name_list,
                    self.coordinate_list,
                    self.noise_im_all,
                    self.stack_index,
                    return_stack_mean=True,
                    stack_means=self.noise_im_means,
                )

            else:
                # spatial* / sparse-mask / temporal_checker: full patch + stack mean
                train_data = trainset_srdtrans(
                    self.name_list,
                    self.coordinate_list,
                    self.noise_im_all,
                    self.stack_index,
                    return_stack_mean=True,
                    stack_means=self.noise_im_means,
                )
            trainloader = DataLoader(train_data, batch_size=self.batch_size,
                                     shuffle=True, num_workers=self.num_workers)
            self.local_model.train()

            def _debug_tensor(name, x):
                xx = x.detach()
                print(
                    f"{name}: shape={tuple(xx.shape)}, "
                    f"min={xx.min().item():.6g}, max={xx.max().item():.6g}, "
                    f"mean={xx.mean().item():.6g}, std={xx.std().item():.6g}"
                )

            
            for iteration, batch in enumerate(trainloader):
                if self._is_unroll_model() and self.lr_decay_every > 0:
                    decay_count = global_iter // int(self.lr_decay_every)

                    current_lr = max(
                        float(self.lr)
                        * (float(self.lr_decay_gamma) ** decay_count),
                        float(self.lr_min),
                    )

                    for param_group in optimizer_G.param_groups:
                        param_group['lr'] = current_lr

                # ── forward pass (sampling-mode-specific) ─────────────────────
                # Patches from SRDTrans process are already stack-mean-centered.
                # stack_global_mean / patch_mean is only for unroll / NLL physics.
                if self.sampling_mode in _SPARSE_MASK_MODES:
                    noisy_centered, stack_global_mean = batch
                    noisy_centered = noisy_centered.cuda() if cuda else noisy_centered

                    patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
                    patch_mean = patch_mean.cuda() if cuda else patch_mean

                    y_raw_centered = noisy_centered

                    y_prime_centered, loss_mask = _make_masked_replacement(
                        noisy_centered,
                        mode=self.sampling_mode,
                        mask_ratio=self.mask_ratio,
                        mask_min_dist=self.mask_min_dist,
                    )

                    if iteration == 0:
                        _check_train_coordinate_consistency(
                            y_raw_centered,
                            y_raw_centered + patch_mean,
                            patch_mean,
                        )

                    fake_B = self._forward_model(
                        y_prime_centered,
                        patch_mean,
                        y_obs=y_prime_centered,
                    )

                elif self.sampling_mode in _FIXED_SINGLE_TARGET_MODES:
                    noisy_centered, stack_global_mean = batch
                    noisy_centered = noisy_centered.cuda() if cuda else noisy_centered

                    patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
                    patch_mean = patch_mean.cuda() if cuda else patch_mean

                    if self.sampling_mode == 'spatial_checker':
                        if hw_axis == 'h':
                            sub1, sub2 = _checkerboard_diag_h_pair(noisy_centered)
                        else:
                            sub1, sub2 = _checkerboard_diag_w_pair(noisy_centered)
                    elif self.sampling_mode == 'spatial_stripe':
                        if hw_axis == 'h':
                            sub1, sub2 = _stripe_h_pair(noisy_centered)
                        else:
                            sub1, sub2 = _stripe_w_pair(noisy_centered)
                    else:  # temporal_checker
                        sub1, sub2 = _stripe_t_pair(noisy_centered)

                    # Randomly swap input and target (per batch element).
                    swap = (torch.rand(sub1.size(0), 1, 1, 1, 1, device=sub1.device) >= 0.5)
                    sub1, sub2 = (
                        torch.where(swap, sub2, sub1),
                        torch.where(swap, sub1, sub2),
                    )
                    sub3 = None

                    if iteration == 0:
                        _check_train_coordinate_consistency(
                            sub1,
                            sub1 + patch_mean,
                            patch_mean,
                        )

                    fake_B = self._forward_model(
                        sub1,
                        patch_mean,
                    )

                elif self.sampling_mode in ('spatial', 'spatial_ori', 'spatial_1target'):
                    noisy_centered, stack_global_mean = batch
                    noisy_centered = noisy_centered.cuda() if cuda else noisy_centered
                    # [B] -> [B, 1, 1, 1, 1] for broadcasting (L1/L2 branch).
                    patch_mean = stack_global_mean.view(-1, 1, 1, 1, 1)
                    patch_mean = patch_mean.cuda() if cuda else patch_mean

                    mask1, mask2, mask3 = _srd_generate_mask_pair(noisy_centered)
                    sub1 = _srd_generate_subimages(noisy_centered, mask1)
                    sub2 = _srd_generate_subimages(noisy_centered, mask2)
                    sub3 = _srd_generate_subimages(noisy_centered, mask3)

                    if iteration == 0:
                        _check_train_coordinate_consistency(
                            sub1,
                            sub1 + patch_mean,
                            patch_mean,
                        )

                    fake_B = self._forward_model(
                        sub1,
                        patch_mean,
                    )

                elif self.sampling_mode == 'temporal_2targets':
                    inp, tgt1, tgt2, patch_mean = batch

                    inp = inp.cuda() if cuda else inp
                    tgt1 = tgt1.cuda() if cuda else tgt1
                    tgt2 = tgt2.cuda() if cuda else tgt2
                    patch_mean = patch_mean.cuda() if cuda else patch_mean

                    fake_B = self._forward_model(
                        Variable(inp),
                        patch_mean,
                    )

                else:  # 'temporal'
                    inp, tgt, patch_mean = batch

                    inp = inp.cuda() if cuda else inp
                    tgt = tgt.cuda() if cuda else tgt
                    patch_mean = patch_mean.view(-1, 1, 1, 1, 1)
                    patch_mean = patch_mean.cuda() if cuda else patch_mean

                    fake_B = self._forward_model(
                        Variable(inp),
                        patch_mean,
                    )

                if iteration == 0:
                    print("\n===== DEBUG FIRST BATCH =====")
                    if self.sampling_mode in _SPARSE_MASK_MODES:
                        _debug_tensor("y_raw_centered", y_raw_centered)
                        _debug_tensor("y_prime_centered", y_prime_centered)
                        _debug_tensor("loss_mask", loss_mask)
                        print(
                            "mask_fraction:",
                            float(loss_mask.float().mean().item())
                        )
                    elif self.sampling_mode in ('spatial', 'spatial_ori', 'spatial_1target') or self.sampling_mode in _FIXED_SINGLE_TARGET_MODES:
                        if self.sampling_mode in _FIXED_HW_SPATIAL_MODES:
                            print("hw_axis:", hw_axis)
                        _debug_tensor("sub1/input", sub1)
                        _debug_tensor("sub2/target_a", sub2)
                        if self.sampling_mode in ('spatial', 'spatial_ori'):
                            _debug_tensor("sub3/target_b", sub3)
                    elif self.sampling_mode == 'temporal_2targets':
                        _debug_tensor("inp", inp)
                        _debug_tensor("tgt1", tgt1)
                        _debug_tensor("tgt2", tgt2)
                    else:
                        _debug_tensor("inp", inp)
                        _debug_tensor("tgt", tgt)
                    print("=============================\n")

                # ── loss: masked exact MPGN NLL on final prior output x_K ───
                valid_mask = torch.ones_like(fake_B[:, :1])

                _nll_kw = dict(
                    pred_img_mean=patch_mean,
                    target_img_mean=patch_mean,
                    alpha=self.mpgn_alpha,
                    beta=self.mpgn_beta,
                    offset=self.mpgn_offset,
                    kmax=int(self.mpgn_kmax),
                    min_signal=float(self.mpgn_min_signal),
                    chunk_t=int(self.mpgn_nll_chunk_t),
                    quant_step=getattr(self, 'mpgn_quant_step', 1.0),
                    clip_low=getattr(self, 'mpgn_clip_low', None),
                    clip_high=getattr(self, 'mpgn_clip_high', None),
                    boundary_atol=float(
                        getattr(self, 'mpgn_boundary_atol', 1e-6)
                    ),
                )

                if self.sampling_mode in _SPARSE_MASK_MODES:
                    valid_mask = loss_mask.to(dtype=fake_B.dtype)

                    data_loss = _mpgn_nll_single_target(
                        fake_B,
                        y_raw_centered,
                        valid_mask,
                        **_nll_kw,
                    )
                    nll_a = data_loss
                    nll_b = None

                elif self.sampling_mode in ('spatial', 'spatial_ori', 'spatial_1target') or self.sampling_mode in _FIXED_SINGLE_TARGET_MODES:
                    nll_a = _mpgn_nll_single_target(
                        fake_B, sub2, valid_mask, **_nll_kw
                    )

                    if self.sampling_mode in ('spatial', 'spatial_ori'):
                        nll_b = _mpgn_nll_single_target(
                            fake_B, sub3, valid_mask, **_nll_kw
                        )
                        data_loss = 0.5 * nll_a + 0.5 * nll_b
                    else:
                        nll_b = None
                        data_loss = nll_a
                elif self.sampling_mode == 'temporal_2targets':
                    nll_a = _mpgn_nll_single_target(
                        fake_B, tgt1, valid_mask, **_nll_kw
                    )
                    nll_b = _mpgn_nll_single_target(
                        fake_B, tgt2, valid_mask, **_nll_kw
                    )
                    data_loss = 0.5 * nll_a + 0.5 * nll_b
                else:  # 'temporal'
                    nll_a = _mpgn_nll_single_target(
                        fake_B, tgt, valid_mask, **_nll_kw
                    )
                    data_loss = nll_a
                    nll_b = None

                Total_loss = data_loss

                optimizer_G.zero_grad()
                Total_loss.backward()

                if self._is_unroll_model() and iteration == 0:
                    unroll_model = self._unwrap_local_model()

                    def module_grad_norm(module):
                        total = 0.0
                        finite = True

                        for parameter in module.parameters():
                            if parameter.grad is None:
                                continue

                            grad = parameter.grad.detach()

                            if not torch.isfinite(grad).all():
                                finite = False

                            total += float(
                                grad.float().pow(2).sum().item()
                            )

                        return math.sqrt(total), finite

                    prior_norm, prior_finite = module_grad_norm(
                        unroll_model.PriorNet
                    )

                    print(
                        '\n[Gradient diagnostic] '
                        'Prior({})={:.3e} finite={}'.format(
                            getattr(
                                unroll_model,
                                'prior_backbone',
                                'unknown',
                            ),
                            prior_norm,
                            prior_finite,
                        )
                    )

                optimizer_G.step()
                global_iter += 1

                # ── shared bookkeeping ────────────────────────────────────────
                batches_done = epoch * len(trainloader) + iteration
                batches_left = self.n_epochs * len(trainloader) - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()

                if iteration % 1 == 0:
                    time_end = time.time()
                    if self.sampling_mode in _SPARSE_MASK_MODES and self._is_unroll_model():
                        log_str = '[Masked exact MPGN NLL: {:.6f}]'.format(
                            Total_loss.item()
                        )
                    else:
                        log_str = '[Exact MPGN NLL: {:.6f}]'.format(
                            Total_loss.item()
                        )
                    print(
                        '\r[Epoch %d/%d] [Batch %d/%d] %s [ETA: %s] [Time cost: %.2d s]     '
                        % (
                            epoch + 1, self.n_epochs,
                            iteration + 1, len(trainloader),
                            log_str, time_left, time_end - time_start
                        ), end=' ')

                # ── iteration-level validation ────────────────────────────────
                if (getattr(self, 'eval_val_per_epoch', False)
                        and int(getattr(self, 'eval_every_iters', 0)) > 0
                        and (global_iter % int(self.eval_every_iters) == 0)):
                    print('\nValidation every {} iters -----> (global_iter: {})'.format(
                        int(self.eval_every_iters), global_iter))
                    self.test(epoch, iteration)
                    self.local_model.train()
                    print('\n', end=' ')

                # ── end of epoch ──────────────────────────────────────────────
                if (iteration + 1) % len(trainloader) == 0:
                    print('\n', end=' ')
                    self.save_model(epoch, iteration)
                    if (self.visualize_images_per_epoch | self.save_test_images_per_epoch) \
                            and not getattr(self, 'eval_val_per_epoch', False):
                        print('Testing model of epoch {} on the first noisy file ----->'.format(epoch + 1))
                        self.test(epoch, iteration)
                        self.local_model.train()
                        print('\n', end=' ')
                    if getattr(self, 'eval_val_per_epoch', False) \
                            and int(getattr(self, 'eval_every_iters', 0)) <= 0:
                        print('Validation on the first noisy file ----->')
                        self.test(epoch, iteration)
                        self.local_model.train()
                        print('\n', end=' ')

        print('Training finished. All models saved to disk.')
        if self.colab_display:
            result_img_list = []
            results_path = self.pth_path
            results_list = list(os.walk(results_path, topdown=False))[-1][-1]
            for i in range(len(results_list)):
                aaa = results_list[i]
                if '.tif' in aaa:
                    result_img_list.append(aaa)
            result_img_list.sort()
            self.result_display = results_path + '/' + result_img_list[-1]

    def save_model(self, epoch, iteration):
        """
        Model storage.
        Args:
           train_epoch : current train epoch number
           train_iteration : current train_iteration number
        """
        model_save_name = self.pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(
            iteration + 1).zfill(4) + '.pth'
        if isinstance(self.local_model, nn.DataParallel):
            torch.save(self.local_model.module.state_dict(), model_save_name)
        else:
            torch.save(self.local_model.state_dict(), model_save_name)

    def test(
        self,
        train_epoch,
        train_iteration,
        metrics_epoch=None,
        viz_epoch=None,
    ):
        """
        Pytorch validation workflow.

        Runs inference on the first val_process_frames frames of the first noisy file,
        then computes SNR metrics on the middle (val_process_frames - 2*snr_margin)
        frames against the provided GT.

        For unroll backbone, reports SNR of final prior x_K and exact-MPGN state z_K.

        Results are appended to val_metrics.md in the checkpoint directory.

        Args:
            train_epoch : current train epoch number (0-based during training)
            train_iteration : current train iteration number
            metrics_epoch : epoch number written to val_metrics.md (default train_epoch+1)
            viz_epoch : epoch number used for unroll_viz filenames (default metrics_epoch)
        """
        if metrics_epoch is None:
            metrics_epoch = train_epoch + 1
        if viz_epoch is None:
            viz_epoch = metrics_epoch
        self.print_img_name = True
        self._prepare_eval_cache()

        name_list = self._eval_name_list
        noise_img = self._eval_noise_img
        coordinate_list = self._eval_coordinate_list
        img_mean = self._eval_img_mean
        input_data_type = self._eval_input_data_type
        ref_img = self._eval_ref_img

        prev_time = time.time()
        time_start = time.time()
        denoise_before_match = np.zeros(noise_img.shape)
        denoise_data = np.zeros(noise_img.shape) if self._is_unroll_model() else None
        input_img = np.zeros(noise_img.shape)

        test_data = testset_srdtrans(name_list, coordinate_list, noise_img)
        testloader = DataLoader(test_data, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)

        cuda = torch.cuda.is_available()
        # noise_img from SRDTrans test preprocess is already centered by img_mean.
        # residual_mean is only an optional second centering term (~0).
        residual_mean = torch.as_tensor(
            float(noise_img.mean()),
            dtype=torch.float32,
        )
        dataset_mean = torch.as_tensor(
            float(img_mean),
            dtype=torch.float32,
        )
        if cuda:
            residual_mean = residual_mean.cuda()
            dataset_mean = dataset_mean.cuda()
        residual_mean = residual_mean.view(1, 1, 1, 1, 1)
        dataset_mean = dataset_mean.view(1, 1, 1, 1, 1)
        # Full scaled-domain mean required by all physical-unit conversions.
        model_mean = dataset_mean + residual_mean

        self.local_model.eval()
        with torch.no_grad():
            for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                noise_patch = noise_patch.cuda() if cuda else noise_patch

                noise_patch_raw = noise_patch.float()
                real_A = noise_patch_raw - residual_mean
                do_unroll_vis = (
                    self._is_unroll_model()
                    and bool(getattr(self, 'visualize_unroll', True))
                    and (viz_epoch % int(getattr(self, 'visualize_every', 1)) == 0)
                    and iteration == 0
                )

                if iteration == 0:
                    print(
                        '[Mean diagnostic] '
                        f'img_mean={float(img_mean):.6f}, '
                        f'residual_mean={residual_mean.item():.6f}, '
                        f'model_mean={model_mean.item():.6f}, '
                        f'noise_img_mean={float(noise_img.mean()):.6f}'
                    )
                    _check_val_coordinate_consistency(
                        real_A,
                        model_mean,
                        dataset_mean,
                        noise_patch_raw,
                    )

                if self._is_unroll_model():
                    final_data, final_prior = self._forward_unroll_data_and_prior(
                        real_A, model_mean,
                    )

                    # Main output = final prior x_K.
                    _accumulate_val_patch(
                        final_prior, residual_mean, noise_patch_raw, single_coordinate,
                        denoise_before_match, denoise_before_match, img_mean, input_img,
                    )

                    # Auxiliary output = final exact-MPGN correction state z_K.
                    _accumulate_val_patch(
                        final_data, residual_mean, noise_patch_raw, single_coordinate,
                        denoise_data, denoise_data, img_mean, input_img=None,
                    )
                    if do_unroll_vis:
                        unroll_model = self._unwrap_local_model()
                        _, debug = unroll_model(
                            real_A[:1],
                            patch_mean=model_mean,
                            y_obs=real_A[:1],
                            return_debug=True,
                        )
                        gt_patch_phys = self._extract_gt_patch(
                            ref_img,
                            single_coordinate,
                        )
                        save_dir = os.path.join(self.pth_path, 'unroll_viz')
                        self._save_unroll_debug_figure(
                            debug=debug,
                            patch_mean=model_mean,
                            gt_patch_phys=gt_patch_phys,
                            save_dir=save_dir,
                            epoch=viz_epoch,
                        )
                        self._save_unroll_diagnostics(
                            debug=debug,
                            save_dir=save_dir,
                            epoch=viz_epoch,
                        )
                else:
                    fake_B = self._forward_model(real_A, model_mean)
                    _accumulate_val_patch(
                        fake_B, residual_mean, noise_patch_raw, single_coordinate,
                        denoise_before_match, denoise_before_match, img_mean, input_img,
                    )

                batches_done = iteration
                batches_left = len(testloader) - batches_done
                time_left_seconds = int(batches_left * (time.time() - prev_time))
                prev_time = time.time()
                if iteration % 1 == 0:
                    time_end = time.time()
                    print(
                        '\r [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                        % (iteration + 1, len(testloader),
                           time_end - time_start, time_left_seconds),
                        end=' ')

                if (iteration + 1) % len(testloader) == 0:
                    print('\n', end=' ')

        # Convert stitched volumes to acquisition units.
        output_prior = denoise_before_match.squeeze().astype(np.float32)
        noisy_full = input_img.squeeze().astype(np.float32)
        del denoise_before_match

        if self._is_unroll_model():
            output_data = denoise_data.squeeze().astype(np.float32)
            del denoise_data

        # Compute SNR on the middle frames (drop snr_margin from each temporal end).
        T = min(output_prior.shape[0], noisy_full.shape[0], ref_img.shape[0])
        if self._is_unroll_model():
            T = min(T, output_data.shape[0])
        s = self.snr_margin
        e = T - self.snr_margin
        if e <= s:
            s, e = 0, T
        snr_prior = cal_snr(output_prior[s:e], ref_img[s:e])
        snr_noisy = cal_snr(noisy_full[s:e], ref_img[s:e])

        if self._is_unroll_model():
            snr_data = cal_snr(output_data[s:e], ref_img[s:e])
            snr_prior_scaled = cal_snr_scaled(output_prior[s:e], ref_img[s:e])
            snr_data_scaled = cal_snr_scaled(output_data[s:e], ref_img[s:e])
            print(
                'SNR (frames {:d}:{:d}, xK_prior / zK_exactMPGN / noisy vs GT) '
                '-----> {:.4f} / {:.4f} / {:.4f} dB '
                '(scaled {:.4f} / {:.4f})'.format(
                    s, e,
                    snr_prior, snr_data, snr_noisy,
                    snr_prior_scaled, snr_data_scaled,
                ))
        else:
            print(
                'SNR (frames {:d}:{:d}, denoised / noisy vs GT) '
                '-----> {:.4f} dB / {:.4f} dB'.format(
                    s, e, snr_prior, snr_noisy))

        # Append to val_metrics.md.
        try:
            metrics_path = os.path.join(self.pth_path, 'val_metrics.md')
            if not os.path.exists(metrics_path):
                with open(metrics_path, 'w') as f:
                    if self._is_unroll_model():
                        f.write(self._unroll_val_metrics_header())
                    else:
                        f.write('| Epoch | Iteration | SNR_denoised (dB) | '
                                'SNR_noisy (dB) |\n')
                        f.write('| ----- | --------- | ----------------- | '
                                '-------------- |\n')
            with open(metrics_path, 'a') as f:
                if self._is_unroll_model():
                    f.write('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n'.format(
                        metrics_epoch, train_iteration + 1,
                        snr_prior, snr_data,
                        snr_prior_scaled, snr_data_scaled,
                        snr_noisy))
                else:
                    f.write('| {} | {} | {:.4f} | {:.4f} |\n'.format(
                        metrics_epoch, train_iteration + 1, snr_prior, snr_noisy))
        except Exception:
            pass

        # Normalize and display inference image (optional).
        if self.visualize_images_per_epoch:
            print('Displaying the first denoised file ----->')
            test_img_display(output_prior, display_length=T, norm_min_percent=1, norm_max_percent=98)

        # Save denoised image cropped to the SNR-metric window.
        if self.save_test_images_per_epoch:
            save_img = output_prior[s:e]
            if input_data_type == 'uint16':
                save_img = np.clip(save_img, 0, 65535).astype('uint16')
            elif input_data_type == 'int16':
                save_img = np.clip(save_img, -32767, 32767).astype('int16')
            else:
                save_img = save_img.astype('int32')

            img_list = list(os.walk(self.datasets_path, topdown=False))[-1][-1]
            img_list.sort()
            test_im_name = img_list[0] if img_list else 'unknown.tif'
            result_name = (self.pth_path + '//' + test_im_name.replace('.tif', '') + '_'
                           + 'E_' + str(metrics_epoch).zfill(2)
                           + '_Iter_' + str(train_iteration + 1).zfill(4) + '.tif')
            io.imsave(result_name, save_img, check_contrast=False)
