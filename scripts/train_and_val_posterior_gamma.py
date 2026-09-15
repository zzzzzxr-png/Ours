"""
Gamma-Poisson dual-context protocol: dual axial replacement + moment-fused
Gamma prior + MPGN posterior correction (train/val share the same chain).

Axis from sampling_mode:
  height_mask / width_mask / temporal_mask  -> dual H / W / T replacements

Example:
  python train_and_val_posterior_gamma.py \\
    --datasets_path /path/to/noisy --gt /path/to/gt.tif \\
    --backbone srdtrans_v2 --sampling_mode height_mask \\
    --mask_ratio 0.05 --mask_min_dist 2 --kappa_mode learned_map
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)


from likelihood.backbone_factory import DEFAULT_SRDTRANS_ROOT
from posterior.trainer_gamma import training_class_srdtrans_gamma


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'SRDTrans-protocol train + validation '
            '(3D U-Net or SRDTrans_v2)'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--datasets_path', type=str, required=True,
                        help='Folder containing noisy .tif file(s) for training')
    parser.add_argument('--gt', type=str, required=True,
                        help='Ground-truth .tif for validation SNR')
    parser.add_argument('--pth_dir', type=str, default='./260615_experiments_srdtrans_protocol_unet',
                        help='Root directory for checkpoints')

    parser.add_argument('--patch_xy', type=int, default=128,
                        help='Patch size in x and y')
    parser.add_argument('--patch_t', type=int, default=128,
                        help='Patch size in t')
    parser.add_argument('--overlap_factor', type=float, default=0.75,
                        help='Training patch overlap (SRDTrans default 0.5)')
    parser.add_argument('--val_overlap_factor', type=float, default=0.5,
                        help='Validation patch overlap (SRDTrans test default 0.5)')

    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Epochs (default: 50 for unet and transformer backbones)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Adam LR (default: 1e-5 for unet, 5e-5 for srdtrans)')
    parser.add_argument('--b1', type=float, default=None,
                        help='Adam beta1 (default: 0.9 for unet, 0.5 for srdtrans)')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--train_datasets_size', type=int, default=None,
                        help='Patches per epoch (default: 5000 for unet, 6000 for srdtrans)')

    parser.add_argument('--sampling_mode', type=str, default='height_mask',
                        choices=[
                            'temporal',
                            'spatial',
                            'temporal_mask',
                            'height_mask',
                            'width_mask',
                            'spatial_mask_mean',
                            'temporal_mask_mean',
                            'spatial_mask_slice',
                            'temporal_mask_slice',
                            'slice_mask',
                            'n2v',
                        ],
                        help='Training sampling strategy. Dual-context Gamma uses '
                             'height_mask (H±), width_mask (W±), or temporal_mask (T±). '
                             'Legacy spatial_mask is removed (split into height/width).')
    parser.add_argument('--backbone', type=str, default='srdtrans_v2',
                        choices=[
                            'unet',
                            'srdtrans_v2',
                        ],
                        help='Denoise backbone (all external backbones use 1ch, '
                             'trained from scratch under SRD protocol)')
    parser.add_argument('--representation', type=str, default='dtcwt', choices=['dtcwt'])
    parser.add_argument('--dtcwt_dim', type=int, default=2, choices=[2, 3])
    parser.add_argument('--dtcwt_levels', type=int, default=3)
    parser.add_argument('--fmap', type=int, default=16,
                        help='3D U-Net feature maps (ignored for transformer backbones)')
    parser.add_argument('--srdtrans-root', type=str, default=DEFAULT_SRDTRANS_ROOT,
                        help='Path to the SRDTrans_v2 package or its parent.')
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--window-size', type=int, default=7)
    parser.add_argument('--num-trans-block', type=int, default=1)
    parser.add_argument('--attn-dropout-rate', type=float, default=0.1)
    parser.add_argument('--input-dropout-rate', type=float, default=0.0)
    parser.add_argument('--srdtrans-f-maps', type=str, default='8,16,32,64')
    parser.add_argument('--temporal_strides', type=str, default=None,
                        help='Comma-separated temporal strides for SRDTrans backbone '
                             '(e.g. 2,2,2,2); ignored for other backbones')
    parser.add_argument('--last_squeeze_op', type=str, default='conv',
                        choices=['conv', 'fold', 'local_attn'],
                        help='Downsampling operator for the last temporal SqueezeLayer only')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--smoke-test-multigpu', action='store_true',
                        help='Run one synthetic DDP forward/backward step and exit')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--select_img_num', type=int, default=100000)
    parser.add_argument('--no_resume', action='store_true',
                        help='Delete existing experiment folder and restart from scratch '
                             '(default: resume from latest E_*_Iter_*.pth if present)')
    parser.add_argument('--mask_ratio', type=float, default=0.05,
                        help='Target mask ratio (~1/num complementary groups). '
                             'Default 0.05 → typically 20 groups via period factorization.')
    parser.add_argument('--mask_min_dist', type=int, default=2,
                        help='Minimum preferred lattice period per axis for complementary groups.')
    parser.add_argument(
        '--lattice-random-phase',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Legacy flag (dual-context uses complementary group sampling instead).',
    )
    parser.add_argument('--slice_axis', type=str, default='random',
                        choices=['random', 't', 'h', 'w'],
                        help='Slice axis for slice_mask mode: random or fixed T/H/W.')

    parser.add_argument('--trans_order', type=str, default='st',
                        choices=['ts', 'st'],
                        help='SRDTrans_v2 attention order: ts=time->space (original), '
                             'st=space->time')
    parser.add_argument('--space_post_norm', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='SRDTrans_v2: LayerNorm after spatial transformer '
                             '(recommended when trans_order=st)')
    parser.add_argument('--space_dropout_rate', type=float, default=0.0,
                        help='SRDTrans_v2: dropout on spatial transformer input')
    parser.add_argument('--use_msconv_before_trans',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='SRDTrans_v2: MSConvBeforeTrans instead of 3x3x3 '
                             'conv_before_trans')
    parser.add_argument(
        '--mask_loss',
        type=str,
        default='nll',
        choices=['l1l2', 'nll'],
        help='Masked self-supervision loss: L1+L2 or Gamma-NB predictive NLL',
    )
    parser.add_argument(
        '--kappa_mode',
        type=str,
        default='learned_map',
        choices=['learned_map', 'fixed'],
        help='learned_map: network predicts per-voxel kappa; fixed: scalar --mpgn_kappa',
    )
    parser.add_argument('--mpgn_alpha', type=float, default=5000.0,
                        help='MPGN NLL alpha (photon gain)')
    parser.add_argument('--mpgn_beta', type=float, default=1600.0,
                        help='MPGN NLL beta (read noise variance)')
    parser.add_argument('--mpgn_offset', type=float, default=0.0,
                        help='MPGN NLL offset')
    parser.add_argument('--mpgn_kmax', type=int, default=512,
                        help='Hard cap for strict adaptive MPGN K selection')
    parser.add_argument('--mpgn_nll_chunk_t', type=int, default=8,
                        help='MPGN NLL temporal chunk size')
    parser.add_argument('--mpgn_k_tail_tol', type=float, default=1e-8,
                        help='Maximum relative omitted positive K-tail')
    parser.add_argument(
        '--mpgn_kappa',
        type=float,
        default=50.0,
        help='Fixed Gamma κ when --kappa_mode fixed.',
    )
    parser.add_argument('--mpgn_kappa_init', type=float, default=50.0,
                        help='Initial κ for learned_map exp head')
    parser.add_argument('--mpgn_kappa_min', type=float, default=1e-4,
                        help='Minimum κ after exp')
    parser.add_argument('--mpgn_prior_var_min', type=float, default=1e-12,
                        help='Floor for fused Gamma variance')
    parser.add_argument('--val_patch_batch', type=int, default=1,
                        help='Number of original patches per val step (network batch=2×)')
    parser.add_argument('--smoke_test_val_batch', action='store_true',
                        help='Run val_patch_batch smoke test then exit')
    parser.add_argument('--smoke_test_batch_candidates', type=str, default='1,2,4,8,16,32')
    parser.add_argument('--smoke_test_memory_fraction', type=float, default=0.85)

    parser.add_argument('--eval_every_iters', type=int, default=0,
                        help='Validate every N iterations (0 = end of epoch only)')
    parser.add_argument('--checkpoint-every-epochs', type=int, default=5,
                        help='Save a periodic checkpoint every N epochs')
    parser.add_argument('--validation-every-epochs', type=int, default=1,
                        help='Run end-of-epoch validation every N epochs')
    parser.add_argument(
        '--val_process_frames', type=int, default=400,
        help='SNR metric window length (frames 0..N). Inference may use '
             'max(N, patch_t) for tiling, but SNR is always on '
             '[snr_margin : N - snr_margin].',
    )
    parser.add_argument('--snr_margin', type=int, default=50)
    parser.add_argument('--save_test_images_per_epoch', action='store_true', default=True,
                        help='Save denoised test images per epoch')
    parser.add_argument('--save_debug_posterior', action='store_true', default=False,
                        help='Save debug information for Gamma posterior')
    parser.add_argument('--eval_val_per_epoch', action='store_true', default=True,
                        help='Run validation per epoch')
    parser.add_argument('--seed', type=int, default=1024,
                        help='Random seed for training (Python/NumPy/PyTorch/DataLoader).')
    parser.add_argument(
        '--eval_ckpt',
        type=str,
        default=None,
        help='Load this E_*_Iter_*.pth, run one validation, then exit. '
             'Does not resume or write into the original experiment folder '
             'unless --pth_dir points there.',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.datasets_path):
        raise ValueError('--datasets_path is not a directory: {}'.format(args.datasets_path))
    if not os.path.isfile(args.gt):
        raise ValueError('--gt does not exist: {}'.format(args.gt))

    if args.backbone == 'unet':
        if args.lr is None:
            args.lr = 1e-5
        if args.b1 is None:
            args.b1 = 0.9
        if args.train_datasets_size is None:
            args.train_datasets_size = 6000
        if args.n_epochs is None:
            args.n_epochs = 100
    else:
        if args.lr is None:
            args.lr = 5e-5
        if args.b1 is None:
            args.b1 = 0.5
        if args.train_datasets_size is None:
            args.train_datasets_size = 6000
        if args.n_epochs is None:
            args.n_epochs = 50

    srdtrans_f_maps = [int(x.strip()) for x in args.srdtrans_f_maps.split(',') if x.strip()]
    temporal_strides = None
    if args.temporal_strides is not None:
        temporal_strides = [
            int(x.strip()) for x in args.temporal_strides.split(',') if x.strip()
        ]

    train_dict = {
        'datasets_path': args.datasets_path,
        'gt_path': args.gt,
        'pth_dir': args.pth_dir,
        'patch_x': args.patch_xy,
        'patch_y': args.patch_xy,
        'patch_t': args.patch_t,
        'overlap_factor': args.overlap_factor,
        'val_overlap_factor': args.val_overlap_factor,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'b1': args.b1,
        'b2': args.b2,
        'train_datasets_size': args.train_datasets_size,
        'sampling_mode': args.sampling_mode,
        'no_resume': args.no_resume,
        'mask_ratio': args.mask_ratio,
        'mask_min_dist': args.mask_min_dist,
        'lattice_random_phase': args.lattice_random_phase,
        'slice_axis': args.slice_axis,
        'fmap': args.fmap,
        'backbone': args.backbone,
        'representation': args.representation,
        'dtcwt_dim': args.dtcwt_dim,
        'dtcwt_levels': args.dtcwt_levels,
        'srdtrans_root': args.srdtrans_root,
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'window_size': args.window_size,
        'num_transBlock': args.num_trans_block,
        'attn_dropout_rate': args.attn_dropout_rate,
        'input_dropout_rate': args.input_dropout_rate,
        'srdtrans_f_maps': srdtrans_f_maps,
        'temporal_strides': temporal_strides,
        'last_squeeze_op': args.last_squeeze_op,
        'GPU': args.gpu,
        'smoke_test_multigpu': args.smoke_test_multigpu,
        'num_workers': args.num_workers,
        'select_img_num': args.select_img_num,
        'no_resume': args.no_resume,
        'eval_val_per_epoch': True,
        'eval_every_iters': args.eval_every_iters,
        'checkpoint_every_epochs': args.checkpoint_every_epochs,
        'validation_every_epochs': args.validation_every_epochs,
        'val_process_frames': args.val_process_frames,
        'snr_margin': args.snr_margin,
        'save_test_images_per_epoch': True,
        'seed': args.seed,
        'trans_order': args.trans_order,
        'space_post_norm': args.space_post_norm,
        'space_dropout_rate': args.space_dropout_rate,
        'use_msconv_before_trans': args.use_msconv_before_trans,
        'mask_loss': args.mask_loss,
        'kappa_mode': args.kappa_mode,
        'mpgn_alpha': args.mpgn_alpha,
        'mpgn_beta': args.mpgn_beta,
        'mpgn_offset': args.mpgn_offset,
        'mpgn_kmax': args.mpgn_kmax,
        'mpgn_nll_chunk_t': args.mpgn_nll_chunk_t,
        'mpgn_k_tail_tol': args.mpgn_k_tail_tol,
        'mpgn_kappa': args.mpgn_kappa,
        'mpgn_kappa_init': args.mpgn_kappa_init,
        'mpgn_kappa_min': args.mpgn_kappa_min,
        'mpgn_prior_var_min': args.mpgn_prior_var_min,
        'val_patch_batch': args.val_patch_batch,
        'smoke_test_val_batch': args.smoke_test_val_batch,
        'smoke_test_batch_candidates': args.smoke_test_batch_candidates,
        'smoke_test_memory_fraction': args.smoke_test_memory_fraction,
        'eval_ckpt': args.eval_ckpt,
    }

    tc = training_class_srdtrans_gamma(train_dict)
    tc.run()


if __name__ == '__main__':
    main()
