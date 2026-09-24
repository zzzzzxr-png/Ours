"""Small real-data DDP trainer smoke test; no validation or persistent output."""

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from likelihood.trainer import training_class_srdtrans


def main():
    with open(os.environ['DDP_TEST_CONFIG']) as stream:
        cfg = yaml.safe_load(stream)
    data = os.environ['DDP_TEST_DATA']
    params = dict(cfg)
    params.update({
        'datasets_path': data,
        'gt_path': os.environ['DDP_TEST_GT'],
        'pth_dir': '/tmp/ddp_trainer_smoke',
        'n_epochs': 1,
        'train_datasets_size': 4,
        'batch_size': 1,
        'num_workers': 0,
        'eval_val_per_epoch': False,
        'no_resume': True,
        'save_test_images_per_epoch': False,
    })
    trainer = training_class_srdtrans(params)
    trainer.run()
    if trainer.distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    if trainer.rank == 0:
        print('DDP trainer smoke PASS: sampler, optimizer, and rank-0 checkpoint path', flush=True)


if __name__ == '__main__':
    main()
