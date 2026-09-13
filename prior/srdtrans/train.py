from operator import imod
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import math
import tifffile as tiff
import numpy as np
import random
import re

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')")

parser.add_argument('--patch_x', type=int, default=128, help="patch size in x and y")
parser.add_argument('--patch_t', type=int, default=128, help="patch size in t")
parser.add_argument('--overlap_factor', type=float, default=0.5, help="the overlap factor between two adjacent patches")
parser.add_argument('--train_datasets_size', type=int, default=6000, help='How many patches will be used for training.')
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='./train', help="A folder containing files for training")
parser.add_argument('--train_data_dir', type=str, default=None,
                    help="full path to the folder containing training .tif files; overrides --datasets_path and --datasets_folder")

parser.add_argument('--pth_path', type=str, default='./pth', help="the root path to save models")
parser.add_argument('--model_output_path', type=str, default=None,
                    help="checkpoint output path. Use a directory to save checkpoints there, or end with .pth to use it as a filename prefix")
parser.add_argument('--output_path', type=str, default='./results', help="output directory")

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--resume', type=str, default=None, help="Resume training from a checkpoint file or checkpoint folder")

parser.add_argument('--select_img_num', type=int, default=10000000000, help='How many frames will be used for training.')
parser.add_argument('--test_datasize', type=int, default=10000000000, help='How many frames will be tested.')
parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU

#############################################################################################################################################
from SRDTrans_v2 import SRDTrans_v2
from data_process import train_preprocess_lessMemoryMulStacks, trainset
from utils import save_yaml_train
from sampling import *

########################################################################################################################
def resolve_dataset_dir(datasets_path, datasets_folder):
    if os.path.isabs(datasets_folder):
        return os.path.normpath(datasets_folder)
    return os.path.normpath(os.path.join(datasets_path, datasets_folder))


def dataset_folder_tag(datasets_folder):
    normalized_path = os.path.normpath(datasets_folder)
    return os.path.basename(normalized_path) or normalized_path


def apply_train_data_override(args):
    if args.train_data_dir is None:
        return

    train_data_dir = os.path.abspath(args.train_data_dir)
    args.datasets_folder = train_data_dir
    args.datasets_path = os.path.dirname(train_data_dir) or '.'


def resolve_checkpoint_output(model_output_path, checkpoint_root, run_name):
    checkpoint_root = os.path.abspath(checkpoint_root)

    if model_output_path is None:
        return os.path.join(checkpoint_root, run_name), None

    if os.path.isabs(model_output_path):
        normalized_path = os.path.normpath(model_output_path)
    else:
        normalized_path = os.path.normpath(os.path.join(checkpoint_root, model_output_path))

    if normalized_path.lower().endswith('.pth'):
        checkpoint_dir = os.path.dirname(normalized_path) or os.getcwd()
        checkpoint_prefix = os.path.splitext(os.path.basename(normalized_path))[0]
    else:
        checkpoint_dir = normalized_path
        checkpoint_prefix = None

    return checkpoint_dir, checkpoint_prefix


def parse_checkpoint_epoch(path):
    match = re.search(r'E_(\d+)_Iter_(\d+)\.pth$', os.path.basename(path))
    if match is None:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def infer_checkpoint_prefix(path):
    base_name = os.path.splitext(os.path.basename(path))[0]

    if base_name.endswith('_latest_resume'):
        prefix = base_name[:-14]
        return prefix if prefix else None

    match = re.search(r'^(.*)_E_\d+_Iter_\d+$', base_name)
    if match is None:
        return None

    prefix = match.group(1)
    return prefix if prefix else None


def find_resume_checkpoint(resume_path):
    resume_path = os.path.abspath(resume_path)

    if os.path.isfile(resume_path):
        return resume_path

    if not os.path.isdir(resume_path):
        raise FileNotFoundError("Resume checkpoint path does not exist: {}".format(resume_path))

    latest_resume_candidates = []
    for file_name in os.listdir(resume_path):
        full_path = os.path.join(resume_path, file_name)
        if os.path.isfile(full_path) and file_name.endswith('latest_resume.pth'):
            latest_resume_candidates.append(full_path)

    if latest_resume_candidates:
        exact_latest_resume = os.path.join(resume_path, 'latest_resume.pth')
        if os.path.isfile(exact_latest_resume):
            return exact_latest_resume
        latest_resume_candidates.sort()
        return latest_resume_candidates[-1]

    checkpoint_list = []
    for file_name in os.listdir(resume_path):
        if not file_name.endswith('.pth'):
            continue
        full_path = os.path.join(resume_path, file_name)
        if os.path.isfile(full_path):
            checkpoint_list.append(full_path)

    if len(checkpoint_list) == 0:
        raise FileNotFoundError("No checkpoint file found in {}".format(resume_path))

    checkpoint_list.sort(key=lambda path: parse_checkpoint_epoch(path))
    return checkpoint_list[-1]


def resolve_resume_start_epoch(checkpoint_path, checkpoint):
    parsed_epoch, _ = parse_checkpoint_epoch(checkpoint_path)
    base_name = os.path.basename(checkpoint_path)

    if not isinstance(checkpoint, dict) or 'epoch' not in checkpoint:
        return parsed_epoch if parsed_epoch > 0 else 0

    saved_epoch = int(checkpoint['epoch'])

    if base_name.endswith('latest_resume.pth'):
        return saved_epoch + 1

    if parsed_epoch > 0:
        if saved_epoch == parsed_epoch - 1:
            return saved_epoch + 1
        return max(saved_epoch, parsed_epoch)

    return saved_epoch


def build_denoise_generator(args):
    return SRDTrans_v2(
        img_dim=args.patch_x,
        img_time=args.patch_t,
        in_channel=1,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=128 * 4,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=[8, 16, 32, 64],
        input_dropout_rate=0,
    )


def get_model_state(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def build_epoch_checkpoint_name(epoch_index, iteration, checkpoint_prefix):
    file_name = 'E_' + str(epoch_index + 1).zfill(2) + '_Iter_' + str(iteration).zfill(4) + '.pth'
    if checkpoint_prefix is not None:
        file_name = checkpoint_prefix + '_' + file_name
    return file_name


def build_latest_resume_name(checkpoint_prefix):
    if checkpoint_prefix is None:
        return 'latest_resume.pth'
    return checkpoint_prefix + '_latest_resume.pth'


apply_train_data_override(opt)

# use isotropic patch size by default
opt.patch_y = opt.patch_x
opt.patch_t = opt.patch_t
opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))
opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
opt.ngpu = opt.GPU.count(',') + 1
opt.batch_size = opt.ngpu
print('\033[1;31mTraining parameters -----> \033[0m')
print(opt)

########################################################################################################################
train_data_dir = resolve_dataset_dir(opt.datasets_path, opt.datasets_folder)
if not os.path.isdir(train_data_dir):
    raise FileNotFoundError("Training data directory not found: {}".format(train_data_dir))

dataset_tag = dataset_folder_tag(train_data_dir)
default_run_name = dataset_tag + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")

output_root = os.path.abspath(opt.output_path)
os.makedirs(output_root, exist_ok=True)

start_epoch = 0
resume_checkpoint = None

if opt.resume is not None:
    resume_checkpoint = find_resume_checkpoint(opt.resume)

    if opt.model_output_path is None:
        pth_path = os.path.dirname(resume_checkpoint)
        checkpoint_prefix = infer_checkpoint_prefix(resume_checkpoint)
        current_time = os.path.basename(os.path.normpath(pth_path))
    else:
        current_time = default_run_name
        pth_path, checkpoint_prefix = resolve_checkpoint_output(opt.model_output_path, opt.pth_path, current_time)

    output_path = os.path.join(output_root, current_time)
    print("Resume checkpoint -----> {}".format(resume_checkpoint))
else:
    current_time = default_run_name
    output_path = os.path.join(output_root, current_time)
    pth_path, checkpoint_prefix = resolve_checkpoint_output(opt.model_output_path, opt.pth_path, current_time)

pth_path = os.path.abspath(pth_path)
output_path = os.path.abspath(output_path)

os.makedirs(pth_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

opt.pth_path = pth_path
opt.output_path = output_path

print("training data is read from {}".format(train_data_dir))
print("ckp is saved in {}".format(pth_path))

train_name_list, train_noise_img, train_coordinate_list, stack_index = train_preprocess_lessMemoryMulStacks(opt)

yaml_name = os.path.join(pth_path, 'para.yaml')
save_yaml_train(opt, yaml_name)
########################################################################################################################

L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()

denoise_generator = build_denoise_generator(opt)

param_num = sum([param.nelement() for param in denoise_generator.parameters()])
print('\033[1;31mParameters of the model is {:.2f} M. \033[0m'.format(param_num / 1e6))

if torch.cuda.is_available():
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
    print('\033[1;31mUsing {} GPU(s) for training -----> \033[0m'.format(torch.cuda.device_count()))
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()

########################################################################################################################
optimizer_G = torch.optim.Adam(
    denoise_generator.parameters(),
    lr=opt.lr,
    betas=(opt.b1, opt.b2)
)

if resume_checkpoint is not None:
    checkpoint = torch.load(resume_checkpoint, map_location='cpu')
    model_to_load = denoise_generator.module if isinstance(denoise_generator, nn.DataParallel) else denoise_generator

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = resolve_resume_start_epoch(resume_checkpoint, checkpoint)
        print("Resume epoch -----> {}".format(start_epoch + 1))
    else:
        model_to_load.load_state_dict(checkpoint)
        resume_epoch, _ = parse_checkpoint_epoch(resume_checkpoint)
        if resume_epoch > 0:
            start_epoch = resume_epoch
        print("Resume from weights only -----> optimizer is reinitialized")
        print("Resume epoch -----> {}".format(start_epoch + 1))

########################################################################################################################
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

prev_time = time.time()
time_start = time.time()

########################################################################################################################
def train_epoch():
    global prev_time
    denoise_generator.train()
    train_data = trainset(train_name_list, train_coordinate_list, train_noise_img, stack_index)
    trainloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    for iteration, noisy in enumerate(trainloader):
        if cuda:
            noisy = noisy.cuda()

        mask1, mask2, mask3 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        noisy_sub3 = generate_subimages(noisy, mask3)

        noisy_output = denoise_generator(noisy_sub1)

        loss2neighbor_1 = 0.5 * L1_pixelwise(noisy_output, noisy_sub2) + 0.5 * L2_pixelwise(noisy_output, noisy_sub2)
        loss2neighbor_2 = 0.5 * L1_pixelwise(noisy_output, noisy_sub3) + 0.5 * L2_pixelwise(noisy_output, noisy_sub3)

        optimizer_G.zero_grad()
        Total_loss = 0.5 * loss2neighbor_1 + 0.5 * loss2neighbor_2
        Total_loss.backward()
        optimizer_G.step()

        batches_done = epoch * len(trainloader) + iteration
        batches_left = opt.n_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
        prev_time = time.time()

        if iteration % 1 == 0:
            time_end = time.time()
            print(
                '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f] [ETA: %s] [Time cost: %.2d s] '
                % (
                    epoch + 1,
                    opt.n_epochs,
                    iteration + 1,
                    len(trainloader),
                    Total_loss.item(),
                    time_left,
                    time_end - time_start
                ),
                end=' '
            )

        if (iteration + 1) % len(trainloader) == 0:
            print('\n', end=' ')

        if (iteration + 1) % len(trainloader) == 0:
            file_save_name = build_epoch_checkpoint_name(epoch, iteration + 1, checkpoint_prefix)
            model_save_name = os.path.join(pth_path, file_save_name)

            model_state_dict = get_model_state(denoise_generator)
            torch.save(model_state_dict, model_save_name)

            latest_resume_path = os.path.join(pth_path, build_latest_resume_name(checkpoint_prefix))
            torch.save(
                {
                    'epoch': epoch,
                    'iteration': iteration + 1,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'opt': vars(opt),
                },
                latest_resume_path
            )


for epoch in range(start_epoch, opt.n_epochs):
    train_epoch()
