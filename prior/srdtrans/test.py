import os
from pyexpat import model
import torch
import torch.nn as nn
import argparse
import time
import datetime
import numpy as np
import math
import tifffile as tiff

from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import save_yaml_test, resolve_dataset_dir, dataset_folder_tag
from skimage import io
from tqdm import tqdm

from SRDTrans_v2 import SRDTrans_v2
from data_process import test_preprocess_lessMemoryNoTail_chooseOne, testset, singlebatch_test_save, multibatch_test_save
from utils import save_yaml_train
from sampling import *

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=str, default='0,1', help="the index of GPU used for computation (e.g., '0', '0,1', '0,1,2')")

parser.add_argument('--denoise_model', type=str, default=None, help='A folder containing models to be tested')
parser.add_argument('--datasets_folder', type=str, default='train', help="A folder containing all *.tif files for training")

parser.add_argument('--patch_x', type=int, default=128, help="patch size in x and y")
parser.add_argument('--patch_t', type=int, default=128, help="patch size in t")
parser.add_argument('--overlap_factor', type=float, default=0.5, help="the overlap factor between two adjacent patches")
parser.add_argument('--datasets_path', type=str, default='./datasets', help="dataset root path")
parser.add_argument('--pth_path', type=str, default='./pth', help="the root path to save models")
parser.add_argument('--output_path', type=str, default='./results', help="output directory")
parser.add_argument('--test_data_dir', type=str, default=None,
                    help="full path to the folder containing test .tif files; overrides --datasets_path and --datasets_folder")
parser.add_argument('--model_input_path', type=str, default=None,
                    help="full path to a .pth model file or a directory containing .pth files; overrides --pth_path and --denoise_model")
parser.add_argument('--result_output_dir', type=str, default=None,
                    help="full path to the directory where denoising results will be saved; overrides --output_path")

parser.add_argument('--test_datasize', type=int, default=1000000, help='how many slices to be tested')
parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
opt = parser.parse_args()


def apply_test_data_override(args):
    if args.test_data_dir is None:
        return

    test_data_dir = os.path.abspath(args.test_data_dir)
    args.datasets_folder = test_data_dir
    args.datasets_path = os.path.dirname(test_data_dir) or '.'


def resolve_model_source(args):
    if args.model_input_path is None:
        model_dir = os.path.abspath(os.path.join(args.pth_path, args.denoise_model))
        model_tag = os.path.basename(os.path.normpath(args.denoise_model))
        return model_dir, model_tag

    model_input_path = os.path.abspath(args.model_input_path)
    if os.path.isdir(model_input_path):
        model_dir = model_input_path
        model_tag = os.path.basename(os.path.normpath(model_input_path))
        return model_dir, model_tag

    if model_input_path.lower().endswith('.pth') and os.path.isfile(model_input_path):
        model_dir = os.path.dirname(model_input_path) or os.getcwd()
        model_tag = os.path.splitext(os.path.basename(model_input_path))[0]
        return model_dir, model_tag

    raise FileNotFoundError("Model path not found: {}".format(model_input_path))


def resolve_model_list(model_dir, model_input_path):
    if model_input_path is not None:
        model_input_path = os.path.abspath(model_input_path)
        if os.path.isfile(model_input_path):
            return [os.path.basename(model_input_path)]

    model_list = [name for name in os.listdir(model_dir) if name.lower().endswith('.pth')]
    model_list.sort()
    if not model_list:
        raise FileNotFoundError("No .pth files found in model directory: {}".format(model_dir))
    print('If there are multiple models, only the last one will be used for denoising.')
    return model_list[-1:]


def resolve_result_output_dir(args, dataset_tag, model_tag):
    if args.result_output_dir is not None:
        return os.path.abspath(args.result_output_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_name = 'DataFolderIs_' + dataset_tag + '_' + current_time + '_ModelFolderIs_' + model_tag
    return os.path.abspath(os.path.join(args.output_path, output_name))


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


def load_model_state(model, model_name):
    ckpt = torch.load(model_name, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


apply_test_data_override(opt)

# use isotropic patch size by default
opt.patch_y = opt.patch_x  # the height of 3D patches (patch size in y)
opt.patch_t = opt.patch_t  # the length of 3D patches (patch size in t)
# opt.gap_t (image gap) is the distance between two adjacent patches
# use isotropic opt.gap by default
opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))
opt.ngpu = str(opt.GPU).count(',') + 1
opt.batch_size = opt.ngpu                       # By default, the batch size is equal to the number of GPU for minimal memory consumption
print('\033[1;31mParameters -----> \033[0m')
print(opt)

########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)

model_path, model_tag = resolve_model_source(opt)
if not os.path.isdir(model_path):
    raise FileNotFoundError("Model directory not found: {}".format(model_path))
model_list = resolve_model_list(model_path, opt.model_input_path)
opt.denoise_model = model_tag
opt.pth_path = model_path

# get stacks for processing
im_folder = resolve_dataset_dir(opt.datasets_path, opt.datasets_folder)
if not os.path.isdir(im_folder):
    raise FileNotFoundError("Test data directory not found: {}".format(im_folder))
opt.datasets_folder_tag = dataset_folder_tag(im_folder)

img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
img_list.sort()

        
print('\033[1;31mStacks to be processed -----> \033[0m')
print('Total stack umber -----> ', len(img_list))
for img in img_list: print(img)

output_path1 = resolve_result_output_dir(opt, opt.datasets_folder_tag, model_tag)
os.makedirs(output_path1, exist_ok=True)
opt.output_path = output_path1

print("test data is read from:", im_folder)
print("model is read from:", model_path)
print("results are saved in:", output_path1)

yaml_name = os.path.join(output_path1, 'para.yaml')

save_yaml_test(opt, yaml_name)

##############################################################################################################################################################

denoise_generator = build_denoise_generator(opt)


if torch.cuda.is_available():
    print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

##############################################################################################################################################################

def test():
    # Start processing
    for pth_index in range(len(model_list)):
        aaa = model_list[pth_index]
        if '.pth' in aaa:
            pth_name = model_list[pth_index]
            output_path = os.path.join(output_path1, pth_name.replace('.pth', ''))
            
            os.makedirs(output_path, exist_ok=True)

            # load model
            model_name = os.path.join(model_path, pth_name)
            load_model_state(denoise_generator, model_name)
            denoise_generator.eval()
            denoise_generator.cuda()

            # test all stacks
            for N in range(len(img_list)):
                name_list, noise_img, coordinate_list, img_mean, input_data_type = test_preprocess_lessMemoryNoTail_chooseOne(opt, N)
                prev_time = time.time()
                time_start = time.time()
                denoise_img = np.zeros(noise_img.shape)
                result_file_name = img_list[N].replace('.tif', '') + '_' + pth_name.replace('.pth','') + '_output.tif'
                result_name = os.path.join(output_path, result_file_name)
                print(os.getcwd())
                print(result_name)

                # print("coordinate_list length:", len(coordinate_list))
                test_data = testset(name_list, coordinate_list, noise_img)
                testloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
                with torch.no_grad():
                    for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                        noise_patch = noise_patch.cuda()

                        real_A = noise_patch
                        real_A = Variable(real_A)
                        fake_B = denoise_generator(real_A)

                        preditc_numpy = fake_B.cpu().detach().numpy().astype(np.float32)
                        ################################################################################################################
                        # Determine approximate time left
                        batches_done = iteration
                        batches_left = 1 * len(testloader) - batches_done
                        time_left_seconds = int(batches_left * (time.time() - prev_time))
                        time_left = datetime.timedelta(seconds=time_left_seconds)
                        prev_time = time.time()
                        ################################################################################################################
                        if iteration % 1 == 0:
                            time_end = time.time()
                            time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                            print(
                                '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                                % (
                                    pth_index + 1,
                                    len(model_list),
                                    pth_name,
                                    N + 1,
                                    len(img_list),
                                    img_list[N],
                                    iteration + 1,
                                    len(testloader),
                                    time_cost,
                                    time_left_seconds
                                ), end=' ')

                        if (iteration + 1) % len(testloader) == 0:
                            print('\n', end=' ')
                        ################################################################################################################
                        output_image = np.squeeze(fake_B.cpu().detach().numpy())
                        raw_image = np.squeeze(real_A.cpu().detach().numpy())
                        if (output_image.ndim == 3):
                            turn = 1
                        else:
                            turn = output_image.shape[0]
                        # print(turn)
                        if (turn > 1):
                            for id in range(turn):
                                # print('shape of output_image -----> ',output_image.shape)
                                aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                                    single_coordinate, id, output_image, raw_image)
                                aaaa=aaaa+img_mean
                                bbbb=bbbb+img_mean
                                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h,
                                stack_start_w:stack_end_w] \
                                    = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5
                        else:
                            aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                                single_coordinate, output_image, raw_image)
                            aaaa=aaaa+img_mean
                            bbbb=bbbb+img_mean
                            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                                = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5

                    del noise_img
                    output_img = denoise_img.squeeze().astype(np.float32) * opt.scale_factor
                    del denoise_img
                    output_img=np.clip(output_img, 0, 65535).astype('int32')
                    # Save inference image
                    if input_data_type == 'uint16':
                        output_img=np.clip(output_img, 0, 65535)
                        output_img = output_img.astype('uint16')

                    elif input_data_type == 'int16':
                        output_img=np.clip(output_img, -32767, 32767)
                        output_img = output_img.astype('int16')

                    else:
                        output_img = output_img.astype('int32')
                            
                    io.imsave(result_name, output_img, check_contrast=False)
                    print("test result saved in:", result_name)


if __name__ == "__main__":
    test()
