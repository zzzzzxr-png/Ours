import os
import datetime
import time

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .backbone_factory import (
    DEFAULT_SRDTRANS_ROOT,
    build_denoise_network,
)
from likelihood.dataset import (
    multibatch_test_save_srdtrans,
    singlebatch_test_save_srdtrans,
    test_preprocess_lessMemoryNoTail_chooseOne_srdtrans,
    testset_srdtrans,
)
from skimage import io
try:
    from likelihood.movie_display import test_img_display, display_img
except ImportError:
    def test_img_display(*args, **kwargs): pass
    def display_img(*args, **kwargs): pass


class testing_class():
    """
    Class implementing testing process
    """

    def __init__(self, params_dict):
        """
        Constructor class for testing process

        Args:
           params_dict: dict
               The collection of testing params set by users
        Returns:
           self

        """
        self.overlap_factor = 0.5
        self.datasets_path = ''
        self.fmap = 16
        self.output_dir = './results'
        self.pth_dir = ''
        self.batch_size = 1
        self.patch_t = 150
        self.patch_x = 150
        self.patch_y = 150
        self.gap_y = 115
        self.gap_x = 115
        self.gap_t = 115
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.test_datasize = 400
        self.denoise_model = ''
        self.visualize_images_per_epoch = False
        self.colab_display = False
        self.result_display = ''
        self.sampling_mode = 'temporal'
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
        self.mpgn_alpha = 5000.0
        self.mpgn_beta = 1600.0
        self.mpgn_offset = 0.0
        self.unroll_steps = 4
        self.unroll_data_weight_scale = 0.0
        self.unroll_data_multiplier_init = 1.0
        self.unroll_prior_multiplier_init = 1.0
        self.unroll_rho_sched_min = 0.1
        self.unroll_rho_sched_max = 100.0
        self.unroll_rho_min = 1e-6
        self.unroll_eps = 1e-6
        self.unroll_gradient_checkpointing = False
        self.set_params(params_dict)

    def run(self):
        """
        General function for testing DeepCAD network.

        """
        self.prepare_file()
        self.read_modellist()
        self.read_imglist()
        self.save_yaml_test()
        self.initialize_network()
        self.distribute_GPU()
        self.test()

    def prepare_file(self):
        """
        Make data folder to store testing results
        Important Fields:
            self.datasets_name: the sub folder of the dataset
            self.pth_path: the folder for pth file storage

        """
        if self.datasets_path[-1] != '/':
            self.datasets_name = self.datasets_path.split("/")[-1]
        else:
            self.datasets_name = self.datasets_path.split("/")[-2]

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.output_path = (self.output_dir + '//'
                            + 'DataFolderIs_' + self.datasets_name + '_' + current_time
                            + '_ModelFolderIs_' + self.denoise_model)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def set_params(self, params_dict):
        """
        Set the params set by user to the testing class object and calculate some default parameters for testing

        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))
        self.ngpu = str(self.GPU).count(',') + 1
        self.batch_size = self.ngpu
        print('\033[1;31mTesting parameters -----> \033[0m')
        print(self.__dict__)

    def read_imglist(self):
        im_folder = self.datasets_path
        self.img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
        self.img_list.sort()
        print('\033[1;31mStacks for processing -----> \033[0m')
        print('Total stack number -----> ', len(self.img_list))
        for img in self.img_list:
            print(img)

    def read_modellist(self):
        import sys
        model_path = self.pth_dir + '//' + self.denoise_model
        model_file_list = list(os.walk(model_path, topdown=False))[-1][-1]
        model_list = [item for item in model_file_list if '.pth' in item]
        model_list.sort()
        try:
            self.model_list = model_list[-1]
        except Exception:
            print('\033[1;31mThere is no .pth file in the models directory! \033[0m')
            sys.exit()
        self.model_list_length = 1

    def initialize_network(self):
        """
        Initialize denoise network (3D U-Net or SRDTrans).

        Important Fields:
           self.fmap: the number of the feature map in U-Net 3D network.
           self.local_model: the denoise network

        """
        self.local_model = build_denoise_network(self)

    def save_yaml_test(self):
        """
        Save some essential params in para.yaml.

        """
        yaml_name = self.output_path + '//para.yaml'
        para = {'datasets_path': 0, 'test_datasize': 0, 'denoise_model': 0,
                'output_dir': 0, 'pth_dir': 0, 'GPU': 0, 'batch_size': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'fmap': 0, 'overlap_factor': 0}
        para["datasets_path"] = self.datasets_path
        para["denoise_model"] = self.denoise_model
        para["test_datasize"] = self.test_datasize
        para["output_dir"] = self.output_dir
        para["pth_dir"] = self.pth_dir
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["fmap"] = self.fmap
        para["overlap_factor"] = self.overlap_factor
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def distribute_GPU(self):
        """
        Allocate the GPU for the testing program.

        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
            self.local_model = nn.DataParallel(self.local_model, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def _is_unroll_model(self):
        return getattr(self, 'backbone', 'unet') in (
            'unet_unroll',
            '3dunet_unroll',
            '3DUNet_unroll',
            'srdtrans_unroll_transformer',
        )

    def _forward_model(self, x, patch_mean):
        if self._is_unroll_model():
            return self.local_model(x, patch_mean=patch_mean)
        return self.local_model(x)

    def test(self):
        """
        Pytorch testing workflow

        """
        pth_count = 0

        if '.pth' in self.model_list:
            pth_count = pth_count + 1
            pth_name = self.model_list
            output_path_name = self.output_path + '//' + pth_name.replace('.pth', '')
            if not os.path.exists(output_path_name):
                os.mkdir(output_path_name)

            model_name = self.pth_dir + '//' + self.denoise_model + '//' + pth_name
            if isinstance(self.local_model, nn.DataParallel):
                self.local_model.module.load_state_dict(torch.load(model_name))
                self.local_model.eval()
            else:
                self.local_model.load_state_dict(torch.load(model_name))
                self.local_model.eval()
            self.local_model.cuda()
            self.print_img_name = False
            print('Testing the last model by default:')
            for N in range(len(self.img_list)):
                # Ensure SRDTrans test preprocess sees the same fields as training.val.
                self.datasets_folder = self.datasets_path
                if not hasattr(self, 'gap_t') or self.gap_t is None:
                    self.gap_t = int(self.patch_t * (1 - self.overlap_factor))
                name_list, noise_img, coordinate_list, img_mean, input_data_type = \
                    test_preprocess_lessMemoryNoTail_chooseOne_srdtrans(self, N)
                prev_time = time.time()
                time_start = time.time()
                denoise_img = np.zeros(noise_img.shape)
                input_img = np.zeros(noise_img.shape)

                test_data = testset_srdtrans(name_list, coordinate_list, noise_img)
                testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                        num_workers=self.num_workers)
                with torch.no_grad():
                    for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                        noise_patch = noise_patch.cuda()
                        residual_mean = torch.as_tensor(
                            float(noise_img.mean()),
                            dtype=torch.float32,
                            device=noise_patch.device,
                        ).view(1, 1, 1, 1, 1)
                        dataset_mean = torch.as_tensor(
                            float(img_mean),
                            dtype=torch.float32,
                            device=noise_patch.device,
                        ).view(1, 1, 1, 1, 1)
                        model_mean = dataset_mean + residual_mean

                        real_A = noise_patch.float() - residual_mean
                        fake_B_centered = self._forward_model(
                            real_A,
                            model_mean,
                        )
                        fake_B = fake_B_centered + residual_mean

                        batches_done = iteration
                        batches_left = 1 * len(testloader) - batches_done
                        time_left_seconds = int(batches_left * (time.time() - prev_time))
                        time_left = datetime.timedelta(seconds=time_left_seconds)
                        prev_time = time.time()

                        if iteration % 1 == 0:
                            time_end = time.time()
                            time_cost = time_end - time_start
                            print(
                                '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] '
                                '[Time Cost: %.0d s] [ETA: %s s]     '
                                % (
                                    pth_count, self.model_list_length, pth_name,
                                    N + 1, len(self.img_list), self.img_list[N],
                                    iteration + 1, len(testloader),
                                    time_cost, time_left_seconds
                                ), end=' ')

                        if (iteration + 1) % len(testloader) == 0:
                            print('\n', end=' ')

                        output_image = np.squeeze(fake_B.cpu().detach().numpy())
                        raw_image = np.squeeze(real_A.cpu().detach().numpy())
                        if output_image.ndim == 3:
                            postprocess_turn = 1
                        else:
                            postprocess_turn = output_image.shape[0]

                        if postprocess_turn > 1:
                            for id in range(postprocess_turn):
                                output_patch, raw_patch, \
                                    stack_start_w, stack_end_w, \
                                    stack_start_h, stack_end_h, \
                                    stack_start_s, stack_end_s = multibatch_test_save_srdtrans(
                                        single_coordinate, id, output_image, raw_image)
                                output_patch = output_patch + img_mean
                                raw_patch = raw_patch + img_mean
                                denoise_img[stack_start_s:stack_end_s,
                                            stack_start_h:stack_end_h,
                                            stack_start_w:stack_end_w] = \
                                    output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                                input_img[stack_start_s:stack_end_s,
                                          stack_start_h:stack_end_h,
                                          stack_start_w:stack_end_w] = raw_patch
                        else:
                            output_patch, raw_patch, \
                                stack_start_w, stack_end_w, \
                                stack_start_h, stack_end_h, \
                                stack_start_s, stack_end_s = singlebatch_test_save_srdtrans(
                                    single_coordinate, output_image, raw_image)
                            output_patch = output_patch + img_mean
                            raw_patch = raw_patch + img_mean
                            denoise_img[stack_start_s:stack_end_s,
                                        stack_start_h:stack_end_h,
                                        stack_start_w:stack_end_w] = \
                                output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                            input_img[stack_start_s:stack_end_s,
                                      stack_start_h:stack_end_h,
                                      stack_start_w:stack_end_w] = raw_patch

                output_img = denoise_img.squeeze().astype(np.float32)
                del denoise_img

                if self.visualize_images_per_epoch:
                    print('Displaying the denoised file ----->')
                    display_length = 200
                    test_img_display(output_img, display_length=display_length,
                                     norm_min_percent=1, norm_max_percent=99)

                output_img = output_img.astype(input_data_type)
                result_name = (output_path_name + '//' + self.img_list[N].replace('.tif', '')
                               + '_' + pth_name.replace('.pth', '') + '_output.tif')
                io.imsave(result_name, output_img, check_contrast=False)

                if pth_count == self.model_list_length:
                    if self.colab_display:
                        self.result_display = (output_path_name + '//'
                                               + self.img_list[N].replace('.tif', '')
                                               + '_' + pth_name.replace('.pth', '') + '_output.tif')

        print('Testing finished. All results saved to disk.')
