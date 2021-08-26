# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from dataset.burstsr_dataset import get_burstsr_val_set, CanonImage

from utils.opencv_plotting import BurstSRVis
import torch
import cv2
import numpy as np
import importlib
import argparse
from admin.environment import env_settings


def visualize_results(setting_name):
    """ Visualize the results on the BurstSR validation set. setting_name denotes
        the name of the experiment setting, which contains the list of methods for which to visualize results.
    """
    expr_module = importlib.import_module('evaluation.burstsr.experiments.{}'.format(setting_name))
    expr_func = getattr(expr_module, 'main')
    network_list = expr_func()

    base_results_dir = env_settings().save_data_path

    dataset = get_burstsr_val_set()

    vis = BurstSRVis(boundary_ignore=40)

    for idx in range(len(dataset)):
        data_batch = dataset[idx]
        frame_gt = data_batch['frame_gt']
        burst = data_batch['burst']
        burst_name = data_batch['burst_name']
        meta_info_burst = data_batch['meta_info_burst']

        pred_all = []
        titles_all = []

        frame_gt_np = CanonImage.generate_processed_image(frame_gt.cpu(), meta_info_burst, return_np=True,
                                                          gamma=True,
                                                          smoothstep=True,
                                                          no_white_balance=False,
                                                          external_norm_factor=None)
        frame_gt_np = cv2.cvtColor(frame_gt_np, cv2.COLOR_RGB2BGR)

        for n in network_list:
            results_dir = '{}/burstsr/{}'.format(base_results_dir, n.get_unique_name())
            net_pred = cv2.imread('{}/{}.png'.format(results_dir, burst_name),
                                  cv2.IMREAD_UNCHANGED)
            pred = (torch.from_numpy(net_pred.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float().to(
                'cuda').unsqueeze(0)

            pred_proc_np = CanonImage.generate_processed_image(pred[0].cpu(), meta_info_burst, return_np=True,
                                                               gamma=True,
                                                               smoothstep=True,
                                                               no_white_balance=False,
                                                               external_norm_factor=None)
            pred_proc_np = cv2.cvtColor(pred_proc_np, cv2.COLOR_RGB2BGR)
            pred_all.append(pred_proc_np)
            titles_all.append(n.get_display_name())

        data = [{'images': [frame_gt_np,] + pred_all,
                 'titles': [burst_name, ] + titles_all}]
        cmd = vis.plot(data)

        if cmd == 'stop':
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the results on the BurstSR validation set. setting_name '
                                                 'denotes the name of the experiment setting, which contains the list '
                                                 'of methods for which to visualize results.')
    parser.add_argument('setting', type=str, help='Name of experiment setting')

    args = parser.parse_args()

    print('Press \'n\' to show next image. '
          'Press \'q\' to quit. '
          'Zoom on a particular region by drawing a box around it (click on the two corner points). '
          'In the zoomed pane (last row), you can click on an image an drag around. '
          'Using \'w\' and \'s\' keys, you can navigate between the two panes (normal pane and zoom pane)'
          'Using the \'space\' key, you can toggle between showing all the images and showing only a single image.'
          'In the single image mode, you can navigate between different images using the \'a\' and \'d\' keys.')

    visualize_results(args.setting)


