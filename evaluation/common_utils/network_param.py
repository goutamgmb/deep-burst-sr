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
from utils.loading import load_network
from admin.environment import env_settings


class NetworkParam:
    def __init__(self, module=None, parameter=None, epoch=None, burst_sz=None, display_name=None, unique_name=None,
                 network_path=None):
        """
        NetworkParam defines a network instance used for evaluations

        args:
            module - Name of the main training module used to train the network
            parameter - Name of the parameter setting used to train the network
            epoch - Which network checkpoint to use for evaluation. If None, the latest epoch is used
            burst_sz - Burst size used for evaluation. If None, the default value for the dataset is used

            display_name - Short name used when displaying results. If None, display_name is generated using
                            module, parameter, epoch, and burst_sz arguments
            unique_name - A unique name which is used when saving predictions of the method. If None, unique_name is
                            generated using module, parameter, epoch, and burst_sz arguments
            network_path - (Only applicable when using downloaded networks) Path to network checkpoint. Can either be
                            the absolute path, or the network name in case it is saved in path pointed by
                            pretrained_nets_dir variable in admin/local.py

        Example use cases:
        1. Evaluating networks trained using the toolkit
        - In this case, one can set the module and parameter names used to train the networks, and optionally epoch,
          burst_sz, display_name, and unique_name.
          e.g. NetworkParam(module='dbsr', parameter='default_synthetic')

        2. Evaluating downloaded pre-trained networks
        - In this case, set the network_path parameter to point to the downloaded checkpoint. Only checkpoint name is
          sufficient if the weights are in path pointed by pretrained_nets_dir variable in admin/local.py. Additionally,
          you need to set the unique_name which will be used when saving generated results. Optinally you can set
          burst_sz, and display_name.
          e.g. If you want to evaluate the network weights dbsr_default_synthetic.pth stored in the directory
               PRETRAINED_NETS_DIR, you can use
               NetworkParam(network_path='dbsr_default_synthetic.pth', unique_name='DBSR')

        2. Evaluating downloaded network predictions
        - In this case, save the downloaded predictions in directory pointed by save_data_path variable in
          admin/local.py. Set the unique_name to the name of the directory which contains predictions. Optinally you can
          set display_name.
          e.g. If the downloaded results are stored in SAVE_DATA_PATH/DBSR_results, then use
               NetworkParam(unique_name='DBSR_results')

        """
        assert network_path is None or (module is None and parameter is None and epoch is None)
        assert network_path is None or (unique_name is not None)

        self.module = module
        self.parameter = parameter
        self.epoch = epoch

        self.display_name = display_name
        self.unique_name = unique_name

        self.burst_sz = burst_sz

        self.network_path = network_path

    def load_net(self):
        if self.network_path is not None:
            if os.path.isabs(self.network_path):
                net, checkpoint_dict = load_network(self.network_path, return_dict=True)
            else:
                network_path = '{}/{}'.format(env_settings().pretrained_nets_dir, self.network_path)
                net, checkpoint_dict = load_network(network_path, return_dict=True)

        elif self.epoch is None:
            net, checkpoint_dict = load_network('{}/{}'.format(self.module, self.parameter), return_dict=True)
        else:
            net, checkpoint_dict = load_network('{}/{}'.format(self.module, self.parameter), checkpoint=self.epoch,
                                                return_dict=True)

        return net

    def get_display_name(self):
        # Display name is used when showing results
        if self.display_name is not None:
            return self.display_name
        else:
            return self.get_unique_name()

    def get_unique_name(self):
        # Unique name is used when saving results
        if self.unique_name is not None:
            return self.unique_name
        else:
            name = '{}_{}'.format(self.module, self.parameter)
            if self.epoch is not None:
                name = '{}_ep{:04d}'.format(name, self.epoch)
            if self.burst_sz is not None:
                name = '{}_bsz{:02d}'.format(name, self.burst_sz)

            return name

