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

import torch
import random
import numpy as np

from admin.tensordict import TensorDict


def no_processing(data):
    return data


class IndexedImage(torch.utils.data.Dataset):
    """ Sequentially load the images from the dataset """
    def __init__(self, dataset, processing=no_processing):
        self.dataset = dataset

        self.processing = processing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        frame, meta_info = self.dataset.get_image(index)

        data = TensorDict({'frame': frame,
                           'dataset': self.dataset.get_name()})

        return self.processing(data)


class RandomImage(torch.utils.data.Dataset):
    """ Randomly sample images from a list of datasets """
    def __init__(self, datasets: list, p_datasets: list, samples_per_epoch, processing=no_processing, fail_safe=False):
        """
        args:
            datasets - list of datasets
            p_datasets - list containing the probabilities by which each dataset will be sampled
            samples_per_epoch - number of sampled loaded per epoch
            processing - the processing function to be applied on the loaded image
            fail_safe - If set to True, the sampler will not crash in case of error when loading an image. Instead it
                        will try to randomly load another image.
        """
        self.datasets = datasets

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch

        self.processing = processing
        self.fail_safe = fail_safe

    def __len__(self):
        return self.samples_per_epoch

    def load_image(self, index):
        # Sample dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        # Sample image
        im_id = random.randint(0, dataset.get_num_images() - 1)

        # Load image
        frame, meta_info = dataset.get_image(im_id)

        data = TensorDict({'frame': frame,
                           'dataset': dataset.get_name()})

        return self.processing(data)

    def __getitem__(self, index):
        if not self.fail_safe:
            return self.load_image(index)
        else:
            for i in range(100):
                try:
                    return self.load_image(index)
                except:
                    print('failed to load')

            raise Exception


class IndexedBurst(torch.utils.data.Dataset):
    """ Sequentially load bursts from a dataset"""
    def __init__(self, dataset, burst_size, processing=no_processing,
                 random_reference_image=False):
        """
        args:
            dataset - dataset to use
            burst_size - number of images sampled for each burst
            processing - the processing function to be applied to the loaded burst
            random_reference_image - Boolean indicating whether the reference (first) image in the burst should be
                                     randomly sampled
        """
        self.dataset = dataset

        # Normalize
        self.burst_size = burst_size

        self.processing = processing
        self.random_reference_image = random_reference_image

    def __len__(self):
        return len(self.dataset)

    def _sample_images(self, burst_info):
        if self.random_reference_image:
            burst_list = list(range(burst_info['burst_size']))

            if len(burst_list) < self.burst_size:
                burst_list = burst_list * ((self.burst_size // len(burst_list)) + 1)

            ids = random.sample(burst_list, k=self.burst_size)
        else:
            burst_list = list(range(1, burst_info['burst_size']))

            if len(burst_list) < (self.burst_size - 1):
                burst_list = burst_list * ((self.burst_size // len(burst_list)) + 1)

            ids = random.sample(burst_list, k=self.burst_size - 1)
            ids = [0, ] + ids
        return ids

    def __getitem__(self, index):
        dataset = self.dataset
        burst_info = dataset.get_burst_info(index)

        # Ids of the images in the burst to be sampled
        im_ids = self._sample_images(burst_info)

        frames, gt, meta_info = dataset.get_burst(index, im_ids, burst_info)

        data = TensorDict({'frames': frames,
                           'gt': gt,
                           'dataset': dataset.get_name(),
                           'burst_name': meta_info['burst_name'],
                           'sig_shot' : meta_info.get('sig_shot', None),
                           'sig_read': meta_info.get('sig_read', None),
                           'meta_info': meta_info})

        return self.processing(data)


class RandomBurst(torch.utils.data.Dataset):
    """ Randomly sample bursts from a list of datasets """
    def __init__(self, datasets: list, p_datasets: list, burst_size, samples_per_epoch, processing=no_processing,
                 random_reference_image=False):
        """
        args:
            datasets - list of datasets
            p_datasets - list containing the probabilities by which each dataset will be sampled
            burst_size - number of images sampled for each burst
            samples_per_epoch - number of sampled loaded per epoch
            processing - the processing function to be applied on the loaded burst
            random_reference_image - Boolean indicating whether the reference (first) image in the burst should be
                                     randomly sampled
        """
        self.datasets = datasets

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.burst_size = burst_size
        self.samples_per_epoch = samples_per_epoch

        self.processing = processing
        self.random_reference_image = random_reference_image

    def __len__(self):
        return self.samples_per_epoch

    def _sample_images(self, burst_info):
        if self.random_reference_image:
            burst_list = list(range(burst_info['burst_size']))

            if len(burst_list) < self.burst_size:
                burst_list = burst_list * ((self.burst_size // len(burst_list)) + 1)

            ids = random.sample(burst_list, k=self.burst_size)
        else:
            burst_list = list(range(1, burst_info['burst_size']))

            if len(burst_list) < (self.burst_size - 1):
                burst_list = burst_list * ((self.burst_size // len(burst_list)) + 1)

            ids = random.sample(burst_list, k=self.burst_size - 1)
            ids = [0, ] + ids
        return ids

    def __getitem__(self, index):
        # Sample dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        # Sample burst id
        burst_id = random.randint(0, dataset.get_num_bursts()-1)

        burst_info = dataset.get_burst_info(burst_id)

        # Sample ids of images from the selected burst to load
        im_ids = self._sample_images(burst_info)

        # Load selected burst images
        frames, gt, meta_info = dataset.get_burst(burst_id, im_ids, burst_info)

        data = TensorDict({'frames': frames,
                           'gt': gt,
                           'dataset': dataset.get_name(),
                           'burst_name': meta_info['burst_name'],
                           'meta_info': meta_info})

        return self.processing(data)
