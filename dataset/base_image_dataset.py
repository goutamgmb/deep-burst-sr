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

import torch.utils.data
from data.image_loader import jpeg4py_loader


class BaseImageDataset(torch.utils.data.Dataset):
    """ Base class for datasets """

    def __init__(self, name, root, image_loader=jpeg4py_loader):
        """
        args:
            root - The root path to the dataset
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
        """
        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.image_list = []     # Contains the list of sequences.

    def __len__(self):
        """ Returns size of the dataset
        returns:
            int - number of samples in the dataset
        """
        return self.get_num_images()

    def __getitem__(self, index):
        """ Not to be used! Check get_frames() instead.
        """
        return None

    def get_name(self):
        """ Name of the dataset

        returns:
            string - Name of the dataset
        """
        return self.name

    def get_num_images(self):
        """ Number of sequences in a dataset

        returns:
            int - number of sequences in the dataset."""
        return len(self.image_list)

    def get_image_info(self, im_id):
        """ Returns information about a particular image,

        args:
            seq_id - index of the image

        returns:
            Dict
            """
        raise NotImplementedError

    def get_image(self, im_id, info=None):
        """ Get a image

        args:
            image_id      - index of image
            anno(None)  - The annotation for the sequence (see get_sequence_info). If None, they will be loaded.

        returns:
            image -
            anno -
            dict - A dict containing meta information about the sequence, e.g. class of the target object.

        """
        raise NotImplementedError

