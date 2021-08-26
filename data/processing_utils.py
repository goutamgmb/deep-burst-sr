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
import torch.nn.functional as F


def random_resized_crop(frames, crop_sz, scale_range=None, ar_range=None):
    """
    :param frames: Input frame as tensor
    :param crop_sz: Output crop sz as (rows, cols)
    :param scale_range: A crop of size scale_factor*crop_sz is first extracted and resized. The scale_range
                        controls the value of scale_factor
    :param ar_range: If none, then a crop of size (rows*scale_factor, cols*scale_factor*ar_factor) will be first
                     extracted.
    :return:
    """
    if not isinstance(crop_sz, (tuple, list)):
        crop_sz = (crop_sz, crop_sz)
    crop_sz = torch.tensor(crop_sz).float()

    shape = frames.shape

    if ar_range is None:
        ar_factor = 1.0
    else:
        ar_factor = random.uniform(ar_range[0], ar_range[1])

    # Select scale_factor. Ensure the crop fits inside the image
    max_scale_factor = torch.tensor(shape[-2:]).float() / (crop_sz * torch.tensor([1.0, ar_factor]))
    max_scale_factor = max_scale_factor.min().item()

    if max_scale_factor < 1.0:
        scale_factor = max_scale_factor
    elif scale_range is not None:
        scale_factor = random.uniform(scale_range[0], min(scale_range[1], max_scale_factor))
    else:
        scale_factor = 1.0

    # Extract the crop
    orig_crop_sz = (crop_sz * torch.tensor([1.0, ar_factor]) * scale_factor).floor()

    assert orig_crop_sz[-2] <= shape[-2] and orig_crop_sz[-1] <= shape[-1], 'Bug in crop size estimation!'

    r1 = random.randint(0, shape[-2] - orig_crop_sz[-2])
    c1 = random.randint(0, shape[-1] - orig_crop_sz[-1])

    r2 = r1 + orig_crop_sz[0].int().item()
    c2 = c1 + orig_crop_sz[1].int().item()

    frames_crop = frames[:, r1:r2, c1:c2]

    # Resize to crop_sz
    frames_crop_resized = F.interpolate(frames_crop.unsqueeze(0), size=crop_sz.int().tolist(), mode='bilinear').squeeze(0)
    return frames_crop_resized


def center_crop(frames, crop_sz):
    """
    :param frames: Input frame as tensor
    :param crop_sz: Output crop sz as (rows, cols)

    :return:
    """
    if not isinstance(crop_sz, (tuple, list)):
        crop_sz = (crop_sz, crop_sz)
    crop_sz = torch.tensor(crop_sz).float()

    shape = frames.shape

    r1 = ((shape[-2] - crop_sz[-2]) // 2).int()
    c1 = ((shape[-1] - crop_sz[-1]) // 2).int()

    r2 = r1 + crop_sz[-2].int().item()
    c2 = c1 + crop_sz[-1].int().item()

    frames_crop = frames[:, r1:r2, c1:c2]

    return frames_crop
