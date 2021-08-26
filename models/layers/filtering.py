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
import math
import torch.nn.functional as F


def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    """ Returns a 1-D Gaussian """
    k = torch.arange(-(sz-1)/2, (sz+1)/2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0/(2*sigma**2) * (k - center.reshape(-1, 1))**2)
    if density:
        gauss /= math.sqrt(2*math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    """ Returns a 2-D Gaussian """
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    if isinstance(sz, int):
        sz = (sz, sz)

    if isinstance(center, (list, tuple)):
        center = torch.tensor(center).view(1, 2)

    return gauss_1d(sz[0], sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1], sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)


def get_gaussian_kernel(sd, ksz=None):
    """ Returns a 2D Gaussian kernel with standard deviation sd """
    if ksz is None:
        ksz = int(4 * sd + 1)

    assert ksz % 2 == 1
    K = gauss_2d(ksz, sd, (0.0, 0.0), density=True)
    K = K / K.sum()
    return K.unsqueeze(0), ksz


def apply_kernel(im, ksz, kernel):
    """ apply the provided kernel on input image """
    shape = im.shape
    im = im.view(-1, 1, *im.shape[-2:])

    pad = [ksz // 2, ksz // 2, ksz // 2, ksz // 2]
    im = F.pad(im, pad, mode='reflect')
    im_out = F.conv2d(im, kernel).view(shape)
    return im_out