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

import numpy as np
import torch


def get_bayer_pattern(color_desc, raw_pattern):
    """ Get bayer pattern from the color_desc field of RAWPy image"""
    colors = color_desc.decode("utf-8")
    pattern = raw_pattern.reshape(-1).tolist()
    bayer_pattern = [colors[i] for i in pattern]
    bayer_pattern = ''.join(bayer_pattern)
    return bayer_pattern


def get_color_map(im):
    colors = im.color_desc.decode("utf-8")

    color_map = {'R': colors.find('R'), 'B': colors.find('B'),
                 'G': colors.find('G')}

    return color_map


def convert_to_rggb(im_raw, four_channel_output=True):
    im = im_raw.raw_image_visible
    bayer_pattern = get_bayer_pattern(im_raw.color_desc, im_raw.raw_pattern)

    if bayer_pattern == 'BGGR':
        # BGGR to RGGB
        im = im[1:-1, 1:-1]
    elif bayer_pattern == 'RGGB':
        pass
    elif bayer_pattern == 'GRBG':
        im = im[:, 1:-1]
    else:
        raise Exception

    if four_channel_output:
        # return as 3d tensor
        im_out = np.zeros_like(im, shape=(4, im.shape[0] // 2, im.shape[1] // 2))
        im_out[0, :, :] = im[0::2, 0::2]
        im_out[1, :, :] = im[0::2, 1::2]
        im_out[2, :, :] = im[1::2, 0::2]
        im_out[3, :, :] = im[1::2, 1::2]
    else:
        im_out = im

    return im_out


def pack_raw_image(im_raw):
    """ Packs a single channel bayer image into 4 channel tensor, where channels contain R, G, G, and B values"""
    if isinstance(im_raw, np.ndarray):
        im_out = np.zeros_like(im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2))
    elif isinstance(im_raw, torch.Tensor):
        im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype)
    else:
        raise Exception

    im_out[0, :, :] = im_raw[0::2, 0::2]
    im_out[1, :, :] = im_raw[0::2, 1::2]
    im_out[2, :, :] = im_raw[1::2, 0::2]
    im_out[3, :, :] = im_raw[1::2, 1::2]
    return im_out


def flatten_raw_image(im_raw_4ch):
    """ unpack a 4-channel tensor into a single channel bayer image"""
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2))
    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros((im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype)
    else:
        raise Exception

    im_out[0::2, 0::2] = im_raw_4ch[0, :, :]
    im_out[0::2, 1::2] = im_raw_4ch[1, :, :]
    im_out[1::2, 0::2] = im_raw_4ch[2, :, :]
    im_out[1::2, 1::2] = im_raw_4ch[3, :, :]

    return im_out
