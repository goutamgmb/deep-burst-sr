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
#
# The file contains code from the unprocessing repo (http://timothybrooks.com/tech/unprocessing )

import torch
import random
import math

""" Based on http://timothybrooks.com/tech/unprocessing 
Functions for forward and inverse camera pipeline. All functions input a torch float tensor of shape (c, h, w). 
Additionally, some also support batch operations, i.e. inputs of shape (b, c, h, w)
"""


def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
               [-0.5625, 1.6328, -0.0469],
               [-0.0703, 0.2188, 0.6406]],
              [[0.4913, -0.0541, -0.0202],
               [-0.613, 1.3513, 0.2906],
               [-0.1564, 0.2151, 0.7183]],
              [[0.838, -0.263, -0.0639],
               [-0.2887, 1.0725, 0.2496],
               [-0.0627, 0.1427, 0.5438]],
              [[0.6596, -0.2079, -0.0562],
               [-0.4782, 1.3016, 0.1933],
               [-0.097, 0.1581, 0.5181]]]

    num_ccms = len(xyz2cams)
    xyz2cams = torch.tensor(xyz2cams)

    weights = torch.FloatTensor(num_ccms, 1, 1).uniform_(0.0, 1.0)
    weights_sum = weights.sum()
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdims=True)
    return rgb2cam


def random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / random.gauss(mu=0.8, sigma=0.1)

    # Red and blue gains represent white balance.
    red_gain = random.uniform(1.9, 2.4)
    blue_gain = random.uniform(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain


def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    image_out = 3 * image**2 - 2 * image**3
    return image_out


def invert_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = image.clamp(0.0, 1.0)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return image.clamp(1e-8) ** 2.2


def gamma_compression(image):
    """Converts from linear to gammaspace."""
    # Clamps to prevent numerical instability of gradients near zero.
    return image.clamp(1e-8) ** (1.0 / 2.2)


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    assert image.dim() == 3 and image.shape[0] == 3

    shape = image.shape
    image = image.view(3, -1)
    ccm = ccm.to(image.device).type_as(image)

    image = torch.mm(ccm, image)

    return image.view(shape)


def apply_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3 and image.shape[0] in [3, 4]

    if image.shape[0] == 3:
        gains = torch.tensor([red_gain, 1.0, blue_gain]) * rgb_gain
    else:
        gains = torch.tensor([red_gain, 1.0, 1.0, blue_gain]) * rgb_gain
    gains = gains.view(-1, 1, 1)
    gains = gains.to(image.device).type_as(image)

    return (image * gains).clamp(0.0, 1.0)


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3 and image.shape[0] == 3

    gains = torch.tensor([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
    gains = gains.view(-1, 1, 1)

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = image.mean(dim=0, keepdims=True)
    inflection = 0.9
    mask = ((gray - inflection).clamp(0.0) / (1.0 - inflection)) ** 2.0

    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains


def mosaic(image, mode='rggb'):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if mode == 'rggb':
        red = image[:, 0, 0::2, 0::2]
        green_red = image[:, 1, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 0::2]
        blue = image[:, 2, 1::2, 1::2]
        image = torch.stack((red, green_red, green_blue, blue), dim=1)
    elif mode == 'grbg':
        green_red = image[:, 1, 0::2, 0::2]
        red = image[:, 0, 0::2, 1::2]
        blue = image[:, 2, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 1::2]

        image = torch.stack((green_red, red, blue, green_blue), dim=1)

    if len(shape) == 3:
        return image.view((4, shape[-2] // 2, shape[-1] // 2))
    else:
        return image.view((-1, 4, shape[-2] // 2, shape[-1] // 2))


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(0.0001)
    log_max_shot_noise = math.log(0.012)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = torch.FloatTensor(image.shape).normal_().to(image.device)*variance.sqrt()
    return image + noise
