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

import torch.nn as nn
import torch.nn.functional as F
import models.layers.blocks as blocks
from models.layers.initializations import ICNR
from models.layers.filtering import gauss_2d


class PixShuffleUpsampler(nn.Module):
    """ Upsampling using sub-pixel convolution """
    @staticmethod
    def _get_gaussian_kernel(ksz, sd):
        assert ksz % 2 == 1
        K = gauss_2d(ksz, sd, (0.0, 0.0), density=True)
        K = K / K.sum()
        return K

    def __init__(self, input_dim, output_dim, upsample_factor=2, use_bn=False, activation='relu',
                 icnrinit=False, gauss_blur_sd=None, gauss_ksz=3):
        super().__init__()
        pre_shuffle_dim = output_dim * upsample_factor ** 2
        self.conv_layer = blocks.conv_block(input_dim, pre_shuffle_dim, 1, stride=1, padding=0, batch_norm=use_bn,
                                            activation=activation, bias=not icnrinit)

        if icnrinit:
            # If enabled, use ICNR initialization proposed in "Checkerboard artifact free sub-pixel convolution"
            # (https://arxiv.org/pdf/1707.02937.pdf) to reduce checkerboard artifacts
            kernel = ICNR(self.conv_layer[0].weight, upsample_factor)
            self.conv_layer[0].weight.data.copy_(kernel)

        if gauss_blur_sd is not None:
            gauss_kernel = self._get_gaussian_kernel(gauss_ksz, gauss_blur_sd).unsqueeze(0)
            self.gauss_kernel = gauss_kernel
        else:
            self.gauss_kernel = None
        self.pix_shuffle = nn.PixelShuffle(upsample_factor)

    def forward(self, x):
        assert x.dim() == 4
        # Increase channel dimension
        out = self.conv_layer(x)

        # Rearrange the feature map to increase spatial size while reducing channel dimension
        out = self.pix_shuffle(out)

        if getattr(self, 'gauss_kernel', None) is not None:
            # If enabled, smooth the output feature map using gaussian kernel to reduce checkerboard artifacts
            shape = out.shape
            out = out.view(-1, 1, *shape[-2:])
            gauss_ksz = getattr(self, 'gauss_ksz', 3)
            out = F.conv2d(out, self.gauss_kernel.to(out.device), padding=(gauss_ksz - 1) // 2)
            out = out.view(shape)
        return out

