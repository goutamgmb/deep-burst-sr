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
from torch import nn


def get_activation(activation, activation_params=None, num_channels=None):
    if activation_params is None:
        activation_params = {}

    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lrelu':
        return nn.LeakyReLU(negative_slope=activation_params.get('negative_slope', 0.1), inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    elif activation == 'none':
        return None
    else:
        raise Exception('Unknown activation {}'.format(activation))


def get_attention(attention_type, num_channels=None):
    if attention_type == 'none':
        return None
    else:
        raise Exception('Unknown attention {}'.format(attention_type))


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, activation='relu', padding_mode='zeros', activation_params=None):
    layers = []

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))

    activation_layer = get_activation(activation, activation_params, num_channels=out_planes)
    if activation_layer is not None:
        layers.append(activation_layer)

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm=False, activation='relu',
                 padding_mode='zeros', attention='none'):
        super(ResBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, dilation=dilation,
                                batch_norm=batch_norm, activation=activation, padding_mode=padding_mode)

        self.conv2 = conv_block(planes, planes, kernel_size=3, padding=1, dilation=dilation, batch_norm=batch_norm,
                                activation='none', padding_mode=padding_mode)

        self.downsample = downsample
        self.stride = stride

        self.activation = get_activation(activation, num_channels=planes)
        self.attention = get_attention(attention_type=attention, num_channels=planes)

    def forward(self, x):
        residual = x

        out = self.conv2(self.conv1(x))

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out += residual

        out = self.activation(out)

        return out

