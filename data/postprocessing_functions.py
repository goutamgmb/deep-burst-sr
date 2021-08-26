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

import utils.data_format_utils as df_utils
from data.camera_pipeline import apply_gains, apply_ccm, apply_smoothstep, gamma_compression


class SimplePostProcess:
    def __init__(self, gains=True, ccm=True, gamma=True, smoothstep=True, return_np=False):
        self.gains = gains
        self.ccm = ccm
        self.gamma = gamma
        self.smoothstep = smoothstep
        self.return_np = return_np

    def process(self, image, meta_info):
        return process_linear_image_rgb(image, meta_info, self.gains, self.ccm, self.gamma,
                                        self.smoothstep, self.return_np)


def process_linear_image_rgb(image, meta_info, gains=True, ccm=True, gamma=True, smoothstep=True, return_np=False):
    if gains:
        image = apply_gains(image, meta_info['rgb_gain'], meta_info['red_gain'], meta_info['blue_gain'])

    if ccm:
        image = apply_ccm(image, meta_info['cam2rgb'])

    image = image.clamp(0.0, 1.0)
    if meta_info['gamma'] and gamma:
        image = gamma_compression(image)

    if meta_info['smoothstep'] and smoothstep:
        image = apply_smoothstep(image)

    image = image.clamp(0.0, 1.0)

    if return_np:
        image = df_utils.torch_to_npimage(image)
    return image


class Identity:
    def __init__(self, return_np=False, clamp=True):
        self.return_np = return_np
        self.clamp = clamp

    def process(self, image, meta_info):
        if self.clamp:
            image = image.clamp(0.0, 1.0)

        if self.return_np:
            image = df_utils.torch_to_npimage(image)
        return image
