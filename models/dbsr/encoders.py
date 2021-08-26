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
import torch.nn as nn
import models.layers.warp as lispr_warp
import models.layers.blocks as blocks


class ResEncoderWarpAlignnet(nn.Module):
    """ Encodes the input images using a residual network. Uses the alignment_net to estimate optical flow between
        reference (first) image and other images. Warps the embeddings of other images to reference frame co-ordinates
        using the estimated optical flow
    """
    def __init__(self, init_dim, num_res_blocks,
                 out_dim, alignment_net, use_bn=False, activation='relu', train_alignmentnet=True,
                 warp_type='bilinear'):
        super().__init__()
        input_channels = 4
        self.warp_type = warp_type

        self.alignment_net = alignment_net
        self.train_alignmentnet = train_alignmentnet

        self.init_layer = blocks.conv_block(input_channels, init_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                            activation=activation)

        res_layers = []
        for _ in range(num_res_blocks):
            res_layers.append(blocks.ResBlock(init_dim, init_dim, stride=1, batch_norm=use_bn, activation=activation))

        self.res_layers = nn.Sequential(*res_layers)

        self.out_layer = blocks.conv_block(init_dim, out_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                           activation=activation)

    def forward(self, x):
        assert x.dim() == 5

        # Compute alignment vectors wrt reference frame
        x_rgb = torch.stack((x[:, :, 0], x[:, :, 1:3].mean(dim=2), x[:, :, 3]), dim=2)
        x_ref = x_rgb[:, :1, ...].repeat(1, x_rgb.shape[1] - 1, 1, 1, 1).contiguous()
        x_oth = x_rgb[:, 1:, ...].contiguous()

        if self.train_alignmentnet:
            offsets = self.alignment_net(x_oth.view(-1, *x_oth.shape[-3:]), x_ref.view(-1, *x_ref.shape[-3:]))
        else:
            with torch.no_grad():
                self.alignment_net = self.alignment_net.eval()
                offsets = self.alignment_net(x_oth.view(-1, *x_oth.shape[-3:]), x_ref.view(-1, *x_ref.shape[-3:]))

        shape = x.shape

        # Extract image embeddings
        x = x.view(-1, *x.shape[-3:])
        out = self.init_layer(x)

        feat = self.res_layers(out)

        feat = self.out_layer(feat)
        feat = feat.view(shape[0], shape[1], *feat.shape[-3:])

        ref_feat = feat[:, :1, ...].contiguous()
        oth_feat = feat[:, 1:, ...].contiguous()

        oth_feat = oth_feat.view(-1, *oth_feat.shape[-3:])

        # Warp embeddings to reference frame co-ordinates
        oth_feat = lispr_warp.warp(oth_feat, offsets, mode=getattr(self, 'warp_type', 'bilinear'))

        oth_feat = oth_feat.view(shape[0], shape[1] - 1, *oth_feat.shape[-3:])
        ref_feat = ref_feat.expand(-1, shape[1] - 1, -1, -1, -1)

        offsets = offsets.view(shape[0], shape[1] - 1, 2, shape[-2], shape[-1])
        return {'ref_feat': ref_feat, 'oth_feat': oth_feat, 'offsets': offsets}
