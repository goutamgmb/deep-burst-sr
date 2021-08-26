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
import models.dbsr.encoders as dbsr_encoders
import models.dbsr.decoders as dbsr_decoders
import models.dbsr.merging as dbsr_merging
from admin.model_constructor import model_constructor
from models.alignment.pwcnet import PWCNet
from admin.environment import env_settings


class DBSRNet(nn.Module):
    """ Deep Burst Super-Resolution model"""
    def __init__(self, encoder, merging, decoder):
        super().__init__()

        self.encoder = encoder      # Encodes input images and performs alignment
        self.merging = merging      # Merges the input embeddings to obtain a single feature map
        self.decoder = decoder      # Decodes the merged embeddings to generate HR RGB image

    def forward(self, im):
        out_enc = self.encoder(im)
        out_merge = self.merging(out_enc)
        out_dec = self.decoder(out_merge)

        return out_dec['pred'], {'offsets': out_enc['offsets'], 'fusion_weights': out_merge['fusion_weights']}


@model_constructor
def dbsrnet_cvpr2021(enc_init_dim, enc_num_res_blocks, enc_out_dim,
                     dec_init_conv_dim, dec_num_pre_res_blocks, dec_post_conv_dim, dec_num_post_res_blocks,
                     upsample_factor=2, activation='relu', train_alignmentnet=False,
                     offset_feat_dim=64,
                     weight_pred_proj_dim=32,
                     num_offset_feat_extractor_res=1,
                     num_weight_predictor_res=1,
                     offset_modulo=1.0,
                     use_offset=True,
                     ref_offset_noise=0.0,
                     softmax=True,
                     use_base_frame=True,
                     icnrinit=False,
                     gauss_blur_sd=None,
                     gauss_ksz=3,
                     ):
    # backbone
    alignment_net = PWCNet(load_pretrained=True,
                           weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))

    encoder = dbsr_encoders.ResEncoderWarpAlignnet(enc_init_dim, enc_num_res_blocks, enc_out_dim,
                                                   alignment_net,
                                                   activation=activation,
                                                   train_alignmentnet=train_alignmentnet)

    merging = dbsr_merging.WeightedSum(enc_out_dim, weight_pred_proj_dim, offset_feat_dim,
                                       num_offset_feat_extractor_res=num_offset_feat_extractor_res,
                                       num_weight_predictor_res=num_weight_predictor_res,
                                       offset_modulo=offset_modulo,
                                       use_offset=use_offset,
                                       ref_offset_noise=ref_offset_noise,
                                       softmax=softmax, use_base_frame=use_base_frame)

    decoder = dbsr_decoders.ResPixShuffleConv(enc_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
                                              dec_post_conv_dim, dec_num_post_res_blocks,
                                              upsample_factor=upsample_factor, activation=activation,
                                              gauss_blur_sd=gauss_blur_sd, icnrinit=icnrinit,
                                              gauss_ksz=gauss_ksz)

    net = DBSRNet(encoder=encoder, merging=merging, decoder=decoder)
    return net
