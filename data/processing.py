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
import data.transforms as transforms
import data.processing_utils as prutils
import data.raw_image_processing as raw_processing

import torch.nn.functional as F
import data.synthetic_burst_generation as syn_burst_generation
from admin.tensordict import TensorDict
import math


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to apply various data augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor()):
        self.transform = transform

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class SyntheticBurstProcessing(BaseProcessing):
    """ The processing class used for training on synthetic bursts. The class generates a synthetic RAW burst using
    a RGB image. This is achieved by i) extracting a crop from the input image, ii) using an inverse camera pipeline to
    convert the RGB crop to linear sensor space, ii) Applying random affine transformations to obtain a burst from the
    single crop, and iii) downsampling the generated burst, applying bayer mosaicking pattern, and adding synthetic
    noise. """
    def __init__(self, crop_sz, burst_size, downsample_factor, crop_scale_range=None, crop_ar_range=None,
                 burst_transformation_params=None, image_processing_params=None,
                 interpolation_type='bilinear', return_rgb_busrt=False, random_crop=True,
                 *args, **kwargs):
        """
        args:
            crop_sz - The size of the image region first cropped from the input image
            burst_size - Number of images in the generated burst.
            downsample_factor - The factor by which the images are downsampled when generating lower-resolution burst
            crop_scale_range - The range (min, max) of random resizing performed when extracting the initial image crop.
                               If None, no resizing is performed.
            crop_ar_range - The range (min, max) of random aspect ratio change performed when extracting the initial
                            image crop. If None, the original aspect ratio is preserved.
            burst_transformation_params - A dict containing the parameters for the affine transformations applied
                                          when generating a burst from a single image.
            image_processing_params - A dict containing the parameters for the inverse camera pipeline used to obtain
                                      linear sensor space image from RGB image.
            interpolation_type - Type of interpolation used when applying the affine transformation and downsampling the
                                 image.
            return_rgb_busrt - Boolean indicating whether to return an RGB burst, in addition to the RAW burst.
            random_crop - Boolean indicating whether to perform random cropping (True) or center cropping (False)
        """
        super().__init__(*args, **kwargs)
        if not isinstance(crop_sz, (tuple, list)):
            crop_sz = (crop_sz, crop_sz)

        self.crop_sz = crop_sz

        self.burst_size = burst_size
        self.downsample_factor = downsample_factor

        self.burst_transformation_params = burst_transformation_params

        self.crop_scale_range = crop_scale_range
        self.crop_ar_range = crop_ar_range
        self.return_rgb_busrt = return_rgb_busrt
        self.interpolation_type = interpolation_type
        self.random_crop = random_crop

        self.image_processing_params = image_processing_params

    def __call__(self, data: TensorDict):
        # Augmentation, e.g. convert to tensor
        if self.transform is not None:
            data['frame'] = self.transform(image=data['frame'])

        # add extra padding to compensate for cropping of border region
        crop_sz = [c + 2 * self.burst_transformation_params.get('border_crop', 0) for c in self.crop_sz]
        if getattr(self, 'random_crop', True):
            # Perform random cropping
            frame_crop = prutils.random_resized_crop(data['frame'], crop_sz,
                                                     scale_range=self.crop_scale_range,
                                                     ar_range=self.crop_ar_range)
        else:
            # Perform center cropping
            assert self.crop_scale_range is None and self.crop_ar_range is None
            frame_crop = prutils.center_crop(data['frame'], crop_sz)

        # Generate synthetic RAW burst
        burst, frame_gt, burst_rgb, flow_vector, meta_info = syn_burst_generation.rgb2rawburst(frame_crop,
                                                                                               self.burst_size,
                                                                                               self.downsample_factor,
                                                                                               burst_transformation_params=self.burst_transformation_params,
                                                                                               image_processing_params=self.image_processing_params,
                                                                                               interpolation_type=self.interpolation_type
                                                                                               )

        # Crop border regions
        if self.burst_transformation_params.get('border_crop') is not None:
            border_crop = self.burst_transformation_params.get('border_crop')
            frame_gt = frame_gt[:, border_crop:-border_crop, border_crop:-border_crop]
        del data['frame']

        if self.return_rgb_busrt:
            data['burst_rgb'] = burst_rgb

        data['frame_gt'] = frame_gt
        data['burst'] = burst
        data['meta_info'] = meta_info
        return data


class BurstSRProcessing(BaseProcessing):
    """ The processing class used for training on BurstSR dataset. """
    def __init__(self, crop_sz=64, substract_black_level=False, white_balance=False,
                 random_flip=False, noise_level=None, random_crop=True, *args, **kwargs):
        """
        args:
            crop_sz - Size of the extracted crop
            substract_black_level - Boolean indicating whether to subtract the black level from the sensor data
            white_balance - Boolean indicating whether to apply white balancing provided by the camera
            random_flip - Boolean indicating whether to apply random flips to sensor data
            noise_level - The parameters of the synthetic noise added to sensor data. If None, no noise is added
            random_crop - Boolean indicating whether to perform random cropping (True) or center cropping (False)
        """
        super().__init__(*args, **kwargs)
        self.substract_black_level = substract_black_level
        self.white_balance = white_balance
        self.crop_sz = crop_sz
        self.noise_level = noise_level
        self.random_crop = random_crop
        self.random_flip = random_flip

    def get_random_noise_level(self):
        """Generates random noise levels from a log-log linear distribution."""
        log_min_shot_noise = math.log(self.noise_level[0])
        log_max_shot_noise = math.log(self.noise_level[1])  # 0.01
        log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
        shot_noise = math.exp(log_shot_noise)

        line = lambda x: 2.18 * x + 1.20
        log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
        read_noise = math.exp(log_read_noise)

        return shot_noise, read_noise

    @staticmethod
    def add_noise(image, shot_noise=0.01, read_noise=0.0005):
        """Adds random shot (proportional to image) and read (independent) noise."""
        variance = image * shot_noise + read_noise
        noise = torch.FloatTensor(image.shape).normal_().to(image.device) * variance.sqrt()
        return image + noise

    def __call__(self, data: TensorDict):
        # Augmentation, e.g. convert to tensor
        if self.transform is not None:
            raise NotImplementedError

        frames = data['frames']
        gt = data['gt']

        # Extract crop
        if frames[0].shape()[-1] != self.crop_sz:
            if not getattr(self, 'random_crop', True):
                r1 = (frames[0].shape()[-2] - self.crop_sz) // 2
                c1 = (frames[0].shape()[-1] - self.crop_sz) // 2
            else:
                r1 = random.randint(0, frames[0].shape()[-2] - self.crop_sz)
                c1 = random.randint(0, frames[0].shape()[-1] - self.crop_sz)
            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz

            scale_factor = gt.shape()[-1] // frames[0].shape()[-1]
            frames = [im.get_crop(r1, r2, c1, c2) for im in frames]

            gt = gt.get_crop(scale_factor * r1, scale_factor * r2, scale_factor * c1, scale_factor * c2)

        # Load the RAW image data
        burst_image_data = [im.get_image_data(normalize=True, substract_black_level=self.substract_black_level,
                                              white_balance=self.white_balance) for im in frames]

        # Convert to tensor
        gt_image_data = gt.get_image_data(normalize=True, white_balance=self.white_balance,
                                          substract_black_level=self.substract_black_level)

        # Perform flipping while preserving the RGGB bayer pattern
        if self.random_flip:
            burst_image_data = [raw_processing.flatten_raw_image(im) for im in burst_image_data]

            pad = [0, 0, 0, 0]
            if random.random() > 0.5:
                burst_image_data = [im.flip([1, ])[:, 1:-1].contiguous() for im in burst_image_data]
                gt_image_data = gt_image_data.flip([2, ])[:, :, 2:-2].contiguous()
                pad[1] = 1

            if random.random() > 0.5:
                burst_image_data = [im.flip([0, ])[1:-1, :].contiguous() for im in burst_image_data]
                gt_image_data = gt_image_data.flip([1, ])[:, 2:-2, :].contiguous()
                pad[3] = 1

            burst_image_data = [raw_processing.pack_raw_image(im) for im in burst_image_data]
            burst_image_data = [F.pad(im.unsqueeze(0), pad, mode='replicate').squeeze(0) for im in burst_image_data]
            gt_image_data = F.pad(gt_image_data.unsqueeze(0), [4*p for p in pad], mode='replicate').squeeze(0)

        burst_image_meta_info = frames[0].get_all_meta_data()

        burst_image_meta_info['black_level_subtracted'] = self.substract_black_level
        burst_image_meta_info['while_balance_applied'] = self.white_balance
        burst_image_meta_info['norm_factor'] = frames[0].norm_factor

        gt_image_meta_info = gt.get_all_meta_data()

        burst = torch.stack(burst_image_data, dim=0)

        # Add additional synthetic noise if specified
        if getattr(self, 'noise_level', None) is not None:
            shot_noise, read_noise = self.get_random_noise_level()
            burst = self.add_noise(burst, shot_noise, read_noise)
            burst = burst.clamp(0.0, 1.0)

        burst_exposure = frames[0].get_exposure_time()
        canon_exposure = gt.get_exposure_time()

        burst_f_number = frames[0].get_f_number()
        canon_f_number = gt.get_f_number()

        burst_iso = frames[0].get_iso()
        canon_iso = gt.get_iso()

        # Normalize the GT image to account for differences in exposure, ISO etc
        light_factor_burst = burst_exposure * burst_iso / (burst_f_number ** 2)
        light_factor_canon = canon_exposure * canon_iso / (canon_f_number ** 2)

        exp_scale_factor = (light_factor_burst / light_factor_canon)
        gt_image_data = gt_image_data * exp_scale_factor

        noise_profile = data['frames'][0].get_noise_profile()[0, :]
        noise_profile = torch.from_numpy(noise_profile).view(-1)

        gt_image_meta_info['black_level_subtracted'] = self.substract_black_level
        gt_image_meta_info['while_balance_applied'] = self.white_balance
        gt_image_meta_info['norm_factor'] = gt.norm_factor / exp_scale_factor

        burst_image_meta_info['exposure'] = burst_exposure
        burst_image_meta_info['f_number'] = burst_f_number
        burst_image_meta_info['iso'] = burst_iso

        gt_image_meta_info['exposure'] = canon_exposure
        gt_image_meta_info['f_number'] = canon_f_number
        gt_image_meta_info['iso'] = canon_iso

        burst_image_meta_info['noise_profile'] = noise_profile

        data['burst'] = burst.float()
        data['frame_gt'] = gt_image_data.float()

        # burst_image_meta_info['burst_name'] = data['burst_name']
        data['meta_info_burst'] = burst_image_meta_info
        data['meta_info_gt'] = gt_image_meta_info
        data['exp_scale_factor'] = exp_scale_factor

        del data['frames']
        del data['gt']

        return data