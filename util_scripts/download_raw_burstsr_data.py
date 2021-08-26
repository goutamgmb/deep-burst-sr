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

import os
import urllib.request
import zipfile
import argparse
from dataset.burstsr_dataset import load_txt


def download_raw_burstsr_data(download_path):
    out_dir = download_path + '/burstsr_full_images'

    # Download train folders
    lispr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

    train_burst_list = load_txt('{}/data_specs/burstsr_{}.txt'.format(lispr_path, 'train'))
    train_out_dir = '{}/train/'.format(out_dir)
    os.makedirs(train_out_dir, exist_ok=True)

    train_burst_list = train_burst_list[:2]
    for burst_id in train_burst_list:
        if not os.path.isfile('{}/{}.zip'.format(train_out_dir, burst_id)):
            print('Downloading {}'.format(burst_id))

            urllib.request.urlretrieve('https://data.vision.ee.ethz.ch/bhatg/burstsr_full_release/train/{}.zip'.format(burst_id),
                                       '{}/tmp.zip'.format(train_out_dir))

            os.rename('{}/tmp.zip'.format(train_out_dir), '{}/{}.zip'.format(train_out_dir, burst_id))

    # Download val folder
    val_burst_list = load_txt('{}/data_specs/burstsr_{}.txt'.format(lispr_path, 'val'))
    val_out_dir = '{}/val/'.format(out_dir)
    os.makedirs(val_out_dir, exist_ok=True)

    val_burst_list = val_burst_list[:1]
    for burst_id in val_burst_list:
        if not os.path.isfile('{}/{}.zip'.format(val_out_dir, burst_id)):
            print('Downloading {}'.format(burst_id))

            urllib.request.urlretrieve('https://data.vision.ee.ethz.ch/bhatg/burstsr_full_release/val/{}.zip'.format(burst_id),
                                       '{}/tmp.zip'.format(val_out_dir))

            os.rename('{}/tmp.zip'.format(val_out_dir), '{}/{}.zip'.format(val_out_dir, burst_id))

    # Unpack train set
    for burst_id in train_burst_list:
        print('Unpacking {}'.format(burst_id))
        with zipfile.ZipFile('{}/{}.zip'.format(train_out_dir, burst_id), 'r') as zip_ref:
            zip_ref.extractall('{}/{}'.format(train_out_dir, burst_id))

    # Unpack val set
    print('Unpacking val')
    for burst_id in val_burst_list:
        print('Unpacking {}'.format(burst_id))
        with zipfile.ZipFile('{}/{}.zip'.format(val_out_dir, burst_id), 'r') as zip_ref:
            zip_ref.extractall('{}/{}'.format(val_out_dir, burst_id))

        os.remove('{}/{}.zip'.format(val_out_dir, burst_id))

    # Delete training zips
    for burst_id in train_burst_list:
        os.remove('{}/{}.zip'.format(train_out_dir, burst_id))

    # Delete val zips
    for burst_id in val_burst_list:
        os.remove('{}/{}.zip'.format(val_out_dir, burst_id))


def main():
    parser = argparse.ArgumentParser(description='Downloads and unpacks the full-sized images used for BurstSR dataset')
    parser.add_argument('path', type=str, help='Path where the dataset will be downloaded')

    args = parser.parse_args()

    download_raw_burstsr_data(args.path)


if __name__ == '__main__':
    main()

