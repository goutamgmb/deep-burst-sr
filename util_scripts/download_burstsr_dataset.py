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
import shutil
import argparse


def download_burstsr_dataset(download_path):
    out_dir = download_path + '/burstsr_dataset'

    os.makedirs(out_dir, exist_ok=True)

    # Download train folders
    for i in range(9):
        if not os.path.isfile('{}/train_{:02d}.zip'.format(out_dir, i)):
            print('Downloading train_{:02d}'.format(i))

            urllib.request.urlretrieve('https://data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/train_{:02d}.zip'.format(i),
                                       '{}/tmp.zip'.format(out_dir))

            os.rename('{}/tmp.zip'.format(out_dir), '{}/train_{:02d}.zip'.format(out_dir, i))

    # Download val folder
    if not os.path.isfile('{}/val.zip'.format(out_dir)):
        print('Downloading val')

        urllib.request.urlretrieve('https://data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/val.zip',
                                   '{}/tmp.zip'.format(out_dir))

        os.rename('{}/tmp.zip'.format(out_dir), '{}/val.zip'.format(out_dir))

    # Unpack train set
    for i in range(9):
        print('Unpacking train_{:02d}'.format(i))
        with zipfile.ZipFile('{}/train_{:02d}.zip'.format(out_dir, i), 'r') as zip_ref:
            zip_ref.extractall('{}'.format(out_dir))

    # Move files to a common directory
    os.makedirs('{}/train'.format(out_dir), exist_ok=True)

    for i in range(9):
        file_list = os.listdir('{}/train_{:02d}'.format(out_dir, i))

        for b in file_list:
            source_dir = '{}/train_{:02d}/{}'.format(out_dir, i, b)
            dst_dir = '{}/train/{}'.format(out_dir, b)

            if os.path.isdir(source_dir):
                shutil.move(source_dir, dst_dir)

    # Delete individual subsets
    for i in range(9):
        shutil.rmtree('{}/train_{:02d}'.format(out_dir, i))

    # Unpack val set
    print('Unpacking val')
    with zipfile.ZipFile('{}/val.zip'.format(out_dir), 'r') as zip_ref:
        zip_ref.extractall('{}'.format(out_dir))


def main():
    parser = argparse.ArgumentParser(description='Downloads and unpacks BurstSR dataset')
    parser.add_argument('path', type=str, help='Path where the dataset will be downloaded')

    args = parser.parse_args()

    download_burstsr_dataset(args.path)


if __name__ == '__main__':
    main()

