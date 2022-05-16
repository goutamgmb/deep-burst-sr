import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import cv2
import numpy as np
import data.raw_image_processing as raw_processing
import argparse
from dataset.synthetic_burst_val_set import SyntheticBurstVal
from data.postprocessing_functions import SimplePostProcess
from utils.opencv_plotting import BurstSRVis


def visualize_synburst_val(mode='srgb'):
    dataset = SyntheticBurstVal()
    process_fn = SimplePostProcess(return_np=True)

    vis = BurstSRVis()

    for idx in range(len(dataset)):
        burst, gt, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        burst = burst[[0, 7, 13]].contiguous()
        burst = burst.clamp(0.0, 1.0)

        titles = [mode + '_00', mode + '_07', mode + '_13']
        if mode == 'raw':
            burst = (burst / burst.max() * 255.0).numpy().astype(np.uint8)
            images = [raw_processing.flatten_raw_image(b) for b in burst]
            images = [np.expand_dims(im, -1) for im in images]
        elif mode == 'linear':
            burst = burst[:, [0, 1, 3]].contiguous()
            burst = (burst / burst.max() * 255.0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            images = [b for b in burst]
        elif mode == 'srgb':
            burst = burst[:, [0, 1, 3]].contiguous()
            images = [process_fn.process(b, meta_info) for b in burst]
        else:
            raise Exception

        gt = process_fn.process(gt, meta_info)
        images = [cv2.resize(im, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) for im in images]

        images = images + [gt, ]
        titles = titles + ['GT']

        data = [{'images': images,
                 'titles': titles}]
        cmd = vis.plot(data)

        if cmd == 'stop':
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize RAW bursts')
    parser.add_argument('mode', type=str, help='Visualization mode, can be raw, linear, or srgb')

    args = parser.parse_args()

    print('The script visualizes the 1st, 8th, and 14th images from the burst. \n'
          'Press \'n\' to show next image. \n'
          'Press \'q\' to quit. \n'
          'Zoom on a particular region by drawing a box around it (click on the two corner points). \n'
          'In the zoomed pane (last row), you can click on an image an drag around. \n'
          'Using \'w\' and \'s\' keys, you can navigate between the two panes (normal pane and zoom pane) \n'
          'Using the \'space\' key, you can toggle between showing all the images and showing only a single image. \n'
          'In the single image mode, you can navigate between different images using the \'a\' and \'d\' keys. \n')
    visualize_synburst_val(args.mode)

