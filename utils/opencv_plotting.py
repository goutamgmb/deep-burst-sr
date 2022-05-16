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

import cv2 as cv
import numpy as np
import utils.data_format_utils as df_utils
import os
import json


class BurstSRVis():
    def __init__(self, data_block_titles=('Prediction',), prediction_block_id=0, display_split=(0.5, 0.5), metric=None,
                 boundary_ignore=None, save_results_path=None):
        num_data_blocks = len(data_block_titles)
        assert len(display_split) == num_data_blocks + 1        # additional element for zoom mode

        self.boundary_ignore = boundary_ignore
        self.num_data_blocks = num_data_blocks
        self.num_panes = num_data_blocks + 1
        self.prediction_block_id = prediction_block_id

        self.pane_titles = data_block_titles + ('Zoom',)
        self.display_size = (960, 1600)
        self.window_size = (960, 1600 + 100)

        self.display_split = display_split
        self.display_split_base = display_split

        self.show_all_images = [True for _ in display_split]    # whether to show all images, or single image
        self.pane_mode = [1 for _ in display_split]

        self.pane_image_id = [0 for _ in display_split]

        self.data = None
        self.data_orig = None

        self.selected_pane_id = 1
        self.click_info = {'mode': 'init', 'pos0': (-1, -1), 'pos1': (-1, 1), 'pane': None, 'im_id': None}
        self.pane_coord_info = {}
        self.button_info = {}

        self.zoom_roi_coords = None
        self.zoom_pane_id = 0

        self.data_sizes = None
        self.padding_amount = 10
        self.boundary_pad = 20
        self.title_pad = 24
        self.selection_color = np.array([192, 240, 208])
        self.selection_pad = 10

        self.metric = metric

        self.save_results_path = save_results_path
    # ************************************************************************************************
    # ********************************** Mouse Callback functions ************************************
    # ************************************************************************************************

    def _toggle_pane_mode(self, pane_id):
        self.pane_mode[pane_id] = (self.pane_mode[pane_id] + 1) % 2

        self.display_split = [self.display_split_base[i] if self.pane_mode[i] != 0 else 0 for i in range(self.num_panes)]
        ss = sum(self.display_split) + 1e-8
        self.display_split = [split / ss for split in self.display_split]

        for _ in range(self.num_panes):
            if self.pane_mode[self.selected_pane_id] == 0:
                self.selected_pane_id = (self.selected_pane_id + 1) % (self.num_panes - 1)
            else:
                break

    def _get_selected_pane(self, x, y):
        pane_id = self.prediction_block_id
        if self.pane_mode[pane_id] != 0:
            rows_flag = self.pane_coord_info[pane_id]['r1'] < y < self.pane_coord_info[pane_id]['r2']
            cols_flag = self.pane_coord_info[pane_id]['c1'] < x < self.pane_coord_info[pane_id]['c2']
            if rows_flag and cols_flag:
                return pane_id

        pane_id = self.num_panes - 1
        if self.zoom_roi_coords is not None and self.pane_mode[pane_id] != 0:
            rows_flag = self.pane_coord_info[pane_id]['r1'] < y < self.pane_coord_info[pane_id]['r2']
            cols_flag = self.pane_coord_info[pane_id]['c1'] < x < self.pane_coord_info[pane_id]['c2']
            if rows_flag and cols_flag:
                return pane_id

        return None

    def _global_xy_to_im_xy(self, x, y, pane_id):
        im_y = y - self.pane_coord_info[pane_id]['r1']
        padded_im_w = self.pane_coord_info[pane_id]['im_w'] + self.pane_coord_info[pane_id]['padding']
        im_x = (x - self.pane_coord_info[pane_id]['c1']) % padded_im_w
        im_id = (x - self.pane_coord_info[pane_id]['c1']) // padded_im_w
        im_x = min(im_x, self.pane_coord_info[pane_id]['im_w'] - 1)

        im_y /= self.pane_coord_info[pane_id]['resize_factor']
        im_x /= self.pane_coord_info[pane_id]['resize_factor']
        return (int(im_x), int(im_y)), im_id

    def _get_pos1(self, x, y):
        shift = (x - self.click_info['pos0_global'][0], y - self.click_info['pos0_global'][1])
        zoom_pane_id = self.click_info['pane']

        resize_factor = self.pane_coord_info[zoom_pane_id]['resize_factor']

        pos1 = (min(self.pane_coord_info[zoom_pane_id]['im_w'] - 1, self.click_info['pos0'][0] * resize_factor + shift[0]),
                min(self.pane_coord_info[zoom_pane_id]['im_h'] - 1, self.click_info['pos0'][1] * resize_factor + shift[1]))

        pos1 = [int(p / resize_factor) for p in pos1]
        return pos1

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.click_info['mode'] == 'init':
                selected_pane = self._get_selected_pane(x, y)

                if selected_pane == self.prediction_block_id:
                    im_xy, im_id = self._global_xy_to_im_xy(x, y, selected_pane)
                    self._reset()

                    self.click_info['pane'] = selected_pane
                    self.click_info['im_id'] = im_id
                    self.click_info['mode'] = 'click1'
                    self.click_info['pos0'] = im_xy
                    self.click_info['pos1'] = (im_xy[0] + 1, im_xy[1] + 1)
                    self.click_info['pos0_global'] = (x, y)

                    self._draw()
                elif selected_pane == (self.num_panes - 1):
                    im_xy, im_id = self._global_xy_to_im_xy(x, y, selected_pane)
                    # self._reset()

                    self.click_info['pane'] = selected_pane
                    self.click_info['im_id'] = im_id
                    self.click_info['mode'] = 'drag'
                    self.click_info['pos0_global'] = (x, y)
                    self.zoom_selection_init = self.zoom_roi_coords
                    self._draw()
                else:
                    # Check buttons
                    for i in range(self.num_panes):
                        rows_flag = self.button_info[i]['r1'] < y < self.button_info[i]['r2']
                        cols_flag = self.button_info[i]['c1'] < x < self.button_info[i]['c2']
                        if rows_flag and cols_flag:
                            self._toggle_pane_mode(i)
                            break
                    self._draw()
            elif self.click_info['mode'] == 'click1':
                self.click_info['mode'] = 'init'

                self.zoom_pane_id = self.click_info['pane']
                self.click_info['pos1'] = self._get_pos1(x, y)

                self.zoom_roi_coords = [self.click_info['pos0'], self.click_info['pos1']]
                self._draw()

        elif self.click_info['mode'] == 'click1' and event == cv.EVENT_MOUSEMOVE:
            self.click_info['pos1'] = self._get_pos1(x, y)

            self._draw()
        elif self.click_info['mode'] == 'drag' and event == cv.EVENT_MOUSEMOVE:
            shift = (x - self.click_info['pos0_global'][0], y - self.click_info['pos0_global'][1])

            zoom_pane_id = self.click_info['pane']
            resize_factor = self.pane_coord_info[zoom_pane_id]['resize_factor']

            shift = [int(float(p) / resize_factor) for p in shift]

            zx1 = self.zoom_selection_init[0][0] - shift[0]
            zx2 = self.zoom_selection_init[1][0] - shift[0]

            zy1 = self.zoom_selection_init[0][1] - shift[1]
            zy2 = self.zoom_selection_init[1][1] - shift[1]

            zoom_selection1 = list(self.zoom_roi_coords[0])
            zoom_selection2 = list(self.zoom_roi_coords[1])

            if zx1 > 0 and zx2 < self.data_sizes[self.prediction_block_id][0]:
                zoom_selection1[0] = zx1
                zoom_selection2[0] = zx2

            if zy1 > 0 and zy2 < self.data_sizes[self.prediction_block_id][1]:
                zoom_selection1[1] = zy1
                zoom_selection2[1] = zy2

            self.zoom_roi_coords = (tuple(zoom_selection1), tuple(zoom_selection2))
            self._draw()
        elif self.click_info['mode'] == 'drag' and event == cv.EVENT_LBUTTONUP:
            self.click_info['mode'] = 'init'

            self._draw()

    # ******************************************************************************************************
    # ***************************** Helper drawing functions ************************************************
    # ******************************************************************************************************

    def _resize_image(self, im, resize_factor):
        im = cv.resize(im, (int(resize_factor * im.shape[1]), int(resize_factor * im.shape[0])), interpolation=cv.INTER_NEAREST)
        return im

    def _insert_rectangle(self, im, pos0, pos1, linewidth):
        cv.rectangle(im, tuple(pos0), tuple(pos1), (0, 0, 255), linewidth)

    def _insert_rois(self, images, resize_factor, pane_id):
        if pane_id != self.prediction_block_id:
            return images

        if self.click_info['pos0'][0] < 0:
            return images

        if self.click_info['mode'] == 'click1':
            pos0 = [int(r * resize_factor) for r in self.click_info['pos0']]
            pos1 = [int(r * resize_factor) for r in self.click_info['pos1']]

            self._insert_rectangle(images[self.click_info['im_id']], pos0, pos1, 1)
        elif self.click_info['mode'] in ['init', 'drag']:
            pos0 = [int(r * resize_factor) for r in self.zoom_roi_coords[0]]
            pos1 = [int(r * resize_factor) for r in self.zoom_roi_coords[1]]

            for im in images:
                self._insert_rectangle(im, pos0, pos1, 3)
        return images

    def _draw_button(self, disp_image, r1, r2, c1, c2, color, text=''):
        disp_image[r1:r2, c1:c2, :] = np.array(color).astype(np.uint8)

        disp_image = cv.putText(disp_image, text=text, org=(c1 + 10, r2 - 10), fontFace=0, fontScale=0.6,
                                color=(0, 0, 0), thickness=1)

        return disp_image

    def _insert_images(self, disp_image, images, titles, r1, c1):
        r1_orig = r1
        for im, title in zip(images, titles):
            r1 = r1 + self.title_pad
            r2 = r1 + im.shape[0]
            c2 = c1 + im.shape[1]

            if im.ndim == 2:
                im = np.expand_dims(im, -1)
            disp_image[r1:r2, c1:c2, :] = im

            r1 = r1 + 20     # -8
            disp_image = cv.putText(disp_image, text=title, org=(c1, r1), fontFace=0, fontScale=0.6,
                                    color=(255, 255, 255), thickness=2)

            c1 = c2 + self.padding_amount
            r1 = r1_orig

        return disp_image

    # ********************************************************************************************************
    # ********************************* Main drawing functions ***********************************************
    # ********************************************************************************************************

    def _draw_buttons(self, disp_image):
        c1 = self.display_size[1]
        c2 = self.window_size[1] - 10
        r1 = 50
        button_h = 40

        on_color = (152, 251, 152)
        off_color = (127, 127, 255)

        for i in range(self.num_panes):
            color = on_color if self.pane_mode[i] == 1 else off_color
            disp_image = self._draw_button(disp_image, r1, r1 + button_h, c1, c2, color, self.pane_titles[i])
            self.button_info[i] = {'r1': r1, 'c1': c1, 'r2': r1 + button_h, 'c2': c2}

            r1 = r1 + button_h + 20

        return disp_image

    def _draw_pane(self, disp_image, images_in, titles_in, pane_id):
        if len(images_in) < 6:
            images = images_in
            titles = titles_in
        else:
            images = images_in[:6]
            titles = titles_in[:6]

        input_sz_cols = len(images) * images[0].shape[1] + self.padding_amount * (len(images) - 1)
        input_sz_rows = images[0].shape[0]

        if self.pane_mode[pane_id] == 0:
            return disp_image

        elif self.pane_mode[pane_id] == 1:
            # Resize
            resize_factor_x = (self.display_size[1] - self.boundary_pad * 2) / input_sz_cols
            resize_factor_y = (self.display_size[0] * self.display_split[pane_id] - 2*self.boundary_pad - self.title_pad) / input_sz_rows
            resize_factor = min(resize_factor_x, resize_factor_y)
            images = [self._resize_image(im, resize_factor) for im in images]
        else:
            raise Exception

        # Draw rois in resized images
        images = self._insert_rois(images, resize_factor, pane_id)

        input_sz_cols = len(images) * images[0].shape[1] + self.padding_amount * (len(images) - 1)
        input_sz_rows = images[0].shape[0]

        left_pad = (self.display_size[1] - input_sz_cols) // 2
        top_pad = int((self.display_size[0] * self.display_split[pane_id] - input_sz_rows - self.title_pad) / 2)
        top_pad = top_pad + int(self.display_size[0] * sum(self.display_split[:pane_id]))
        r1 = top_pad
        c1 = left_pad

        self.pane_coord_info[pane_id] = {'r1': r1 + self.title_pad, 'c1': c1,
                                         'r2': r1 + input_sz_rows + self.title_pad, 'c2': c1 + input_sz_cols,
                                         'num_images': len(images), 'padding': self.padding_amount,
                                         'im_w': images[0].shape[1], 'im_h': images[0].shape[0],
                                         'resize_factor': resize_factor}
        if self.selected_pane_id == pane_id:
            disp_image[r1 - self.selection_pad:r1 + images[0].shape[0] + self.selection_pad + self.title_pad,
            c1 - self.selection_pad:c1 + input_sz_cols + self.selection_pad, :] = self.selection_color

        disp_image = self._insert_images(disp_image, images, titles, r1, c1)
        disp_image = self._draw_buttons(disp_image)

        return disp_image

    def _draw(self):
        disp_image = np.ones((self.window_size[0], self.window_size[1], 3)).astype(np.uint8) * 255

        for i in range(self.num_data_blocks):
            all_images = self.data[i]['images']
            all_titles = self.data[i]['titles']
            im_id = self.pane_image_id[i]
            images = all_images if self.show_all_images[i] else all_images[im_id:im_id + 1]
            titles = all_titles if self.show_all_images[i] else all_titles[im_id:im_id + 1]

            if i == self.prediction_block_id and self.metric is not None:
                scores = ['{:.3f}'.format(self.metric(df_utils.npimage_to_torch(im),
                                                      df_utils.npimage_to_torch(images[0]))) for im in images[1:]]
                scores = [''] + scores
                titles = ['{} Score: {}'.format(tt, sc) for tt, sc in zip(titles, scores)]

            # Handle zoom
            if i != self.prediction_block_id and self.zoom_roi_coords is not None:
                norm_factor = self.data[i]['images'][0].shape[0] / self.data[self.prediction_block_id]['images'][0].shape[0]
                r1 = int(norm_factor * self.zoom_roi_coords[0][1])
                r2 = int(norm_factor * self.zoom_roi_coords[1][1])
                c1 = int(norm_factor * self.zoom_roi_coords[0][0])
                c2 = int(norm_factor * self.zoom_roi_coords[1][0])
                images = [im[r1:r2, c1:c2, :] for im in images]

            disp_image = self._draw_pane(disp_image, images, titles, i)

        # ****************************** Zoom selection **************************************
        self.zoom_images = None
        if self.zoom_roi_coords is not None:
            prediction_images_orig = self.data_orig[self.prediction_block_id]['images']
            self.pane_image_id[-1] = self.pane_image_id[-1] % len(prediction_images_orig)

            zoom_show_all = self.show_all_images[self.num_panes - 1]
            zoom_images = prediction_images_orig if zoom_show_all else prediction_images_orig[self.pane_image_id[-1]:self.pane_image_id[-1] + 1]
            zoom_images = [im[self.zoom_roi_coords[0][1]:self.zoom_roi_coords[1][1],
                           self.zoom_roi_coords[0][0]:self.zoom_roi_coords[1][0], :] for im in zoom_images]

            self.zoom_images = zoom_images
            prediction_titles = self.data_orig[self.prediction_block_id]['titles']
            zoom_titles = prediction_titles if zoom_show_all else prediction_titles[self.pane_image_id[-1]:self.pane_image_id[-1] + 1]

            if self.metric is not None:
                scores = ['{:.3f}'.format(self.metric(df_utils.npimage_to_torch(im),
                                                      df_utils.npimage_to_torch(zoom_images[0]))) for im in zoom_images[1:]]
                scores = [''] + scores
                zoom_titles = ['{} Score: {}'.format(tt, sc) for tt, sc in zip(zoom_titles, scores)]

            disp_image = self._draw_pane(disp_image, zoom_images, zoom_titles, self.num_panes - 1)

        cv.imshow('Display', disp_image)
        cv.setMouseCallback("Display", self._mouse_callback)
        return

    def _reset(self):
        self.data = [{'images': [im.copy() for im in d['images']], 'titles': d['titles']} for d in self.data_orig]
        self.click_info = {'mode': 'init', 'pos0': (-1, -1), 'pos1': (-1, -1), 'pane': None, 'im_id': None}

        self._draw()

    def save_data(self):
        base_dir = self.save_results_path
        save_dir = '{}/{}'.format(base_dir, self.data[0]['titles'][0])
        os.makedirs(save_dir, exist_ok=True)
        for idx in range(len(self.data[0]['titles'])):
            title = self.data[0]['titles'][idx]
            im = self.data[0]['images'][idx]
            cv.imwrite('{}/{}.png'.format(save_dir, title), im)

            if self.zoom_images is not None:
                zoom_im = self.zoom_images[idx]
                zoom_im = cv.resize(zoom_im, dsize=None, fx=8, fy=8, interpolation=cv.INTER_NEAREST)
                cv.imwrite('{}/{}_crop.png'.format(save_dir, title), zoom_im)

        norm_factor = self.data[0]['images'][0].shape[0] / self.data[self.prediction_block_id]['images'][0].shape[0]
        r1 = int(norm_factor * self.zoom_roi_coords[0][1])
        r2 = int(norm_factor * self.zoom_roi_coords[1][1])
        c1 = int(norm_factor * self.zoom_roi_coords[0][0])
        c2 = int(norm_factor * self.zoom_roi_coords[1][0])

        with open('{}/crop_info.json'.format(save_dir), 'w') as outfile:
            json.dump({'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2}, outfile)

    def plot(self, data):
        if not isinstance(data, (tuple, list)):
            data = [data, ]

        if self.boundary_ignore is not None:
            bi = self.boundary_ignore
            for d_ in data:
                d_['images'] = [d[bi:-bi, bi:-bi] for d in d_['images']]

        self.data_orig = [{'images': [im.copy() for im in d['images']], 'titles': d['titles']} for d in data]
        self.data = [{'images': [im.copy() for im in d['images']], 'titles': d['titles']} for d in data]

        self.data_sizes = [[d['images'][0].shape[1], d['images'][0].shape[0]] for d in data]

        self._draw()

        while True:
            key = cv.waitKey(0)
            if key == ord('n'):
                return 'continue'
            if key == ord('y'):
                return 'save'
            elif key == ord('q'):
                return 'stop'
            elif key == ord('r'):
                self._reset()
            elif key == ord(' '):
                self.show_all_images[self.selected_pane_id] = not self.show_all_images[self.selected_pane_id]
                self._draw()
            elif key == 83 or key == ord('d'):     # right
                pane_num_images = len(data[self.selected_pane_id]['images']) if self.selected_pane_id != (self.num_panes - 1) else len(data[self.prediction_block_id]['images'])
                self.pane_image_id[self.selected_pane_id] = (self.pane_image_id[self.selected_pane_id] + 1) % pane_num_images
                self._draw()
            elif key == 81 or key == ord('a'):     # left
                pane_num_images = len(data[self.selected_pane_id]['images']) if self.selected_pane_id != (
                            self.num_panes - 1) else len(data[self.prediction_block_id]['images'])
                self.pane_image_id[self.selected_pane_id] = (self.pane_image_id[
                                                                 self.selected_pane_id] - 1) % pane_num_images
                self._draw()
            elif key == 82 or key == ord('w'):     # up
                num_panes = self.num_panes if self.zoom_roi_coords is not None else self.num_data_blocks
                self.selected_pane_id = (self.selected_pane_id - 1) % num_panes
                self._draw()
            elif key == 84 or key == ord('s'):     # down
                num_panes = self.num_panes if self.zoom_roi_coords is not None else self.num_data_blocks
                self.selected_pane_id = (self.selected_pane_id + 1) % num_panes
                self._draw()
            elif key == ord('p'):     # save
                self.save_data()


