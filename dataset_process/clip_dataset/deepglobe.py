
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('--dataset_path',default=r'deepglobe\train',
                        help='potsdam folder path')
    # parser.add_argument('--tmp_dir', default=r'../../data/temp',
    #                     help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=306)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=306)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, args, data_type, to_label=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    image = mmcv.imread(image_path)

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size
    if data_type=='test' and stride_size<clip_size:
        stride_size = clip_size
        import warnings
        warnings.warn('stride_size < clip_size ===> stride_size has been set equal to clip_size for test dataset')

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    # xmin = x * clip_size
    # ymin = y * clip_size

    xmin = x * stride_size
    ymin = y * stride_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    if to_label:
        # 'urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren', 'unknown'
        color_map = np.array([[255, 255, 0], [0, 255, 255], [255, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255],
                              [0, 0, 0]])
        label = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in zip(label,color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for i, box in enumerate(boxes):
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        idx = osp.basename(image_path)[:3]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{idx}_{str(i).zfill(3)}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


def main():
    args = parse_args()

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join(r'../../data/deepglobe', f'deepglobe_{args.clip_size}_{args.stride_size}')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    img_list = [os.path.join(dataset_path,f'{name}') for name in ['img','label']]
    print('Find the data', img_list)

    for img_dir in img_list:
        src_path_list = glob.glob(os.path.join(img_dir, '*.png'))
        if not src_path_list:
            src_path_list = glob.glob(os.path.join(img_dir, '*.jpg'))

        prog_bar = mmcv.ProgressBar(len(src_path_list))
        for i, src_path in enumerate(src_path_list):
            idx = osp.basename(src_path)[:3]
            if i<563: data_type = 'train'
            elif i<683: data_type = 'val'
            else: data_type = 'test'
            if 'label' in src_path:
                dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                clip_big_image(src_path, dst_dir, args, data_type, to_label=True)
                dst_dir_color = osp.join(out_dir, 'ann_dir', data_type+'_color')
                clip_big_image(src_path, dst_dir_color, args, data_type, to_label=False)
            else:
                dst_dir = osp.join(out_dir, 'img_dir', data_type)
                clip_big_image(src_path, dst_dir, args, data_type, to_label=False)
            prog_bar.update()

    print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
