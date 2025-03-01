# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

# from ..builder import PIPELINES
from mmseg.datasets.builder import PIPELINES
import cv2
from osgeo import gdal

@PIPELINES.register_module()
class LoadImageFromFileCustom(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # print('--------------------------------------------:%s', filename)
        img = self.readImage(filename)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

    def readImage(self, dirname):
        if dirname.endswith('.png'):
            image = cv2.imread(dirname, -1)
        elif dirname.endswith('.tif'):
            Img = gdal.Open(dirname)
            image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
            if len(image.shape) == 3:
                image = np.rollaxis(image, 0, 3)
        return image



@PIPELINES.register_module()
class LoadAnnotationsCustom(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        gt_semantic_seg = self.readImage(filename)

        # modify if custom classes
        # if results.get('label_map', None) is not None:
        #     gt_semantic_seg = self.encode_segmap(gt_semantic_seg, results['label_map'])
        #     # reduce zero_label
        # if self.reduce_zero_label:
        #     # avoid using underflow conversion
        #     gt_semantic_seg[gt_semantic_seg == 0] = 255
        #     gt_semantic_seg = gt_semantic_seg - 1
        #     gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        # print('---------------------------------label_unique:{}'.format(np.unique(gt_semantic_seg)))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

    def readImage(self, dirname):
        if dirname.endswith('.png'):
            image = cv2.imread(dirname, -1)
        elif dirname.endswith('.tif'):
            Img = gdal.Open(dirname)
            image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
            if len(image.shape) == 3:
                image = np.rollaxis(image, 0, 3)
        return image

    def encode_segmap(self, mask, classes_index):
        if len(mask.shape) == 2:
            encode_mask = np.zeros(mask.shape, dtype=np.uint8)
            for class_index, pixel_value in classes_index.items():
                encode_mask[mask == pixel_value] = class_index
            encode_mask[mask == 0] = 255
        else:
            encode_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            for class_index, pixel_value in classes_index.items():
                encode_mask[np.all(mask == pixel_value, 2)] = class_index
            encode_mask = np.array(encode_mask)
        return encode_mask
