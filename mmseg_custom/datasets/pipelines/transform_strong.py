import mmcv
import numpy as np
import pickle
from mmseg.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmcv.utils import deprecated_api_warning, is_tuple_of
from mmseg.datasets.pipelines.formatting import to_tensor
import albumentations as A
import cv2
import json
from osgeo import gdal
import pandas as pd
import random
import os
import json
from copy import deepcopy
from skimage.measure import label as sklabel

@PIPELINES.register_module()
class DefaultFormatBundleStrong(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'img_full' in results:
            img = results['img_full']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img_full'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(to_tensor(
                results['gt_semantic_seg'][None, ...].astype(np.int64)),
                                            stack=True)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(to_tensor(results['gt_masks']))
        if 'gt_labels' in results:
            results['gt_labels'] = DC(to_tensor(results['gt_labels']))

        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class WeakToStrong(object):
    def __init__(self):
        pass

    def __call__(self, results):

        img = results['img']
        results['img_full'] = img.copy()

        if 'gt_semantic_seg' in results:
            gt_semantic_seg = results['gt_semantic_seg']
            results['gt_semantic_seg_full'] = gt_semantic_seg.copy()
            results['seg_fields'].append('gt_semantic_seg_full')

        return results


@PIPELINES.register_module(force=True)
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        if 'img_full' in results:
            results['img_full'] = mmcv.imnormalize(results['img_full'], self.mean, self.std,
                                              self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


# @PIPELINES.register_module(force=True)
# class Pad(object):
#     """Pad the image & mask.
#
#     There are two padding modes: (1) pad to a fixed size and (2) pad to the
#     minimum size that is divisible by some number.
#     Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
#
#     Args:
#         size (tuple, optional): Fixed padding size.
#         size_divisor (int, optional): The divisor of padded size.
#         pad_val (float, optional): Padding value. Default: 0.
#         seg_pad_val (float, optional): Padding value of segmentation map.
#             Default: 255.
#     """
#
#     def __init__(self,
#                  size=None,
#                  size_divisor=None,
#                  pad_val=0,
#                  seg_pad_val=255):
#         self.size = size
#         self.size_divisor = size_divisor
#         self.pad_val = pad_val
#         self.seg_pad_val = seg_pad_val
#         # only one of size and size_divisor should be valid
#         assert size is not None or size_divisor is not None
#         assert size is None or size_divisor is None
#
#     def _pad_img(self, results):
#         """Pad images according to ``self.size``."""
#         if self.size is not None:
#             padded_img = mmcv.impad(
#                 results['img'], shape=self.size, pad_val=self.pad_val)
#             # if 'img_full' in results:
#             #     padded_img_full = mmcv.impad(
#             #         results['img_full'], shape=self.size, pad_val=self.pad_val)
#         elif self.size_divisor is not None:
#             padded_img = mmcv.impad_to_multiple(
#                 results['img'], self.size_divisor, pad_val=self.pad_val)
#             if 'img_full' in results:
#                 padded_img_full = mmcv.impad_to_multiple(
#                     results['img_full'], self.size_divisor, pad_val=self.pad_val)
#
#         results['img'] = padded_img
#         # results['img_full'] = padded_img_full
#         results['pad_shape'] = padded_img.shape
#         results['pad_fixed_size'] = self.size
#         results['pad_size_divisor'] = self.size_divisor
#
#     def _pad_seg(self, results):
#         """Pad masks according to ``results['pad_shape']``."""
#         for key in results.get('seg_fields', []):
#             results[key] = mmcv.impad(
#                 results[key],
#                 shape=results['pad_shape'][:2],
#                 pad_val=self.seg_pad_val)
#
#     def __call__(self, results):
#         """Call function to pad images, masks, semantic segmentation maps.
#
#         Args:
#             results (dict): Result dict from loading pipeline.
#
#         Returns:
#             dict: Updated result dict.
#         """
#
#         self._pad_img(results)
#         self._pad_seg(results)
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
#                     f'pad_val={self.pad_val})'
#         return repr_str


@PIPELINES.register_module(force=True)
class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1. and 'gt_semantic_seg' in results:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        if 'crop_bbox' not in results:
            results['crop_bbox'] = crop_bbox
        elif 'img_full' in results:   #直接else
            results['crop_bbox_full'] = results['crop_bbox']
            results['crop_bbox'] = crop_bbox

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module(force=True)
class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
        min_size (int, optional): The minimum size for input and the shape
            of the image and seg map will not be less than ``min_size``.
            As the shape of model input is fixed like 'SETR' and 'BEiT'.
            Following the setting in these models, resized images must be
            bigger than the crop size in ``slide_inference``. Default: None
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 min_size=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.min_size = min_size

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            if self.min_size is not None:
                # TODO: Now 'min_size' is an 'int' which means the minimum
                # shape of images is (min_size, min_size, 3). 'min_size'
                # with tuple type will be supported, i.e. the width and
                # height are not equal.
                if min(results['scale']) < self.min_size:
                    new_short = self.min_size
                else:
                    new_short = min(results['scale'])

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        elif 'img_full' in results:
            results['scale_full'] = results['scale']
            results['scale_idx_full'] = results['scale_idx']
            results['scale'] = None
            results['scale_idx'] = None
            self._random_scale(results)

        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module(force=True)
class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        elif 'img_full' in results:
            results['flip_full'] = results['flip']
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip

        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        elif 'img_full' in results:
            results['flip_direction_full'] = results['flip_direction']
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'

@PIPELINES.register_module()
class RandMixPseudo():
    def __init__(self, psd_root, classes, choice_p):
        self.psd_root = psd_root
        self.classes = classes
        self.choice_p = choice_p
        self.candidate_name = 'CTR'
        self.file_p = os.path.join(self.psd_root, self.candidate_name + '.p')
        if os.path.isfile(self.file_p):
            self.label_to_file, _ = pickle.load(open(self.file_p, "rb"))
    def readImage(self, dirname):
        if dirname.endswith('.png'):
            image = cv2.imread(dirname, -1)
        elif dirname.endswith('.tif'):
            Img = gdal.Open(dirname)
            image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
            if len(image.shape) == 3:
                image = np.rollaxis(image, 0, 3)
        return image

    def mix(self, in_imgs, image_root, psd_root, classes, choice_p):
        # in_imgs :  PIL or numpy   (w, h, 3)
        # classes :  candidate category for copy-paste
        # choice_p :  sample prob for candidate category
        # return :
        #       out_imgs:    (w, h, 3)
        #       out_labels:  (w, h, 3)
        in_imgs = np.array(in_imgs)
        out_imgs = deepcopy(in_imgs)
        out_lbls = np.ones(in_imgs.shape[:2]) * 255
        in_w, in_h = in_imgs.shape[:2]
        # if len(classes) == 0:
        #     # print('-----------------######-------------------------')
        #     return out_imgs, out_lbls
        class_idx = np.random.choice(classes, size=1, p=choice_p)[0]
        ins_pix_num = 50
        while True:
            name = random.sample(self.label_to_file[class_idx], 1)
            cdd_img_path = os.path.join(image_root, name[0]).replace('_labelTrainIds.png', '.png')   #isprs:.replace('_labelTrainIds.tif', '.tif')
            # print(f'\n------------------------cdd_img_path:{cdd_img_path}')
            cdd_label_path = os.path.join(psd_root, self.candidate_name, name[0])
            # print(f'------------------------------cdd_root:{cdd_label_path}')
            cdd_img = self.readImage(cdd_img_path) #np.asarray(Image.open(cdd_img_path).convert('RGB'), np.uint8)
            cdd_lbl = self.readImage(cdd_label_path)  #np.asarray(Image.open(cdd_label_path))

            x1 = random.randint(0, cdd_img.shape[0] - in_h)
            y1 = random.randint(0, cdd_img.shape[1] - in_w)
            cdd_img = cdd_img[x1:x1 + in_h, y1:y1 + in_w, :]
            cdd_lbl = cdd_lbl[x1:x1 + in_h, y1:y1 + in_w]


            mask = cdd_lbl == class_idx
            if np.sum(mask) >= ins_pix_num:
                break
        # if self.join_mode == 'direct':
        #     if self.class_num == 19:
        #         # 12: rider  17: bike   18: moto-
        #         if class_idx == 12:
        #             if 17 in cdd_lbl:
        #                 mask += cdd_lbl == 17
        #             if 18 in cdd_lbl:
        #                 mask += cdd_lbl == 18
        #         if class_idx == 17 or class_idx == 18:
        #             mask += cdd_lbl == 12
        #         # 5: pole  6: light   7: sign
        #         if class_idx == 5:
        #             if 6 in cdd_lbl:
        #                 mask += cdd_lbl == 6
        #             if 7 in cdd_lbl:
        #                 mask += cdd_lbl == 7
        #         if class_idx == 6 or class_idx == 7:
        #             mask += cdd_lbl == 5
        #     else:
        #         # 11: rider  14: bike 15: moto-
        #         if class_idx == 11:
        #             if 14 in cdd_lbl:
        #                 mask += cdd_lbl == 14
        #             if 15 in cdd_lbl:
        #                 mask += cdd_lbl == 15
        #         if class_idx == 14 or class_idx == 15:
        #             mask += cdd_lbl == 11
        #         # 5: pole  6: light   7: sign
        #         if class_idx == 5:
        #             if 6 in cdd_lbl:
        #                 mask += cdd_lbl == 6
        #             if 7 in cdd_lbl:
        #                 mask += cdd_lbl == 7
        #         if class_idx == 6 or class_idx == 7:
        #             mask += cdd_lbl == 5

        masknp = mask.astype(int)
        seg, forenum = sklabel(masknp, background=5, return_num=True, connectivity=2)  #background=0
        filled_mask = np.zeros(in_imgs.shape[:2])
        filled_boxes = []
        for i in range(forenum):
            instance_id = i + 1
            if np.sum(seg == instance_id) < 20:
                continue
            ins_mask = (seg == instance_id).astype(np.uint8)
            cont, hierarchy = cv2.findContours(ins_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cont = list(cont)   #后加，查是否为opencv版本问题
            cont.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            x, y, w, h = cv2.boundingRect(cont[0])
            #### rescale instance
            # randscale = 0.5 + np.random.rand() * 1.5
            randscale = 1.0 + np.random.rand() * 1.5
            resized_crop_ins = cv2.resize(cdd_img[y:y + h, x:x + w], None, fx=randscale, fy=randscale,
                                          interpolation=cv2.INTER_NEAREST)  #cv2.INTER_NEAREST  #此处似乎有误,应为cv2.INTER_LINEAR
            resized_crop_mask = cv2.resize(ins_mask[y:y + h, x:x + w], None, fx=randscale, fy=randscale,
                                           interpolation=cv2.INTER_NEAREST)
            resized_crop_mask_cdd = cv2.resize(cdd_lbl[y:y + h, x:x + w], None, fx=randscale, fy=randscale,
                                               interpolation=cv2.INTER_NEAREST)
            new_w, new_h = resized_crop_ins.shape[:2]
            if in_w <= new_w or in_h <= new_h:
                continue
            ##### cal new axis
            cnt = 100
            while cnt > 1:
                ##### 判断共现类，生成paste位置，穿插到单实例中
                # if class_idx not in resized_crop_mask_cdd and len(filled_boxes) > 0 and random.random() < 0.5:
                if class_idx not in resized_crop_mask_cdd[resized_crop_mask > 0] and len(filled_boxes) > 0:  #这段判断语句应该可以忽略   此句不可以忽略，可能resized_crop_mask > 0的部分是背景部分
                    rand_box = random.sample(filled_boxes, 1)[0]
                    x1 = random.randint(rand_box[0], rand_box[1] - 1)
                    y1 = random.randint(rand_box[2], rand_box[3] - 1)
                    if x1 + new_w < in_w - 1 and y1 + new_h < in_h - 1:
                        break
                x1 = random.randint(0, in_w - new_w - 1)
                y1 = random.randint(0, in_h - new_h - 1)
                if filled_mask[x1, y1] == 0 and filled_mask[x1 + new_w, y1 + new_h] == 0:
                    break
                cnt -= 1
            ##### paste
            if cnt > 1:
                out_imgs[x1:x1 + new_w, y1:y1 + new_h][resized_crop_mask > 0] = resized_crop_ins[resized_crop_mask > 0]
                out_lbls[x1:x1 + new_w, y1:y1 + new_h][resized_crop_mask > 0] = resized_crop_mask_cdd[resized_crop_mask > 0]
                filled_mask[x1:x1 + new_w, y1:y1 + new_h] = 1
                ##### 将单实例的存入filled_boxes
                if len(np.unique(resized_crop_mask_cdd[resized_crop_mask > 0])) == 1 and class_idx in \
                        resized_crop_mask_cdd[resized_crop_mask > 0]:
                    filled_boxes.append([x1, x1 + new_w, y1, y1 + new_h])

        return out_imgs, out_lbls

    def __call__(self, results):
        img = results['img']
        img_root = results['img_prefix']
        mix_image, mix_label = self.mix(results['img'], img_root, self.psd_root, self.classes, self.choice_p)

        results['img'] = mix_image
        results['mix_label'] = mix_label

        return results

@PIPELINES.register_module()
class RandomROIBbox(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, scale, roi_num=10, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.scale = scale
        self.roi_num = roi_num
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    # def get_crop_bbox_list(self, img, count):
    #     roi_list = []
    #     for it in range(count):
    #         margin_h = max(img.shape[0] - self.crop_size[0], 0)
    #         margin_w = max(img.shape[1] - self.crop_size[1], 0)
    #         offset_h = np.random.randint(0, margin_h + 1)
    #         offset_w = np.random.randint(0, margin_w + 1)
    #         crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
    #         crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
    #         roi_list.append(np.array([crop_x1, crop_y1,  crop_x2, crop_y2]))
    #     roi_list = np.stack(roi_list, axis=0)
    #     return roi_list

    def get_crop_bbox_list(self, img, count):
        roi_list = []
        ratio = np.random.random_sample() * (max(self.scale) - min(self.scale)) + min(self.scale)
        scale = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
        for it in range(count):
            margin_h = max(img.shape[0] - scale[0], 0)
            margin_w = max(img.shape[1] - scale[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + scale[0]
            crop_x1, crop_x2 = offset_w, offset_w + scale[1]
            roi_list.append(np.array([crop_x1, crop_y1,  crop_x2, crop_y2]))
        roi_list = np.stack(roi_list, axis=0)
        return roi_list

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        # crop_bbox = self.get_crop_bbox(img)
        roi_bbox_list = self.get_crop_bbox_list(img, self.roi_num)

        # crop the image
        # img = self.crop(img, crop_bbox)
        # img_shape = img.shape
        # results['img'] = img
        # results['img_shape'] = img_shape

        # if 'crop_bbox' not in results:
        #     results['crop_bbox'] = crop_bbox
        # elif 'img_full' in results:   #直接else
        #     results['crop_bbox_full'] = results['crop_bbox']
        #     results['crop_bbox'] = crop_bbox

        results['crop_bbox_list'] = roi_bbox_list
        # # crop semantic seg
        # for key in results.get('seg_fields', []):
        #     results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
