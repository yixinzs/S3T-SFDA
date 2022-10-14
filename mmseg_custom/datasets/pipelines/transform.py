import mmcv
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES
import albumentations as A
import cv2
import json
from osgeo import gdal
import pandas as pd
import random
import os

# @PIPELINES.register_module()
class SETR_Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 crop_size=None,
                 setr_multi_scale=False):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.crop_size = crop_size
        self.setr_multi_scale = setr_multi_scale

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
                and uper bound of image scales.

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
            if self.setr_multi_scale:
                if min(results['scale']) < self.crop_size[0]:
                    new_short = self.crop_size[0]
                else:
                    new_short = min(results['scale'])

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(results['img'],
                                               results['scale'],
                                               return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(results['img'],
                                                  results['scale'],
                                                  return_scale=True)
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
                gt_seg = mmcv.imrescale(results[key],
                                        results['scale'],
                                        interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(results[key],
                                       results['scale'],
                                       interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

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


@PIPELINES.register_module()
class PadShortSide(object):
    """Pad the image & mask.

    Pad to the minimum size that is equal or larger than a number.
    Added keys are "pad_shape", "pad_fixed_size",

    Args:
        size (int, optional): Fixed padding size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """
    def __init__(self, size=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        h, w = results['img'].shape[:2]
        new_h = max(h, self.size)
        new_w = max(w, self.size)
        padded_img = mmcv.impad(results['img'],
                                shape=(new_h, new_w),
                                pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        # results['unpad_shape'] = (h, w)

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key],
                                      shape=results['pad_shape'][:2],
                                      pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        h, w = results['img'].shape[:2]
        if h >= self.size and w >= self.size:  # 短边比窗口大，跳过
            pass
        else:
            self._pad_img(results)
            self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class MapillaryHack(object):
    """map MV 65 class to 19 class like Cityscapes."""
    def __init__(self):
        self.map = [[13, 24, 41], [2, 15], [17], [6], [3],
                    [45, 47], [48], [50], [30], [29], [27], [19], [20, 21, 22],
                    [55], [61], [54], [58], [57], [52]]

        self.others = [i for i in range(66)]
        for i in self.map:
            for j in i:
                if j in self.others:
                    self.others.remove(j)

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        gt_map = results['gt_semantic_seg']
        # others -> 255
        new_gt_map = np.zeros_like(gt_map)

        for value in self.others:
            new_gt_map[gt_map == value] = 255

        for index, map in enumerate(self.map):
            for value in map:
                new_gt_map[gt_map == value] = index

        results['gt_semantic_seg'] = new_gt_map

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ClassMix(object):
    def __init__(self,
                 prob,
                 small_class=None,
                 file=None,
                 ):
        assert prob >= 0 and prob <= 1
        self.prob = prob
        self.small_class = small_class
        self.file = file

        with open(file, 'r') as of:
            sample_class_stats = json.load(of)
        self.sample_class_stats = sample_class_stats

    def generate_class_mask(self, label, classes):
        if not isinstance(classes, list):
            classes = np.array([classes])
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=0)
        # debug = classes.unsqueeze(1).unsqueeze(2)
        label, classes = np.broadcast_arrays(label, classes[:, None, None])
        class_mask = np.sum(label == classes, axis=0).astype(np.uint8)

        return class_mask

    def one_mix(self, mask, origin_data=None, cut_data=None, origin_target=None, cut_target=None):
        if mask is None:
            return origin_data, origin_target
        if not (origin_data is None):
            stackedMask0, _ = np.broadcast_arrays(mask[:, :, None], origin_data)
            data = (stackedMask0 * cut_data +
                    (1 - stackedMask0) * origin_data)
        if not (origin_target is None):
            stackedMask0, _ = np.broadcast_arrays(mask, origin_target)
            target = (stackedMask0 * cut_target +
                      (1 - stackedMask0) * origin_target)
        return data, target

    def __call__(self, results):
        cut_mix = True if np.random.rand() < self.prob else False

        if cut_mix:
            origin_img = results['img']
            origin_label = results['gt_semantic_seg']

            class_choice = np.random.choice(self.small_class, replace=False)
            index = np.random.choice(len(self.sample_class_stats[str(class_choice)]))
            f1 = self.sample_class_stats[str(class_choice)][index]
            # print('------choice:{}, len:{}, index:{}, f1:{}'.format(class_choice,
            #                                                  len(self.sample_class_stats[str(class_choice)]),
            #                                                  index,
            #                                                  f1))
            cut_img = self.readImage(f1[0])
            cut_label = self.readImage(f1[1])
            # cut_img = cut_img[64:448, 64:448, :]
            # cut_label = cut_label[64:448, 64:448]

            if results['label_map'] is not None:
                class_choice = results['label_map'][class_choice]
                # print('-----------map:{}'.format(class_choice))
            class_mask = self.generate_class_mask(cut_label, class_choice)
            mix_img, mix_label = self.one_mix(class_mask, origin_img, cut_img, origin_label, cut_label)
            # print('----------------:mask:{}'.format(np.unique(class_mask)))
            results['img'] = mix_img
            for key in results.get('seg_fields', []):
                results[key] = mix_label
            # print('---pixels:{}'.format(np.unique(mix_label)))
            # results['gt_semantic_seg'] = mix_label
            #
            # results['cut_img'] = cut_img
            # results['cut_label'] = cut_label
            # results['class_mask'] = class_mask

        return results

    def readImage(self, dirname):
        if dirname.endswith('.png'):
            image = cv2.imread(dirname, -1)
        elif dirname.endswith('.tif'):
            Img = gdal.Open(dirname)
            image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
            if len(image.shape) == 3:
                image = np.rollaxis(image, 0, 3)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(small_class={self.small_class}, file={self.file})'
        return repr_str


@PIPELINES.register_module()
class AlbumentationAug(object):
    def __init__(self, p=1):
        self.p = p
        self.simple_tranforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]

        self.tranforms = [
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.4, rotate_limit=90, p=0.3),
            A.RandomBrightnessContrast(p=0.5),  # ------------1
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),  # -------2
            A.OneOf(
                [
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),  # @色调变换1
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # #@色调变换1
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),  # @色调变换1
                ],
                p=0.6,
            ),  # -------------3
            # A.GaussNoise(var_limit=(0.0, 15.0), mean=0, p=0.2),  # @噪声变换  #-----------4
            # A.OneOf(
            #     [
            #         A.GaussianBlur(blur_limit=(3, 5), p=0.5),  # @边缘变换1
            #         A.Blur(blur_limit=3, p=0.5),  # @边缘变换1
            #         A.Sharpen(alpha=(0.1, 0.3), lightness=(0.1, 0.3), p=0.5)  # @边缘变换1
            #     ],
            #     p=0.6,
            # ),        #---------------5
            # A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.2),  #----------6
            # A.RandomCrop(height=384, width=384, always_apply=True)
        ]

        self.tranforms_v2 = [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),  # ------------1
            A.GaussNoise(p=0.2),
            # A.GaussNoise(var_limit=(0.0, 15.0), mean=0, p=0.2),  # @噪声变换  #-----------4
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=1.0),  # @边缘变换
                    A.MedianBlur(blur_limit=3, p=1.0)
                ],
                p=0.1,
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),  # -------2
            A.OneOf(
                [
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),  # @色调变换1
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # #@色调变换1
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),  # @色调变换1
                ],
                p=0.6,
            )  # -------------3
        ]

    def TrainAugmentation(self, p=1):

        return A.Compose(self.tranforms_v2, p=p)

    def __call__(self, results):

        img = results['img']
        gt_map = results['gt_semantic_seg']

        sample = self.TrainAugmentation(p=self.p)(image=img, mask=gt_map)
        img, gt_map = sample['image'], sample['mask']


        results['img'] = img
        results['gt_semantic_seg'] = gt_map

        return results



@PIPELINES.register_module()
class LabelEncode(object):
    def __init__(self):
        pass


    def __call__(self, results):

        gt_map = results['gt_semantic_seg']

        if results.get('label_map', None) is not None:
            gt_map = self.encode_segmap(gt_map, results['label_map'])

        results['gt_semantic_seg'] = gt_map
        # print('----------------------------:{}'.format(np.unique(gt_map)))
        return results

    def encode_segmap(self, mask, classes_index):
        if len(mask.shape) == 2:
            encode_mask = np.zeros(mask.shape, dtype=np.uint8)
            for class_index, pixel_value in classes_index.items():
                encode_mask[mask == pixel_value] = class_index
            # encode_mask[mask == 0] = 255
        else:
            encode_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            for class_index, pixel_value in classes_index.items():
                encode_mask[np.all(mask == pixel_value, 2)] = class_index
            encode_mask = np.array(encode_mask)
        return encode_mask


@PIPELINES.register_module()
class FDA(object):
    def __init__(self,
                 prob,
                 amp_thred=0.006,
                 img_dir=None,
                 file=None,
                 ):
        assert prob >= 0 and prob <= 1
        self.prob = prob
        self.amp_thred = amp_thred
        self.img_dir = img_dir
        self.file = file

        df = pd.read_csv(file)
        self.img_names = df['name']

    def __call__(self, results):
        fda_trans = True if np.random.rand() < self.prob else False

        if fda_trans:
            origin_img = results['img']

            f1 = random.choice(self.img_names)
            img_dirname = os.path.join(self.img_dir, f1)
            target_image = self.readImage(img_dirname)

            if np.sum(np.all(origin_img == [0, 0, 0], 2)) / (origin_img.shape[0] * origin_img.shape[1]) > 0.05:
                # print('----------------------------------------:1')
                return results
            if np.sum(np.all(target_image == [0, 0, 0], 2)) / (target_image.shape[0] * target_image.shape[1]) > 0.05:
                # print('----------------------------------------:2')
                return results

            aug = A.Compose([A.FDA([target_image], beta_limit=self.amp_thred, read_fn=lambda x: x, p=1)])
            origin_img = aug(image=origin_img)['image']

            results['img'] = origin_img

        return results

    def readImage(self, dirname):
        if dirname.endswith('.png'):
            image = cv2.imread(dirname, -1)
        elif dirname.endswith('.tif'):
            Img = gdal.Open(dirname)
            image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
            if len(image.shape) == 3:
                image = np.rollaxis(image, 0, 3)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_dir={self.img_dir}, file={self.file})'
        return repr_str


@PIPELINES.register_module()
class ClassMixFDA(object):
    def __init__(self,
                 prob,
                 small_class=None,
                 amp_thred=0.006,
                 file=None,
                 ):
        assert prob >= 0 and prob <= 1
        self.prob = prob
        self.small_class = small_class
        self.amp_thred = amp_thred
        self.file = file

        with open(file, 'r') as of:
            sample_class_stats = json.load(of)
        self.sample_class_stats = sample_class_stats

    def generate_class_mask(self, label, classes):
        if not isinstance(classes, list):
            classes = np.array([classes])
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=0)
        # debug = classes.unsqueeze(1).unsqueeze(2)
        label, classes = np.broadcast_arrays(label, classes[:, None, None])
        class_mask = np.sum(label == classes, axis=0).astype(np.uint8)

        return class_mask

    def one_mix(self, mask, origin_data=None, cut_data=None, origin_target=None, cut_target=None):
        if mask is None:
            return origin_data, origin_target
        if not (origin_data is None):
            stackedMask0, _ = np.broadcast_arrays(mask[:, :, None], origin_data)
            data = (stackedMask0 * cut_data +
                    (1 - stackedMask0) * origin_data)
        if not (origin_target is None):
            stackedMask0, _ = np.broadcast_arrays(mask, origin_target)
            target = (stackedMask0 * cut_target +
                      (1 - stackedMask0) * origin_target)
        return data, target

    def get_classLayout(self, class_value):
        if class_value == 511:
            return [511, 512, 613, 614]
        elif class_value == 410:
            return [410, 409]
        elif class_value == 303:
            return [303, 512]
        
    def __call__(self, results):
        cut_mix = True if np.random.rand() < self.prob else False

        if cut_mix:
            origin_img = results['img']
            origin_label = results['gt_semantic_seg']

            class_choice = np.random.choice(self.small_class, replace=False)
            index = np.random.choice(len(self.sample_class_stats[str(class_choice)]))
            f1 = self.sample_class_stats[str(class_choice)][index]
            # print('------choice:{}, len:{}, index:{}, f1:{}'.format(class_choice,
            #                                                  len(self.sample_class_stats[str(class_choice)]),
            #                                                  index,
            #                                                  f1))
            cut_img = self.readImage(f1[0])
            cut_label = self.readImage(f1[1])
            # cut_img = cut_img[64:448, 64:448, :]
            # cut_label = cut_label[64:448, 64:448]
            if np.sum(np.all(origin_img == [0, 0, 0], 2)) / (cut_img.shape[0] * cut_img.shape[1]) > 0.05:
                # print('----------------------------------------:1')
                return results
            if np.sum(np.all(cut_img == [0, 0, 0], 2)) / (cut_img.shape[0] * cut_img.shape[1]) > 0.05:
                # print('----------------------------------------:2')
                return results
            if np.sum(cut_label == 100) / (cut_img.shape[0] * cut_img.shape[1]) > 0.999:
                # print('----------------------------------------:3')
                return results
            # print('----------------------------------------:4')
            aug = A.Compose([A.FDA([origin_img], beta_limit=self.amp_thred, read_fn=lambda x: x, p=1)])
            cut_img = aug(image=cut_img)['image']

            if results['label_map'] is not None:
                class_choice = results['label_map'][class_choice]
                # print('-----------map:{}'.format(class_choice))
            class_mask = self.generate_class_mask(cut_label, class_choice)
            mix_img, mix_label = self.one_mix(class_mask, origin_img, cut_img, origin_label, cut_label)
            # print('----------------:mask:{}'.format(np.unique(class_mask)))
            results['img'] = mix_img
            for key in results.get('seg_fields', []):
                results[key] = mix_label
            # print('---pixels:{}'.format(np.unique(mix_label)))
            # results['gt_semantic_seg'] = mix_label
            #
            # results['cut_img'] = cut_img
            # results['cut_label'] = cut_label
            # results['class_mask'] = class_mask

        return results

    def readImage(self, dirname):
        if dirname.endswith('.png'):
            image = cv2.imread(dirname, -1)
        elif dirname.endswith('.tif'):
            Img = gdal.Open(dirname)
            image = Img.ReadAsArray(0, 0, Img.RasterXSize, Img.RasterYSize)
            if len(image.shape) == 3:
                image = np.rollaxis(image, 0, 3)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(small_class={self.small_class}, file={self.file})'
        return repr_str


