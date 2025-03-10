# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging

import os
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce
import pandas as pd

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
import pandas as pd
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg_custom.core import SegmentationMetric
from mmseg.utils import get_root_logger
# from .builder import DATASETS
# from .pipelines import Compose
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose
from mmseg_custom.datasets.pipelines import LoadAnnotationsCustom
from mmseg_custom.datasets.pipelines import LabelEncode

import cv2

@DATASETS.register_module()
class CustomBaseDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 fold=0,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 label_map=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.fold = fold
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = label_map #{0: 0, 1: 100, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600, 7: 700, 8: 800}  #{0: 100, 1: 200, 2: 300, 3: 400, 4: 500, 5: 600, 6: 700, 7: 800}
        self.custom_classes = True
        # self.CLASSES, self.PALETTE = self.get_classes_and_palette(
        #     classes, palette)

        self.gt_seg_map_loader = LoadAnnotationsCustom(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotationsCustom(
            **gt_seg_map_loader_cfg)
        self.gt_encode_label = LabelEncode()

        # join paths if data_root is specified
        if self.data_root is not None:  # 训练和验证时
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            # if not (self.split is None or osp.isabs(self.split)):
            #     self.split = osp.join(self.data_root, self.split)

        # self.img_dir = self.data_root   #测试时
        # self.ann_dir = None
        # self.img_suffix = 'E080.tif'
        # self.split = None

        # self.img_dir = self.data_root  # 文件加载数据时
        # self.ann_dir = self.data_root

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    # def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
    #                      split):
    #     """Load annotation from directory.
    #
    #     Args:
    #         img_dir (str): Path to image directory
    #         img_suffix (str): Suffix of images.
    #         ann_dir (str|None): Path to annotation directory.
    #         seg_map_suffix (str|None): Suffix of segmentation maps.
    #         split (str|None): Split txt file. If split is specified, only file
    #             with suffix in the splits will be loaded. Otherwise, all images
    #             in img_dir/ann_dir will be loaded. Default: None
    #
    #     Returns:
    #         list[dict]: All image info of dataset.
    #     """
    #
    #     img_infos = []
    #     if split is not None:
    #         with open(split) as f:
    #             for line in f:
    #                 img_name = line.strip()
    #                 img_info = dict(filename=img_name + img_suffix)
    #                 if ann_dir is not None:
    #                     seg_map = img_name + seg_map_suffix
    #                     img_info['ann'] = dict(seg_map=seg_map)
    #                 img_infos.append(img_info)
    #     else:
    #         for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
    #             img_info = dict(filename=img)  # 存放底图文件名
    #             if ann_dir is not None:
    #                 seg_map = img.replace(img_suffix, seg_map_suffix)
    #                 img_info['ann'] = dict(seg_map=seg_map)  # 存放标签文件名
    #             img_infos.append(img_info)
    #
    #     print_log(
    #         f'Loaded {len(img_infos)} images from {img_dir}',
    #         logger=get_root_logger())
    #     return img_infos

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            fold = self.fold
            print('-----------------------------------------------------------fold:{}'.format(fold))
            if split == 'train':
                df = pd.read_csv('/workspace/mmsegmentation_rsipac/data_config/split.csv')
                for row in range(len(df)):
                    if int(df['fold'][row]) != fold:
                        img = df['name'][row]
                        img_info = dict(filename=img)  # 存放底图文件名
                        if ann_dir is not None:
                            seg_map = img.replace(img_suffix, seg_map_suffix)
                            img_info['ann'] = dict(seg_map=seg_map)  # 存放标签文件名
                        img_infos.append(img_info)

            if split == 'val':
                df = pd.read_csv('/workspace/mmsegmentation_rsipac/data_config/split.csv')
                for row in range(len(df)):
                    if int(df['fold'][row]) == fold:
                        img = df['name'][row]
                        img_info = dict(filename=img)  # 存放底图文件名
                        if ann_dir is not None:
                            seg_map = img.replace(img_suffix, seg_map_suffix)
                            img_info['ann'] = dict(seg_map=seg_map)  # 存放标签文件名
                        img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)   #存放底图文件名
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)   #存放标签文件名
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        # debug = self.pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        # debug = self.pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        self.gt_encode_label(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def pre_eval(self, preds, indices, metric=None):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            if metric is not None:
                metric.update(pred, seg_map)
            else:
                pre_eval_results.append(
                    intersect_and_union(pred, seg_map, len(self.CLASSES),
                                        self.ignore_index, None,  # self.label_map
                                        self.reduce_zero_label))  # self.reduce_zero_label
        if metric is not None:
            return metric
        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if len(results) == 1:
            Metrics = SegmentationMetric(len(self.CLASSES), None)
            ret_metrics = Metrics.get_result(results[0])

            ret_metrics_class = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
                if ret_metric in ['IoU', 'Precision', 'Recall', 'F1']})
            ret_metrics_class.update({'Class': self.CLASSES})
            ret_metrics_class.move_to_end('Class', last=False)
            class_table_data = PrettyTable()
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)

            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
                if ret_metric in ['pixAcc', 'mIoU', 'FWIoU']})
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            # print_log('+++++++++++++++++++++++++++++++++++++++++++++++++++++++', logger)
            print_log('per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

            eval_results = OrderedDict({'Acc': ret_metrics['pixAcc']})
            if metric == 'mIoU':
                eval_results['mIoU'] = ret_metrics['mIoU']
                eval_results['IoU'] = ret_metrics['IoU']
            elif metric == 'Fscore':
                pass
                eval_results['Fscore'] = ret_metrics['f1']
                eval_results['Precision'] = ret_metrics['precision']
                eval_results['Recall'] = ret_metrics['recall']
            elif metric == 'FWIoU':
                eval_results['IoU'] = ret_metrics['IoU']
                eval_results['FWIoU'] = ret_metrics['FWIoU']

            '''write as csv'''
            metric_dict = {}
            for key, value in ret_metrics_summary.items():
                metric_dict[key] = [value]

            class_names = ret_metrics_class.pop('Class', None)
            for key, value in ret_metrics_class.items():
                metric_dict.update({
                    key + '.' + str(name): [value[idx]]
                    for idx, name in enumerate(class_names)
                })

            df = pd.DataFrame(metric_dict)
            out_file = r'/data_zs/output/loveDA_uda/rural2urban/st-sfda_roi-prop_loveda_rural2urban_deeplab_resnet_512x512_b4_dtu_lr6e-5_total_augv2_dt-st/metric.csv' #r'/irsa/data_zs/output/tmp_debug/onlysource_potsdamIRRG_deeplabv2_resnet101_512x512_lr5e-4_b4_vaihingen-train-val_pmd/metric.csv' #r'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_dfc22_convnext_mitb2_512x512_b4_pmd-albu_addauxdecode_train-val-Clermont-Ferrand_3004_1231/metric.csv'
            if not osp.exists(out_file):
                df.to_csv(out_file, index=False)
            else:
                df.to_csv(out_file, mode='a', index=False, header=False)

            return eval_results

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })
        # print('--------------------------------eval_results:{}'.format(eval_results))


        return eval_results

    def decode_segmap(self, mask, class_index=None):

        assert len(mask.shape) == 2, "the len of mask unexpect"
        # assert cfg.DATASET.NUM_CLASSES == len(cfg.DATASET.CLASS_INDEX[0]), "the value of NUM_CLASSES do not equal the len of CLASS_INDEX"
        height, width = mask.shape

        if isinstance(class_index[0], int):
            decode_mask = np.zeros((height, width), dtype=np.uint16)

            for pixel_value, class_index in class_index.items():  # range(self.config.num_classes):
                decode_mask[mask == pixel_value] = class_index
        else:
            decode_mask = np.zeros((height, width, 3), dtype=np.uint8)
            for pixel, color in class_index.items():
                if isinstance(color, list) and len(color) == 3:
                    decode_mask[np.where(mask == int(pixel))] = color
                else:
                    print("unexpected format of color_map in the config json:{}".format(color))

        return decode_mask.astype(np.uint8)  #[:, :, ::-1]  #.astype(dtype=np.uint32)


    def show_result(self, pred, out_file, label_map=None):
        if label_map is None:
            label_map = self.label_map
        mask = self.decode_segmap(pred, label_map)   #[:, :, ::-1]
        cv2.imwrite(out_file, mask)  # .replace('.tif', '.png')
