# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

# from .builder import DATASETS
from mmseg.datasets.builder import DATASETS
# from .custom import CustomDataset
from .custom_base import CustomBaseDataset

@DATASETS.register_module()
class RSIPACDataset(CustomBaseDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('水体', '道路', '建筑物', '机场', '火车站',
             '光伏', '停车场', '操场', '普通耕地', '农业大棚',
             '自然草地', '绿地绿化', '自然林', '人工林', '自然裸土',
             '人为裸土', '其它无法确定归属地物	')
    # background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7. And the no-data regions were assigned 0 which should be ignored.
    PALETTE = None  # [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],

    # [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    # [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    # [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    # [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    # background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7. And the no-data regions were assigned 0 which should be ignored.

    def __init__(self, **kwargs):
        super(RSIPACDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.png',
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files