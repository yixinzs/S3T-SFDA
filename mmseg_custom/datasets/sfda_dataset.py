import json
import os.path as osp

import mmcv
import numpy as np
import torch


from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose

@DATASETS.register_module()
class SFDADataset(object):
    def __init__(self, **cfg):
        self.target = DATASETS.build(cfg['target'])
        self.CLASSES = self.target.CLASSES
        self.PALETTE = self.target.PALETTE
        self.pipeline = Compose(cfg['pipeline'])

    def __getitem__(self, idx):
        pass
        result = self.target[idx]
        result = self.pipeline(result)

        return result


    def __len__(self):
        return len(self.target)