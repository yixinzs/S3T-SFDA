import json
import os.path as osp

import mmcv
import numpy as np
import torch
import pickle
import os
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose


@DATASETS.register_module()
class SFDADataset(object):
    def __init__(self, **cfg):
        self.target = DATASETS.build(cfg['target'])
        self.CLASSES = self.target.CLASSES
        self.PALETTE = self.target.PALETTE
        self.pipeline = Compose(cfg['pipeline'])

        max_iters = cfg.get('max_iters', None)
        tcr_file = cfg.get('tcr_file', None)
        tcr_o_file = cfg.get('tcr_o_file', None)

        self.file_to_idx = {}
        for i, dic in enumerate(self.target.img_infos):
            file = dic['ann']['seg_map']
            self.file_to_idx[file] = i

        self.num_classes = len(self.CLASSES)
        self.img_ids = []
        if max_iters is None:
            self.img_ids = range(len(self.target))
        else:
            # print(f'-----------------------------max_iters----------------------:{max_iters}')
            assert os.path.exists(tcr_o_file), f'do not exist the file:{tcr_o_file}'
            self.label_to_file, self.file_to_label = pickle.load(open(tcr_o_file, "rb"))
            self.img_ids = []
            SUB_EPOCH_SIZE = 500
            tmp_list = []
            ind = dict()
            for i in range(self.num_classes):
                ind[i] = 0
            for e in range(int(max_iters / SUB_EPOCH_SIZE) + 1):
                cur_class_dist = np.zeros(self.num_classes)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist / cur_class_dist.sum()
                    w = 1 / np.log(1 + 1e-2 + dist1)
                    w = w / w.sum()
                    c = np.random.choice(self.num_classes, p=w)

                    if len(self.label_to_file[c]) == 0 or len(self.label_to_file[c]) == 1:
                        continue

                    if ind[c] > (len(self.label_to_file[c]) - 1):
                        np.random.shuffle(self.label_to_file[c])
                        ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)

                    if len(self.label_to_file[c]) == 0:
                        continue
                    c_file = self.label_to_file[c][ind[c]]
                    tmp_list.append(c_file)
                    self.img_ids.append(self.file_to_idx[c_file])
                    ind[c] = ind[c] + 1
                    cur_class_dist[self.file_to_label[c_file]] += 1

    def __getitem__(self, idx):
        # print(f'-------------------------idx:{idx}')
        # print(f'----------------------------img_ids:{len(self.img_ids)}')
        img_id = self.img_ids[idx]
        # print(f'-------------------------------img_id:{img_id}')
        result = self.target[img_id]
        result = self.pipeline(result)

        return result

    def __len__(self):
        return len(self.img_ids)
