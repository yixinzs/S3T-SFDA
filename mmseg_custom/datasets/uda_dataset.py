import json
import os.path as osp

import mmcv
import numpy as np
import torch


from mmseg.datasets.builder import DATASETS

def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}  #整体数据集的各类别占比
    for s in sample_class_stats:  #遍历所有标签文件，并进行统计
        s.pop('file')
        for c, n in s.items():   #c为此文件中对应类别train_id, n为此类别对应的像素个数
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }      #将类别按像素个数排列   #生成的新字典的value值，应该越来越大
    freq = torch.tensor(list(overall_class_stats.values()))   #取字典的value，列表元素值应该递增
    freq = freq / torch.sum(freq)     #列表元素值应该递增，总和为1
    freq = 1 - freq    #列表元素值应该递减
    freq = torch.softmax(freq / temperature, dim=-1)   #列表元素值应该递减，，temperature越大，曲线越应该越平滑（推测）

    return list(overall_class_stats.keys()), freq.numpy()   #overall_class_stats：返回排序后的类别列表，以及对应类别的采样权重列表


@DATASETS.register_module()
class UDADataset(object):

    def __init__(self, **cfg):
        self.source = DATASETS.build(cfg['source'])
        self.target = DATASETS.build(cfg['target'])
        self.ignore_index = self.target.ignore_index
        self.CLASSES = self.target.CLASSES
        self.PALETTE = self.target.PALETTE
        assert self.target.ignore_index == self.source.ignore_index
        assert self.target.CLASSES == self.source.CLASSES
        assert self.target.PALETTE == self.source.PALETTE

        rcs_cfg = cfg['rare_class_sampling']  #min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                    osp.join(cfg['source']['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}   #samples_with_class:针对每个类别，将标签文件中此类别像素个数大于self.rcs_min_pixels的文件筛选出来
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:  #samples_with_class_and_n[c]：类别c对应的标签文件列表（file，n）
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])  #file.split('/')[-2] + '/' + file.split('/')[-1]  #file.split('/')[-1]
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                # if isinstance(self.source, SourceDataset):
                #     file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]   #self.source---Dataset形式类，self.source[i1]选择索引为i1的数据，由self.source的__getitem__()输出得到，
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:  #self.source[i1]数据增强裁剪后，c类可能会裁掉，如果裁掉重新选择s1
                    break
                s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]

        return {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']
        }

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.target[idx % len(self.target)]
            return {
                **s1, 'target_img_metas': s2['img_metas'],
                'target_img': s2['img']
            }

    def __len__(self):
        return len(self.source) * len(self.target)
