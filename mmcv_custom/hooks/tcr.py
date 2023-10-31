from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner import BaseModule, auto_fp16
from .logger import log_every_n
from torch.nn import Module
from copy import deepcopy
import torch
import torch.nn.functional as F
import mmcv
import os
import os.path as osp
import time
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import pickle

from mmcv.parallel import collate, scatter
from mmcv.runner.iter_based_runner import IterLoader
from mmseg.datasets import build_dataloader, build_dataset

@HOOKS.register_module()
class TCRHook(Hook):
    def __init__(self,
                 output_dir,
                 data_train,
                 data_pseudo,
                 topk_candidate=0.5,
                 update_frequency=3000,
                 start=None,
                 by_epoch=False
                 ):
        self.data_train = data_train
        self.data_presudo_test = data_pseudo
        self.topk_candidate = topk_candidate
        self.update_frequency = update_frequency
        self.output_dir = output_dir
        self.by_epoch = by_epoch
        self.start = start
        self.save_folder = os.path.join(self.output_dir, 'CTR')
        os.makedirs(self.save_folder, exist_ok=True)
        self.save_folder_O = os.path.join(self.output_dir, 'CTR_O')
        os.makedirs(self.save_folder_O, exist_ok=True)

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        device = next(model.parameters()).device
        data_test = self.data_presudo_test['pseudo_test']
        data_test['pseudo'] = False
        data_test['test_mode'] = True
        dataset = build_dataset(data_test)
        data_loader = build_dataloader(dataset,
                                             1,
                                             4,
                                             1,
                                             dist=False,
                                             shuffle=False)

        prog_bar = mmcv.ProgressBar(len(dataset))
        loader_indices = data_loader.batch_sampler

        name_list = []
        predicted_label = np.zeros((len(dataset), 256, 512))
        predicted_prob = np.zeros((len(dataset), 256, 512))

        for batch_indices, data in zip(loader_indices, data_loader):
            with torch.no_grad():
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                name = img_metas[0]['ori_filename']
                if next(model.parameters()).is_cuda:
                    # scatter to specified GPU
                    data = scatter(data, [device])[0]
                else:
                    data['img_metas'] = [i.data[0] for i in data['img_metas']]

                probs = self.inference(model, **data)
                pred = probs.max(1)[1]
                prob = probs.max(1)[0]
                predicted_prob[batch_indices] = F.interpolate(prob.unsqueeze(0).float(), size=[256, 512]).cpu().numpy().squeeze()
                predicted_label[batch_indices] = F.interpolate(pred.unsqueeze(0).float(), size=[256, 512]).cpu().numpy().squeeze()
                name_list.append(name)
                prog_bar.update()

        thres = []
        for i in range(model.num_classes):
            x = predicted_prob[predicted_label == i]
            # print('------------------------x:{}'.format(x.shape))
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[int(np.round(len(x) * self.topk_candidate))])
        thres = np.array(thres)
        runner.logger.info(f'init prob thres is {thres}')

        for index in range(len(name_list)):
            name = name_list[index]
            label = predicted_label[index]
            output = np.asarray(label, dtype=np.uint8)
            mask = cv2.resize(output, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)

            out_file = osp.join(self.save_folder_O, name)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            # print(f'--------------out_file:{out_file}')
            cv2.imwrite(out_file.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png'), mask)

            prob = predicted_prob[index]
            for i in range(model.num_classes):
                label[(prob<thres[i])*(label==i)] = 255
            output = np.asarray(label, dtype=np.uint8)
            mask = cv2.resize(output, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)

            out_file = osp.join(self.save_folder, name)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            # print(f'--------------out_file_:{out_file}')
            cv2.imwrite(out_file.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png'), mask)
        runner.logger.info('run_candidate over !!!')

        out_file_ctr = osp.join(self.output_dir, 'CTR.p')
        out_file_ctr_o = osp.join(self.output_dir, 'CTR_O.p')
        self.gen_lb_info(self.save_folder, out_file_ctr, model.num_classes)
        self.gen_lb_info(self.save_folder, out_file_ctr_o, model.num_classes)

        data_train = self.data_train['train']
        data_train['max_iters'] = 40000
        data_train['tcr_file'] = out_file_ctr
        data_train['tcr_o_file'] = out_file_ctr_o
        dataset_train = build_dataset(data_train)
        data_loader_train = build_dataloader(dataset_train,
                                       self.data_train['samples_per_gpu'],
                                       self.data_train['workers_per_gpu'],
                                       1,
                                       seed=42,
                                       dist=False,
                                       shuffle=True)

        # runner.run([data_loader_train], [('train', 1)])
        runner.iter_loaders = [IterLoader(data_loader_train)]

    def after_train_iter(self, runner):
        if not self.by_epoch and self._should_evaluate(runner):
            self.run_candidate(runner)

    def after_train_epoch(self, runner):
        if self.by_epoch and self._should_evaluate(runner):
            self.run_candidate(runner)


    def run_candidate(self, runner, init_candidate=False):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        device = next(model.parameters()).device
        data_test = self.data_presudo_test['pseudo_test']
        data_test['pseudo'] = True
        data_test['test_mode'] = True
        data_test['pseudo_dir'] = self.save_folder_O
        dataset = build_dataset(data_test)

        data_loader = build_dataloader(dataset,
                                       1,
                                       4,
                                       1,
                                       dist=False,
                                       shuffle=False)

        prog_bar = mmcv.ProgressBar(len(dataset))
        loader_indices = data_loader.batch_sampler

        name_list = []
        predicted_label = np.zeros((len(dataset), 512, 1024))
        single_iou_list = np.zeros((len(dataset), model.num_classes))

        for batch_indices, data in zip(loader_indices, data_loader):
            with torch.no_grad():
                img_tensor = data['img'][0]
                # print(f'------------@@@@----------------batch_indices:{batch_indices}')
                y = self.get_gt(dataset, batch_indices)
                y = torch.from_numpy(y).to(device)
                size = y.shape[-2:]

                img_metas = data['img_metas'][0].data[0]
                name = img_metas[0]['ori_filename']
                if next(model.parameters()).is_cuda:
                    # scatter to specified GPU
                    data = scatter(data, [device])[0]
                else:
                    data['img_metas'] = [i.data[0] for i in data['img_metas']]

                probs = self.inference(model, **data)
                pred = probs.max(1)[1]
                # print(f'-----------@@@@@-----------------pred:{pred.shape},y:{y.shape}')
                intersection, union, target = self.intersectionAndUnionGPU(pred.clone(), y, model.num_classes)
                single_iou = intersection / (union + 1e-8)
                single_iou_list[batch_indices] = single_iou.cpu().numpy()
                predicted_label[batch_indices] = F.interpolate(pred.unsqueeze(0).float(),
                                                               size=[xx // 2 for xx in size]).cpu().numpy().squeeze()
                name_list.append(name)
                prog_bar.update()

        thres = []
        for i in range(model.num_classes):
            x = single_iou_list[:, i]
            x = x[x > 0]
            x = np.sort(x)
            if len(x) == 0:
                thres.append(0)
            else:
                thres.append(x[int(np.round(len(x) * self.topk_candidate))])
        thres = np.array(thres)
        runner.logger.info(f'ReL thres is:{thres}')

        for index in range(len(name_list)):
            name = name_list[index]
            label = predicted_label[index]
            t = np.asarray(label, dtype=np.uint8)
            mask = cv2.resize(t, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

            out_file = osp.join(self.save_folder_O, name)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            cv2.imwrite(out_file.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png'), mask)

            ReL = single_iou_list[index]
            for i in range(model.num_classes):  ## masking
                if ReL[i]<thres[i]:
                    label[label==i] = 255
            output = np.asarray(label, dtype=np.uint8)
            mask = cv2.resize(output, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

            out_file = osp.join(self.save_folder, name)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            cv2.imwrite(out_file.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png'), mask)
        runner.logger.info('run_candidate over !!!')

        out_file_ctr = osp.join(self.output_dir, 'CTR.p')
        out_file_ctr_o = osp.join(self.output_dir, 'CTR_O.p')
        self.gen_lb_info(self.save_folder, out_file_ctr, model.num_classes)
        self.gen_lb_info(self.save_folder, out_file_ctr_o, model.num_classes)

        data_train = self.data_train['train']
        data_train['max_iters'] = 40000
        data_train['tcr_file'] = out_file_ctr
        data_train['tcr_o_file'] = out_file_ctr_o
        dataset_train = build_dataset(data_train)
        data_loader_train = build_dataloader(dataset_train,
                                             self.data_train['samples_per_gpu'],
                                             self.data_train['workers_per_gpu'],
                                             1,
                                             seed=42,
                                             dist=False,
                                             shuffle=True)
        # runner.run([data_loader_train], [('train', 1)])
        runner.iter_loaders = [IterLoader(data_loader_train)]

    def intersectionAndUnionGPU(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        # https://github.com/pytorch/pytorch/issues/1382
        area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
        area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
        area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target

    # @auto_fp16(apply_to=('img',))
    def inference(self, model, img, img_metas, rescale=True, **kwargs):
        self.format_vaildate(img, img_metas)
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = model.inference(img[0], img_metas[0], rescale)
        for i in range(1, len(img)):
            cur_seg_logit = model.inference(img[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(img)
        # seg_pred = seg_logit.argmax(dim=1)
        # seg_pred = seg_pred.cpu().numpy()
        # # unravel batch dim
        # seg_pred = list(seg_pred)
        return seg_logit  # seg_pred

    def format_vaildate(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.update_frequency):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.update_frequency:
                return False
        return True

    def get_gt(self, dataset, indices):
        sep_map_list = []
        for index in indices:
            seg_map = dataset.get_gt_seg_map_by_idx(index)
            sep_map_list.append(seg_map)

        return np.stack(sep_map_list, axis=0)

    def gen_lb_info(self, img_path, out_file, num_classes, nprocs=1, suffix='.png'):

        labfiles = []
        file_client = mmcv.FileClient.infer_client(dict(backend='disk'))
        for img in file_client.list_dir_or_file(
                dir_path=img_path,
                list_dir=False,
                suffix=suffix,
                recursive=True):
            labfiles.append(img)

        id_to_trainid = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            13: 13,
            14: 14,
            15: 15,
            16: 16,
            17: 17,
            18: 18
        }

        label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
        file_to_label = {e: [] for e in labfiles}

        def generate_label_info():
            label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
            file_to_label = {e.strip(): [] for e in labfiles}

            for labfile in tqdm(labfiles):
                label_ = np.array(cv2.imread(os.path.join(img_path, labfile)), dtype=np.uint8)
                label = np.unique(label_)
                # print(label)

                for lab in label:
                    if 255 == lab: continue
                    if np.sum(label_ == lab) < 50: continue  # 像素个数小于50
                    label_to_file[lab].append(labfile)
                    file_to_label[labfile].append(
                        lab)

            return label_to_file, file_to_label

        def _foo(i):
            label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
            file_to_label = dict()
            labfile = labfiles[i]
            file_to_label[labfile] = []
            label = np.unique(cv2.imread(os.path.join(img_path, labfile)), dtype=np.float32)
            for lab in label:
                label_to_file[int(lab)].append(labfile)
                file_to_label[labfile].append(lab)
            return label_to_file, file_to_label

        if nprocs == 1:
            label_to_file, file_to_label = generate_label_info()
        else:
            with Pool(nprocs) as p:
                r = list(tqdm(p.imap(_foo, range(len(labfiles))), total=len(labfiles)))
            for l2f, f2l in r:
                for lab in range(len(l2f)):
                    label_to_file[lab].extend(l2f[lab])
                for fname in f2l.keys():
                    if fname in file_to_label:
                        file_to_label[fname].extend(f2l[fname])

        with open(out_file, 'wb') as f:
            pickle.dump((label_to_file, file_to_label), f)