# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align    #https://blog.csdn.net/rocking_struggling/article/details/131330266

from mmcv.parallel import MMDistributedDataParallel

from mmseg.core import add_prefix
from mmseg.ops import resize

from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor

from mmcv_custom.checkpoint import load_checkpoint
from mmseg.ops import resize

def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))



def pseudo_labels_probs(probs, running_conf, THRESHOLD_BETA, RUN_CONF_UPPER=0.80, ignore_augm=None,
                        discount=True):  # 查da-sac的公式2-公式5
    """Consider top % pixel w.r.t. each image"""

    RUN_CONF_UPPER = RUN_CONF_UPPER
    RUN_CONF_LOWER = 0.20

    B, C, H, W = probs.size()
    max_conf, max_idx = probs.max(1, keepdim=True)  # B,1,H,W

    probs_peaks = torch.zeros_like(probs)
    probs_peaks.scatter_(1, max_idx, max_conf)  # B,C,H,W
    top_peaks, _ = probs_peaks.view(B, C, -1).max(-1)  # B,C

    # top_peaks 是一张图上每个类的最大置信度
    top_peaks *= RUN_CONF_UPPER

    if discount:
        # discount threshold for long-tail classes
        top_peaks *= (1. - torch.exp(- running_conf / THRESHOLD_BETA)).view(1, C)  #

    top_peaks.clamp_(RUN_CONF_LOWER)  # in-place
    probs_peaks.gt_(top_peaks.view(B, C, 1, 1))

    # ignore if lower than the discounted peaks
    ignore = probs_peaks.sum(1, keepdim=True) != 1

    # thresholding the most confident pixels
    pseudo_labels = max_idx.clone()
    pseudo_labels[ignore] = 255  #

    pseudo_labels = pseudo_labels.squeeze(1)
    # pseudo_labels[ignore_augm] = 255

    return pseudo_labels, max_conf, max_idx



def update_running_conf(probs, running_conf, THRESHOLD_BETA, tolerance=1e-8):
    """Maintain the moving class prior"""
    STAT_MOMENTUM = 0.9

    B, C, H, W = probs.size()
    probs_avg = probs.mean(0).view(C, -1).mean(-1)

    # updating the new records: copy the value
    update_index = probs_avg > tolerance
    new_index = update_index & (running_conf == THRESHOLD_BETA)
    running_conf[new_index] = probs_avg[new_index]

    # use the moving average for the rest (Eq. 2)
    running_conf *= STAT_MOMENTUM
    running_conf += (1 - STAT_MOMENTUM) * probs_avg
    return running_conf

def full2strong(feats, img_metas, down_ratio=1, nearest=False):
    tmp = []
    for i in range(feats.shape[0]):
        # print('---------------------:{}'.format(img_metas))
        #### rescale
        w, h = img_metas[i]['scale'][0], img_metas[i]['scale'][1]
        if nearest:
            feat_ = F.interpolate(feats[i:i+1], size=[int(h/down_ratio), int(w/down_ratio)])
        else:
            feat_ = F.interpolate(feats[i:i+1], size=[int(h/down_ratio), int(w/down_ratio)], mode='bilinear', align_corners=True)

        if img_metas[i]['flip']:
            if img_metas[i]['flip_direction'] == 'horizontal':
                inv_idx = torch.arange(feat_.size(3)-1,-1,-1).long().to(feat_.device)
                feat_ = feat_.index_select(3,inv_idx)
            elif img_metas[i]['flip_direction'] == 'vertical':
                inv_idx = torch.arange(feat_.size(2) - 1, -1, -1).long().to(feat_.device)
                feat_ = feat_.index_select(2, inv_idx)

        #### then crop
        y1, y2, x1, x2 = img_metas[i]['crop_bbox'][0], img_metas[i]['crop_bbox'][1], img_metas[i]['crop_bbox'][
            2], img_metas[i]['crop_bbox'][3]
        y1, th, x1, tw = int(y1 / down_ratio), int((y2 - y1) / down_ratio), int(x1 / down_ratio), int(
            (x2 - x1) / down_ratio)
        feat_ = feat_[:, :, y1:y1 + th, x1:x1 + tw]

        tmp.append(feat_)
    feat = torch.cat(tmp, 0)
    return feat


def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


class BinaryCrossEntropy(torch.nn.Module):

    def __init__(self, size_average=True, ignore_index=255):
        super(BinaryCrossEntropy, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_index

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.binary_cross_entropy_with_logits(predict, target.unsqueeze(-1), pos_weight=weight, size_average=self.size_average)
        return loss

class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

    # def step(self):
    #     one_minus_alpha = 1.0 - self.alpha
    #     for p, src_p in zip(self.params, self.src_params):
    #         p.data.mul_(self.alpha)
    #         p.data.add_(src_p.data * one_minus_alpha)

    def step(self, iter_=None):
        if iter_ is None:
            alpha_= self.alpha
            one_minus_alpha = 1.0 - self.alpha
        else:
            alpha_ = min(1 - 1 / (iter_ + 1), self.alpha)
            one_minus_alpha = 1.0 - alpha_
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(alpha_)
            p.data.add_(src_p.data * one_minus_alpha)

@SEGMENTORS.register_module()
class MVCSFDAEncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, **cfg):
        super(MVCSFDAEncoderDecoder, self).__init__()
        # if pretrained is not None:
        #     assert backbone.get('pretrained') is None, \
        #         'both backbone and segmentor set pretrained weight'
        #     backbone.pretrained = pretrained
        self.local_iter = 0
        self.dtu_dynamic = cfg['dtu_dynamic']
        self.dtu_query_step = cfg['query_step']
        self.dtu_query_start = cfg['query_start']
        self.dtu_meta_max_update = cfg['meta_max_update']
        self.dtu_proxy_metric = cfg['proxy_metric']
        self.dtu_ema_weight = cfg['ema_weight']
        self.dtu_fix_iteration = cfg['fix_iteration']
        self.topk_candidate = cfg['topk_candidate']
        self.update_frequency = cfg['update_frequency']
        self.threshold_beta = cfg['threshold_beta']
        self.roi_size = cfg['roi_size'] #(512, 1024)

        model = build_segmentor(deepcopy(cfg['model']))
        self.model = get_module(model)
        # print('-------------resume_file:{}'.format(cfg['resume_file']))
        load_checkpoint(self.model, cfg['resume_file'])

        ema_model = build_segmentor(deepcopy(cfg['model']))
        self.ema_model =get_module(ema_model)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.binary_ce = BinaryCrossEntropy(ignore_index=255)

        self.running_conf = torch.zeros(self.model.decode_head.num_classes) #.to(torch.device('cuda:1')) #.cuda() #TODO:确认是否需要加cuda(),或者是否可以把running_conf分布到多卡上
        self.running_conf.fill_(self.threshold_beta)
        if self.dtu_dynamic:
            self.stu_eval_list = []
            self.stu_score_buffer = []
            self.res_dict = {'stu_ori': [], 'stu_now': [], 'update_iter': []}

        self.ema_model_optimizer = WeightEMA(list(self.ema_model.parameters()), list(self.model.parameters()), self.dtu_ema_weight)

    def _init_ema_weights(self):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(self.model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()   #初始化，直接将权重赋给动量编码器
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def extract_feat(self, img):
        """Extract features from images."""
        return self.model.extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.model.encode_decode(img, img_metas)

    @property
    def num_classes(self):
        return self.model.decode_head.num_classes


    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def evel_stu(self, stu_eval_list, dtu_proxy_metric, device=torch.device('cuda:0')):
        eval_result = []
        # self.eval()
        with torch.no_grad():
            for i, (x, permute_index) in enumerate(stu_eval_list):
                # print('------------------------i:{},x:{},permute_index:{}'.format(i, x.shape, permute_index))
                output = self.model.decode_head.forward(self.model.extract_feat(x.to(device))) #  .cuda()  .to(device)  torch.device('cuda:1')
                output = resize(input=output, size=x.shape[2:], mode='bilinear', align_corners=self.model.align_corners)
                output = F.softmax(output, dim=1)
                if dtu_proxy_metric == 'ENT':
                    out_max_prob = output.max(1)[0]
                    uc_map_prob = 1 - (-torch.mul(out_max_prob, torch.log2(out_max_prob)) * 2)
                    eval_result.append(uc_map_prob.mean().item())
                elif dtu_proxy_metric == 'SND':
                    pred1 = output.permute(0, 2, 3, 1)
                    pred1 = pred1.reshape(-1, pred1.size(3))
                    pred1_rand = permute_index
                    # select_point = pred1_rand.shape[0]
                    select_point = 100
                    pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                    pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
                    eval_result.append(pred1_en.item())
        # self.train()
        return eval_result

    def get_rois(self, imgs, img_metas):

        device = imgs.device

        img_rois_list = []
        for i in range(imgs.shape[0]):
            img_list, feat_list = [], []
            img_rois = img_metas[i]['crop_bbox_list']
            img_rois = torch.from_numpy(img_rois).to(device).float()
            img_rois_list.append(img_rois)

        return img_rois_list

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      img_full=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        device = img.device

        if self.local_iter == 0:
            self._init_ema_weights()
            self.running_conf = self.running_conf.to(device)

        # x = self.model.extract_feat(img)
        # target_pred = self.model.decode_head.forward(x)
        # target_pred = resize(input=target_pred, size=img.shape[2:], mode='bilinear', align_corners=self.model.align_corners)
        if self.dtu_dynamic:
            with torch.no_grad():
                x_full = self.model.extract_feat(img_full.detach())
                target_pred_full = self.model.decode_head.forward(x_full)
                target_pred_full = resize(input=target_pred_full, size=img_full.shape[2:], mode='bilinear', align_corners=self.model.align_corners)
                output = F.softmax(target_pred_full.clone().detach(), dim=1).detach()
                if self.dtu_proxy_metric == 'ENT':
                    out_max_prob = output.max(1)[0]
                    uc_map_prob = 1 - (-torch.mul(out_max_prob, torch.log2(out_max_prob)) * 2)
                    self.stu_score_buffer.append(uc_map_prob.mean().item())
                    self.stu_eval_list.append([img_full.clone().detach().cpu()])
                elif self.dtu_proxy_metric == 'SND':
                    pred1 = output.permute(0, 2, 3, 1)
                    pred1 = pred1.reshape(-1, pred1.size(3))
                    pred1_rand = torch.randperm(pred1.size(0))
                    # select_point = pred1_rand.shape[0]
                    select_point = 100
                    pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                    pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
                    self.stu_score_buffer.append(pred1_en.item())
                    self.stu_eval_list.append([img_full.clone().detach().cpu(), pred1_rand.cpu()])
                else:
                    print('no support')
                    return

        rois_coords = self.get_rois(img, img_metas)
        # print(f'----------------------------img:{img.shape}')
        # print(f'-------------img_type:{img_full.dtype},img_shape:{img_full.shape},rois_type:{rois_coords[0].dtype},rois_shape:{rois_coords[0].shape}')
        rois_imgs = roi_align(img, rois_coords, self.roi_size)   #roi_align原理：https://zhuanlan.zhihu.com/p/73113289
        # print(f'-----------------rois_images_shape:{rois_imgs.shape}')
        x = self.model.extract_feat(rois_imgs)   #TODO:rois_imgs  img
        target_pred = self.model.decode_head.forward(x)
        target_pred = resize(input=target_pred, size=rois_imgs.shape[2:], mode='bilinear', align_corners=self.model.align_corners)  #TODO:rois_imgs  img
        with torch.no_grad():

            size = img_full.shape[-2:]
            ema_x_full = self.ema_model.extract_feat(img_full.detach())
            ema_target_pred_full = self.ema_model.decode_head.forward(ema_x_full)
            ema_target_pred_full = resize(input=ema_target_pred_full, size=img_full.shape[2:], mode='bilinear', align_corners=self.model.align_corners)
            ema_target_prob = F.softmax(full2strong(ema_target_pred_full, img_metas), dim=1)
            # print(f'---------------------ema_target_prob_shape:{ema_target_prob.shape}')
            rois_prob = roi_align(ema_target_prob, rois_coords, self.roi_size)   #TODO:
            # print(f'---------------------rois_prob:{rois_prob.shape}')
            # pos label
            running_conf = update_running_conf(F.softmax(ema_target_pred_full, dim=1), self.running_conf, self.threshold_beta)
            psd_label, _, _ = pseudo_labels_probs(rois_prob, running_conf, self.threshold_beta)  #TODO:rois_prob  ema_target_prob

            b, c, h, w = rois_prob.size()  #TODO:rois_prob  ema_target_prob
            t_neg_label = rois_prob.clone().detach().view(b * c, h, w)  #TODO:rois_prob  ema_target_prob
            rois_prob = rois_prob.view(b * c, h, w)    #TODO:rois_prob   rois_prob   ema_target_prob = ema_target_prob.view
            thr = 1 / self.model.decode_head.num_classes
            t_neg_label[rois_prob > thr] = 255   #TODO:rois_prob  ema_target_prob
            t_neg_label[rois_prob <= thr] = 0    #TODO:rois_prob  ema_target_prob

            # label mix
            # psd_label = psd_label * (mix_label == 255) + mix_label * ((mix_label != 255))
            uc_map_eln = torch.ones_like(psd_label).float()

        target_losses = dict()
        # print('--------------------------target_pred:{}'.format(target_pred.shape))
        # print('--------------------------psd_label:{}'.format(psd_label.shape))
        # st_loss = self.model.decode_head.losses(target_pred, psd_label.unsqueeze(1).long())  #调用mmseg内部损失含函数
        st_loss = self.criterion(target_pred, psd_label.long())
        pesudo_p_loss = (st_loss * (-(1 - uc_map_eln)).exp()).mean()  #TODO:删掉是否合理
        target_losses['pesudo_pos.loss'] = pesudo_p_loss
        pesudo_n_loss = self.binary_ce(target_pred.view(b * c, 1, h, w), t_neg_label) * 1  # 忽略255，专注于负样本，即t_neg_label[tgt_prob_his <= thr] = 0部分
        target_losses['pesudo_neg.loss'] = pesudo_n_loss
        # target_losses.update(add_prefix(st_loss, 'decode'))    #调用mmseg内部损失含函数

        losses, target_log_vars = self._parse_losses(target_losses)
        log_vars.update(target_log_vars)
        losses.backward()

        if self.dtu_dynamic:
            if len(self.stu_score_buffer) >= self.dtu_query_start and int(len(self.stu_score_buffer) - self.dtu_query_start) % self.dtu_query_step == 0:
                all_score = self.evel_stu(self.stu_eval_list, self.dtu_proxy_metric, device)
                compare_res = np.array(all_score) - np.array(self.stu_score_buffer)
                if np.mean(compare_res > 0) > 0.5 or len(self.stu_score_buffer) > self.dtu_meta_max_update:
                    update_iter = len(self.stu_score_buffer)

                    self.ema_model_optimizer.step()   #self.local_iter


                    self.res_dict['stu_ori'].append(np.array(self.stu_score_buffer).mean())
                    self.res_dict['stu_now'].append(np.array(all_score).mean())
                    self.res_dict['update_iter'].append(update_iter)

                    df = pd.DataFrame(self.res_dict)
                    df.to_csv('dyIter_FN.csv')

                    ## reset
                    self.stu_eval_list = []
                    self.stu_score_buffer = []
        else:
            if self.local_iter % self.dtu_fix_iteration == 0:
                self.ema_model_optimizer.step()  #self.local_iter

        self.local_iter += 1
        return log_vars


    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        return self.model.inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""

        return self.model.simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now

        return self.model.aug_test(imgs, img_metas, rescale)
