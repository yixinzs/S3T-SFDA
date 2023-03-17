# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss

from typing import Optional
from torch import nn, Tensor

from abc import ABC
from .soft_cross_entropy_loss import SoftCrossEntropyLoss

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, temperature=0.07, base_temperature=0.07, max_samples=1024, max_views=18, ignore_label=255):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_label

        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(X.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(y.device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().to(labels_.device)

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).to(mask.device),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        # print('------------------------feats:{}'.format(feats.shape))
        # print('------------------------labels:{}'.format(labels.shape))
        # print('------------------------predict:{}'.format(predict.shape))
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # print('------------------------feats:{}'.format(feats.shape))
        # print('------------------------labels:{}'.format(labels.shape))
        # print('------------------------predict:{}'.format(predict.shape))
        # print('------------------------feats_v3:{}'.format(feats.device))
        # print('------------------------labels_v3:{}'.format(labels.device))
        # print('------------------------predict:{}'.format(predict.device))
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        # print('------------------------feats_v2:{}'.format(feats_.device))
        # print('------------------------labels_v2:{}'.format(labels_.device))
        loss = self._contrastive(feats_, labels_)
        # print('------------------------loss_v2:{}'.format(loss.device))
        return loss

@LOSSES.register_module()
class ContrastCELoss(nn.Module, ABC):
    def __init__(self, loss_weight=0.1, ignore_index=255, loss_name='loss_contrast_ce'):
        super(ContrastCELoss, self).__init__()

        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name

        self.seg_criterion = SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=255, loss_weight=1.0)
        self.contrast_criterion = PixelContrastLoss()

    def forward(self, preds, target,
                class_weight=None,
                weight=None,
                **kwargs
                ):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds

        seg = preds['seg']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        return loss + self.loss_weight * loss_contrast

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

@LOSSES.register_module()
class ContrastLoss(nn.Module, ABC):
    def __init__(self, loss_weight=0.1, ignore_index=255, loss_name='loss_contrast'):
        super(ContrastLoss, self).__init__()

        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name

        self.contrast_criterion = PixelContrastLoss()

    def forward(self, preds,
                target: torch.Tensor,
                class_weight=None,
                weight=None,
                **kwargs
                ) -> torch.Tensor:
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds

        seg = preds['seg']
        embedding = preds['embed']
        # print('------------------------seg:{}'.format(seg.device))
        # print('------------------------embedding:{}'.format(embedding.device))
        # print('------------------------target:{}'.format(target.device))
        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        # print('------------------------loss_contrast:{}'.format(loss_contrast.device))
        return self.loss_weight * loss_contrast

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name



def label_smoothed_nll_loss(
        lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


@LOSSES.register_module()
class SoftCrossEntropyLossV2(nn.Module):
    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
            self,
            reduction: str = "mean",
            smooth_factor: Optional[float] = None,
            ignore_index: Optional[int] = 255,
            dim: int = 1,
            loss_weight=1.0,
            loss_name='loss_softce'
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        self._loss_name = loss_name
        self.loss_weight = loss_weight

    def forward(self, y_preds,
                y_true: torch.Tensor,
                class_weight=None,
                weight=None,
                **kwargs
                ) -> torch.Tensor:

        assert "seg" in y_preds
        y_pred = y_preds['seg']
        # print('------------------------y_pred:{}'.format(y_pred.device))
        # print('------------------------y_true:{}'.format(y_true.device))
        y_pred = F.interpolate(input=y_pred, size=y_true.shape[-2:], mode='bilinear', align_corners=True)
        # print('------------------------y_true:{}'.format(y_true.shape))
        # print('------------------------y_pred:{}'.format(y_pred.shape))
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        loss = label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        ) * self.loss_weight
        # print('------------------------loss:{}'.format(loss.device))
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name