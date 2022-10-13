"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np
from torch import distributed as dist
import copy
from collections import OrderedDict

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']

#https://blog.csdn.net/m0_47355331/article/details/119972157  可忽略指定类别的混淆矩阵计算

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, distributed):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.distributed = distributed
        self.reset()
        print('------------------------------------:{}'.format(self.nclass))
        #self.confusion_matrix = torch.zeros((nclass, nclass)).cuda()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def reduce_tensor(tensor):
            rt = tensor.clone()                          #dist.all_reduce(rt, op=dist.ReduceOp.SUM)：https://blog.csdn.net/husthy/article/details/108226256
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)    #dist.all_reduce：https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/dist_tuto.md
            return rt

        def evaluate_worker(self, pred, label):

            current_confusion_matrix = _confusion_matrix(pred, label, self.nclass)

            # print('---------------------current_confusion_matrix:{}'.format(current_confusion_matrix))
            if self.distributed:
                current_confusion_matrix = reduce_tensor(current_confusion_matrix.cuda())
            torch.cuda.synchronize()    #https://blog.csdn.net/qq_23981335/article/details/105709273

            if self.confusion_matrix.device != current_confusion_matrix.device:
                self.confusion_matrix = self.confusion_matrix.to(current_confusion_matrix.device)
            self.confusion_matrix += current_confusion_matrix

        if not isinstance(preds, torch.Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        evaluate_worker(self, preds, labels)
        # if isinstance(preds, torch.Tensor):
        #     evaluate_worker(self, preds, labels)
        # elif isinstance(preds, (list, tuple)):
        #     for (pred, label) in zip(preds, labels):
        #         evaluate_worker(self, pred, label)

    def F1_score(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        true_pos = torch.diag(confusion_matrix)
        false_pos = torch.sum(confusion_matrix, dim=0) - true_pos
        false_neg = torch.sum(confusion_matrix, dim=1) - true_pos

        precision = true_pos / (true_pos + false_pos + 1e-6)
        recall = true_pos / (true_pos + false_neg + 1e-6)

        return 2 * precision * recall / (precision + recall + 1e-6), torch.mean(2 * precision * recall / (precision + recall + 1e-6))

    def kappa(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        pe_rows = torch.sum(confusion_matrix, dim=0)
        pe_cols = torch.sum(confusion_matrix, dim=1)
        sum_total = sum(pe_cols)
        pe = torch.dot(pe_rows, pe_cols) / float(sum_total ** 2)

        po = torch.trace(confusion_matrix) / float(sum_total)

        return (po - pe) / (1 - pe)

    def pixel_recall(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        return torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=1) + 1e-6)

    def FW_IoU(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        freq = torch.sum(confusion_matrix, dim=1) / torch.sum(confusion_matrix)

        iou = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, dim=0) + torch.sum(confusion_matrix, dim=1) - torch.diag(confusion_matrix))

        return torch.sum(freq[freq > 0] * iou[freq > 0])

    def pixel_precision(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        return torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=0) + 1e-6)

    def pixel_accuracy(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        accuracy = torch.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-6)

        return accuracy


    def mean_IoU(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        true_pos = torch.diag(confusion_matrix)
        false_pos = torch.sum(confusion_matrix, dim=0) - true_pos
        false_neg = torch.sum(confusion_matrix, dim=1) - true_pos
        tp_fp_fn = true_pos + false_pos + false_neg

        # exist_class_mask = tp_fp_fn > 0
        # true_pos, tp_fp_fn = true_pos[exist_class_mask], tp_fp_fn[exist_class_mask]
        true_pos, tp_fp_fn = true_pos, tp_fp_fn + 1e-6
        return true_pos / tp_fp_fn, torch.mean(true_pos / tp_fp_fn)

    def get_conf_matrix(self):
        return [self.confusion_matrix]

    def get_result(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix

        pixAcc, pixAcc_class = self.pixel_accuracy(confusion_matrix), self.pixel_recall(confusion_matrix)
        IoU, mIoU = self.mean_IoU(confusion_matrix)
        FWIoU = self.FW_IoU(confusion_matrix)
        f1_class, fl_mean = self.F1_score(confusion_matrix)
        precision_class = self.pixel_precision(confusion_matrix)
        recall_class = self.pixel_recall(confusion_matrix)

        scores = OrderedDict({'pixAcc':pixAcc, 'Precision':precision_class, 'Recall': recall_class,
                  'F1':f1_class, 'mIoU': mIoU, 'FWIoU': FWIoU, 'IoU':IoU})
        # scores = [pixAcc, precision_class, recall_class, f1_class, mIoU, FWIoU]

        scores = {
            metric: value.numpy()
            for metric, value in scores.items()
        }

        return scores

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.confusion_matrix = torch.zeros((self.nclass, self.nclass), dtype=torch.float64).cuda() #torch.zeros((self.nclass, self.nclass)).cuda()


def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    #predict = torch.argmax(output.long(), 1) + 1
    predict = output.long() + 1
    target = target.long() + 1   #此处需要加.long()，否则遇到忽略值255会溢出

    pixel_labeled = torch.sum(target > 0)#.item()
    pixel_correct = torch.sum((predict == target) * (target > 0))#.item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

# def _confusion_matrix(y_pred, y, nclass):
#     ''' Computes the confusion matrix between two predicitons '''
#     b_size = y_pred.shape[0]
#     y, y_pred = _one_hot(y, nclass), _one_hot(y_pred, nclass)
#     y, y_pred = y.reshape(b_size, nclass, -1), y_pred.reshape(b_size, nclass, -1)
#     return torch.einsum('iaj,ibj->ab', y.float(), y_pred.float())    #https://zhuanlan.zhihu.com/p/27739282

def _confusion_matrix(y_pred, y, nclass):
    mask = (y >= 0) & (y < nclass)
    label = nclass * y[mask] + y_pred[mask]
    count = torch.bincount(label, minlength=nclass ** 2)
    confusion_matrix = count.view(nclass, nclass)

    return confusion_matrix

def pixel_accuracy_class(output, target, nclass):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    #predict = torch.argmax(output.long(), 1)
    predict = output
    target = target.long()
    confusion_matrix = _confusion_matrix(predict, target, nclass)
    return torch.diag(confusion_matrix), confusion_matrix.sum(dim=1)


def _one_hot(labels, nclass, class_dim=1):
    labels = labels.long()
    #labels = torch.LongTensor(labels)
    labels = torch.unsqueeze(labels, class_dim)
    #print(range(len(labels.shape)))
    labels_one_hot = torch.zeros_like(labels).repeat([nclass if d == class_dim else 1 for d in range(len(labels.shape))])
    # labels_one_hot = torch.tensor(labels_one_hot)
    labels_one_hot.scatter_(class_dim, labels, 1)  # https://www.cnblogs.com/dogecheng/p/11938009.html
    return labels_one_hot

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    #predict = torch.argmax(output, 1) + 1
    predict = output + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # print(intersection.shape)
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    # freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc
