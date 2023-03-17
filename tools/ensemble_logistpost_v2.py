import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import mmcv
import torch
import argparse

cur_path = os.path.abspath(os.path.dirname(__file__))  #https://www.cnblogs.com/joldy/p/6144813.html
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)    #sys.path.append:https://blog.csdn.net/zxyhhjs2017/article/details/80582246?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242

from mmcv.runner import (get_dist_info, init_dist, #load_checkpoint,
                         wrap_fp16_model)
from mmcv_custom.checkpoint import load_checkpoint

from mmcv.image import tensor2imgs
import os.path as osp
from mmcv.parallel import collate, scatter
from mmcv.runner import (get_dist_info, init_dist, #load_checkpoint,
                         wrap_fp16_model)

cur_path = os.path.abspath(os.path.dirname(__file__))  #https://www.cnblogs.com/joldy/p/6144813.html
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)    #sys.path.append:https://blog.csdn.net/zxyhhjs2017/article/details/80582246?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import mmcv_custom
import  mmseg_custom

def collect_model(cfg_path_list, model_path_list, device, fp16=True):
    assert len(cfg_path_list) == len(model_path_list), 'error'
    model_list = []
    for config, checkpoint in zip(cfg_path_list, model_path_list):
        cfg = mmcv.Config.fromfile(config)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None

        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        if fp16:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # model.CLASSES = checkpoint['meta']['CLASSES']
        # model = MMDataParallel(model, device_ids=[0])
        model.to(device)
        model.eval()

        model_list.append(model)
    return model_list

def get_dataset(img_path, aug_test=False, distributed=None):
    test_cfg = dict(
        type='RSIPACDataset',
        data_root=img_path,
        img_dir='images',
        ann_dir='labels',
        pipeline=[
            dict(type='LoadImageFromFileCustom'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[61.455, 64.868, 74.614],   #[61.455, 64.868, 74.614]
                        std=[32.379, 35.219, 42.206],    #[32.379, 35.219, 42.206]
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ])
    if aug_test:
        test_cfg['pipeline'][1]['img_ratios'] = [
            1.0, 1.25, 1.5, 1.75, 2.0  #, 2  0.75,
        ]   #0.5, 0.75, 1.0, 1.25, 1.5, 1.75   #1.0, 1.25, 1.5   #0.5,
        test_cfg['pipeline'][1]['flip'] = True
        test_cfg['pipeline'][1]['flip_direction'] = ["horizontal", "vertical"]
    test_cfg['test_mode'] = True
    print('-------------------:{}'.format(test_cfg))
    dataset = build_dataset(test_cfg)

    return dataset
from mmcv.runner import BaseModule, auto_fp16

@auto_fp16(apply_to=('img', ))
def inference(model, img, img_metas, rescale=True, **kwargs):
    format_vaildate(img, img_metas)
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
    return seg_logit #seg_pred

def format_vaildate(imgs, img_metas, **kwargs):
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


# def logistic_thresd(logists):
#     index_one, index_two = 13, 11
#     twoindex_thresd = 1.5
#     preds = []
#     # print('----------------logists:{}'.format(logists.shape))
#     logists = torch.softmax(logists, dim=1)
#     for logist_output in list(logists):
#         # print('----------------logist_output:{}'.format(logist_output.shape))
#         top2_logist, top2_pred = logist_output.topk(2, dim=0, largest=True, sorted=True)
#         # print('----------------top2_logist:{},top2_pred:{}'.format(top2_logist.shape, top2_pred.shape))
#         prop_lowthresd = torch.where((top2_logist[0] < top2_logist[1] * twoindex_thresd), 1, 0)
#         mask = torch.zeros((top2_pred.shape[-2], top2_pred.shape[-1]), dtype=torch.uint8).to(logists.device)
#         mask_index_one = (prop_lowthresd == 1) & (top2_pred[0] == index_one) & (top2_pred[1] == index_two)
#         mask[mask_index_one] = 1
#         # print('-----------------------mask:{}'.format(torch.sum(mask)))
#         pred = torch.where(mask == 0, top2_pred[0], top2_pred[1])
#         # print('----------------pred:{}'.format(pred.shape))
#         preds.append(pred)
#     # print('----------------preds:{}'.format(torch.stack(preds, dim=0).shape))
#     return torch.stack(preds, dim=0)

def logistic_thresd(logists):
    index_list = [11, 12, 13, 14]
    factor_list = [1.3, 1.3, 0.9, 0.9]

    for index, factor in zip(index_list, factor_list):
        logists[:, index, :, :] = logists[:, index, :, :] * factor

    return logists

def gpu_test(model_list, data_loader, show_dir):
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            device = next(model_list[0].parameters()).device
            # print('----------------------:{}'.format(device))
            # data = collate([data], samples_per_gpu=1)
            if next(model_list[0].parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                data['img_metas'] = [i.data[0] for i in data['img_metas']]

            logist_output = inference(model_list[0], **data)
            for model in model_list[1:]:
                output = inference(model, **data)
                logist_output = logist_output + output
            logist_output = logist_output / len(model_list)

        result = logistic_thresd(logist_output).argmax(dim=1)
        result = result.cpu().numpy()
        result = list(result)

        if show_dir:

            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0]
            # print('-----------------------:{}'.format(img_metas))
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta, pred in zip(imgs, img_metas, result):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                out_file = osp.join(show_dir, img_meta['ori_filename'])

                dataset.show_result(pred, out_file=out_file)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default='', help='test config file path')
    parser.add_argument('--checkpoint', default='', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        default='',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--gpu-id',
        type=int,
        # default=3,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    config_path = args.config
    model_path = args.checkpoint
    save_path = args.work_dir
    config_path_list = config_path.split('@')  #[config_path]  #['/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda-cutreplace-classmix14_448s1.25_swa_fold0/upernet_convnext_tiny_rsipac_s1.5_ms.py'] #['/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold0/upernet_convnext_tiny_rsipac_s1.5_ms.py', '/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold4/upernet_convnext_tiny_rsipac_s1.5_ms.py']
    model_path_list = model_path.split('@')   #[model_path]  #['/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda-cutreplace-classmix14_448s1.25_swa_fold0/swa_final.pth'] #['/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold0/latest.pth', '/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold4/latest.pth']
    # save_path = r'/data_zs/output/rsipac_semi/test/poly_Softce_fda-cutreplace-classmix14_448s1.25_swa_fold0-10_v3'
    print('-------------config_path_list:{}'.format(config_path_list))
    print('-------------model_path_list:{}'.format(model_path_list))
    input_path = r'/test' #r'/test'
    aug_test = True
    device_id = args.gpu_id

    os.makedirs(save_path, exist_ok=True)

    dataset = get_dataset(input_path, aug_test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)

    # device = 'cuda:{}'.format(device_id)
    device = 'cuda'
    model_list = collect_model(config_path_list, model_path_list, device)

    gpu_test(model_list, data_loader, save_path)
