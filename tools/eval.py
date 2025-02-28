import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import mmcv
import torch

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

from mmseg_custom.apis.score import SegmentationMetric

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
        print(f'----------------------------:{checkpoint}')
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # model.CLASSES = checkpoint['meta']['CLASSES']
        # model = MMDataParallel(model, device_ids=[0])
        model.to(device)
        model.eval()

        model_list.append(model)
    return model_list

def get_dataset(img_path, aug_test=False, distributed=None):
    test_cfg = dict(
        type='LOVEDADataset',    #IsprsDataset  DFCDataset
        data_root=img_path,
        img_dir='images_png',   #images     images_png   images_512_384-384
        ann_dir='masks_png',   #labels_gray  labels_new
        pipeline=[
            dict(type='LoadImageFromFileCustom'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),   #(1024, 1024)  #(512, 512)  #
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    # dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],   #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],   mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206],
                        to_rgb=False),     #mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206]
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ])
    if aug_test:
        test_cfg['pipeline'][1]['img_ratios'] = [
            1.0, 1.25, 1.5, #1.75, 2.0, 2.25  #, 2  0.75,
        ]   #0.5, 0.75, 1.0, 1.25, 1.5, 1.75   #1.0, 1.25, 1.5   #0.5,
        test_cfg['pipeline'][1]['flip'] = True
        test_cfg['pipeline'][1]['flip_direction'] = ["horizontal", "vertical"]  #, "vertical"
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

def gpu_test(model_list, data_loader, show_dir):
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    # metric = SegmentationMetric(12, False)
    # metric.reset()
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

            dirpath = r'/irsa/data_zs/output/loveDA_uda/rural2urban/st-sfda_roi-prop_loveda_rural2urban_daformer_mitb2_512x512_b4_dtu_lr6e-5_tmp_out_npy_v2/urban_total_1024x1024_6300_44.45_npy' #r'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/sfda_dfc22train2val_daformer_mitb2_512x512_b4_dtu_lr6e-5_neg-loass-w1_resume17000_train-val-same_3004/Clermont-Ferrand_1024x1024_14000_npy_v2'  #r'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_dfc22_daformer_mitb2_512x512_b4_pmd_train-val-Clermont-Ferrand_3004_v3/Clermont-Ferrand_1024x1024_20000_npy' #  # 1
            save_logit(logist_output, data['img_metas'], dirpath)

        result = logist_output.argmax(dim=1)
        # metric.update(result, targets)
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

                # label_map = {
                #            0: 0,
                #            1: 1,
                #            2: 2,
                #            3: 3,
                #            4: 4,
                #            5: 5,
                #            6: 6,
                #            7: 7,
                #            8: 8,
                #            9: 9,
                #            10: 10,
                #            11: 11
                #              }

                label_map = {
                           0: 0,
                           1: 1,
                           2: 2,
                           3: 3,
                           4: 4,
                           5: 5,
                           6: 6
                             }

                # label_map = {0: [255, 255, 255],
                #              1: [0, 0, 255],
                #              2: [0, 255, 255],
                #              3: [0, 255, 0],
                #              4: [255, 255, 0],
                #              5: [255, 0, 0]}
                dataset.show_result(pred, out_file, label_map)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

def save_logit(logit, img_metas, dirpath):
    import numpy as np
    import os.path as osp
    os.makedirs(dirpath, exist_ok=True)
    logits_numpy = logit.cpu().numpy()
    logits_numpy = list(logits_numpy)
    img_metas = img_metas[0]
    # print('------------------img_metas:{}'.format(img_metas))
    for logit_numpy, img_meta in zip(logits_numpy, img_metas):
        # print('------------------:shape:{}'.format(logit_numpy.shape))
        # print('------------------img_meta:{}'.format(img_meta))
        out_file = osp.join(dirpath, img_meta['ori_filename'])
        np.save(out_file, logit_numpy)


if __name__ == '__main__':
    input_path = r'/irsa/data_zs/open_datasets/2021LoveDA/Train/Urban' #r'/irsa/data_zs/open_datasets/2021LoveDA/Test/Rural'  # #r'/irsa/data_zs/open_datasets/2021LoveDA/Val/Urban'  #/irsa/data_zs/open_datasets/2021LoveDA/Test/Urban  #r'/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/vaihingen/train' # #r'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_dfc22_convnext_mitb2_512x512_b4_pmd_addauxdecode_val-Clermont-Ferrand_3004/tmp' #r'/irsa/data_zs/open_datasets/minfrance/dataset_crop/val/Clermont-Ferrand'  #'/data_zs/open_datasets/2021LoveDA/Test/Urban'
    save_path = r'/irsa/data_zs/output/loveDA_uda/rural2urban/st-sfda_roi-prop_loveda_rural2urban_daformer_mitb2_512x512_b4_dtu_lr6e-5_tmp_out_npy_v2/urban_total_1024x1024_6300_44.45' #r'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/sfda_dfc22train2val_daformer_mitb2_512x512_b4_dtu_lr6e-5_neg-loass-w1_resume17000_train-val-same_3004/Clermont-Ferrand_1024x1024_14000_v2' #r'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/sfda_dfc22train2val_daformer_mitb2_512x512_b4_dtu_lr6e-5_neg-loass-w1_resume17000_train-val-same_3004/Clermont-Ferrand_1024x1024_3000'    #'/data_zs/output/loveDA_uda/advmatch/visual/rural2urban_convnext_small_80k_b12_Softce_augv2_totaldata'
    config_path_list = ['/irsa/data_zs/output/loveDA_uda/rural2urban/onlysource_rural_daformer_mitb2_512x512_b4_val-totalurban-traintotal_augv2/onlysource_rural_daformer.py'] #['/irsa/data_zs/output/tmp_debug/onlysource_potsdamIRRG2vaihingen_deeplabv2_stri16-headlr1.0/onlysource_potsdamIRRG2vaihingen_deeplabv2.py'] #['/data_zs/code/source_free_da/mmsegmentation_sfda/configs_custom/sfda/onlysource_zkxt2potsdamRGB_deeplabv2.py']  #['/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_dfc22_daformer_mitb2_512x512_b4_pmd_val-Clermont-Ferrand_3004/onlysource_dfc22_daformer.py'] #['/data/code/domain_adaptation/mmsegmentation_domain_MuGCDA/configs_custom/loveDA/upernet_convnext_small_512_loveDA_ms.py'] #['/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold0/upernet_convnext_tiny_rsipac_s1.5_ms.py', '/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold4/upernet_convnext_tiny_rsipac_s1.5_ms.py']
    model_path_list = ['/irsa/data_zs/output/loveDA_uda/rural2urban/st-sfda_roi-prop_loveda_rural2urban_daformer_mitb2_512x512_b4_dtu_lr6e-5_tmp_out_npy_v2/iter_6300.pth'] #['/irsa/data_zs/output/tmp_debug/onlysource_potsdamIRRG2vaihingen_deeplabv2_stri16-headlr1.0/checkpoints/4500_46.23_mIoU_checkpoint.pth']  #['/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_zkxt2potsdamRGB/onlysource_zkxt2potsdamRGB_deeplabv2_resnet101_512x512_lr5e-4_b4_pmd/checkpoint/iter_14200_0.38067709069790445_checkpoint.pth']  #['/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/sfda_dfc22train2val_daformer_mitb2_512x512_b4_dtu_lr6e-5_neg-loass-w1_resume17000_train-val-same_3004/iter_14000.pth']  #['/data_zs/output/loveDA_uda/rural2urban/rural2urban_convnext_small_80k_b12_Softce_augv2_totaldata/checkpoint/iter_4000_0.5840717357121707_checkpoint.pth'] #['/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold0/latest.pth', '/data_zs/output/rsipac_semi/checkpoint/convnext_tiny_80k_b10_poly_Softce_fda0.001_448s1.25_fold4/latest.pth']
    aug_test = False #True
    device_id = 0

    os.makedirs(save_path, exist_ok=True)

    dataset = get_dataset(input_path, aug_test)
    data_loader = build_dataloader(dataset, samples_per_gpu=4, workers_per_gpu=1, dist=False, shuffle=False)

    device = 'cuda:{}'.format(device_id)
    model_list = collect_model(config_path_list, model_path_list, device)

    gpu_test(model_list, data_loader, save_path)

    '''
    rural2urban:
        daformer: /irsa/data_zs/output/loveDA_uda/rural2urban/onlysource_rural_daformer_mitb2_512x512_b4_val-totalurban-traintotal/onlysource_rural_daformer.py    #/irsa/data_zs/output/loveDA_uda/rural2urban/onlysource_rural_daformer_mitb2_512x512_b4_val-totalurban-traintotal_augv2/onlysource_rural_daformer.py
        convnext: /irsa/data_zs/output/loveDA_uda/rural2urban/onlysource_rural_convnext_512x512_b4_val-totalurban-traintotal/onlysource_rural_convnext.py         #/irsa/data_zs/output/loveDA_uda/rural2urban/onlysource_rural_convnext_512x512_b4_val-totalurban-traintotal_augv2/onlysource_rural_convnext.py
    urban2rural:
        daformer: /irsa/data_zs/output/loveDA_uda/urban2rural/onlysource_urban_daformer_mitb2_512x512_b4_val-totalrural/onlysource_urban_daformer.py         #/irsa/data_zs/output/loveDA_uda/urban2rural/onlysource_urban_daformer_mitb2_512x512_b4_val-totalrural-traintotal_augv2/onlysource_urban_daformer.py
        convnext: /irsa/data_zs/output/loveDA_uda/urban2rural/onlysource_urban_convnext_512x512_b4_val-totalrural-traintotal/onlysource_urban_convnext.py   #/irsa/data_zs/output/loveDA_uda/urban2rural/onlysource_urban_convnext_512x512_b4_val-totalrural-traintotal_augv2/onlysource_urban_convnext.py
    '''
