# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
# recommand use this config for BEiT models which are self-supervised pretrained and then intermediate fine-tuned on imagenet
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/rsipac.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

pretrained = '/data_zs/data/pretrained_models/beit_base_patch16_224_pt22k_ft22k.pth'
model = dict(
    pretrained=pretrained,
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))

# optimizer = dict(
#     type='AdamW',
#     lr=3e-5,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

optimizer = dict(
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)),
            head=dict(lr_mult=10.0)
            ))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

size = 512
img_norm_cfg = dict(
    mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206], to_rgb=False)  #True
train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),    #, reduce_zero_label=True
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='Resize', img_scale=(size, size), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(size, size), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]


# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4,
          train=dict(pipeline=train_pipeline))

# runner = dict(type='EpochBasedRunner', max_epochs=40)
# checkpoint_config = dict(by_epoch=True, interval=1000000, max_keep_ckpts=1)
# evaluation = dict(interval=1, metric='FWIoU', save_best='FWIoU', greater_keys='FWIoU')

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=3)
evaluation = dict(interval=10000, metric='FWIoU', save_best='FWIoU', greater_keys='FWIoU')
# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
# optimizer_config = {}

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()

name = 'upernet_beit_base_512_slide_80000_rsipac'
