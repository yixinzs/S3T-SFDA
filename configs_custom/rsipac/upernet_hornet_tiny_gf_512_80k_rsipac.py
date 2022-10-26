# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_hornet.py', '../_base_/datasets/rsipac.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

model = dict(
    pretrained='/data_zs/data/pretrained_models/hornet_tiny_gf.pth',
    backbone=dict(
        depths=[2, 3, 18, 2],
        base_dim=64,
        block='Block',
        gnconv=[
            'partial(gnconv, order=2, s=1/3)',
            'partial(gnconv, order=3, s=1/3)',
            'partial(gnconv, order=4, s=1/3, h=24, w=13, gflayer=GlobalLocalFilter)',
            'partial(gnconv, order=5, s=1/3, h=12, w=7, gflayer=GlobalLocalFilter)',
        ],
        drop_path_rate=0.4,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=18,
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=18
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# optimizer = dict(constructor='LearningRateDecayOptimizerConstructorHorNet', type='AdamW',   #, _delete_=True
#                  lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg={'decay_rate': 0.9,
#                                 'decay_type': 'stage_wise',
#                                 'num_layers': 6})

optimizer = dict(
    type='AdamW',
    lr=2e-04,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)),
            head=dict(lr_mult=10.0)
            ))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

img_norm_cfg = dict(
    mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206], to_rgb=False)  #True
train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),    #, reduce_zero_label=True
    # dict(type='ClassMixFDA', prob=0.5, small_class=[1, 2, 3, 4, 5, 6, 7, 8], amp_thred=0.006, file='/data_zs/output/rsipac/config/small_class_with_samples_512x512_fold0.json'),   #[1, 3, 4, 5, 6]
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
data=dict(samples_per_gpu=12,
          workers_per_gpu=12,
          train=dict(pipeline=train_pipeline))

runner = dict(type='IterBasedRunner', max_iters=80000)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))
# fp16 placeholder
fp16 = dict()
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=3)
evaluation = dict(interval=10000, metric='FWIoU', save_best='FWIoU', greater_keys='FWIoU')
# evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

name = 'hornet_tiny_80k_b12_ce_augv2_s1.25'
