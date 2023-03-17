# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/rsipac.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

model = dict(
    pretrained='/data_zs/data/pretrained_models/convnext_tiny_22k_224.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='ContrastUPerHead',
        in_channels=[96, 192, 384, 768],
        num_classes=18,
        loss_decode=[dict(type='SoftCrossEntropyLossV2', smooth_factor=0.1, ignore_index=255, loss_weight=1.0),
                     dict(type='ContrastLoss', ignore_index=255, loss_weight=0.05)
                     ],
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
    ),
    auxiliary_head=dict(
        type='ContrastFCNHead',
        in_channels=384,
        num_classes=18,
        loss_decode=[dict(type='SoftCrossEntropyLossV2', smooth_factor=0.1, ignore_index=255, loss_weight=0.4)],
    ), 
)

# optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
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

# lr_config = dict(_delete_=True, policy='CosineAnnealing',
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6,
#                  min_lr=0.0, by_epoch=False)

# lr_config = dict(_delete_=True, policy='PolyRestart',
#                  warmup='linear',
#                  warmup_iters=1500,   #1500   #20
#                  warmup_ratio=1e-6,
#                  periods=[70000, 5000, 5000],  #[65000, 5000, 5000, 5000]   #[50, 50, 50, 50]
#                  restart_weights=[1, 0.5, 0.5],  #[1, 0.3, 0.3, 0.3]
#                  min_lr=0.0, by_epoch=False)

img_norm_cfg = dict(
    mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206], to_rgb=False)  #True
train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),    #, reduce_zero_label=True
    dict(type='CutReplace', prob=0.5, img_dir='/data_zs/data/open_datasets/fusai_release/train/images', lbl_dir='/data_zs/data/open_datasets/fusai_release/train/labels', file='/data_zs/data/open_datasets/fusai_release/train/artificial_forest.csv'),
    dict(type='FDA', prob=0.2, amp_thred=0.001, img_dir='/data_zs/data/open_datasets/fusai_release/train/images', file='/data_zs/data/open_datasets/fusai_release/train/split.csv'),
    # dict(type='ClassMixFDA', prob=0.2, small_class=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], amp_thred=0.001, file='/data_zs/output/rsipac_semi/config/all_class_with_samples_512x512_fold0.json'),   #[1, 3, 4, 5, 6]
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='RandomCrop', crop_size=(448, 448), cat_max_ratio=0.75),   #crop_size  (448, 448)  (384, 384)  (448, 448)
    dict(type='Resize', ratio_range=(1.25, 1.25)),   #(1.25, 1.25)   (1.5, 1.5)
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=1.25,  #1.25  1.5
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=10,
          workers_per_gpu=10,
          train=dict(pipeline=train_pipeline),
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))

runner = dict(type='IterBasedRunner', max_iters=80000)   #80000

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=3)
evaluation = dict(interval=10000, metric='FWIoU', save_best='FWIoU', greater_keys='FWIoU')
# evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

# custom_hooks = [dict(
#                     type='SWAHook',
#                     swa_start=70000,  #65000   #50
#                     swa_freq=5000  #5000    #50
#                     )]

name = 'convnext_tiny_80k_b10_poly_contrastSoftce0.05_fda0.001_448s1.25_fold0'
