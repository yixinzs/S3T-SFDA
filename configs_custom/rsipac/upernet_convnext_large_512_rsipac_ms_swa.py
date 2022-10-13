# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/rsipac.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

# checkpoint_file = '/data_zs/data/pretrained_models/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth' #'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'  # noqa
# model = dict(
#     backbone=dict(
#         type='mmcls.ConvNeXt',
#         arch='large',
#         out_indices=[0, 1, 2, 3],
#         drop_path_rate=0.4,
#         layer_scale_init_value=1.0,
#         gap_before_final_norm=False,
#         init_cfg=dict(
#             type='Pretrained', checkpoint=checkpoint_file,
#             prefix='backbone.')),
#     decode_head=dict(
#         in_channels=[192, 384, 768, 1536],
#         num_classes=9,
#     ),
#     auxiliary_head=dict(
#         in_channels=768,
#         num_classes=9
#     ),
#     test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)),
# )

model = dict(
    pretrained='/data_zs/data/pretrained_models/convnext_large_22k_224.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=17,
        # loss_decode=[dict(type='LovaszLoss', per_image=False, reduction='none', loss_weight=1.0),
        #              dict(type='FocalLoss', loss_weight=1.0)],
        # loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        #              dict(type='LovaszLoss', per_image=False, reduction='none', loss_weight=1.0)],
        # loss_decode=dict(_delete_=True, type='SoftCrossEntropyLoss', smooth_factor=0.1, ignore_index=255, loss_weight=1.0),
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=17,
        # loss_decode=[dict(type='LovaszLoss', per_image=False, reduction='none', loss_weight=0.4),
        #              dict(type='FocalLoss', loss_weight=0.4)],
        # loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        #              dict(type='LovaszLoss', per_image=False, reduction='none', loss_weight=0.4)],
        # loss_decode=dict(_delete_=True, type='SoftCrossEntropyLoss', smooth_factor=0.1, ignore_index=255, loss_weight=0.4)
    ),
)

# optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
#                  lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg={'decay_rate': 0.9,
#                                 'decay_type': 'stage_wise',
#                                 'num_layers': 12})

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

# lr_config = dict(_delete_=True, policy='poly',
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)

# lr_config = dict(_delete_=True, policy='CosineRestart',
#                  warmup='linear',
#                  warmup_iters=1500,   #1500   #20
#                  warmup_ratio=1e-6,
#                  periods=[65000, 5000, 5000, 5000],  #[65000, 5000, 5000, 5000]   #[50, 50, 50, 50]
#                  restart_weights=[1, 0.3, 0.3, 0.3],  #[1, 0.3, 0.3, 0.3]
#                  min_lr=1e-6, by_epoch=False)

lr_config = dict(_delete_=True, policy='PolyRestart',
                 warmup='linear',
                 warmup_iters=1500,   #1500   #20
                 warmup_ratio=1e-6,
                 periods=[65000, 5000, 5000, 5000],  #[65000, 5000, 5000, 5000]   #[50, 50, 50, 50]
                 restart_weights=[1, 0.3, 0.3, 0.3],  #[1, 0.3, 0.3, 0.3]
                 min_lr=1e-6, by_epoch=False)

#
# lr_config = dict(_delete_=True, policy='OneCycle',
#                  max_lr=2e-04,
#                  by_epoch=False)

img_norm_cfg = dict(
    mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206], to_rgb=False)  #True
train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),    #, reduce_zero_label=True
    # dict(type='ClassMixFDA', prob=0.4, small_class=[4, 2, 7, 8], amp_thred=0.006, file='/data_zs/output/rsipac/config/small_class_with_samples_512x512_fold0.json'),   #[1, 3, 4, 5, 6]
    # dict(type='AlbumentationAug'),
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
data=dict(samples_per_gpu=10,
          workers_per_gpu=10,
          train=dict(pipeline=train_pipeline))

runner = dict(type='IterBasedRunner', max_iters=80000)   #80000   #200

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=3)
evaluation = dict(interval=10000, metric='FWIoU', save_best='FWIoU', greater_keys='FWIoU')
# evaluation = dict(interval=100, metric='mIoU', pre_eval=True)

# custom_hooks_config = [dict(
#                     type='LinearMomentumEMAHook',
#                     momentum=0.001,
#                     interval=1,
#                     warm_up=100,
#                     resume_from=None
#                     )]
# custom_hooks = [dict(
#                     type='SWAHook',
#                     swa_start=65000,  #65000   #50
#                     swa_freq=5000  #5000    #50
#                     )]

name = 'large_80k_rsipac_ms_swa_polyrestart_ce_fold0'  #upernet_convnext_
