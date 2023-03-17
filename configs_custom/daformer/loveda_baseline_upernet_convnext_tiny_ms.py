# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/loveda.py',
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
        in_channels=[96, 192, 384, 768],
        num_classes=7,
        # loss_decode=[dict(type='SoftCrossEntropyLoss', smooth_factor=0.1, ignore_index=255, loss_weight=1.0)],
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=7,
        # loss_decode=[dict(type='SoftCrossEntropyLoss', smooth_factor=0.1, ignore_index=255, loss_weight=0.4)],
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

runner = dict(type='IterBasedRunner', max_iters=40000)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
# evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

name = 'loveda_baseline_convnext_tiny_40k_b22_poly'
