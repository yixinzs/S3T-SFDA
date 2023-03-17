_base_ = [
    '../_base_/models/mask2former_swin.py', '../_base_/datasets/rsipac.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

checkpoint_file = '/data_zs/data/pretrained_models/swin_tiny_patch4_window7_224.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=18)
)

# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             backbone=dict(lr_mult=0.1),
#             absolute_pos_embed=dict(decay_mult=0.0),
#             relative_position_bias_table=dict(decay_mult=0.0),
#             norm=dict(decay_mult=0.0)),
#             embed=dict(decay_mult=0.0)
#             # head=dict(lr_mult=10.0)
#             ))

optimizer = dict(
    type='AdamW',
    lr=0.00001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
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
    dict(type='CutReplace', prob=0.5, img_dir='/data_zs/data/open_datasets/fusai_release/train/images', lbl_dir='/data_zs/data/open_datasets/fusai_release/train/labels', file='/data_zs/data/open_datasets/fusai_release/train/artificial_forest.csv'),
    dict(type='ClassMixFDA', prob=1, small_class=[14], amp_thred=0.001, file='/data_zs/output/rsipac_semi/config/sample_small26214-52428_class_stats_512x512_fold0.json'),   #[1, 3, 4, 5, 6]  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    dict(type='FDA', prob=0.2, amp_thred=0.001, img_dir='/data_zs/data/open_datasets/fusai_release/train/images', file='/data_zs/data/open_datasets/fusai_release/train/split.csv'),
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='RandomCrop', crop_size=(448, 448), cat_max_ratio=0.75),   #crop_size  (448, 448)  (384, 384)  (448, 448)
    dict(type='Resize', ratio_range=(1.25, 1.25)),   #(1.25, 1.25)   (1.5, 1.5)
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
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
data=dict(samples_per_gpu=4,
          workers_per_gpu=4,
          train=dict(pipeline=train_pipeline),
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))

runner = dict(type='IterBasedRunner', max_iters=80000)   #80000

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=3)
evaluation = dict(interval=1000, metric='FWIoU', save_best='FWIoU', greater_keys='FWIoU')

name = 'mask2former_tiny_80k_b10_poly_fda-cutreplace-classmix14_448s1.25_fold0'