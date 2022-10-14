_base_ = [
    '../_base_/models/upernet_cswin.py', '../_base_/datasets/rsipac.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
pretrained = '/data_zs/data/pretrained_models/cswin_base_224.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        type='CSWin',
        embed_dim=96,
        depth=[2,4,32,2],
        num_heads=[4,8,16,32],
        split_size=[1,2,7,7],
        drop_path_rate=0.6,
        use_chk=False,
    ),
    decode_head=dict(
        in_channels=[96,192,384,768],
        num_classes=18
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=18
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)},
                                    head=dict(lr_mult=10.0)))

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
    # dict(type='ClassMixFDA', prob=0.4, small_class=[4, 2, 7, 8], amp_thred=0.006, file='/data_zs/output/rsipac/config/small_class_with_samples_512x512_fold0.json'),   #[1, 3, 4, 5, 6]
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=10,
          workers_per_gpu=10,
          train=dict(pipeline=train_pipeline))

runner = dict(type='IterBasedRunner', max_iters=80000)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
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

name = 'cswin_base_b10_ce_augv2'