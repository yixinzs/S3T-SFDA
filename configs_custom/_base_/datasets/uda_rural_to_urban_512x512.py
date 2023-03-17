# dataset settings
dataset_type = 'LOVEDADataset'
data_root = '/data_zs/open_datasets/2021LoveDA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
crop_size = (512, 512)
rural_train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),  # , reduce_zero_label=True
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    # dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
urban_train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),  # , reduce_zero_label=True
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    # dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),    #(1024, 512)
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,  #2
    workers_per_gpu=8,  #4
    train=dict(
        type='UDADataset',
        source=dict(
            type='LOVEDADataset',
            data_root='/data_zs/open_datasets/2021LoveDA/Train/Rural',
            img_dir='images_png',
            ann_dir='masks_png',
            pipeline=rural_train_pipeline),
        target=dict(
            type='LOVEDADataset',
            data_root='/data_zs/open_datasets/2021LoveDA/Train/Urban',
            img_dir='images_png',
            ann_dir='masks_png',
            pipeline=urban_train_pipeline)),
    val=dict(
        type='LOVEDADataset',
        data_root='/data_zs/open_datasets/2021LoveDA/Val/Urban',
        img_dir='images_png',
        ann_dir='masks_png',
        pipeline=test_pipeline),
    test=dict(
        type='LOVEDADataset',
        data_root='/data_zs/open_datasets/2021LoveDA/Val/Urban',
        img_dir='images_png',
        ann_dir='masks_png',
        pipeline=test_pipeline))
