# dataset settings
dataset_type = 'RSIPACDataset'
data_root = 'data/ade/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206], to_rgb=False)  #True
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),    #, reduce_zero_label=True
    # dict(type='ClassMix', prob=0.5, small_class=[1, 3, 4, 5, 6], file='/data_zs/output/rsipac/config/small_class_with_samples_512x512.json'),
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]


test_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root='/data/train_fusai',
        img_dir='images',
        ann_dir='labels',
        split='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root='/data/train_fusai',  #/dataset/data/DFCT/dataset_patch_512x512/val  #/dataset/data/DFCT/dataset_patch_512x512_9-1/val
        img_dir='images',
        ann_dir='labels',
        split='val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root='/test',
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline))
