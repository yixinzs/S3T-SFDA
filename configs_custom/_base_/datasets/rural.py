# dataset settings
dataset_type = 'GTADataset'
data_root = 'data/gta/'
img_norm_cfg = dict(
    mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206], to_rgb=False)  #True  mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
crop_size = (512, 512)  #(1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 1.5)),   #(1024, 1024)  (720, 720)  (1536, 1536)  #(1024, 1024)
    # dict(type='Resize', ratio_range=(1, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size),   #, cat_max_ratio=0.75
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]


test_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),   #(1024, 1024)
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
    samples_per_gpu=8,  #4
    workers_per_gpu=8,  #4
    train=dict(
        type='LOVEDADataset',
        data_root='/irsa/data_zs/open_datasets/2021LoveDA_All/Rural/Total',    #/data_zs/open_datasets/2021LoveDA/Train/Rural
        img_dir='images_png',
        ann_dir='masks_png',
        # label_map={0: [255, 0, 0],
        #            1: [0, 0, 255],
        #            2: [255, 255, 255]},
        pipeline=train_pipeline),
    val=dict(
        type='LOVEDADataset',
        data_root='/data_zs/open_datasets/2021LoveDA_All/Urban/Total',  #Val/Urban   #/data_zs/open_datasets/2021LoveDA/Val/Urban
        img_dir='images_png',
        ann_dir='masks_png',
        # label_map={0: [255, 0, 0],
        #            1: [0, 0, 255],
        #            2: [255, 255, 255]},
        pipeline=test_pipeline),
    test=dict(
        type='LOVEDADataset',
        data_root='/data_zs/open_datasets/2021LoveDA_All/Urban/Total',  #Val/Urban    #/data_zs/open_datasets/2021LoveDA/Val/Urban
        img_dir='images_png',
        ann_dir='masks_png',
        # label_map={0: [255, 0, 0],
        #            1: [0, 0, 255],
        #            2: [255, 255, 255]},
        pipeline=test_pipeline))
