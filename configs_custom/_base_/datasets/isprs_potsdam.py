# dataset settings
dataset_type = 'GTADataset'
data_root = 'data/gta/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)  #True
crop_size = (512, 512)  #(1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),
    dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='Resize', img_scale=(512, 512)),   #(2560, 1440)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='IsprsDataset',
        data_root='/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/postman/train',  #'/irsa/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/postman/train_IRRG'
        img_dir='images',
        ann_dir='labels_gray',
        # label_map={0: [255, 0, 0],
        #            1: [0, 0, 255],
        #            2: [255, 255, 255]},
        pipeline=train_pipeline),
    val=dict(
        type='IsprsDataset',
        data_root='/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/vaihingen/test',  #'/irsa/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/vaihingen/train'  #zurich_patch  #/irsa/data_zs/data/expert_datasets/ZhongKeXingTu/train_and_val
        img_dir='images',
        ann_dir='labels_gray',
        # label_map={0: [255, 0, 0],
        #            1: [0, 0, 255],
        #            2: [255, 255, 255]},
        pipeline=test_pipeline),
    test=dict(
        type='IsprsDataset',
        data_root='/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/vaihingen/test', # #zurich_patch   #/irsa/data_zs/data/expert_datasets/ZhongKeXingTu/train_and_val
        img_dir='images',
        ann_dir='labels_gray',
        # label_map={0: [255, 0, 0],
        #            1: [0, 0, 255],
        #            2: [255, 255, 255]},
        pipeline=test_pipeline))
