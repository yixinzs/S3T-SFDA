# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/data_zs/open_datasets/2021LoveDA'
img_norm_cfg = dict(
    mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206], to_rgb=False)  #mean=[61.455, 64.868, 74.614], std=[32.379, 35.219, 42.206]   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
crop_size = (512, 512)  #(512, 512)
train_weak_pipeline = [
    dict(type='LoadImageFromFileCustom'),
    dict(type='LoadAnnotationsCustom'),
    # dict(type='AlbumentationAug'),
    dict(type='LabelEncode'),
    dict(type='Resize', img_scale=(1024, 1024)),  # (2560, 1440)  (1280, 720)  img_scale=(w, h)
    dict(type='RandomCrop', crop_size=crop_size),  #, cat_max_ratio=0.75
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
]

meta_keys=('filename', 'ori_filename', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction', 'img_norm_cfg', 'scale', 'crop_bbox', 'crop_bbox_list')

train_strong_pipeline = [
    dict(type='WeakToStrong'),
    dict(type='AlbumentationAug'),
    dict(type='Resize', ratio_range=(1, 1.25)),
    dict(type='RandomCrop', crop_size=(512, 512)),  #, cat_max_ratio=0.75  crop_size=(h, w)
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandMixPseudo', psd_root='/data_zs/output/loveDA_uda/rural2urban/st-sfda_roi-prop_loveda_rural2urban_deeplab_resnet_512x512_b4_dtu_lr6e-5_total_augv2_dt-st', classes=[1, 2, 3, 4, 5, 6], choice_p=[1 / 6] * 6),      #/irsa/data_zs/output/tmp_debug/sfda_isprs_potsdamIRRG2vaihingen_deeplabv2_512x512_b4_lr5e-4_neg-loass-w1_resume9600_mixlabel
    dict(type='RandomROIBbox', crop_size=(512, 512), scale=(0.5, 1), roi_num=4),   #(512, 1024)  (512, 512)
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundleStrong'),
    dict(type='Collect', keys=['img', 'img_full', 'gt_semantic_seg', 'mix_label'], meta_keys=meta_keys)  #   #'gt_semantic_seg',
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
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,  #2
    workers_per_gpu=4,  #4
    train=dict(
        type='SFDADataset',
        target=dict(
            type='LOVEDADataset',
            data_root='/data_zs/open_datasets/2021LoveDA/Train/Urban',
            img_dir='images_png',
            ann_dir='masks_png',
            pipeline=train_weak_pipeline),
        pipeline=train_strong_pipeline),
    val=dict(
        type='LOVEDADataset',
        data_root='/data_zs/open_datasets/2021LoveDA/Val/Urban',    #/data_zs/open_datasets/2021LoveDA/Val/Urban
        img_dir='images_png',
        ann_dir='masks_png',
        pipeline=test_pipeline),
    test=dict(
        type='LOVEDADataset',
        data_root='/data_zs/open_datasets/2021LoveDA/Val/Urban',    #/data_zs/open_datasets/2021LoveDA/Val/Urban
        img_dir='images_png',
        ann_dir='masks_png',
        pipeline=test_pipeline))
