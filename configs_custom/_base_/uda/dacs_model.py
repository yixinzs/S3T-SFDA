
baseline = dict(
    type='DACSEncoderDecoder',
    pretrained='/data_zs/data/pretrained_models/mit_b2.pth',
    backbone=dict(type='mit_b2', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True))),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5)
        ]),
    train_cfg=dict(
        work_dir='/data_zs/output/loveDA_uda/daformer_b2-rcs-tmp0.1_ce_focalloss_pseudo0.9_loveda'),
    test_cfg=dict(mode='whole'))

model = dict(
    type='DACS',
    alpha=0.99,
    pseudo_threshold=0.9,   #0.968
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,  #1000
    print_grad_magnitude=False,
    model=baseline
)

