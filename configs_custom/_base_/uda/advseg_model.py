
baseline = dict(
    type='HRDAEncoderDecoder',
    pretrained='/data_zs/data/pretrained_models/mit_b2.pth',
    backbone=dict(type='mit_b2', style='pytorch'),
    decode_head=dict(
        type='HRDAHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
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
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1),
    train_cfg=dict(work_dir='/data_zs/output/loveDA_uda/hrda_loveda_crop512'),
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]),
    scales=[1, 0.5],
    hr_crop_size=[512, 512],
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True)

model = dict(
    type='AdvSeg',
    discriminator_type='LS',
    lr_D=1e-4,
    lr_D_power=0.9,
    lr_D_min=0,
    lambda_adv_target=dict(main=0.001, aux=0.0002),
    debug_img_interval=1000,
    model=baseline,)

