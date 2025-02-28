
baseline = dict(
    type='EncoderDecoder',
    pretrained='/data_zs/data/pretrained_models/mit_b2.pth',
    backbone=dict(type='mit_b2', style='pytorch'),  #'mit_b5'
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,   #19  #3  #6  7  #12
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
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model = dict(
    type='SFDAEncoderDecoder',   #SFDAEncoderDecoder   MVCSFDAEncoderDecoder
    dtu_dynamic=True,
    query_step=3,
    query_start=9,
    meta_max_update=30,
    ema_weight=0.99,
    proxy_metric='SND',
    fix_iteration=12,
    topk_candidate=0.5,
    update_frequency=3000,
    threshold_beta=0.001,
    feat_channels=64 + 128 + 320 + 512,
    model=baseline
)