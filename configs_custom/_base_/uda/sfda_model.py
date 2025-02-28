
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
        num_classes=6,   #19  #3  #6  7  #12
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

# baseline = dict(
#     type='EncoderDecoder',
#     pretrained='/data_zs/data/pretrained_models/convnext_small_22k_224.pth',
#     backbone=dict(
#         type='ConvNeXt',
#         in_chans=3,
#         depths=[3, 3, 27, 3],
#         dims=[96, 192, 384, 768],
#         drop_path_rate=0.3,
#         layer_scale_init_value=1.0,
#         out_indices=[0, 1, 2, 3]),
#     decode_head=dict(
#         type='UPerHead',
#         in_channels=[96, 192, 384, 768],
#         in_index=[0, 1, 2, 3],
#         pool_scales=(1, 2, 3, 6),
#         channels=512,
#         dropout_ratio=0.1,
#         num_classes=12,  #7
#         norm_cfg=dict(type='SyncBN', requires_grad=True),
#         align_corners=False,
#         loss_decode=[
#             dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
#         ]),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))

model = dict(
    type='SFDAEncoderDecoder',
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
    model=baseline
)