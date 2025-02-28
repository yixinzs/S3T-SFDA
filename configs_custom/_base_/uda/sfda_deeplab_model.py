
# baseline = dict(
#     type='EncoderDecoder',
#     pretrained='/data_zs/data/pretrained_models/resnet101_v1c-e67eebb6.pth',
#     backbone=dict(
#         type='ResNetV1c',
#         depth=101,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         dilations=(1, 1, 2, 4),
#         strides=(1, 2, 1, 1),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=False,
#         style='pytorch',
#         contract_dilation=True),
#     decode_head=dict(
#         type='DLV2Head',
#         in_channels=2048,
#         in_index=3,
#         dilations=(6, 12, 18, 24),
#         num_classes=6,
#         align_corners=False,
#         init_cfg=dict(
#             type='Normal', std=0.01, override=dict(name='aspp_modules')),
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))

# baseline = dict(
#     type='EncoderDecoder',
#     pretrained='/data_zs/data/pretrained_models/resnet101-5d3b4d8f.pth',
#     backbone=dict(type='resnet101'),
#     decode_head=dict(
#         type='DLV2Head',
#         in_channels=2048,
#         in_index=3,
#         dilations=(6, 12, 18, 24),
#         num_classes=6,
#         align_corners=False,
#         init_cfg=dict(
#             type='Normal', std=0.01, override=dict(name='aspp_modules')),
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))

baseline = dict(
    type='EncoderDecoder',
    pretrained='/data_zs/data/pretrained_models/resnet101-5d3b4d8f.pth',
    backbone=dict(type='ResNet101'),   #'resnet101'
    decode_head=dict(
        type='ASPP_Classifier_V2',  # ASPP_Classifier_V2
        in_channels=2048,
        in_index=3,
        dilations=(6, 12, 18, 24),
        paddings=(6, 12, 18, 24),
        num_classes=7,
        align_corners=False,
        # init_cfg=dict(
        #     type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


model = dict(
    type='SFDAEncoderDecoder',   #SFDAEncoderDecoder  MVCSFDAEncoderDecoder
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
    feat_channels=256 + 512 + 1024 + 2048,
    model=baseline
)