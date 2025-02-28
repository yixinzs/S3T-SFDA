_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/isprs_vaihingen.py',
    # Basic UDA Self-Training
    '../_base_/models/upernet_convnext.py',
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0

# model = dict(
#     pretrained='/data_zs/data/pretrained_models/convnext_tiny_22k_224.pth',
#     backbone=dict(
#         type='ConvNeXt',
#         in_chans=3,
#         depths=[3, 3, 9, 3],
#         dims=[96, 192, 384, 768],
#         drop_path_rate=0.4,
#         layer_scale_init_value=1.0,
#         out_indices=[0, 1, 2, 3],
#     ),
#     decode_head=dict(
#         in_channels=[96, 192, 384, 768],
#         num_classes=7,
#         loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)],
#     ),
#     auxiliary_head=dict(
#         in_channels=384,
#         num_classes=7,
#         loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)],
#     ),
# )

model = dict(
    pretrained='/data_zs/data/pretrained_models/convnext_small_22k_224.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=6,
        # loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        #              dict(type='LovaszLoss', per_image=False, reduction='none', loss_weight=1.0)],
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)],
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=6,
        # loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        #              dict(type='LovaszLoss', per_image=False, reduction='none', loss_weight=0.4)],
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)],
    ),
)

# optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
#                  lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg={'decay_rate': 0.9,
#                                 'decay_type': 'stage_wise',
#                                 'num_layers': 6})

optimizer = dict(
    type='AdamW',
    lr=2e-04,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)),
            head=dict(lr_mult=10.0)
            ))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=25000)
# runner = dict(type='EpochBasedRunner', max_epochs=160)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=25000, max_keep_ckpts=1)  #False 40000
evaluation = dict(interval=300, metric='mIoU', save_best='mIoU')  #, pre_eval=True

# evaluation = dict(interval=100, metric='mIoU')   #4000
# Meta Information for Result Analysis
work_dir = r'/data_zs/output/loveDA_uda/vaihingen2potsdamIRRG/onlysource_vaihingen2potsdamIRRG_convnext_small_512x512_b4'  #TODO:remeber update the dirpath in custom_base.py
name = 'onlysource_vaihingen2potsdamIRRG_convnext_small_512x512_b4'

