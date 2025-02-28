_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/urban.py',
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0

model = dict(
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

# Optimizer Hyperparameters
# optimizer = dict(
#     type='SGD',
#     lr=2.5e-04,
#     momentum=0.9,
#     weight_decay=0.0005,
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             head=dict(lr_mult=10.0),
#             pos_block=dict(decay_mult=0.0),
#             norm=dict(decay_mult=0.0)))
# )

optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0005)  #0.01
optimizer_config = dict()

# lr_config = dict(
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-06,
#     power=0.9,
#     min_lr=0,
#     by_epoch=False)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-06, by_epoch=False)  #0.0001

runner = dict(type='IterBasedRunner', max_iters=40000)
# runner = dict(type='EpochBasedRunner', max_epochs=160)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)  #False 40000
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')  #, pre_eval=True


# evaluation = dict(interval=100, metric='mIoU')   #4000
# Meta Information for Result Analysis
work_dir = r'/data_zs/output/loveDA_uda/urban2rural/onlysource_urban_deeplab_512x512_b4_val-totalrural-traintotal_augv2'  #TODO:remeber update the dirpath in custom_base.py
name = 'onlysource_urban_deeplab_512x512_b4_val-totalrural-traintotal_augv2'

