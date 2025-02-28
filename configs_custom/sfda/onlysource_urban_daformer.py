_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/urban.py',
    # Basic UDA Self-Training
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0

model = dict(
    pretrained='/data_zs/data/pretrained_models/mit_b2.pth',   #'pretrained/mit_b5.pth'
    backbone=dict(type='mit_b2', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        num_classes=7,   #7  #19
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1)),  #'CrossEntropyLoss'  FocalLoss:use_sigmoid需为True  DiceLoss  #dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )

# Optimizer Hyperparameters
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
# runner = dict(type='EpochBasedRunner', max_epochs=160)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)  #False 40000
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')  #, pre_eval=True

# evaluation = dict(interval=100, metric='mIoU')   #4000
# Meta Information for Result Analysis
work_dir = r'/data_zs/output/loveDA_uda/urban2rural/onlysource_urban_daformer_mitb2_512x512_b4_val-totalrural-traintotal_augv2'  #TODO:remeber update the dirpath in custom_base.py
name = 'onlysource_urban_daformer_mitb2_512x512_b4_val-totalrural-traintotal_augv2'

