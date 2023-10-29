_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/sfda_gta2cityscapes.py',
    # Basic UDA Self-Training
    '../_base_/uda/sfda_model.py',
    '../_base_/schedules/schedule_160k.py'
]
# Random Seed
seed = 0

resume = '/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_gta_daformer_mitb5_512x512_b4_pmd/best_mIoU_iter_38000.pth'
model = dict(
    resume_file=resume,
    )

# Optimizer Hyperparameters
optimizer_config = None
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
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)  #False 40000
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')  #, pre_eval=True

# evaluation = dict(interval=100, metric='mIoU')   #4000
# Meta Information for Result Analysis
name = 'sfda_gta_daformer_mitb5_512x512_b4_dtu'

