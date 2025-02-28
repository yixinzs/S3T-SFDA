_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/sfda_dfctrain2val.py',
    # Basic UDA Self-Training
    '../_base_/uda/sfda_model.py',
    '../_base_/schedules/schedule_160k.py'
]


# Random Seed
# seed = 0

# from configs_custom._base_.datasets.sfda_gta2cityscapes import *


# data_ = dict(
#     pseudo_test=dict(
#             type='PseudoCityscapesDataset',
#             data_root='/data_zs/code/source_free_da/mmsegmentation_sfda/data/cityscapes/',
#             img_dir='leftImg8bit/train',
#             ann_dir='gtFine/train',
#             pipeline=test_pipeline)
# )

resume = '/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_dfc22_daformer_mitb2_512x512_b4_pmd_val-Clermont-Ferrand_3004/checkpoint/iter_17000_0.3471175714059836_checkpoint.pth' #'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_rural_daformer_mitb2_512x512_b4_pmd_valurban_3004/checkpoint/best_0.5494988853356404_iter_5000.pth' #'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_gta_daformer_mitb5_512x512_b4_pmd/best_mIoU_iter_38000.pth'
model = dict(
    resume_file=resume,
    )

# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    type='AdamW',
    lr=6e-05,  #6e-05
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
runner = dict(type='IterBasedRunner', max_iters=25000)
# runner = dict(type='EpochBasedRunner', max_epochs=160)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)  #False 40000
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')  #, pre_eval=True

work_dir = r'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/sfda_dfc22train2val_daformer_mitb2_512x512_b4_dtu_lr6e-5_neg-loass-w1_resume17000_train-val-same_3004'
name = 'sfda_dfc22train2val_daformer_mitb2_512x512_b4_dtu_lr6e-5_neg-loass-w1_resume17000_train-val-same_3004'  #_tcr-iter3000

# custom_hooks = [dict(
#                     type='TCRHook',
#                     data_train=data,
#                     data_pseudo=data_,
#                     topk_candidate=0.5,
#                     update_frequency=3000,
#                     output_dir=work_dir
#                     )]




