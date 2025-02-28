_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/st_sfda_loveda_rural2urban.py',
    # Basic UDA Self-Training
    '../_base_/uda/mvc_sfda_uperhead_model.py',
    '../_base_/schedules/schedule_160k.py'
]


# Random Seed
# seed = 0
work_dir = r'/irsa/data_zs/output/loveDA_uda/rural2urban/st-sfda_roi-prop_loveda_rural2urban_convnext_small_512x512_b4_dtu_lr6e-5_total_augv2'
name = 'st-sfda_roi-prop_loveda_rural2urban_convnext_small_512x512_b4_dtu_lr6e-5_total_augv2'  #_tcr-iter3000

from configs_custom._base_.datasets.st_sfda_loveda_rural2urban import *

data_ = dict(
    pseudo_test=dict(
            type='PseudoLoveDataset',
            data_root='/data_zs/open_datasets/2021LoveDA_All/Urban/Total',  #_addtest  #/data_zs/open_datasets/2021LoveDA/Train/Urban
            img_dir='images_png',
            ann_dir='masks_png',
            pipeline=test_pipeline)
)

# data_ = dict(
#     pseudo_test=dict(
#             type='PseudoCityscapesDataset',
#             data_root='/data_zs/code/source_free_da/mmsegmentation_sfda/data/cityscapes/',
#             img_dir='leftImg8bit/train',
#             ann_dir='gtFine/train',
#             pipeline=test_pipeline)
# )

resume = '/irsa/data_zs/output/loveDA_uda/rural2urban/onlysource_rural_convnext_512x512_b4_val-totalurban-traintotal_augv2/checkpoint/iter_27000_0.5121293196055698_checkpoint.pth' #'/data_zs/output/loveDA_uda/rural2urban/onlysource_rural_convnext_512x512_b4_val-urban/checkpoint/iter_14000_0.5402216528724442_checkpoint.pth'
model = dict(
    resume_file=resume,
    work_dir=work_dir,
    roi_size=(256, 256)  # equals the crop size of RandomROIBbox  (512, 1024)  (512, 512)
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
checkpoint_config = dict(by_epoch=False, interval=300, max_keep_ckpts=3)  #False 40000
evaluation = dict(interval=300, metric='mIoU', save_best='mIoU')  #, pre_eval=True
#
custom_hooks = [dict(
                    type='TCRHook',
                    data_train=data,
                    data_pseudo=data_,
                    topk_candidate=0.5,
                    update_frequency=300,  #3000  500
                    output_dir=work_dir
                    )]




