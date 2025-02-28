_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/st_sfda_loveda_urban2rural.py',
    # Basic UDA Self-Training
    '../_base_/uda/sfda_deeplab_model.py',
    '../_base_/schedules/schedule_160k.py'
]


# Random Seed
# seed = 0
work_dir = r'/data_zs/output/loveDA_uda/urban2rural/st-sfda_roi-prop_loveda_urban2rural_deeplab_resnet_512x512_b4_dtu_lr6e-5_total_augv2_dt-st'  #_dtu-snd_nosample  _add-model-eval
name = 'st-sfda_roi-prop_loveda_urban2rural_deeplab_resnet_512x512_b4_dtu_lr6e-5_total_augv2'  #_dtu-snd_nosample   _add-model-eval

from configs_custom._base_.datasets.st_sfda_loveda_urban2rural import *

data_ = dict(
    pseudo_test=dict(
            type='PseudoLoveDataset',
            data_root='/data_zs/open_datasets/2021LoveDA_All/Rural/Total', #_addtest #'/data_zs/code/source_free_da/mmsegmentation_sfda/data/cityscapes/'  #'/irsa/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/postman/train_IRRG'
            img_dir='images_png',
            ann_dir='masks_png',
            pipeline=test_pipeline)
)

resume = r'/irsa/data_zs/output/loveDA_uda/urban2rural/onlysource_urban_deeplab_512x512_b4_val-totalrural-traintotal_augv2/checkpoint/iter_5000_0.494709269468064_checkpoint.pth' #r'/irsa/data_zs/output/tmp_debug/onlysource_potsdamIRRG_deeplabv2_resnet101_512x512_lr5e-4_b4_vaihingen-train-val_pmd/checkpoint/iter_9600_0.44926494457061356_checkpoint.pth'
model = dict(
    resume_file=resume,
    work_dir=work_dir,
    roi_size=(256, 256)  # equals the crop size of RandomROIBbox  (512, 1024)  (512, 512)
    )

# Optimizer Hyperparameters
optimizer_config = None
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
optimizer = dict(type='SGD', lr=2.5e-4, momentum=0.9, weight_decay=0.0005)  #0.01
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=0.9,
    min_lr=0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=25000)
# runner = dict(type='EpochBasedRunner', max_epochs=160)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=300, max_keep_ckpts=3)  #False 40000
evaluation = dict(interval=300, metric='mIoU', save_best='mIoU')  #, pre_eval=True

custom_hooks = [dict(
                    type='TCRHook',
                    data_train=data,
                    data_pseudo=data_,
                    topk_candidate=0.5,
                    update_frequency=300,  #3000
                    output_dir=work_dir
                    )]




