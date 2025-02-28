_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/st_sfda_isprs_potsdamIRRG2vaihingenIRRG.py',
    # Basic UDA Self-Training
    '../_base_/uda/mvc_sfda_model.py',
    '../_base_/schedules/schedule_160k.py'
]


# Random Seed
# seed = 0
work_dir = r'/data_zs/output/tmp_debug/st-sfda_roi-prop_isprs_potsdamIRRG2vaihingenIRRG_daformer_mitb2_512x512_b4_dtu_lr6e-5_total'
name = 'st-sfda_roi-prop_isprs_potsdamIRRG2vaihingenIRRG_daformer_mitb2_512x512_b4_dtu_lr6e-5_total'  #_tcr-iter3000

from configs_custom._base_.datasets.st_sfda_isprs_potsdamIRRG2vaihingenIRRG import *

data_ = dict(
    pseudo_test=dict(
            type='PseudoISPRSDataset',
            data_root='/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/vaihingen/train',  #'/data_zs/code/source_free_da/mmsegmentation_sfda/data/cityscapes/'  #'/irsa/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/postman/train_IRRG'
            img_dir='images',
            ann_dir='labels_gray',
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

resume = '/data_zs/output/tmp_debug/onlysource_isprs_potsdamIRRG2vaihingen_daformer_mitb2_512x512_b4_pmd/checkpoint/iter_6300_0.5586992954116728_checkpoint.pth' #'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_isprs_potsdamIRRG_daformer_mitb2_512x512_b4_pmd_val-vaihingen_3004/checkpoint/iter_3500_0.5566446137951381_checkpoint.pth' #'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_rural_daformer_mitb2_512x512_b4_pmd_valurban_3004/checkpoint/best_0.5494988853356404_iter_5000.pth' #'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_gta_daformer_mitb5_512x512_b4_pmd/best_mIoU_iter_38000.pth'
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
checkpoint_config = dict(by_epoch=False, interval=300, max_keep_ckpts=20)  #False 40000
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




