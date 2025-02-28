_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/sfda_isprs_potsdamIRRG2vaihingenIRRG.py',
    # Basic UDA Self-Training
    '../_base_/uda/sfda_deeplab_model.py',
    '../_base_/schedules/schedule_160k.py'
]


# Random Seed
# seed = 0
work_dir = r'/irsa/data_zs/output/tmp_debug/sfda_isprs_potsdamIRRG2vaihingen_deeplabv2_512x512_b4_lr2.5e-4_neg-loass-w1_resume9600_mixlabel'  #_dtu-snd_nosample  _add-model-eval
name = '/irsa/data_zs/output/tmp_debug/sfda_isprs_potsdamIRRG2vaihingen_deeplabv2_512x512_b4_lr2.5e-4_neg-loass-w1_resume9600_mixlabel'  #_dtu-snd_nosample   _add-model-eval

from configs_custom._base_.datasets.sfda_isprs_potsdamIRRG2vaihingenIRRG import *


data_ = dict(
    pseudo_test=dict(
            type='PseudoISPRSDataset',
            data_root='/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/vaihingen/train',  #'/data_zs/code/source_free_da/mmsegmentation_sfda/data/cityscapes/'  #'/irsa/data_zs/data/expert_datasets/isprs_2d_semantic_splitbyfile/postman/train_IRRG'
            img_dir='images',
            ann_dir='labels_gray',
            pipeline=test_pipeline)
)

resume = r'/irsa/data_zs/output/tmp_debug/onlysource_potsdamIRRG_deeplabv2_resnet101_512x512_lr5e-4_b4_vaihingen-train-val_pmd/checkpoint/iter_9600_0.44926494457061356_checkpoint.pth' #r'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/potsdamIRRG2vaihingen/onlysource_potsdamIRRG_deeplabv2_resnet101_512x512_lr5e-4_b4_pmd/checkpoint/iter_11000_0.4420795725077639_checkpoint.pth' #'/irsa/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/potsdamIRRG2vaihingen/onlysource_potsdamIRRG_deeplabv2_resnet101_512x512_b4_pmd/checkpoint/iter_7500_0.42894648486939363_checkpoint.pth' #'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_rural_daformer_mitb2_512x512_b4_pmd_valurban_3004/checkpoint/best_0.5494988853356404_iter_5000.pth' #'/data_zs/code/source_free_da/mmsegmentation_sfda/work_dirs/onlysource_gta_daformer_mitb5_512x512_b4_pmd/best_mIoU_iter_38000.pth'
model = dict(
    resume_file=resume,
    work_dir=work_dir
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
    # warmup='linear',
    # warmup_iters=0,
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
                    update_frequency=500,  #3000
                    output_dir=work_dir
                    )]




