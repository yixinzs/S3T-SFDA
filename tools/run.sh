#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#single model
config_dir='/workspace/mmsegmentation_rsipac/configs_custom/rsipac/'
work_dir='/workspace/checkpoint/'
show_dir='/output'

name='Softce_fda-cutreplace-classmix14_448s1.25p12_swa_fold0'


$PYTHON /workspace/mmsegmentation_rsipac/tools/train.py --config $config_dir'upernet_convnext_tiny_rsipac_s1.5_ms_single.py' \
                 --work-dir $work_dir$name
#                 --gpu-id 0

config_path=$work_dir$name'/upernet_convnext_tiny_rsipac_s1.5_ms_single.py'  #upernet_convnext_tiny_rsipac_s1.5_ms_single.p
model_path=$work_dir$name'/swa_final.pth'

$PYTHON /workspace/mmsegmentation_rsipac/tools/ensemble_logistpost.py --config $config_path \
        --checkpoint $model_path \
        --work-dir $show_dir
#          --gpu-id 0



#$PYTHON ensemble_logistpost.py --config /data_zs/output/rsipac_semi/checkpoint/Softce_fda-cutreplace-classmix14_448s1.25p12_swa_fold0/upernet_convnext_tiny_rsipac_s1.5_ms.py \
#        --checkpoint /data_zs/output/rsipac_semi/checkpoint/Softce_fda-cutreplace-classmix14_448s1.25p12_swa_fold0/swa_final.pth \
#        --work-dir /data_zs/output/rsipac_semi/test/Softce_fda-cutreplace-classmix14_448s1.25p12_swa_fold0_v2 \
#          --gpu-id 1




