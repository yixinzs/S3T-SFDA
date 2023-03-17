#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#ensemble model path:https://blog.51cto.com/welcomeweb/2163585
config_dir='/workspace/mmsegmentation_rsipac/configs_custom/rsipac/'
work_dir='/workspace/checkpoint/'
show_dir='/output'

name_1='Softce_fda-cutreplace-classmix14_448s1.25p8_swa_fold0'
name_2='Softce_fda-cutreplace-classmix14_448s1.25p8_swa_fold10'

$PYTHON /workspace/mmsegmentation_rsipac/tools/train.py --config $config_dir'upernet_convnext_tiny_rsipac_s1.5_ms.py'  \
                 --work-dir $work_dir$name_1 &
$PYTHON /workspace/mmsegmentation_rsipac/tools/train.py --config $config_dir'upernet_convnext_tiny_rsipac_s1.5_ms_debug.py'  \
                 --work-dir $work_dir$name_2

config_path_1=$work_dir$name_1'/upernet_convnext_tiny_rsipac_s1.5_ms.py'
config_path_2=$work_dir$name_2'/upernet_convnext_tiny_rsipac_s1.5_ms_debug.py'

model_path_1=$work_dir$name_1'/swa_final.pth'
model_path_2=$work_dir$name_2'/swa_final.pth'

config_path_list=$config_path_1'@'$config_path_2
model_path_list=$model_path_1'@'$model_path_2

$PYTHON /workspace/mmsegmentation_rsipac/tools/ensemble_logistpost_v2.py --config $config_path_list \
        --checkpoint $model_path_list \
        --work-dir $show_dir




