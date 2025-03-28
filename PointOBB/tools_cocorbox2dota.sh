#!/bin/bash
# conda activate openmmlab

cd PointOBB/
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
export CUDA_VISIBLE_DEVICES=3


# python tools_data_trans/test_cocorbox2dota.py \
#        --json_name xxx/work_dir/test_pointobb_r50_fpn_2x_dior/pseudo_obb_result_ann_1.json\
#        --txt_root ../Dataset/DIOR/Annotations/pseudo_obb_labelTxt_dior_pointobb/



nohup python tools_data_trans/test_cocorbox2dota.py \
      --json_name xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result_ann.json\
      --txt_root ../DOTAv10/data/split_ss_dota_1024_200/trainval/pseudo_obb_labelTxt_dota_pointobb/ > "logs/transform/test_cocorbox2dota_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &
