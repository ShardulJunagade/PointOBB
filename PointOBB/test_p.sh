#!/bin/bash
# conda activate mmdet20_2

cd PointOBB/
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
export CUDA_VISIBLE_DEVICES=3


### Inference
# ## way1
# Note: This method gave a logging error when I tried to run it. Some Lis t/ int error while setting max_epochs = 0 in terminal.
nohup python tools/train.py \
    --config configs2/pointobb/pointobb_r50_fpn_2x_dota10_inf.py \
    --work-dir xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/ \
    --cfg-options evaluation.save_result_file='xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result.json', \
    evaluation.do_first_eval=True, \
    runner.max_epochs=0, \
    load_from='xxx/work_dir_epoch_14/pointobb_r50_fpn_2x_dota/epoch_13.pth' >  "logs_inference/inference_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

## way2 
# Note: You need to uncomment the Inference section in configs2\pointobb\pointobb_r50_fpn_2x_dior.py and run the following command: 
nohup python tools/train.py\
       --config configs2/pointobb/pointobb_r50_fpn_2x_dota10_inf.py \
       --work-dir xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/ \
       --cfg-options evaluation.save_result_file='xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result.json' > "logs/inference/inference_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &





### Transform Fmt

# # Transform detection results into COCO-style annotations
# python exp/tools/result2ann_obb.py \
#        ../DOTAv10/data/split_ss_dota_1024_200/trainval/trainval_1024_P2Bfmt_dotav10_rbox.json \
#        xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result.json \
#        xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result_ann_1.json

# Transform detection results into COCO-style annotations
nohup python exp/tools/result2ann_obb.py \
    --ori_ann "../DOTAv10/data/split_ss_dota_1024_200/trainval/trainval_1024_P2Bfmt_dotav10_rbox.json" \
    --det_file "xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result.json" \
    --save_ann "xxx/work_dir_epoch_14/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result_ann.json" > "logs/transform/result2ann_obb_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &