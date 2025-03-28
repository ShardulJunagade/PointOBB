
cd PointOBB/
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# Train with single GPU

nohup python tools/train.py \
    --config configs2/pointobb/pointobb_r50_fpn_2x_dota10.py \
    --work-dir xxx/work_dir_subset/pointobb_r50_fpn_2x_dota \
    --cfg-options evaluation.save_result_file='xxx/work_dir_subset/pointobb_r50_fpn_2x_dota/pseudo_obb_result.json' >  "logs/train/train_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &



# Inference
# add checkpoint path in configs2/pointobb/pointobb_r50_fpn_2x_dota10_inf.py
nohup python tools/train.py\
    --config configs2/pointobb/pointobb_r50_fpn_2x_dota10_inf.py \
    --work-dir xxx/work_dir_epoch_23/test_pointobb_r50_fpn_2x_dota/ \
    --cfg-options evaluation.save_result_file='xxx/work_dir_epoch_23/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result_test.json' > "logs/inference/inference_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &



# Transform detection results into COCO-style annotations

nohup python exp/tools/result2ann_obb.py \
    --ori_ann "../DOTAv10/data/split_ss_dota_1024_200/test/test_1024_P2Bfmt_dotav10_rbox.json" \
    --det_file "xxx/work_dir_epoch_23/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result_test.json" \
    --save_ann "xxx/work_dir_epoch_23/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result_test_ann.json" > "logs/transform/result2ann_obb_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &


# Convert COCO format to DOTA format

nohup python tools_data_trans/test_cocorbox2dota.py \
    --json_name xxx/work_dir_epoch_23/test_pointobb_r50_fpn_2x_dota/pseudo_obb_result_test_ann.json\
    --txt_root ../DOTAv10/data/split_ss_dota_1024_200/test/pseudo_obb_labelTxt_dota_pointobb/ > "logs/transform/test_cocorbox2dota_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

