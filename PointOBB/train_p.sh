
#!/bin/bash
# conda activate openmmlab

source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# DOTA
nohup python tools/train.py \
    --config configs2/pointobb/pointobb_r50_fpn_2x_dota10.py \
    --work-dir xxx/work_dir_subset/pointobb_r50_fpn_2x_dota \
    --cfg-options evaluation.save_result_file='xxx/work_dir_subset/pointobb_r50_fpn_2x_dota/pseudo_obb_result.json' >  "logs/train/train_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &
