
#!/bin/bash
# conda activate openmmlab
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab

export CUDA_VISIBLE_DEVICES=1,2
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# # DIOR
# WORK_DIR='work_dirs/pointobb_r50_fpn_2x_dior/$(date +%Y-%m-%d_%H-%M-%S)/'
# nohup tools/dist_train.sh --config configs2/pointobb/pointobb_r50_fpn_2x_dior.py 2 \
#                     --work-dir ${WORK_DIR}\
#                     --cfg-options evaluation.save_result_file=${WORK_DIR}'pseudo_obb_result.json' >  "logs/train/dist_train_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

# DOTA
WORK_DIR="work_dirs/pointobb_r50_fpn_2x_dota10/$(date +%Y-%m-%d_%H-%M-%S)/"
nohup tools/dist_train.sh configs2/pointobb/pointobb_r50_fpn_2x_dota10.py 2 \
                    --work-dir ${WORK_DIR}\
                    --cfg-options evaluation.save_result_file=${WORK_DIR}'pseudo_obb_result.json' >  "logs/train/dist_train_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &
