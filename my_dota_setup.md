# My Setup

## Environment Setup

```sh
/home/shardul.junagade/miniconda3/bin/conda create -n open-mmlab python=3.8 -y

source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
# source /home/shardul.junagade/miniconda3/bin/activate pointobb


conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip uninstall pycocotools
pip install -r requirements/build.txt
pip install -v -e . --user
chmod +x tools/dist_train.sh

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mmengine
mim install mmengine

pip install shapely
```

### Remove Environment

```sh
conda deactivate
/home/shardul.junagade/miniconda3/bin/conda env remove -n open-mmlab
```


## DOTA Dataset Preparation

### Split DOTA dataset

[Dataset Preparation](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/recommended_topics/dataset_preparation.md)

Script `tools/dataset_converters/dota/dota_split.py` can split and prepare DOTA dataset.

```shell
python tools/dataset_converters/dota/dota_split.py \
    [--split-config ${SPLIT_CONFIG}] \
    [--data-root ${DATA_ROOT}] \
    [--out-dir ${OUT_DIR}] \
    [--ann-subdir ${ANN_SUBDIR}] \
    [--phase ${DATASET_PHASE}] \
    [--nproc ${NPROC}] \
    [--save-ext ${SAVE_EXT}] \
    [--overwrite]
```

shapely is required, please install shapely first by pip install shapely.

**Description of all parameters：**

- `--split-config`: The split config for image slicing.
- `--data-root`: Root dir of DOTA dataset.
- `--out-dir`: Output dir for split result.
- `--ann-subdir`: The subdir name for annotation. Defaults to labelTxt-v1.0.
- `--phase`: Phase of the data set to be prepared. Defaults to trainval test
- `--nproc`: Number of processes. Defaults to 8.
- `--save-ext`: Extension of the saved image. Defaults to png
- `--overwrite`: Whether to allow overwrite if annotation folder exist.

Based on the configuration in the DOTA paper, we provide two commonly used split config.

- `./split_config/single_scale.json` means single-scale split.
- `./split_config/multi_scale.json` means multi-scale split.


#### Example for DOTA V1.0
Split DOTA V1.0 dataset into trainval and test set with single scale.
```sh
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
python tools/dataset_converters/dota/dota_split.py \
    tools/dataset_converters/dota/split_config/single_scale.json \
    "../data/DOTA_V1.0" \
    "../DOTAv10/data/split_ss_dota_1024_200" \
    --nproc 40
```

Splitting a subset of DOTA V1.0 dataset into train val and test set with single scale.
```sh
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
python tools/dataset_converters/dota/dota_split.py \
    tools/dataset_converters/dota/split_config/single_scale.json \
    "../data/dota_subset" \
    "../DOTAv10/data_subset/split_ss_dota_1024_200" \
    --phase "train" "val" "test" \
    --nproc 30
```

## Convert DOTA to COCO format
```sh
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
# Generate 'obb+pt' Format:
python tools_data_trans/test_dota2dota_obbpt_viaobb.py
# Generate COCO Format:
python tools_data_trans/test_dota2coco_P2B_obb-pt.py
```



## Train
To train the model, follow these steps:
```sh
cd PointOBB
# train with single GPU, note adjust learning rate or batch size accordingly
python tools/train.py \
    --config configs2/pointobb/pointobb_r50_fpn_2x_dota10.py \
    --work-dir xxx/work_dir_subset/pointobb_r50_fpn_2x_dota \
    --cfg-options evaluation.save_result_file='xxx/work_dir_subset/pointobb_r50_fpn_2x_dota_dist/pseudo_obb_result.json'

# train with multiple GPUs
sh train_p_dist.sh
```



## Inference
To inference (generate pseudo obb label), follow these steps:
```sh
# obtain COCO format pseudo label for the training set 
sh test_p.sh
# convert COCO format to DOTA format 
sh tools_cocorbox2dota.sh
# train standard oriented object detectors 
# Please use algorithms in mmrotate (https://github.com/open-mmlab/mmrotate)
```




## Resume Training

Yes, you **can** continue training from a certain epoch.  

Your `train.py` script supports resuming training via the `--resume-from` argument. In your config file, you also have a commented-out `load_from` option, which allows loading a specific checkpoint.  

### How to Resume Training  
1. **Use the `--resume-from` Argument:**  
   Run the script with:  
   ```bash
   python train.py --config your_config.py --resume-from path/to/checkpoint.pth
   ```
   This will continue training from the specified checkpoint.  

2. **Manually Set `resume_from` in Config:**  
   If you prefer, you can uncomment and set `resume_from` in your config file:  
   ```python
   resume_from = '../TOV_mmdetection_cache/work_dir/dota/epoch_22.pth'
   ```
   Then run:  
   ```bash
   python train.py --config your_config.py
   ```

### Difference Between `resume_from` and `load_from`
- **`resume_from`** → Continues training from the exact state (including optimizer state and learning rate).  
- **`load_from`** → Loads model weights but starts training from epoch 0 (useful for fine-tuning).  