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


## Remove Environment

```sh
conda deactivate
/home/shardul.junagade/miniconda3/bin/conda env remove -n open-mmlab
```




## Split DOTA dataset

[Dataset Preparation](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/recommended_topics/dataset_preparation.md)

Script `tools/dataset_converters/dota/dota_split.py` can split and prepare DOTA dataset.

```shell
python tools/dataset_converters/dota/dota_split.py \
    [--splt-config ${SPLIT_CONFIG}] \
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


### Example for DOTA V1.0
Split DOTA V1.0 dataset into trainval and test set with single scale.
```sh
source /home/shardul.junagade/miniconda3/bin/activate open-mmlab
python tools/dataset_converters/dota/dota_split.py \
    tools/dataset_converters/dota/split_config/single_scale.json \
    "../data/DOTA_V1.0" \
    "../data/split_ss_dota_1024_200" \
    --nproc 40
```



