# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation
Please follow the OpenPCDet [tutorial](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to 
prepare needed datasets.

## Training & Testing
[//]: # ( TODO)
### Step 1: Train a teacher model (CP-voxel as example)
```shell
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/
```

### Step 2: Distillation
```shell
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/ --teacher_ckpt ${TEACHER_CKPT} \
    --pretrained_model ${TEACHER_CKPT}
```

## Calculate Efficiency Metrics

### Prepare
Make sure you have installed our customized Thop as [INSTALL.md](./INSTALL.md).
To calculate the Flops and Acts for spconv-based models, you also need to replace original `conv.py` in spconv
with our modified one.
```shell
# replace our modified conv file for 
# make sure your spconv is at least 2.1.20
cp extra_files/conv.py ${CONDA_PATH}/envs/${ENV_NAME}/lib/${PYTHON_VERSION}/site-packages/spconv/pytorch/
```

### Command
```shell
# Take Waymo as an example
# This command have to be executed on single gpu only
python test.py --cfg_file ${CONFIG_PATH} --batch_size 1 --ckpt ${CKPT_PATH} --infer_time --cal_params \
  --set DATA_CONFIG.DATA_SPLIT.test infer_time DATA_CONFIG.SAMPLED_INTERAVL.test 2
``` 
